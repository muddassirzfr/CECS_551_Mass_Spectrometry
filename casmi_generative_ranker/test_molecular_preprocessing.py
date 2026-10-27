import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import pandas as pd
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize

import molecular_preprocessing as prep
import pipeline


class MolecularPreprocessingTests(unittest.TestCase):
    def test_matches_scoring_and_equivalent_forms(self):
        pairs = [("CC(=O)C", "C=C(O)C"),
                 ("C[C@H](O)C(=O)O", "C[C@@H](O)C(=O)O"),
                 ("CCO", "OCC")]
        for left, right in pairs:
            a, b = prep.normalize_smiles(left), prep.normalize_smiles(right)
            self.assertEqual(a["status"], "valid")
            self.assertEqual(a["competition_identity"], b["competition_identity"])
            self.assertEqual(a["candidate_target"], b["candidate_target"])
            for smiles, result in ((left, a), (right, b)):
                mol = rdMolStandardize.TautomerEnumerator().Canonicalize(Chem.MolFromSmiles(smiles))
                self.assertEqual(result["competition_identity"], Chem.MolToInchiKey(mol)[:14])
        self.assertNotEqual(prep.normalize_smiles("CCO")["competition_identity"],
                            prep.normalize_smiles("CCN")["competition_identity"])
        # Do not collapse salts through an unrequested fragment-removal policy.
        self.assertNotEqual(prep.normalize_smiles("CCO.[Na+]")["competition_identity"],
                            prep.normalize_smiles("CCO")["competition_identity"])

    def test_invalids_and_version_guard(self):
        for value in (None, "", " ", "C1CC", "CCO molecule_name"):
            self.assertEqual(prep.normalize_smiles(value)["status"], "invalid")
        with patch.object(prep.rdBase, "rdkitVersion", "2026.09.1"):
            with self.assertRaisesRegex(RuntimeError, "requires RDKit"):
                prep.require_competition_rdkit()

    def test_manifest_all_spectra_share_identity_and_target(self):
        smiles = ["CC(=O)C", "C=C(O)C", "CC(=O)C", "C1CC", None,
                  "C[C@H](O)C(=O)O", "C[C@@H](O)C(=O)O"]
        smiles += ["C" * n for n in range(1, 40)]
        data = pd.DataFrame({"normalized_smiles": smiles,
                             "inchikey14": [f"source-{i}" for i in range(len(smiles))]})
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            path = directory / "train.parquet"
            data.to_parquet(path)
            output = directory / "results"
            report = prep.build_preprocessing(path, output, holdout=.2, val_frac=.3, workers=1)
            mapping, loaded_report = prep.load_preprocessing(output)
            frame = prep.apply_preprocessing(data, mapping)
            self.assertEqual(report, loaded_report)
            self.assertEqual(report["rows_scanned"], len(data))
            self.assertEqual(report["invalid_spectra"], 2)
            self.assertEqual(len(frame), len(data) - 2)
            self.assertTrue((frame.groupby("competition_identity").split.nunique() == 1).all())
            self.assertTrue((frame.groupby("competition_identity").normalized_smiles.nunique() == 1).all())
            self.assertEqual(frame.iloc[0].competition_identity, frame.iloc[1].competition_identity)
            self.assertEqual(set(frame.split), {"Training", "Validation", "Ranker holdout"})
            assignments = pd.read_parquet(output / "spectrum_partitions.parquet")
            self.assertEqual(assignments.source_row.tolist(), list(range(len(data))))
            self.assertEqual(int(assignments.split.isna().sum()), 2)
            errors = [json.loads(line) for line in (output / "invalid_molecules.jsonl").read_text().splitlines()]
            self.assertEqual(len(errors), 2)
            self.assertEqual(frame.iloc[0].source_inchikey14, "source-0")
            with self.assertRaisesRegex(ValueError, "missing from preprocessing map"):
                prep.apply_preprocessing(pd.DataFrame({"normalized_smiles": ["BrCCBr"]}), mapping)
            # Rerunning from the cache must preserve the target and partition map.
            before = mapping.sort_values("input_smiles").reset_index(drop=True)
            prep.build_preprocessing(path, output, holdout=.2, val_frac=.3, workers=1)
            after, _ = prep.load_preprocessing(output)
            pd.testing.assert_frame_equal(before, after.sort_values("input_smiles").reset_index(drop=True))

    def test_generated_tautomers_deduplicate_and_label_positive(self):
        with tempfile.TemporaryDirectory() as directory:
            tokenizer = pipeline.train_tokenizer(pd.DataFrame({
                "normalized_smiles": ["CC(=O)C", "C=C(O)C", "CCN"]}),
                Path(directory) / "tokenizer.json", 64)
            class Sampler:
                def sample(self, batch, *args):
                    return [[(tokenizer.encode(s).ids, -1., len(tokenizer.encode(s).ids) - 1)
                             for s in ("CC(=O)C", "C=C(O)C", "CCN")]]
            data = pd.DataFrame({"molecule_id": ["query"], "ms2_mzs": [[40., 60.]],
                                 "ms2_normalized_intensities": [[.5, 1.]],
                                 "precursor_mz": [59.049], "adduct": ["[M+H]+"]})
            args = SimpleNamespace(infer_batch=1, workers=0, device="cpu", bf16=False,
                                   samples=3, max_tokens=100, temperature=1., sample_chunk=3,
                                   progress_every=0)
            groups = pipeline.build_candidates(data, Sampler(), tokenizer, 8, args)
            self.assertEqual(len(groups["query"]["candidates"]), 2)
            truth = {"query": prep.normalize_smiles("C=C(O)C")["competition_identity"]}
            frame = pipeline.candidate_frame(groups, truth)
            self.assertEqual(frame.label.sum(), 1)
            self.assertAlmostEqual(frame.loc[frame.label == 1, "candidate_frequency"].iloc[0], 2 / 3)
            with self.assertRaisesRegex(ValueError, "before splitting"):
                pipeline.split_structures(pd.DataFrame({"inchikey14": ["raw"]}), 42, .2)

    def test_training_uses_manifest_and_checkpoints_require_policy(self):
        smiles = ["C" * n for n in range(1, 80)] + ["CC(=O)C", "C=C(O)C", "C1CC"]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = pd.DataFrame({"normalized_smiles": smiles,
                                 "inchikey14": ["raw"] * len(smiles),
                                 "ms2_mzs": [[20., 30.]] * len(smiles),
                                 "ms2_normalized_intensities": [[.5, 1.]] * len(smiles),
                                 "precursor_mz": [100.] * len(smiles),
                                 "adduct": ["[M+H]+"] * len(smiles)})
            source = root / "train.parquet"
            data.to_parquet(source)
            prep.build_preprocessing(source, root / "prep", holdout=.2, val_frac=.2, workers=1)
            command = [sys.executable, str(Path(pipeline.__file__)), "train", "--train", str(source),
                       "--preprocessing", str(root / "prep"), "--output", str(root / "model"),
                       "--device", "cpu", "--epochs", "1", "--d", "16", "--heads", "2",
                       "--enc-layers", "1", "--dec-layers", "1", "--ff", "32", "--batch", "16",
                       "--accum", "1", "--max-len", "128", "--max-peaks", "8", "--eval-every", "0"]
            subprocess.run(command, check=True, capture_output=True, text=True)
            pipeline.load_generator(root / "model", "cpu")
            frames = [pd.read_parquet(root / "model" / name) for name in (
                "generator_train.parquet", "generator_val.parquet", "ranker_holdout.parquet")]
            identities = [set(frame.competition_identity) for frame in frames]
            self.assertFalse(identities[0] & identities[1] or identities[0] & identities[2] or identities[1] & identities[2])
            self.assertEqual(sum(map(len, frames)), len(smiles) - 1)
            # Even when all supplied keys are identical, recomputed identities
            # govern partitions. A valid old checkpoint must not bypass the gate.
            checkpoint_path = root / "model" / "generator.pt"
            checkpoint = pipeline.torch.load(checkpoint_path, weights_only=False)
            del checkpoint["normalization_policy"]
            pipeline.torch.save(checkpoint, checkpoint_path)
            with self.assertRaisesRegex(ValueError, "normalization policy"):
                pipeline.load_generator(root / "model", "cpu")


if __name__ == "__main__":
    unittest.main()
