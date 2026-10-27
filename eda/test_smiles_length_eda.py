"""Regression checks for leakage, molecule weighting, and length accounting."""
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from smiles_length_eda import (
    collect_molecules, fit_training_tokenizer, measure_molecules,
    split_for_key, summarize, validate_smiles,
)


class SmilesLengthTests(unittest.TestCase):
    def test_duplicate_measurements_and_missing_values(self):
        data = pd.DataFrame({
            "inchikey14": ["a"] * 10 + ["b", "b", None, "c"],
            "normalized_smiles": ["CCO"] * 10 + ["O", "CC", "N", None],
        })
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train.parquet"
            data.to_parquet(path)
            frame, audit = collect_molecules(path, batch_size=3)
        self.assertEqual(len(frame), 3)
        self.assertEqual(frame.set_index("identity").loc["b", "smiles"], "CC")
        self.assertEqual(audit["rows_scanned"], 14)
        self.assertEqual(audit["rows_missing_identity"], 1)
        self.assertEqual(audit["identities_with_multiple_smiles"], 1)
        self.assertEqual(frame.set_index("identity").loc["c", "smiles"], None)

    def test_training_only_vocab_and_untruncated_roundtrip(self):
        tokenizer = fit_training_tokenizer(["CCO", "CC", "O"], 32)
        self.assertIsNone(tokenizer.token_to_id("N"))
        self.assertEqual(tokenizer.encode("N").ids, [1, 3, 2])
        long_smiles = "C" * 3000
        encoding = tokenizer.encode(long_smiles)
        self.assertGreater(len(encoding.ids), 512)
        self.assertEqual(encoding.ids[0], 1)
        self.assertEqual(encoding.ids[-1], 2)
        self.assertEqual(tokenizer.decode(encoding.ids).replace(" ", ""), long_smiles)
        frame = pd.DataFrame({"split": ["Training", "Validation"],
                              "smiles": ["CCO", "N"], "status": ["valid", "valid"]})
        measured = measure_molecules(frame, tokenizer, batch_size=1)
        validation = summarize(measured).set_index("split").loc["Validation"]
        self.assertEqual(validation.unknown_token_pct, 100)
        self.assertEqual(validation.molecules_with_unknown_pct, 100)
        self.assertEqual(validation["median"], 3)

    def test_limits_are_strict_and_invalids_have_separate_denominator(self):
        frame = pd.DataFrame({
            "split": ["Training"] * 4 + ["Validation"],
            "status": ["valid"] * 3 + ["invalid_smiles", "missing_or_empty"],
            "token_length": [128, 129, 512, pd.NA, pd.NA],
            "content_tokens": [126, 127, 510, pd.NA, pd.NA],
            "unknown_tokens": [0, 1, 0, pd.NA, pd.NA],
            "roundtrip_ok": pd.array([True, False, True, None, None], dtype="boolean"),
        })
        result = summarize(frame).set_index("split")
        self.assertAlmostEqual(result.loc["Training", "exceed_128_pct"], 200 / 3)
        self.assertEqual(result.loc["Training", "exceed_512_pct"], 0)
        self.assertEqual(result.loc["Training", "invalid_or_missing_pct"], 25)
        self.assertTrue(pd.isna(result.loc["Validation", "median"]))

    def test_smiles_validation_and_stable_group_assignment(self):
        for smiles in (None, "", " "):
            self.assertEqual(validate_smiles(smiles), "missing_or_empty")
        for smiles in ("C1CC", "CCO name", "not_smiles"):
            self.assertEqual(validate_smiles(smiles), "invalid_smiles")
        for smiles in ("CCO", "[Na+].[Cl-]", "C[C@H](N)C(=O)O"):
            self.assertEqual(validate_smiles(smiles), "valid")
        self.assertEqual(split_for_key("same-key"), split_for_key("same-key"))


if __name__ == "__main__":
    unittest.main()
