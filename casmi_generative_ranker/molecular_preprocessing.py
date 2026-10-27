"""Competition molecular targets, identities, audit logs, and grouped partitions.

No spectra are decoded. Run this module directly with --help, or use
`pipeline.py preprocess`. The RDKit version is deliberately pinned to scoring.
"""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache
import hashlib
import json
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from rdkit import Chem, rdBase
from rdkit.Chem.MolStandardize import rdMolStandardize

RDKIT_VERSION = "2026.03.3"
POLICY = "casmi26-tautomer-inchikey14-v1"
SOURCE_URL = "https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/overview"


def require_competition_rdkit():
    if rdBase.rdkitVersion != RDKIT_VERSION:
        raise RuntimeError(f"Competition normalization requires RDKit {RDKIT_VERSION}; "
                           f"found {rdBase.rdkitVersion}. Install rdkit==2026.3.3.")


@lru_cache(maxsize=65536)
def normalize_smiles(smiles):
    """Shared training/candidate policy. Return strings, never a mutable Mol.

    Identity: default RDKit canonical tautomer -> first InChIKey block.
    Target: canonical SMILES of that tautomer with stereochemistry removed;
    preserve exact input in rare enumeration-capped identity-unstable cases.
    Do not uncharge, remove fragments, or apply extra chemical cleanup: those
    operations would change the documented competition matching policy.
    """
    require_competition_rdkit()
    result = dict(input_smiles=smiles, status="invalid", error=None,
                  competition_identity=None, candidate_target=None,
                  tautomer_status="not_attempted", target_fallback=None)
    if not isinstance(smiles, str) or not smiles.strip():
        result["error"] = "missing_or_empty_smiles"
        return result
    try:
        with rdBase.BlockLogs():
            params = Chem.SmilesParserParams()
            params.parseName = False
            mol = Chem.MolFromSmiles(smiles, params)
            if mol is None or mol.GetNumAtoms() == 0:
                result["error"] = "invalid_smiles"
                return result
            enumerator = rdMolStandardize.TautomerEnumerator()
            # Canonicalize uses this same enumeration and default scoring. Keeping
            # the result also exposes limits instead of silently hiding warnings.
            enumeration = enumerator.Enumerate(mol)
            result["tautomer_status"] = str(enumeration.status)
            canonical = enumerator.PickCanonical(enumeration)
            key = Chem.MolToInchiKey(canonical)
            if not key or len(key.split("-")[0]) != 14:
                result["error"] = "inchi_generation_failed"
                return result
            identity = key.split("-")[0]
            # Canonicalize identity BEFORE removing stereo to match scoring order.
            Chem.RemoveStereochemistry(canonical)
            target = Chem.MolToSmiles(canonical, canonical=True, isomericSmiles=True)
            # Verify serialized targets reproduce the same scoring identity.
            check = Chem.MolFromSmiles(target)
            # A completed enumeration already chose the canonical tautomer; only
            # capped enumeration needs another pass to detect non-idempotence.
            if enumeration.status != rdMolStandardize.TautomerEnumeratorStatus.Completed:
                check = enumerator.Canonicalize(check)
            if Chem.MolToInchiKey(check).split("-")[0] != identity:
                # Bounded tautomer enumeration can be atom-order dependent. The
                # exact supplied string reproduces the identity computed above;
                # preserve it instead of discarding a chemically valid molecule.
                target = smiles
                result["target_fallback"] = "original_smiles_due_to_tautomer_limit"
        result.update(status="valid", competition_identity=identity, candidate_target=target)
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def canonical_candidate(smiles):
    normalized = normalize_smiles(smiles)
    if normalized["status"] != "valid":
        return None
    target = normalized["candidate_target"]
    return target, normalized["competition_identity"], Chem.MolFromSmiles(target)


def normalize_batch(strings):
    return [normalize_smiles(smiles) for smiles in strings]


def partition_identity(identity, seed=42, holdout=0.1, val_frac=0.02):
    """Assign ALL occurrences using only competition identity, independently of rows."""
    if not 0 < holdout < 1 or not 0 < val_frac < 1:
        raise ValueError("holdout and val_frac must be between zero and one")
    def fraction(salt):
        digest = hashlib.blake2b(f"{salt}:{identity}".encode(), digest_size=8).digest()
        return int.from_bytes(digest, "big") / 2**64
    if fraction(seed) < holdout:
        return "Ranker holdout"
    return "Validation" if fraction(seed + 1) < val_frac else "Training"


def scan_source_pairs(path, batch_size=8192):
    counts = Counter()
    scanned = 0
    with pq.ParquetFile(path, pre_buffer=False) as source:
        columns = ["normalized_smiles"]
        if "inchikey14" in source.schema_arrow.names:
            columns.append("inchikey14")
        for batch in source.iter_batches(batch_size=batch_size, columns=columns, use_threads=False):
            strings = batch.column(0).to_pylist()
            keys = batch.column(1).to_pylist() if len(columns) == 2 else [None] * len(strings)
            counts.update(zip(strings, keys))
            scanned += len(strings)
            if scanned % 500_000 < len(strings):
                print(f"Scanned molecular columns: {scanned:,} / {source.metadata.num_rows:,}", flush=True)
    return counts, scanned


def build_preprocessing(train_path, output_dir, seed=42, holdout=0.1, val_frac=0.02,
                        workers=4, batch_size=8192):
    require_competition_rdkit()
    partition_identity("configuration-check", seed, holdout, val_frac)
    if workers < 1 or batch_size < 1:
        raise ValueError("workers and batch_size must be positive")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    policy = dict(policy=POLICY, rdkit_version=RDKIT_VERSION, source=SOURCE_URL)
    policy_path = output / "normalization_policy.json"
    if policy_path.exists() and json.loads(policy_path.read_text()) != policy:
        raise ValueError("Existing normalization cache has a different policy/version")
    policy_path.write_text(json.dumps(policy, indent=2) + "\n")
    # The final report is written only after every output has been completed.
    report_path = output / "preprocessing_report.json"
    if report_path.exists():
        report_path.unlink()
    pairs, scanned = scan_source_pairs(train_path, batch_size)
    strings = sorted({s for s, _ in pairs if s is not None})
    if any(s is None for s, _ in pairs):
        strings.append(None)
    normalized = {}
    cache_path = output / "normalization_cache.jsonl"
    if cache_path.exists():
        with cache_path.open() as cache:
            for line in cache:
                record = json.loads(line)
                normalized[record["input_smiles"]] = record
    missing = [s for s in strings if s not in normalized]
    print(f"Normalizing {len(strings):,} unique SMILES ({len(missing):,} uncached) "
          f"with RDKit {RDKIT_VERSION}, workers={workers}", flush=True)
    pool = ProcessPoolExecutor(max_workers=workers) if workers > 1 and missing else None
    try:
        if pool:
            futures = [pool.submit(normalize_batch, missing[i:i+16]) for i in range(0, len(missing), 16)]
            results = (record for future in as_completed(futures) for record in future.result())
        else:
            results = map(normalize_smiles, missing)
        with cache_path.open("a") as cache:
            for i, record in enumerate(results, 1):
                normalized[record["input_smiles"]] = record
                cache.write(json.dumps(record) + "\n")
                if i % 5000 == 0:
                    cache.flush()
                    print(f"Normalized {i:,} / {len(missing):,} unique SMILES", flush=True)
    finally:
        if pool:
            pool.shutdown()
    records = []
    for (smiles, key), n in pairs.items():
        rec = dict(normalized[smiles])
        rec.update(source_inchikey14=key, n_spectra=n)
        records.append(rec)
    mapping = pd.DataFrame(records)
    valid = mapping[mapping.status == "valid"]
    if valid.empty:
        raise ValueError("No valid molecules found; inspect normalization_cache.jsonl")
    # InChIKey14 can also merge forms differing in protonation/isotopic layers.
    # One deterministic observed target per scoring identity avoids conflicting targets.
    targets = valid.groupby("competition_identity").candidate_target.min().to_dict()
    mapping["target_smiles"] = mapping.competition_identity.map(targets)
    assignment = {identity: partition_identity(identity, seed, holdout, val_frac) for identity in targets}
    mapping["split"] = mapping.competition_identity.map(assignment)
    mapping.to_parquet(output / "molecule_mapping.parquet", index=False)
    invalid = mapping[mapping.status != "valid"]
    (output / "invalid_molecules.jsonl").write_text(
        invalid.to_json(orient="records", lines=True) if not invalid.empty else "")
    molecules = valid.groupby("competition_identity").agg(
        n_spectra=("n_spectra", "sum"), n_source_smiles=("input_smiles", "nunique"),
        n_source_identities=("source_inchikey14", "nunique")).reset_index()
    molecules["target_smiles"] = molecules.competition_identity.map(targets)
    molecules["split"] = molecules.competition_identity.map(assignment)
    molecules.to_parquet(output / "molecules.parquet", index=False)
    molecules.to_csv(output / "molecules.csv", index=False)
    # A compact row-index manifest keeps every spectrum, without duplicating 3 GB.
    row_start = 0
    schema = pa.schema([("source_row", pa.int64()), ("competition_identity", pa.string()),
                        ("split", pa.string()), ("status", pa.string())])
    with pq.ParquetFile(train_path, pre_buffer=False) as source, pq.ParquetWriter(
            output / "spectrum_partitions.parquet", schema) as writer:
        for batch in source.iter_batches(batch_size=batch_size, columns=["normalized_smiles"], use_threads=False):
            strings_batch = batch.column(0).to_pylist()
            rows = [normalized[s] for s in strings_batch]
            identities = [r["competition_identity"] for r in rows]
            table = pa.Table.from_arrays([
                pa.array(range(row_start, row_start + len(rows)), type=pa.int64()),
                pa.array(identities, type=pa.string()),
                pa.array([assignment.get(k) for k in identities], type=pa.string()),
                pa.array([r["status"] for r in rows], type=pa.string())], schema=schema)
            writer.write_table(table)
            row_start += len(rows)
    valid_keys = valid.dropna(subset=["source_inchikey14"])
    split_counts = molecules.groupby("split").agg(
        identities=("competition_identity", "size"), spectra=("n_spectra", "sum")).to_dict("index")
    input_stat = Path(train_path).stat()
    report = dict(**policy, input_path=str(Path(train_path).resolve()),
                  input_size_bytes=input_stat.st_size, input_mtime_ns=input_stat.st_mtime_ns,
                  rows_scanned=scanned,
                  unique_input_smiles=len(strings), unique_source_identity_smiles_pairs=len(mapping),
                  valid_identities=len(molecules), invalid_unique_smiles=int(invalid.input_smiles.nunique(dropna=False)),
                  invalid_spectra=int(invalid.n_spectra.sum()),
                  tautomer_status_counts=mapping.tautomer_status.value_counts().to_dict(),
                  target_fallback_unique_smiles=int(mapping.loc[mapping.get("target_fallback", pd.Series(index=mapping.index, dtype=str)).notna(), "input_smiles"].nunique()),
                  source_identities_merged=int((molecules.n_source_identities > 1).sum()),
                  identities_with_multiple_source_smiles=int((molecules.n_source_smiles > 1).sum()),
                  source_identity_mismatches=int((valid_keys.source_inchikey14 != valid_keys.competition_identity).sum()),
                  source_ids_mapping_to_multiple_identities=int((valid_keys.groupby("source_inchikey14").competition_identity.nunique() > 1).sum()),
                  seed=seed, holdout=holdout, val_frac=val_frac, partitions=split_counts,
                  target_policy="Canonical tautomer, remove stereo, canonical SMILES; exact input fallback if capped enumeration changes target identity; minimum verified target per scoring identity",
                  spectrum_manifest="source_row is the zero-based row in the original parquet; invalid rows have no partition")
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    return report


def load_preprocessing(directory):
    require_competition_rdkit()
    directory = Path(directory)
    report = json.loads((directory / "preprocessing_report.json").read_text())
    if report["policy"] != POLICY or report["rdkit_version"] != RDKIT_VERSION:
        raise ValueError("Preprocessing policy/version mismatch; regenerate artifacts")
    return pd.read_parquet(directory / "molecule_mapping.parquet"), report


def apply_preprocessing(df, mapping):
    """Map every spectrum; preserve supplied fields as source_* audit columns."""
    if "source_normalized_smiles" in df:
        raise ValueError("Input is already preprocessed; supply the original parquet")
    lookup = mapping.drop_duplicates("input_smiles").set_index("input_smiles")
    source = df.normalized_smiles
    status = source.map(lookup.status)
    if status.isna().any():
        raise ValueError("Input SMILES missing from preprocessing map; regenerate for this data")
    result = df.loc[status == "valid"].copy()
    result["source_normalized_smiles"] = result.normalized_smiles
    if "inchikey14" in result:
        result["source_inchikey14"] = result.inchikey14
    result["normalized_smiles"] = result.source_normalized_smiles.map(lookup.target_smiles)
    result["competition_identity"] = result.source_normalized_smiles.map(lookup.competition_identity)
    result["inchikey14"] = result.competition_identity  # compatibility with ranker feature tables
    result["split"] = result.source_normalized_smiles.map(lookup.split)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "preprocessing_results")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--holdout", type=float, default=0.1)
    parser.add_argument("--val-frac", type=float, default=0.02)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    build_preprocessing(args.train, args.output, args.seed, args.holdout, args.val_frac, args.workers)


if __name__ == "__main__":
    main()
