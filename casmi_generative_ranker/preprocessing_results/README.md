# Molecular preprocessing results

Processed all **2,539,608 spectra** and **277,566 distinct supplied SMILES** with
RDKit **2026.03.3**, following the competition's default tautomer canonicalization
and InChIKey14 matching rule. No invalid SMILES or rejected spectra were found.
There are **274,193 competition-normalized identities**.

| Partition | Identities | Spectra |
|---|---:|---:|
| Generator training | 242,056 | 2,242,315 |
| Generator validation | 4,914 | 49,951 |
| Ranker holdout | 27,223 | 247,342 |

The same scoring identity always has one target and one partition. The split uses
seed 42, approximately 10% identity-level ranker holdout, and 2% of the remaining
identities for validation. All spectra are retained; spectral arrays remain in
the original parquet.

1,542 normalized identities combine multiple supplied identities. The scan found
4,469 supplied identity/SMILES pairs whose recomputed identity differs from the
supplied key. Eleven supplied keys map to more than one scoring identity; the
recomputed competition identity governs their assignment.

RDKit reached its default tautomer limits for 3,033 input strings. Sixty targets
retain the exact original SMILES because rewriting them would change the scoring
identity under bounded enumeration. These are valid molecules, and their fallback
and enumeration status are recorded in `molecule_mapping.parquet`. Limits are not
increased and extra chemical cleanup is not applied.

- `molecules.csv` / `molecules.parquet`: one consistent target per normalized identity.
- `molecule_mapping.parquet`: every supplied identity/SMILES pair, count, target,
  normalized identity, partition, and normalization audit fields.
- `spectrum_partitions.parquet`: all original zero-based row indices and their
  normalized identities/partitions, without copying spectral arrays.
- `invalid_molecules.jsonl`: invalid-entry log, empty for this dataset.
- `preprocessing_report.json`: configuration, provenance, counts, and rule version.
- `preprocessing_audit.json`: full-manifest consistency checks and scoring parity checks.
- `normalization_cache.jsonl`: resumable normalization cache.

The original parquet is unchanged. Training applies the map to its spectra, keeps
source strings/keys for audit, and uses the normalized target and identity. The
shared normalizer also processes generated candidates before deduplication and
ranker labeling. Existing raw-target tokenizers/checkpoints must be regenerated;
the earlier 80/20 decoder-length EDA is historical.

Source: [competition matching rules](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/overview).
