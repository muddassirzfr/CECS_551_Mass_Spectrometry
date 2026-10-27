# CASMI 2026 retrieval-free generator + LightGBM ranker

Architecture: MS/MS peak pairs -> sinusoidal peak embedding -> Transformer encoder -> causal Transformer SMILES decoder -> sampled candidates -> shared tautomer normalization / competition InChIKey14 deduplication -> candidate features -> LightGBM LambdaRank -> top 25.

This is an **implementation scaffold**, not a trained or benchmarked competition solution. All three commands have been run end to end on a small synthetic dataset (CPU), but not on real CASMI data. You must verify the actual competition schema (column names `normalized_smiles`, `inchikey14`, `ms2_mzs`, `ms2_normalized_intensities`, `precursor_mz`, `adduct`, `molecule_id`), adduct vocabulary and sample-submission format. The submission writer only supports a 25-column wide format plus `molecule_id`; it deliberately errors on other formats instead of producing a misleading CSV.

## Installation

`pip install -r requirements.txt`

## Consistent molecular preprocessing

The [competition matching rule](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/overview)
is RDKit tautomer canonicalization followed by the first InChIKey block, pinned to
**RDKit 2026.03.3**. `molecular_preprocessing.py` implements that rule for both
training structures and generated candidates. No additional salt stripping,
fragment selection, or uncharging is applied.

Run preprocessing once, without loading spectral arrays:

```bash
python molecular_preprocessing.py --train /path/to/train.parquet --output preprocessing_results --workers 4
# Equivalent entry point:
python pipeline.py preprocess --train /path/to/train.parquet --output preprocessing_results
python pipeline.py train --train /path/to/train.parquet --preprocessing preprocessing_results --output artifacts
```

Without `--preprocessing`, `train` builds this manifest in `artifacts/preprocessing`
automatically. Existing preprocessing explicitly supplied with `--preprocessing`
retains its saved partitions, seed, and split fractions. Default identity-level
fractions are 10% ranker holdout, then 2% of the remainder for validation. Hash
assignment is stable under row reordering. Spectrum caps and length filtering
happen after identity assignment, and never move an identity to another split.

Each distinct supplied SMILES is validated and normalized once. The generator
uses canonical SMILES of the canonical tautomer with stereo removed. When several
representations share a scoring identity, every spectrum receives the same
lexicographically smallest verified observed target. For a molecule that reaches
RDKit’s default tautomer limits, rewriting the canonical tautomer can occasionally
change its scoring identity. Such cases retain the exact supplied SMILES as a
logged target fallback. The scoring identity always follows the pinned metric;
limits are not increased. Original strings and keys
are preserved as `source_normalized_smiles` / `source_inchikey14` in training
artifacts; `competition_identity` is explicit and `inchikey14` is updated for
compatibility. Generated candidates use this same identity function for
deduplication and positive/negative labels.

Outputs:

- `molecule_mapping.parquet`: every supplied identity/SMILES pair, spectrum count,
  validity/error, tautomer-limit status, target fallback, normalized identity, target, and partition.
- `molecules.parquet` / `molecules.csv`: one row per competition identity.
- `spectrum_partitions.parquet`: every original zero-based `source_row`, normalized
  identity, and partition. Invalid entries remain in the audit with no partition.
- `invalid_molecules.jsonl`: invalid or failed-normalization entries and counts.
- `preprocessing_report.json`: full-scan counts, merges, identity mismatches,
  partition sizes, exact version, and policy. Written only after completion.
- `normalization_cache.jsonl`: resumable per-SMILES results; repeat the same command
  to resume. Do not share a cache between different normalization policies.

Old raw-key splits/checkpoints/rankers must be regenerated and retrained. The
pipeline rejects checkpoint/ranker artifacts without the matching policy/version.
Rerun decoder-length EDA on the new targets and exact training partition; the
previous EDA measured supplied strings in a different 80/20 partition.

Regression tests cover invalid logging, shared identity/target assignment,
no identity leakage, scoring parity, generated-candidate deduplication and labels:

```bash
python -m unittest discover -s . -p 'test_molecular_preprocessing.py' -v
```

## Recommended first run: measure candidate recall at small scale

```
python pipeline.py train --train train.parquet --output artifacts --limit 200000 --epochs 10 --bf16
python pipeline.py ranker --output artifacts --ranker-limit 500 --bf16
```

Then read `ranker_metrics.json`. `candidate_recall_all` is the fraction of held-out structures whose correct molecule appears anywhere among the sampled candidates. If it is near zero, the ranker cannot help; improve the generator first.

## Train generator

`python pipeline.py train --train /path/to/train.parquet --output artifacts --epochs 10 --batch 32 --accum 4 --bf16`

For the original notebook-scale architecture use `--d 768 --heads 12 --enc-layers 6 --dec-layers 6 --ff 3072 --batch 8 --accum 16` (significantly more GPU memory and time).

Data is normalized before splitting and grouped by recomputed competition identity into three disjoint sets: generator train, generator validation (`--val-frac`, capped at `--val-max` spectra), and ranker holdout (`--holdout`).

- **Length limit:** SMILES longer than `--max-len` tokens (default 160, including BOS/EOS) are dropped from training and validation, with the count printed. The same limit caps generation length.
- **Optimizer:** AdamW with linear warmup (`--warmup` steps, capped at 10% of training) then cosine decay from peak `--lr` (default 3e-4). Layers are pre-LayerNorm by default; `--post-ln` restores the original layout.
- **Validation and checkpoints:** token-level validation loss every `--eval-every` optimizer steps and at each epoch end. `generator.pt` is the best-validation checkpoint, `generator_last.pt` the latest. Every evaluation is appended to `train_log.jsonl`.

Outputs: `generator.pt`, `generator_last.pt`, `tokenizer.json`, `train_log.jsonl`, `generator_train.parquet`, `generator_val.parquet`, `ranker_holdout.parquet`.

## Generate held-out candidates, evaluate, and train ranker

`python pipeline.py ranker --output artifacts --samples 100 --bf16`

Sampling every holdout spectrum would take days without a KV cache, so by default this uses `--ranker-limit 2000` random holdout structures and at most `--spectra-per-structure 5` spectra each. All `--samples` for a spectrum are drawn in one batch (`--sample-chunk 100`, `--infer-batch 8`); lower these if GPU memory is short.

The ranker structures are split again (`--ranker-eval-frac`, default 0.3). LightGBM is trained on the train part and evaluated on the eval part against generator-only baselines (sorting by log-probability, length-normalized log-probability, or sampling frequency). `ranker_metrics.json` reports MRR@25, top-1/10/25 and candidate recall; molecules with no valid candidate count as misses. After evaluation the saved `ranker.txt` is refit on all ranker structures (`--no-refit` keeps the train-only model).

If fewer than `--min-positive-groups` training structures have a correct candidate, the metrics are still written and the command stops before training LightGBM.

## Predict and submit

`python pipeline.py submit --test /path/to/test.parquet --sample-submission /path/to/sample_submission.csv --output artifacts --samples 100 --bf16`

Writes `ranked_candidates_debug.csv` and, if the format is supported, `submission.csv`. This is an inference output, not an official score; Kaggle computes the score.

## Known limitations and next improvements

- The model is randomly initialized and requires real training; no pretrained weights are included.
- The sampling decoder recomputes the full prefix and has **no KV cache**. Sampling 100 candidates per spectrum is still slow at competition scale.
- Multi-spectrum grouping is done by `molecule_id`, but the generator itself still sees each spectrum separately. A learned multi-spectrum fusion encoder is a possible upgrade.
- Chemical compatibility uses only a handful of common singly charged adducts; unknown adducts get no precursor mass constraint.
- `n_spectra`, `n_peaks`, `peak_entropy` and `adduct_known` are constant within a molecule, so they carry little ranking signal. Features do not include a fragmentation forward-model score, which will likely limit ranking quality.
- Sampling does not guarantee 25 valid, unique candidates per molecule.
- The final generator is not retrained on the validation and ranker-holdout structures.
- Large dataset loading currently uses pandas in memory; 2.5M spectra may require a large RAM machine or streaming implementation.
- LightGBM runs single-threaded to avoid an OpenMP crash seen when it runs after PyTorch in the same process.
