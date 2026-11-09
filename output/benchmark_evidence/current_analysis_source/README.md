# CASMI 2026 retrieval-free generator + LightGBM ranker

Architecture: MS/MS peak pairs plus spectrum metadata -> sinusoidal peak embedding -> Transformer encoder -> causal Transformer SMILES decoder -> sampled or beam candidates -> shared tautomer normalization / competition InChIKey14 deduplication -> optional LightGBM LambdaRank -> top 25.

This is an **experimental implementation**, not a trained competition solution. Correctness tests include synthetic CPU training and checkpoint reload. The local data schema has been checked. The submission writer supports the competition's `molecule_id,smiles` format, with up to 25 guesses joined by semicolons. It refuses to write when any required molecule has no completed valid candidate.

## Computational benchmark before full training

Commit to the planned **CNN + RNN baseline (`--architecture cnn_gru`)** and one Transformer. The baseline uses two channels on a 0.1-m/z grid over [0, 4096), Conv1D channels 32/64/128 with kernel 9 and stride 4, pooling to 16 cells, and a two-layer width-128 GRU decoder. Both models share partitions, tokenizer, metadata, targets, source peak cap and a 160-token output cap. Reproduce the baseline with `--architecture cnn_gru --d 128 --dec-layers 2 --cnn-max-mz 4096 --cnn-bin-width 0.1 --cnn-pool-bins 16`.

The earlier **pooled peak embeddings + GRU** measurement is auxiliary only, excluded from the required comparison and budget. CNN + GRU was separately benchmarked in PBS job 159856. No pooled-GRU timings are transferred. The [resource report](benchmark_results/report.md) states the configuration and procedure directly. Prediction quality remains untested.

From the repository root, prepare a reproducible sample without loading the full spectra file into RAM:

```bash
python casmi_generative_ranker/benchmark.py prepare --output artifacts/benchmark_data
python casmi_generative_ranker/benchmark.py run --data artifacts/benchmark_data \
  --output artifacts/benchmark --bf16 --steps 150 --warmup 20
# Recalculate the execution plan once the total allocation is known:
python casmi_generative_ranker/benchmark.py report --output artifacts/benchmark \
  --gpu-budget-hours YOUR_TOTAL_GPU_HOURS --spent-gpu-hours HOURS_ALREADY_USED \
  --evaluation-retry-reserve-hours 2
```

The default Transformer matches training defaults: width 384, eight heads, four encoder/four decoder layers, feed-forward width 1536, microbatch 32, accumulation four, 129 total encoder positions (127 fragments + precursor + metadata) and a 160-token cap. `benchmark.py` measures 150 full optimizer updates after 20 warm-up updates, including loading and device transfer. It also measures full-cap training memory, validation, checkpoint writing, beam decoding (width 25), and ranker candidate sampling (100 samples per spectrum). All configuration, observed lengths, GPU identity, allocated/reserved memory and raw timings are saved to JSON.

`hpc_benchmark.pbs` targets the CSULB HPC2 Torque/PBS `batch` queue with one GPU, eight CPU cores and a 30-minute walltime, using the `hpc2-coe-users` group. `setup_hpc_benchmark.sh` installs dependencies in a fresh task directory using an existing Python runtime, verifies imports, and publishes a `dependencies_current` symlink; it does not modify existing environments. Stage `code/`, prepared `data/`, and the PBS script under a new run directory, run setup, then run `qsub hpc_benchmark.pbs` there. A CUDA allocation is mandatory; `--allow-cpu-smoke --device cpu` only tests the harness.

`report.md` contains the requested six-column stage table. Total hours use the full length-filtered training count × planned epochs / measured throughput, measured validation/checkpoint costs, and an explicit extra loading allowance. Queue wait is excluded. CPU ranker timing is a **synthetic workload proxy**, not a quality result; GPU candidate generation and CPU chemistry/fitting are reported separately. Short-run decoding weights do not establish candidate recall, so decoding must be remeasured on a validated checkpoint.

Default ranker estimates cover **at most 10,000 spectra** (2,000 structures × up to five spectra), not all 247,342 reserved spectra. The report separately estimates full-holdout generation. The base beam estimate covers **553 validation spectra plus 1,213 test spectra**; the full validation partition has 49,951 spectra and is shown as a separate alternative. Training estimates include at most 5,000 teacher-forced validation spectra per event.

The confirmed 1.5-hour allowance was for benchmarking only. A **proposed** 12-GPU-hour total plan reserves two hours for additional evaluation/retries, leaving about 0.01 hours after 0.060 hours of measurement allocation usage and the recalculated 9.93-hour CNN + GRU/Transformer base plan. The minimum total including the two-hour reserve and recorded measurement spending is 11.99 hours. The capped ranker would need about 0.56 allocation-hours when inline CPU processing and the fitting proxy retain the GPU, so it does not fit that proposal. Neither the proposed budget nor its reserve is an approved full-training allocation. Change `--proposed-gpu-budget-hours` and `--evaluation-retry-reserve-hours` to explore other plans; set `--gpu-budget-hours` only when an actual full-run allocation is known.

Cycling 4,096 spectra and the original 2.19 GiB host-RAM peak **support preliminary compute-feasibility estimates**. They do not establish full-dataset loading speed or RAM usage. Profile the full pandas load on a compute node under a RAM limit or use streaming before full training.

Add a ranker or ensemble only when held-out MRR@25 improves and the remaining GPU-hour budget covers it. **If the estimated additional cost exceeds the remaining budget, retain the single Transformer.** No full training is launched by the benchmark.

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
- **Metadata:** a separate encoder token embeds stable hashed adduct, ionization-mode and instrument categories, plus minimum/median/maximum collision energy and a missingness flag. Hashing uses 128 buckets per field, so category collisions are possible. Missing fields are supported. Legacy policy-compatible checkpoints still load with metadata disabled; retraining is required to learn the new metadata inputs.
- **Optimizer:** AdamW with linear warmup (`--warmup` steps, capped at 10% of training) then cosine decay from peak `--lr` (default 3e-4). Layers are pre-LayerNorm by default; `--post-ln` restores the original layout. Gradient accumulation uses the actual target-token count, including incomplete final groups.
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

Use `--decoding beam --beam-size 25` for beam search. Sequences that hit the token cap without EOS are excluded from candidates under either decoding method. Fewer than 25 guesses are allowed; missing coverage raises an error instead of writing empty cells. The submission format follows the [competition specification](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/).

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
