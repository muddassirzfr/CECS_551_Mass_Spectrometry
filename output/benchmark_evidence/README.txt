COMPUTATIONAL BENCHMARK EVIDENCE

Planned comparison: CNN + GRU versus one Transformer encoder-decoder.
Pooled peak embeddings + GRU is AUXILIARY ONLY and excluded from the base budget.
Prediction quality, full-data pandas loading, and full-data RAM remain untested.

Read analysis/report.md for configurations, timing procedure, workload scopes and
allocation accounting. analysis/model_measurements.csv contains exact timing values.
records/ contains hardware/package records, per-model JSON, cost calculations,
PBS output/accounting, and the original startup failure and successful run logs.
records/cnn_run/ contains the UNCHANGED CNN raw JSON and exact GPU source snapshot.
original_gpu_run_source/ contains the original Transformer/pooled-GRU source.
current_analysis_source/ contains the current corrected analysis and reproduction code.
Re-running requires the prepared spectral samples or the competition source data,
the pinned normalization environment, and an allocated CUDA GPU. Spectral rows and
model checkpoints are not part of this raw-timing evidence archive.

Important CNN raw-record clarification:
full_cap_memory.peak_positions=129 is an inherited shared-harness counter for
retained peaks plus optional metadata. It is NOT the CNN convolution axis.
This CNN actually uses two channels of 40,960 m/z bins, with metadata fused after
pooling. records/cnn_gru.json is an annotated copy of the unchanged raw record at
records/cnn_run/results/cnn_gru.json. Timing and memory values are identical.

Units: spectra/s counts input spectra, not sampled candidates. One update includes
four microbatches of 32 spectra. Each model has 20 warm-up and 150 timed updates.
GPU allocated/reserved memory is distinct from host RAM and GPU allocation-hours.
CPU ranker labels are synthetic and provide no evidence of predictive quality.

The original measurement allowance was 1.5 hours only; total GPU allocations were
216 seconds (0.060 hours), including the startup retry. A 12-hour total training
plan and two-hour extra evaluation/retry reserve are PROPOSED, not approved.

Verify the files after extraction with: sha256sum -c CHECKSUMS.sha256
(On macOS, use: shasum -a 256 -c CHECKSUMS.sha256)
Checksums establish file integrity, not independent attestation of measurements.
The records are supplied for inspection/reproduction; no independent verification
or model-quality result is claimed. The revised PDF is delivered separately and
contains this same ZIP as an embedded file attachment.
