# Google Colab v5e TPU: feasibility and benchmark plan

**Status: not benchmarked.** No TPU training speed, decoding speed, memory peak, or full-run time was measured in this session. This is the TPU counterpart to the [measured RTX 3090 table](RTX_3090_Benchmark.md), with missing measurements stated explicitly. RTX timings, CUDA memory peaks and the RTX 9.93-hour base estimate are **not transferred to TPU**. Prediction quality remains untested on both platforms.

The user supplied [this Colab notebook](https://colab.research.google.com/drive/1IdUN7rXFv_R4zGgcF6PgwIMS_k4uTzc1). The browser connector failed before opening it, so the session's selected accelerator, topology, installed backend, permissions and runtime state could not be inspected. No notebook cells were executed or changed. The intended target is one v5e chip (`v5e-1` if that is the selected runtime); verify the actual device instead of assuming an eight-chip TPU.

Google documents **16 GB nominal HBM per v5e chip**, with some memory reserved by the system. This is accelerator memory, not the Colab VM's host RAM. Neither Colab host RAM nor this model's memory fit can be inferred from the full-host Cloud TPU specifications or from RTX allocations. [Google v5e specifications](https://docs.cloud.google.com/tpu/docs/v5e)

Google currently recommends **TorchTPU** for new PyTorch TPU workloads. It provides a native `tpu` device, BF16 autocast and eager execution, with optional compilation. PyTorch/XLA is now described as its predecessor. Inspect the actual Colab image, select one supported backend/version combination, and record it; the existing CUDA benchmark cannot simply be relabeled as TPU. [PyTorch on Cloud TPU](https://docs.cloud.google.com/tpu/docs/pytorch-on-tpu)

The TorchTPU support matrix lists v5e, CNNs and Transformers as validated categories, while **LSTM/GRU support is experimental**. This makes a complete forward/backward/optimizer compatibility test a prerequisite for the planned CNN + GRU baseline; general CNN support alone does not validate that decoder. [TorchTPU feature support](https://torchtpu.readthedocs.io/en/0.2.0/feature_set/)

| Stage | Priority | Hardware | Measured speed/memory | Estimated total time | Proceed/fallback condition | Planned epochs | Checkpoint policy |
| --- | --- | --- | --- | --- | --- | --- | --- |
| CNN + GRU (planned baseline) training | Required compatibility gate | Colab v5e TPU; actual device count unverified | **Unmeasured**; GRU compatibility and memory untested | **Not yet estimated**; 10-epoch reference; 2,242,315 spectra/epoch; use measured `r_C` in the formula below | First pass finite-loss/gradient and optimizer-update checks; if unsupported, retain the planned baseline on RTX and report a hardware-split experiment explicitly | Initially 1–3; extend while validation improves, within budget. | Save each epoch for resumption; retain best validation MRR@25. |
| Transformer encoder-decoder training | Required | Colab v5e TPU; actual device count unverified | **Unmeasured**; BF16 and memory must be checked | **Not yet estimated**; 10-epoch reference; 2,242,315 spectra/epoch; use measured `r_T` | Proceed only after the complete-step benchmark, memory check and an actual training allocation | Initially 1–3; extend while validation improves, within budget. | Save each epoch for resumption; retain best validation MRR@25. |
| Pooled peak embeddings + GRU | Auxiliary only | Not scheduled on TPU | **Unmeasured on TPU** | Excluded from the required model comparison and budget | Do not substitute the RTX pooled-GRU measurement for CNN + GRU | — | — |
| CNN + GRU beam validation | Base plan | TPU if compatible; otherwise separately measured fallback | **Unmeasured**; target beam 25 and batch eight | `553 / (b_C x 3600)` TPU-hours, plus startup overhead | Validate the same 553 held-out spectra as the Transformer; compare outputs and record backend fallbacks | — | — |
| Transformer beam decoding: validation subset + test | Base plan | Colab v5e TPU | **Unmeasured**; target beam 25, batch eight, no KV cache | `1,766 / (b_T x 3600)` TPU-hours; 553 validation + 1,213 test spectra | Benchmark decoding separately; do not infer it from training speed | — | — |
| Transformer beam decoding: full validation + test | Alternative scope | Colab v5e TPU | **Unmeasured** | `51,164 / (b_T x 3600)` TPU-hours; 49,951 validation + 1,213 test spectra | Replaces the subset Transformer beam row; repeated evaluations cost extra | — | — |
| Ranker candidate generation: capped subset | Optional | Colab v5e TPU + Colab CPU | **Unmeasured**; 100 samples/spectrum on at most 10,000 spectra | `10,000 / (s x 3600)` TPU compute-hours + measured CPU chemistry time + startup/loading | Require useful recall, held-out MRR@25 gain and sufficient allocation; count inline CPU time if the TPU remains allocated | — | — |
| Ranker candidate generation: full reserved partition | Deferred alternative | Colab v5e TPU + Colab CPU | **Unmeasured** | `247,342 / (s x 3600)` TPU compute-hours + CPU processing; full-holdout fitting unmeasured | Replaces the capped subset; do not label a 10,000-spectrum estimate as a full-partition cost | — | — |
| Ranker fitting + refit | Optional | Colab CPU, initially one thread | **Unmeasured on the Colab host** | Time the 2,000-group x 500-candidate synthetic fixture separately; no full-holdout estimate yet | The RTX host's 58.8-second CPU result is not a Colab measurement; synthetic labels provide no quality evidence | — | — |
| Ensemble | Deferred | Depends on measured training/inference placement | **Unmeasured** | Unestimated | If estimated additional cost exceeds the remaining budget, retain the single Transformer; otherwise still require held-out validation improvement | — | — |

Here `r_C` and `r_T` are input **spectra/second for complete CNN and Transformer optimizer updates**; `b_C` and `b_T` are input **spectra/second for beam decoding**; and `s` is input **spectra/second for drawing 100 candidates per spectrum**. It is not candidates/second. All rates must come from the selected TPU runtime, not hardware peak TFLOPs.

## Match the RTX experiment

- **Transformer:** four encoder and four decoder layers; width 384; eight heads; feed-forward width 1536; dropout 0.1; pre-LayerNorm; vocabulary 512; 17,614,208 parameters. Input cap **129 total encoder positions** = 127 fragment peaks + precursor + metadata. Output cap **160 SMILES tokens including BOS/EOS**.
- **CNN + GRU:** two channels of 40,960 mass bins over [0, 4096) at 0.1 m/z; Conv1D channels 32/64/128, kernel nine, stride four; adaptive pooling to 16 cells; width-128 projection; two GRU layers; shared metadata conditioning. Keep this architecture unless a change is disclosed and separately validated.
- **Data and optimization:** the same 4,096-spectrum prepared sample, training-only BPE tokenizer and identity partitions; microbatch 32, accumulation four, effective batch 128; AdamW and the same target-token normalization/clipping; BF16 autocast with FP32 parameter/optimizer state.
- **Training timing:** 20 warm-up optimizer updates followed by 150 complete measured updates, or 19,200 processed spectra. Include transfer, loading/tokenization, forward pass, backward pass and optimizer work. Record startup/warm-up wall time separately and synchronize the TPU before starting and after ending each timed interval. Use the selected backend's supported synchronization and memory APIs. [TorchTPU public APIs](https://github.com/google-pytorch/torch_tpu/blob/0.2.0-docs/docs/public_apis.md)
- **Compilation and shape policy:** begin with the RTX dynamic-padding workload. Record any compilation/cache effects that remain in the timed interval. A fixed-padding or bucketed TPU optimization is a different measured profile; identify it and do not claim a pure hardware speedup against the RTX dynamic-padding rate. Confirm finite gradients and that the intended operations actually execute on TPU rather than relying on hidden CPU fallbacks.
- **Decode timing:** measure beam width 25 on 64 validation spectra, and 100-sample candidate generation on 16 reserved spectra, with inference batch eight and sampling chunk 100. Record cold and repeat timing separately because prefix lengths and early EOS affect execution. Keep CPU parsing/tautomer normalization separate from TPU generation.
- **Resources:** check actual chip count, usable HBM, backend/package versions, host RAM and peak device memory. Start data-loader workers at zero, limit PyTorch/BLAS CPU threads, and fit the CPU ranker with one thread. Compilation can have additional backend-controlled CPU activity. Use only the prepared sample for this pilot; full-dataset host memory/I/O remain unmeasured.

## Cost calculation after measurements

For each required model, the ten-epoch reference scenario is:

```text
training spectra = 2,242,315
planned epochs   = 10
total spectra    = 22,423,150

training hours = 22,423,150 / (measured spectra_per_second x 3600)
               = 6,228.6528 / measured spectra_per_second
```

Add measured startup/compilation cost, validation, checkpoint writing, and loading costs. For the RTX-matched validation schedule, project **97 teacher-forced loss-validation events capped at 5,000 spectra each**, plus one beam pass on the stated validation subset for each required model and Transformer test decoding. Measure CPU transfer/checkpoint storage on the actual Colab path; local runtime storage and Google Drive need not have the same I/O cost. A hypothetical 10% loading allowance may be shown separately for planning, but it cannot demonstrate full-data behavior.

```text
TPU base allocation hours = CNN training + Transformer training
                          + specified validation/test decoding
                          + startup, loading and checkpoint overhead

remaining allocation = actual total project allowance
                     - measured prior usage
                     - TPU base allocation hours
                     - reserved extra evaluation/retry time
```

Count wall-clock **allocation time**, not just TPU-busy time, if CPU chemistry or fitting runs while the accelerator stays connected. Keep TPU and GPU allocation-hours distinct in any hybrid plan, and also state elapsed wall time; one TPU-hour is not a monetary equivalent of one GPU-hour.

Only the earlier **1.5-hour measurement allowance** was confirmed. The RTX jobs used 0.060 hours; the **12-hour project proposal and two-hour reserve were RTX-based planning assumptions**, not an approved TPU allocation or a TPU performance estimate. Recalculate the base budget and reserve decision after the TPU measurements. Colab hardware availability and runtime duration depend on the service/account; do not assume a continuous ten-epoch run or transfer Cloud TPU pricing to Colab compute units. [Colab FAQ](https://research.google.com/colaboratory/faq.html)

**Practical conclusion:** v5e is a candidate for the Transformer experiment, subject to measured compatibility, memory and total cost. It is not yet a demonstrated replacement for the RTX pipeline. The CNN + GRU compatibility gate and separate autoregressive decoding timings matter. Prediction quality and full-dataset loading/RAM feasibility remain untested; no TPU speedup or completion time is claimed here.
