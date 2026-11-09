"""Prepare real, identity-disjoint samples and benchmark a baseline plus one Transformer.

GPU throughput includes data loading, transfer, forward/backward, clipping and AdamW.
This short run measures execution cost, not generalization or competition accuracy.
"""
import argparse
from functools import partial
import gc
import json
import math
import os
from pathlib import Path
import platform
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader
from tokenizers import Tokenizer

import pipeline as p


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def quantiles(values):
    return dict(zip(('min', 'p50', 'p95', 'p99', 'max'),
                    map(float, np.percentile(values, [0, 50, 95, 99, 100]))))


def prepare(args):
    started = time.perf_counter()
    folder = Path(args.output)
    folder.mkdir(parents=True, exist_ok=True)
    mapping, source_report = p.load_preprocessing(args.preprocessing)
    valid = mapping[mapping.status == 'valid'].drop_duplicates('competition_identity')
    training = valid[valid.split == 'Training']
    tk = p.train_tokenizer(pd.DataFrame({'normalized_smiles': training.target_smiles}),
                           folder / 'tokenizer.json', args.vocab)
    lengths = p.token_lengths(tk, valid.target_smiles)
    length_lookup = dict(zip(valid.competition_identity, lengths))
    rng = np.random.default_rng(args.seed)
    chosen = {}
    for split in ('Validation', 'Ranker holdout'):
        keys = valid.loc[valid.split == split, 'competition_identity'].to_numpy()
        chosen[split] = set(rng.choice(keys, min(args.eval_structures, len(keys)), replace=False))
    samples = {s: pd.DataFrame() for s in ('Training', 'Validation', 'Ranker holdout')}
    counts = dict.fromkeys(samples, 0)
    dropped = dict.fromkeys(samples, 0)
    columns = ['normalized_smiles', 'ms2_mzs', 'ms2_normalized_intensities', 'precursor_mz',
               'adduct', 'ionization_mode', 'instrument_type', 'collision_energy_ev']
    with pq.ParquetFile(args.train) as source:
        if source.metadata.num_rows != source_report['rows_scanned']:
            raise ValueError('Source row count differs from preprocessing')
        columns = [c for c in columns if c in source.schema_arrow.names]
        for number, batch in enumerate(source.iter_batches(batch_size=16384, columns=columns), 1):
            data = p.apply_preprocessing(batch.to_pandas(), mapping)
            data = data.dropna(subset=['ms2_mzs', 'ms2_normalized_intensities', 'precursor_mz'])
            allowed = data.competition_identity.map(length_lookup) <= args.max_len
            for split in samples:
                dropped[split] += int(((data.split == split) & ~allowed).sum())
            data = data[allowed]
            for split in samples:
                part = data[data.split == split].copy()
                counts[split] += len(part)
                if split != 'Training':
                    part = part[part.competition_identity.isin(chosen[split])]
                part['_priority'] = rng.random(len(part))
                pool = pd.concat([samples[split], part], ignore_index=True)
                if split == 'Training':
                    samples[split] = pool.nsmallest(args.train_sample, '_priority')
                else:
                    samples[split] = pool.sort_values('_priority').groupby('competition_identity').head(args.spectra_per_structure)
            if number % 25 == 0:
                print(f'Sampled {number * 16384:,} source rows', flush=True)
    for split, name in [('Training', 'train'), ('Validation', 'validation'), ('Ranker holdout', 'holdout')]:
        frame = samples[split].drop(columns='_priority').reset_index(drop=True)
        if frame.empty:
            raise ValueError(f'Empty {split} benchmark sample')
        frame['molecule_id'] = frame.competition_identity
        frame.to_parquet(folder / f'{name}.parquet', index=False)
    manifest = dict(seed=args.seed, source_rows=source_report['rows_scanned'],
                    normalization_policy=p.POLICY, rdkit_version=p.RDKIT_VERSION,
                    vocab_size=tk.get_vocab_size(), max_len=args.max_len,
                    training_spectra=counts['Training'], validation_spectra=counts['Validation'],
                    holdout_spectra=counts['Ranker holdout'], dropped_long_spectra=dropped,
                    sample_spectra={s: len(f) for s, f in samples.items()},
                    training_identity_token_lengths=quantiles(lengths[valid.split == 'Training']),
                    preparation_seconds=time.perf_counter()-started,
                    sampling='Uniform spectrum reservoir for train; uniform identities then spectrum reservoirs for evaluation',
                    tokenizer='Fitted on all normalized training identities only; shared by both architectures')
    dump(folder / 'manifest.json', manifest)
    print(json.dumps(manifest, indent=2), flush=True)


def synchronize(args):
    if args.device.startswith('cuda'):
        torch.cuda.synchronize()


def peak_memory(args):
    if not args.device.startswith('cuda'):
        return dict(allocated_gib=None, reserved_gib=None)
    return dict(allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                reserved_gib=torch.cuda.max_memory_reserved()/2**30)


def reset_memory(args):
    if args.device.startswith('cuda'):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def cycle(loader):
    while True:
        yield from loader


def training_step(model, optimizer, scaler, batches, args):
    count = spectra = 0
    last_loss = None
    for _ in range(args.accum):
        batch = p.move(next(batches), args.device)
        spectra += len(batch['meta'])
        count += int((batch['tokens'][:, 1:] != p.PAD).sum())
        with p.amp(args):
            loss = model(batch, reduction='sum')/args.accum
        scaler.scale(loss).backward()
        last_loss = loss.detach()
    updated = p.optimizer_update(model, optimizer, scaler, count, args.accum)
    if not updated or not torch.isfinite(last_loss):
        raise RuntimeError('Nonfinite or skipped optimizer update; throughput would be misleading')
    return spectra


def benchmark_model(architecture, frames, tk, args):
    p.seed_all(args.seed)
    folder = Path(args.output)
    config = dict(vocab=tk.get_vocab_size(), max_len=args.max_len, architecture=architecture)
    if architecture == 'transformer':
        config.update(d=args.d, heads=args.heads, enc_layers=args.enc_layers,
                      dec_layers=args.dec_layers, ff=args.ff)
    else:
        config.update(d=args.baseline_d, heads=2, enc_layers=1, dec_layers=args.baseline_layers,
                      ff=args.baseline_d*4)
    if architecture == 'cnn_gru':
        config.update(cnn_max_mz=args.cnn_max_mz, cnn_bin_width=args.cnn_bin_width, cnn_pool_bins=args.cnn_pool_bins)
    model = p.Generator(**config).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=.01)
    scaler = torch.amp.GradScaler('cuda', enabled=args.device.startswith('cuda') and not args.bf16)
    collator = partial(p.collate, tokenizer=tk)
    # Match the production loader: dynamic padding, transfer and tokenization are timed.
    loader = DataLoader(p.Spectra(frames['train'], args.max_peaks), batch_size=args.batch,
                        shuffle=True, drop_last=True, collate_fn=collator, num_workers=args.workers)
    if not len(loader):
        raise ValueError('Train sample must contain at least one complete batch')
    batches = cycle(loader)
    model.train()
    for _ in range(args.warmup):
        training_step(model, optimizer, scaler, batches, args)
    synchronize(args)
    reset_memory(args)
    started = time.perf_counter()
    spectra = 0
    for index in range(args.steps):
        spectra += training_step(model, optimizer, scaler, batches, args)
        if (index+1) % 25 == 0:
            print(f'{architecture}: {index+1}/{args.steps} complete optimizer updates', flush=True)
    synchronize(args)
    elapsed = time.perf_counter()-started
    train = dict(optimizer_steps=args.steps, warmup_steps=args.warmup,
                 spectra=spectra, seconds=elapsed, spectra_per_second=spectra/elapsed,
                 peak_memory=peak_memory(args), includes_loading_transfer_and_optimizer=True)
    # Full-cap stress test measures memory without implying all spectra have this length.
    synchronize(args)
    reset_memory(args)
    stress = dict(mz=torch.rand(args.batch, args.max_peaks, device=args.device)*1000,
                  ints=torch.rand(args.batch, args.max_peaks, device=args.device),
                  mask=torch.ones(args.batch, args.max_peaks, dtype=torch.bool, device=args.device),
                  tokens=torch.randint(4, tk.get_vocab_size(), (args.batch, args.max_len), device=args.device),
                  meta=[{}]*args.batch)
    stress['tokens'][:, 0] = p.BOS
    stress['tokens'][:, -1] = p.EOS
    try:
        training_step(model, optimizer, scaler, iter([stress]*args.accum), args)
        synchronize(args)
        stress_result = dict(status='passed', peak_memory=peak_memory(args),
                             source_spectral_positions=args.max_peaks, target_tokens=args.max_len)
        if architecture == 'cnn_gru':
            stress_result['cnn_grid_bins_per_channel'] = model.cnn_bins
        else:
            stress_result['encoder_positions'] = args.max_peaks+int(model.use_metadata)
    except torch.cuda.OutOfMemoryError:
        stress_result = dict(status='out_of_memory', peak_memory=peak_memory(args))
        optimizer.zero_grad(set_to_none=True)
    del stress
    reset_memory(args)
    vdl = DataLoader(p.Spectra(frames['validation'], args.max_peaks), batch_size=args.batch,
                     collate_fn=collator, num_workers=args.workers)
    synchronize(args)
    started = time.perf_counter()
    loss = p.validation_loss(model, vdl, args)
    synchronize(args)
    validation = dict(spectra=len(frames['validation']), seconds=time.perf_counter()-started,
                      token_loss=loss, accuracy_claim=False)
    validation['spectra_per_second'] = validation['spectra']/validation['seconds']
    started = time.perf_counter()
    for name in ('generator_last.pt', 'generator.pt'):
        p.save_generator(model, folder/f'{architecture}_{name}', args)
    synchronize(args)
    checkpoint_seconds = time.perf_counter()-started
    del optimizer, scaler, batches, loader, vdl
    gc.collect()
    reset_memory(args)
    beam_frame = frames['validation'].head(args.decode_spectra)
    beam_loader = DataLoader(p.Spectra(beam_frame, args.max_peaks), batch_size=args.infer_batch,
                             collate_fn=p.collate, num_workers=args.workers)
    model.eval()
    first = p.move(next(iter(beam_loader)), args.device)
    with p.amp(args):
        model.beam(first, args.beam_size, min(8, args.max_len-1), args.length_penalty)
    synchronize(args)
    reset_memory(args)
    started = time.perf_counter()
    outputs = []
    for batch in beam_loader:
        with p.amp(args):
            outputs.extend(model.beam(p.move(batch, args.device), args.beam_size,
                                      args.max_tokens, args.length_penalty))
    synchronize(args)
    beam_seconds = time.perf_counter()-started
    all_sequences = [entry for group in outputs for entry in group]
    beam = dict(spectra=len(beam_frame), beam_size=args.beam_size, seconds=beam_seconds,
                spectra_per_second=len(beam_frame)/beam_seconds, peak_memory=peak_memory(args),
                completed_fraction=float(np.mean([p.EOS in x[0] for x in all_sequences])),
                generated_token_lengths=quantiles([x[2] for x in all_sequences]),
                weights='Short benchmark run; remeasure decoding on a validated checkpoint', kv_cache=False)
    # Ranker generation uses sampling, which is a different workload from beam search.
    holdout = frames['holdout'].head(args.candidate_spectra)
    timing = {}
    warmup_batch = p.collate([p.Spectra(holdout, args.max_peaks)[0]])
    with p.amp(args):
        model.sample(p.move(warmup_batch, args.device), args.samples, min(8, args.max_len-1),
                     args.temperature, args.sample_chunk)
    synchronize(args)
    reset_memory(args)
    started = time.perf_counter()
    groups = p.build_candidates(holdout, model, tk, args.max_peaks, args, timing)
    synchronize(args)
    candidate_seconds = time.perf_counter()-started
    candidate = dict(spectra=len(holdout), samples_per_spectrum=args.samples,
                     seconds=candidate_seconds, **timing, peak_memory=peak_memory(args),
                     valid_unique_candidates=sum(len(g['candidates']) for g in groups.values()),
                     weights='Short benchmark run; candidate recall is not evidence for ranker inclusion')
    result = dict(architecture=architecture, settings=model.settings,
                  parameters=sum(x.numel() for x in model.parameters()),
                  batch=args.batch, accumulation=args.accum, effective_batch=args.batch*args.accum,
                  max_peaks=args.max_peaks, max_len=args.max_len, training=train,
                  full_cap_memory=stress_result, validation=validation,
                  two_checkpoint_seconds=checkpoint_seconds, beam=beam, candidate_generation=candidate)
    dump(folder/f'{architecture}.json', result)
    del model, outputs, first, groups
    gc.collect()
    reset_memory(args)
    return result


def ranker_proxy(args):
    # Synthetic labels are explicitly a CPU cost proxy, never a quality evaluation.
    rng = np.random.default_rng(args.seed)
    groups, candidates = args.ranker_structures, args.ranker_candidates
    frame = pd.DataFrame(rng.normal(size=(groups*candidates, len(p.FEATURES))), columns=p.FEATURES)
    frame['molecule_id'] = np.repeat([f'g{i:06d}' for i in range(groups)], candidates)
    frame['label'] = np.tile([1]+[0]*(candidates-1), groups)
    started = time.perf_counter()
    p.train_lambdarank(frame[frame.molecule_id < f'g{int(groups*.7):06d}'], args.seed)
    p.train_lambdarank(frame, args.seed)
    return dict(seconds=time.perf_counter()-started, groups=groups, candidates_per_group=candidates,
                rows=len(frame), threads=1, kind='synthetic CPU timing proxy: train fit plus all-group refit',
                accuracy_claim=False)


def run(args):
    if not args.device.startswith('cuda') and not args.allow_cpu_smoke:
        raise ValueError('GPU benchmarking requires CUDA; use --allow-cpu-smoke only to test the harness')
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU unavailable; run inside a PBS GPU allocation')
    if not args.allow_cpu_smoke and not 100 <= args.steps <= 200:
        raise ValueError('Use 100–200 measured complete optimizer updates')
    if min(args.accum, args.batch, args.steps, args.warmup, args.decode_spectra, args.candidate_spectra) < 1:
        raise ValueError('Batch sizes, steps, warm-up and sample counts must be positive')
    if args.device.startswith('cuda'):
        device = torch.device(args.device)
        torch.cuda.set_device(device.index if device.index is not None else 0)
        if args.bf16 and not torch.cuda.is_bf16_supported():
            raise ValueError('BF16 unsupported on this GPU; rerun without --bf16')
    torch.set_num_threads(args.cpu_threads)
    folder = Path(args.output)
    folder.mkdir(parents=True, exist_ok=True)
    source = Path(args.data)
    started = time.perf_counter()
    frames = {name: pd.read_parquet(source/f'{name}.parquet') for name in ('train', 'validation', 'holdout')}
    tk = Tokenizer.from_file(str(source/'tokenizer.json'))
    manifest = json.loads((source/'manifest.json').read_text())
    if args.max_len != manifest['max_len']:
        raise ValueError('Benchmark max_len must match prepared data filtering')
    load_seconds = time.perf_counter()-started
    metadata = dict(host=platform.node(), cpu=platform.processor(), torch=torch.__version__,
                    cuda=torch.version.cuda, gpu=torch.cuda.get_device_name() if args.device.startswith('cuda') else None,
                    gpu_total_gib=torch.cuda.get_device_properties(0).total_memory/2**30 if args.device.startswith('cuda') else None,
                    device=args.device, precision='bf16' if args.bf16 else ('fp16' if args.device.startswith('cuda') else 'fp32'),
                    pbs_job_id=os.environ.get('PBS_JOBID'), cpu_smoke_only=args.allow_cpu_smoke,
                    data_load_seconds=load_seconds, manifest=manifest,
                    sample_token_lengths={k: quantiles(p.token_lengths(tk, v.normalized_smiles)) for k, v in frames.items()})
    dump(folder/'hardware.json', metadata)
    for architecture in args.architectures:
        benchmark_model(architecture, frames, tk, args)
    if not args.skip_ranker_proxy:
        proxy = ranker_proxy(args)
        dump(folder/'ranker_cpu.json', proxy)
    if not args.skip_report:
        report(args)


def report(args):
    folder = Path(args.output)
    hardware = json.loads((folder/'hardware.json').read_text())
    manifest = hardware['manifest']
    if not (folder/'cnn_gru.json').exists():
        raise ValueError('CNN + GRU is the planned baseline and needs separate measurements; pooled-GRU cannot supply its budget')
    models = {name: json.loads((folder/f'{name}.json').read_text()) for name in ('cnn_gru', 'transformer')}
    if (folder/'pooled_gru.json').exists():
        models['pooled_gru'] = json.loads((folder/'pooled_gru.json').read_text())
    cpu = json.loads((folder/'ranker_cpu.json').read_text())
    reserve = args.evaluation_retry_reserve_hours
    proposed_budget = args.proposed_gpu_budget_hours
    if reserve < 0 or proposed_budget < 0:
        raise ValueError('Budget and reserve must be nonnegative')
    rows = []
    estimates = {}
    for name, model in models.items():
        train_seconds = manifest['training_spectra']*args.epochs/model['training']['spectra_per_second']
        steps = args.epochs*math.ceil(math.ceil(manifest['training_spectra']/model['batch'])/model['accumulation'])
        validations = args.epochs + (steps//args.eval_every if args.eval_every else 0)
        val_count = min(args.val_max, manifest['validation_spectra']) if args.val_max else manifest['validation_spectra']
        validation_seconds = validations*val_count/model['validation']['spectra_per_second']
        checkpoint_seconds = validations*model['two_checkpoint_seconds']
        # Steady-state loading is included in throughput. Extra allowance covers
        # full-data RAM/I/O changes; preparation is conservatively charged per model.
        loading_allowance = train_seconds*args.loading_allowance
        total = train_seconds+validation_seconds+checkpoint_seconds+loading_allowance+manifest['preparation_seconds']
        estimates[name] = dict(training_hours=train_seconds/3600, validation_hours=validation_seconds/3600,
                               checkpoint_hours=checkpoint_seconds/3600, loading_allowance_hours=loading_allowance/3600,
                               total_hours=total/3600, validation_events=validations, planned_epochs=args.epochs,
                               training_spectra_per_epoch=manifest['training_spectra'],
                               validation_spectra_per_event=val_count)
        estimates[name]['included_in_base_plan'] = name in ('cnn_gru', 'transformer')
        speed = model['training']['spectra_per_second']
        memory = model['training']['peak_memory']['allocated_gib']
        memory_text = f'{memory:.2f} GiB allocated' if memory is not None else 'GPU memory unmeasured'
        cap_memory = model['full_cap_memory']['peak_memory']['allocated_gib']
        if cap_memory is not None:
            memory_text += f'; {cap_memory:.2f} GiB at full caps ({model["full_cap_memory"]["status"]})'
        labels = {'cnn_gru':'CNN + GRU (planned baseline)', 'pooled_gru':'Pooled peak embeddings + GRU (auxiliary)', 'transformer':'Transformer encoder–decoder'}
        condition = 'Auxiliary measurement only; excluded from the planned baseline and budget' if name=='pooled_gru' else 'Require a full-training allocation and held-out MRR@25 evaluation'
        rows.append([labels[name]+' training', 'Auxiliary only' if name=='pooled_gru' else 'Required', hardware['gpu'] or 'CPU smoke only',
                     f'{speed:.1f} spectra/s; {memory_text}',
                     f'{total/3600:.2f} h estimated; {manifest["training_spectra"]:,} spectra × {args.epochs} epochs', condition])
    transformer = models['transformer']
    beam = transformer['beam']
    beam_eval_spectra = manifest['sample_spectra']['Validation']
    beam_hours = (args.prediction_spectra+beam_eval_spectra)/beam['spectra_per_second']/3600
    full_beam_hours = (args.prediction_spectra+manifest['validation_spectra'])/beam['spectra_per_second']/3600
    beam_memory = beam['peak_memory']['allocated_gib']
    beam_memory_text = f'{beam_memory:.2f} GiB allocated' if beam_memory is not None else 'GPU memory unmeasured'
    baseline_beam = models['cnn_gru']['beam']
    baseline_beam_hours = beam_eval_spectra/baseline_beam['spectra_per_second']/3600
    baseline_beam_memory = baseline_beam['peak_memory']['allocated_gib']
    baseline_beam_memory_text = f'{baseline_beam_memory:.2f} GiB allocated' if baseline_beam_memory is not None else 'GPU memory unmeasured'
    rows.append(['CNN + GRU beam validation', 'Base plan', hardware['gpu'] or 'CPU smoke only',
                 f"{baseline_beam['spectra_per_second']:.2f} spectra/s; {baseline_beam_memory_text}; beam {baseline_beam['beam_size']}",
                 f'{baseline_beam_hours:.5f} GPU h (~{baseline_beam_hours*3600:.1f} s); {beam_eval_spectra:,} validation spectra',
                 'Evaluate the baseline and Transformer on the same held-out identities'])
    rows.append(['Beam decoding: validation subset + test', 'Base plan', hardware['gpu'] or 'CPU smoke only',
                 f"{beam['spectra_per_second']:.2f} spectra/s; {beam_memory_text}; beam {beam['beam_size']}",
                 f'{beam_hours:.4f} GPU h (~{beam_hours*3600:.0f} s); {beam_eval_spectra:,} validation + {args.prediction_spectra:,} test spectra',
                 'Remeasure on validated weights; lower beam width or inference batch if needed'])
    rows.append(['Beam decoding: full validation + test', 'Alternative scope', hardware['gpu'] or 'CPU smoke only',
                 f"Same {beam['spectra_per_second']:.2f} spectra/s extrapolated from {beam['spectra']} measured spectra",
                 f'{full_beam_hours:.3f} GPU h (~{full_beam_hours*60:.1f} min); {manifest["validation_spectra"]:,} validation + {args.prediction_spectra:,} test spectra',
                 'Replaces the subset beam row; additional evaluation passes need additional budget'])
    generation = transformer['candidate_generation']
    subset_spectra = args.ranker_structures*args.spectra_per_structure
    generation_hours = subset_spectra*generation['generation_seconds']/generation['spectra']/3600
    chemistry_hours = subset_spectra*generation['chemistry_seconds']/generation['spectra']/3600
    subset_inline_hours = subset_spectra*generation['seconds']/generation['spectra']/3600
    ranker_allocation_hours = subset_inline_hours+cpu['seconds']/3600
    full_generation_hours = manifest['holdout_spectra']*generation['generation_seconds']/generation['spectra']/3600
    full_chemistry_hours = manifest['holdout_spectra']*generation['chemistry_seconds']/generation['spectra']/3600
    full_inline_hours = manifest['holdout_spectra']*generation['seconds']/generation['spectra']/3600
    rows.append([f'Ranker generation: capped subset (≤{subset_spectra:,} spectra)', 'Optional', hardware['gpu'] or 'CPU smoke only',
                 f"{generation['spectra']/generation['generation_seconds']:.2f} spectra/s; {generation['samples_per_spectrum']} samples/spectrum",
                 f'{generation_hours:.2f} GPU compute h + {chemistry_hours:.2f} CPU chemistry h; {args.ranker_structures:,} structures × up to {args.spectra_per_structure} spectra',
                 'Only if candidate recall is useful and held-out ranker MRR@25 improves; remeasure with trained weights'])
    rows.append([f'Ranker generation: all {manifest["holdout_spectra"]:,} reserved spectra', 'Deferred alternative', hardware['gpu'] or 'CPU smoke only',
                 f"Same {generation['spectra']/generation['generation_seconds']:.2f} spectra/s extrapolated from {generation['spectra']} measured spectra",
                 f'{full_generation_hours:.2f} GPU compute h + {full_chemistry_hours:.2f} CPU chemistry h; full-holdout fitting unmeasured',
                 'Replaces the capped-subset row; requires a separate allocation and trained-checkpoint timing'])
    rows.append(['Ranker fitting + refit', 'Optional', 'CPU, one thread',
                 f"{cpu['seconds']:.1f} s synthetic proxy; {cpu['rows']:,} candidate rows",
                 f"{cpu['seconds']/3600:.3f} CPU h proxy; {cpu['groups']:,} groups × {cpu['candidates_per_group']} candidates/group", 'Proxy for the capped subset; cannot transfer to full-holdout fitting; require validation gain'])
    base_hours = estimates['cnn_gru']['total_hours']+estimates['transformer']['total_hours']+beam_hours+baseline_beam_hours
    remaining = None if args.gpu_budget_hours is None else args.gpu_budget_hours-args.spent_gpu_hours-base_hours-reserve
    proposed_remaining = proposed_budget-args.spent_gpu_hours-base_hours-reserve
    ranker_fits = remaining is not None and ranker_allocation_hours <= remaining
    rows.append(['Ensemble', 'Deferred', hardware['gpu'] or 'GPU', 'Unmeasured', 'Unestimated; no extra model committed',
                 'If estimated additional cost exceeds remaining budget, retain the single Transformer; otherwise require held-out MRR@25 gain'])
    projections = dict(estimates=estimates, beam_hours=beam_hours, cnn_validation_beam_hours=baseline_beam_hours, ranker_gpu_hours=generation_hours,
                       ranker_chemistry_cpu_hours=chemistry_hours, ranker_fit_cpu_proxy_hours=cpu['seconds']/3600,
                       full_validation_beam_hours=full_beam_hours,
                       full_holdout_gpu_compute_hours=full_generation_hours,
                       full_holdout_chemistry_cpu_hours=full_chemistry_hours,
                       full_holdout_inline_allocation_hours_before_fit=full_inline_hours,
                       ranker_subset_inline_allocation_hours_including_fit_proxy=ranker_allocation_hours,
                       base_plan_models=['cnn_gru','transformer'], base_plan_gpu_hours=base_hours, evaluation_retry_reserve_hours=reserve,
                       minimum_total_budget_with_reserve_and_prior_spend=base_hours+reserve+args.spent_gpu_hours,
                       proposed_gpu_budget_hours=proposed_budget,
                       proposed_remaining_gpu_hours_after_base_and_reserve=proposed_remaining,
                       ranker_within_proposed_budget=ranker_allocation_hours<=proposed_remaining,
                       workloads=dict(training_spectra_per_epoch=manifest['training_spectra'], epochs=args.epochs,
                                      validation_partition_spectra=manifest['validation_spectra'],
                                      teacher_forced_validation_spectra_per_event=estimates['transformer']['validation_spectra_per_event'],
                                      subset_beam_validation_spectra=beam_eval_spectra, test_spectra=args.prediction_spectra,
                                      ranker_subset_spectra_cap=subset_spectra, ranker_subset_structures=args.ranker_structures,
                                      ranker_subset_spectra_per_structure_cap=args.spectra_per_structure,
                                      ranker_full_holdout_spectra=manifest['holdout_spectra']),
                       benchmark_budget_hours=args.benchmark_budget_hours,
                       gpu_budget_hours=args.gpu_budget_hours, spent_gpu_hours=args.spent_gpu_hours,
                       remaining_gpu_hours_after_base_plan=remaining, ranker_within_budget=ranker_fits,
                       estimates_only=True, queue_wait_included=False, full_training_authorized=False)
    dump(folder/'projections.json', projections)
    settings = transformer['settings']
    baseline = models['cnn_gru']['settings']
    timing = transformer['training']
    lengths = hardware['sample_token_lengths']['train']
    lines = ['# Computational benchmark and execution plan', '',
             '**Prediction quality remains untested.** These measurements **support preliminary compute-feasibility estimates**; they do not establish structure-level recall, MRR@25, or generalization. All projected total times are estimates and exclude queue wait.', '',
             f'The planned baseline is **CNN + GRU**, a CNN-to-RNN model with a {baseline["d"]}-dimensional context and {baseline["dec_layers"]} GRU decoder layers. '
             'The CNN captures patterns along a fixed mass axis; the recurrent decoder generates SMILES token by token. '
             'The primary model retains the sinusoidal peak encoder, spectrum metadata, Transformer encoder, and causal SMILES decoder: attention can combine fragment information across the peak set while conditioning each output token on the spectrum. '
             '**Pooled peak embeddings + GRU is auxiliary only**, not the planned baseline or a required training run. Its measurements are preserved but excluded from the CNN + GRU plus Transformer budget. No pooled-GRU timings are transferred to CNN + GRU.', '',
             f'The CNN baseline was benchmarked separately using two input channels (fragment intensities and precursor indicator) on **[0, {baseline["cnn_max_mz"]:g}) m/z** with **{baseline["cnn_bin_width"]:g}-m/z bins** '
             f'({math.ceil(baseline["cnn_max_mz"]/baseline["cnn_bin_width"]):,} bins/channel). Its three Conv1D layers have channels **32, 64, {baseline["d"]}**, kernel size **9** and stride **4** each, '
             f'followed by adaptive average pooling to {baseline["cnn_pool_bins"]} cells and projection to width {baseline["d"]}. Metadata conditions the initial GRU state. '
             f'It uses the same retained spectral peaks, tokenizer, batch and token cap as the Transformer; input masses outside the grid fail explicitly. The CNN has {models["cnn_gru"]["parameters"]:,} parameters. '
             'The supplied PDF specified CNN-to-RNN without these layer/binning details; these are the concrete pilot settings.', '',
             f'The measured Transformer has **{settings["enc_layers"]} encoder layers, {settings["dec_layers"]} decoder layers, width {settings["d"]}, '
             f'{settings["heads"]} attention heads, feed-forward width {settings["ff"]}, dropout {settings["dropout"]}, '
             f'pre-LayerNorm={settings["norm_first"]}, {settings["vocab"]} vocabulary tokens, and {transformer["parameters"]:,} parameters**. '
             f'The Transformer cap is **{transformer["max_peaks"]+1} total encoder positions**: {transformer["max_peaks"]-1} fragment peaks + one precursor + one metadata token. '
             f'The decoder limit is **{transformer["max_len"]} SMILES tokens including BOS/EOS**. These settings supersede the original PDF\'s earlier 128-total-input/128-output-token description. '
             f'Microbatch **{transformer["batch"]}**, accumulation **{transformer["accumulation"]}**, effective batch **{transformer["effective_batch"]}**; '
             f'**{hardware["precision"].upper()} autocast** with FP32 model/optimizer state on one **{hardware["gpu"] or "CPU smoke test"}** '
             f'(PyTorch {hardware["torch"]}, CUDA {hardware["cuda"]}).', '',
             f'Timing procedure: {timing["warmup_steps"]} warm-up optimizer updates, then **{timing["optimizer_steps"]} timed complete updates per model**, '
             f'processing **{timing["spectra"]:,} spectra** by cycling the **{manifest["sample_spectra"]["Training"]:,}-spectrum** training sample. '
             'Each update includes forward loss, backward pass, token-normalized accumulation, gradient clipping, and AdamW; loading/tokenization and device transfer are included. '
             'CUDA is synchronized before and after the measured interval. Both models share identity-disjoint partitions, a training-only tokenizer, metadata, and preprocessing. '
             f'Padding is dynamic: the sampled training targets have median {lengths["p50"]:.0f}, P95 {lengths["p95"]:.0f}, and maximum {lengths["max"]:.0f} tokens. '
             'The full-cap memory stress check is one separate optimizer update; its memory result does not imply full-cap training throughput.', '',
             '**Full-dataset loading and host-memory feasibility have not been demonstrated.** Cycling 4,096 spectra measures a reused in-memory sample, not a 2,539,608-row Parquet load or sustained full-dataset I/O. '
             'The original 2.19 GiB PBS peak host-RAM figure applies only to that small-sample job. It is not a bound or estimate for full-dataset RAM. '
             'The current training loader reads the full Parquet file into pandas. A compute-node full-load/RAM profile or a streaming loader is needed before a production run; the extra loading allowance below is an assumption, not evidence of full-data behavior.', '',
             f'Beam timing uses **{beam["spectra"]} validation spectra**, beam width **{beam["beam_size"]}**, inference batch **{args.infer_batch}**, '
             f'and no KV cache. Sampling uses **{generation["spectra"]} reserved spectra**, **{generation["samples_per_spectrum"]} samples/spectrum**, '
             f'inference batch **{args.infer_batch}**, and sample chunk **{args.sample_chunk}**. The short-run weights do not establish prediction quality; remeasure decoding on a validated checkpoint. '
             'Memory in the table is PyTorch peak allocated memory; peak reserved memory is preserved in the linked raw records.', '',
             '| Stage | Priority | Hardware | Measured speed/memory | Estimated total time | Proceed/fallback condition |',
             '|---|---|---|---|---|---|']
    lines += ['| '+' | '.join(row)+' |' for row in rows]
    if hardware['cpu_smoke_only']:
        lines += ['', '**CPU smoke test only: these rates must not be used as GPU cost estimates.**']
    lines += ['', f"Training uses {manifest['training_spectra']:,} length-filtered spectra × {args.epochs} planned epochs. "
              f"Formula: spectra × epochs / measured spectra/s / 3600, plus measured validation and checkpoint costs, "
              f"{args.loading_allowance:.0%} extra loading allowance, and preparation time. Steady-state data loading is already timed.",
              '', f'The **{estimates["transformer"]["total_hours"]:.2f}-hour Transformer estimate** is an extrapolation for the stated {args.epochs}-epoch case, not a completed training run. '
              f'Training overhead uses teacher-forced loss validation on at most **{estimates["transformer"]["validation_spectra_per_event"]:,} spectra per event**, '
              f'for **{estimates["transformer"]["validation_events"]} events** (periodic checks plus epoch ends), extrapolated from {manifest["sample_spectra"]["Validation"]} measured validation spectra. '
              f'The validation partition contains **{manifest["validation_spectra"]:,} spectra**. The base plan beam-decodes only **{beam_eval_spectra} validation spectra**, '
              'once per planned model, plus Transformer test decoding. Full-partition Transformer beam evaluation is a separate alternative in the table; it is not silently included at every checkpoint.',
              '', f'The capped ranker workload is **{args.ranker_structures:,} structures × up to {args.spectra_per_structure} spectra = at most {subset_spectra:,} spectra**; '
              'its estimate assumes the cap is reached. It is a subset of the reserved partition. '
              f'All **{manifest["holdout_spectra"]:,} reserved spectra** would instead require approximately **{full_generation_hours:.2f} GPU compute-hours**, '
              f'excluding **{full_chemistry_hours:.2f} estimated CPU chemistry-hours** and unmeasured full-holdout ranker fitting. These alternatives are not additive.',
              '', f'GPU compute time and GPU allocation time differ when CPU stages run inline. With the current sequential pipeline, retaining the GPU during subset chemistry and fitting would cost approximately '
              f'**{ranker_allocation_hours:.2f} allocation-hours**, including the fitting proxy. Full-holdout inline generation alone would cost approximately **{full_inline_hours:.2f} allocation-hours before fitting**. '
              'Budget checks conservatively use allocation time for the capped ranker, not GPU compute time alone.',
              '', 'CPU ranker timing uses synthetic labels only; it provides no evidence of recall, ranking quality or validation gain. '
              'It times a fit on 70% of synthetic groups plus a refit on all groups, each with 250 boosting trees, 15 leaves and one CPU thread. '
              'Decoding speed is conditional on beam width, generated lengths and benchmark weights. Remeasure on the first validated checkpoint.',
              '', 'No ranker or ensemble is committed until the budget is specified and held-out quality supports it. '
              '**If the estimated additional cost exceeds the remaining budget, retain the single Transformer.**']
    lines += ['', f'The recalculated base plan (**CNN + GRU plus Transformer**, each {args.epochs} epochs, and the subset beam evaluation/test pass) requires an estimated **{base_hours:.2f} GPU allocation-hours**. '
              f'Reserve an additional **{reserve:.2f} GPU-hours** for extra structure-level evaluation and retries, separate from already estimated validation, checkpointing, and loading costs. '
              'Limit extra evaluations and partial retries to this reserve; a full repeated training run requires a revised allocation. '
              f'Including **{args.spent_gpu_hours:.3f} hours** already spent, the minimum total would be **{base_hours+reserve+args.spent_gpu_hours:.2f} GPU-hours**. '
              'The earlier 9.60-hour base and 11.65-hour minimum budgeted pooled-GRU plus Transformer; those figures do not describe the planned CNN + RNN comparison and are superseded here.',
              '', f'A **proposed {proposed_budget:.2f}-GPU-hour total plan**, not an approved allocation, would leave **{proposed_remaining:.2f} GPU-hours** after the base plan, prior spend, and reserve. '
              f'The capped ranker needs approximately **{ranker_allocation_hours:.2f} allocation-hours**; '
              + ('it fits this proposed budget but still requires validation gain.' if ranker_allocation_hours<=proposed_remaining else 'it does not fit, so retain the single Transformer and defer the ranker/ensemble under this proposal.')]
    if remaining is None:
        lines += ['', '**Actual full-training GPU allocation: not specified or approved.** The proposed plan cannot be treated as available resources; full training and optional stages remain deferred.']
    else:
        lines += ['', f'Actual stated total project budget: {args.gpu_budget_hours:.2f} GPU-hours. Remaining after the base plan, prior spend, and evaluation/retry reserve: {remaining:.2f} GPU-hours.']
    if args.benchmark_budget_hours is not None:
        lines += ['', f'The **confirmed measurement-only allowance is {args.benchmark_budget_hours:.2f} hours**. '
                  f'Recorded GPU allocation usage, including the failed startup attempt: **{args.spent_gpu_hours:.3f} hours**. '
                  f'The unspent measurement allowance is {args.benchmark_budget_hours-args.spent_gpu_hours:.3f} hours; it is not an approved full-training budget.']
    context_path = folder/'run_context.json'
    if context_path.exists():
        context = json.loads(context_path.read_text())
        lines += ['', f'PBS job {context["successful_pbs_job"]} exited successfully. Total GPU allocation time including retry: '
                  f'{context["total_gpu_allocation_seconds"]} seconds across recorded jobs. Original small-sample job peak host RAM: {context["peak_host_ram_gib"]:.2f} GiB; this does not characterize full-dataset loading. '
                  f'The job reserved {context["pbs_cpu_cores_requested"]} cores; PyTorch/BLAS were capped at {context["pytorch_and_blas_thread_cap"]} threads and ranker fitting at {context["ranker_fit_thread_cap"]}. '
                  'Heavy computation ran on the allocated compute node. The benchmark’s 10 original correctness checks passed locally and in PBS; these do not establish prediction quality.']
        if 'cnn_successful_pbs_job' in context:
            lines += ['', f'The separate CNN + GRU job **{context["cnn_successful_pbs_job"]}** exited successfully after {context["cnn_job_wall_seconds"]} allocation seconds, '
                      f'with {context["cnn_peak_host_ram_gib"]:.2f} GiB peak host RAM for the small sample. Its 12 correctness checks passed in PBS. '
                      'This second small-sample RAM figure also does not bound full-dataset loading.']
    lines += ['', 'Supporting records: [hardware and data scope](hardware.json), [CNN + GRU timings](cnn_gru.json), [CNN job hardware](cnn_hardware.json), [Transformer timings](transformer.json), '
              '[pooled-GRU timings](pooled_gru.json), [CPU fitting proxy](ranker_cpu.json), [cost calculations](projections.json), '
              'and [PBS accounting](pbs_accounting.txt). Configuration and procedure are stated above so this report is readable independently of those files. '
              'The separately provided evidence ZIP contains these raw records, logs, source snapshots and checksums for inspection.']
    (folder/'report.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('command', choices=['prepare', 'run', 'report'])
    ap.add_argument('--train', default='dataset/train.parquet')
    ap.add_argument('--preprocessing', default='casmi_generative_ranker/preprocessing_results')
    ap.add_argument('--data', default='artifacts/benchmark_data')
    ap.add_argument('--output', default='artifacts/benchmark')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--vocab', type=int, default=512)
    ap.add_argument('--max-len', type=int, default=160)
    ap.add_argument('--max-peaks', type=int, default=128)
    ap.add_argument('--train-sample', type=int, default=4096)
    ap.add_argument('--eval-structures', type=int, default=128)
    ap.add_argument('--spectra-per-structure', type=int, default=5)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--allow-cpu-smoke', action='store_true')
    ap.add_argument('--bf16', action='store_true')
    ap.add_argument('--cpu-threads', type=int, default=4)
    ap.add_argument('--workers', type=int, default=0)
    ap.add_argument('--d', type=int, default=384)
    ap.add_argument('--heads', type=int, default=8)
    ap.add_argument('--enc-layers', type=int, default=4)
    ap.add_argument('--dec-layers', type=int, default=4)
    ap.add_argument('--ff', type=int, default=1536)
    ap.add_argument('--baseline-d', type=int, default=128)
    ap.add_argument('--baseline-layers', type=int, default=2)
    ap.add_argument('--architectures', nargs='+', choices=['cnn_gru', 'transformer', 'pooled_gru'], default=['cnn_gru', 'transformer'])
    ap.add_argument('--skip-ranker-proxy', action='store_true')
    ap.add_argument('--skip-report', action='store_true')
    ap.add_argument('--cnn-max-mz', type=float, default=4096.)
    ap.add_argument('--cnn-bin-width', type=float, default=.1)
    ap.add_argument('--cnn-pool-bins', type=int, default=16)
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--accum', type=int, default=4)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--warmup', type=int, default=20)
    ap.add_argument('--steps', type=int, default=150)
    ap.add_argument('--decode-spectra', type=int, default=64)
    ap.add_argument('--candidate-spectra', type=int, default=16)
    ap.add_argument('--infer-batch', type=int, default=8)
    ap.add_argument('--beam-size', type=int, default=25)
    ap.add_argument('--length-penalty', type=float, default=1.)
    ap.add_argument('--max-tokens', type=int, default=0)
    ap.add_argument('--samples', type=int, default=100)
    ap.add_argument('--sample-chunk', type=int, default=100)
    ap.add_argument('--temperature', type=float, default=1.)
    ap.add_argument('--decoding', choices=['sample'], default='sample')
    ap.add_argument('--progress-every', type=int, default=0)
    ap.add_argument('--ranker-structures', type=int, default=2000)
    ap.add_argument('--ranker-candidates', type=int, default=500)
    ap.add_argument('--prediction-spectra', type=int, default=1213)
    ap.add_argument('--epochs', type=int, default=10)
    ap.add_argument('--eval-every', type=int, default=2000)
    ap.add_argument('--val-max', type=int, default=5000)
    ap.add_argument('--loading-allowance', type=float, default=.1)
    ap.add_argument('--gpu-budget-hours', type=float, help='Actual total project GPU allocation, including already recorded measurement spend')
    ap.add_argument('--proposed-gpu-budget-hours', type=float, default=12., help='Illustrative full-run total; not an allocation or authorization')
    ap.add_argument('--evaluation-retry-reserve-hours', type=float, default=2., help='Additional GPU allocation time kept for extra evaluation and retries')
    ap.add_argument('--benchmark-budget-hours', type=float, help='Measurement-only allowance; enforce job runtime with PBS walltime')
    ap.add_argument('--spent-gpu-hours', type=float, default=0.)
    args = ap.parse_args()
    if args.command == 'prepare':
        prepare(args)
    elif args.command == 'run':
        run(args)
    else:
        report(args)


if __name__ == '__main__':
    main()
