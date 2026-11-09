"""Package the raw compute records separately from (and for attachment to) the PDF."""
import csv
import hashlib
import io
import json
from pathlib import Path
import tarfile
import zipfile

ROOT = Path(__file__).resolve().parent.parent
RECORDS = ROOT/'casmi_generative_ranker/benchmark_results'
OUTPUT = ROOT/'output/benchmark_evidence.zip'


def main():
    files = {}
    for name in ('hardware.json','cnn_hardware.json','cnn_gru.json','transformer.json',
                 'pooled_gru.json','ranker_cpu.json','data_manifest.json','projections.json',
                 'run_context.json','python_packages.txt','pbs_accounting.txt',
                 'casmi-bench.o159854','casmi-bench.o159855'):
        files[f'records/{name}'] = (RECORDS/name).read_bytes()
    for path in sorted((RECORDS/'cnn_run').rglob('*')):
        if path.is_file() and '__pycache__' not in path.parts and path.suffix != '.pyc':
            files['records/'+str(path.relative_to(RECORDS))] = path.read_bytes()
    with tarfile.open(ROOT/'artifacts/original_gpu_source.tar.gz') as archive:
        for member in archive.getmembers():
            if member.isfile():
                files['original_gpu_run_source/'+member.name] = archive.extractfile(member).read()
    for name in ('pipeline.py','benchmark.py','molecular_preprocessing.py','test_molecular_preprocessing.py','test_pipeline.py','test_benchmark_report.py','README.md',
                 'requirements.txt','hpc_cnn_benchmark.pbs','revise_phase3_pdf.py','package_benchmark_evidence.py'):
        files[f'current_analysis_source/{name}'] = (ROOT/'casmi_generative_ranker'/name).read_bytes()
    files['analysis/report.md'] = (RECORDS/'report.md').read_bytes()
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(['model','role','GPU','batch','accumulation','warmup_updates','timed_updates',
                     'timed_spectra','training_seconds','training_spectra_per_second',
                     'train_peak_allocated_GiB','full_cap_peak_allocated_GiB',
                     'beam_spectra','beam_width','beam_seconds','beam_spectra_per_second'])
    for model, role in [('cnn_gru','planned baseline'),('transformer','primary'),('pooled_gru','auxiliary only')]:
        row = json.loads((RECORDS/f'{model}.json').read_text())
        train, beam = row['training'], row['beam']
        writer.writerow([model,role,'NVIDIA GeForce RTX 3090',row['batch'],row['accumulation'],
                         train['warmup_steps'],train['optimizer_steps'],train['spectra'],
                         train['seconds'],train['spectra_per_second'],train['peak_memory']['allocated_gib'],
                         row['full_cap_memory']['peak_memory']['allocated_gib'],beam['spectra'],beam['beam_size'],
                         beam['seconds'],beam['spectra_per_second']])
    files['analysis/model_measurements.csv'] = buffer.getvalue().encode()
    files['README.txt'] = b'''COMPUTATIONAL BENCHMARK EVIDENCE

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
'''
    original_pdf = Path('/Users/muddassirzafar/Downloads/Phase 3 Prelimary Data Analysis and Feasibility.pdf')
    files['manifest.json'] = (json.dumps(dict(
        original_pdf_name=original_pdf.name,
        original_pdf_sha256=hashlib.sha256(original_pdf.read_bytes()).hexdigest(),
        baseline='cnn_gru',primary='transformer',auxiliary=['pooled_gru'],
        measurement_allowance_hours=1.5,recorded_gpu_allocation_seconds=216,
        full_training_authorized=False,prediction_quality_tested=False,
        full_dataset_loading_tested=False,independent_verification_claimed=False,
    ),indent=2)+'\n').encode()
    files['CHECKSUMS.sha256'] = ''.join(f'{hashlib.sha256(data).hexdigest()}  {name}\n'
                                      for name,data in sorted(files.items())).encode()
    OUTPUT.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(OUTPUT,'w',zipfile.ZIP_DEFLATED) as archive:
        for name,data in sorted(files.items()):
            archive.writestr(name,data)
    with zipfile.ZipFile(OUTPUT) as archive:
        assert archive.testzip() is None
        for line in archive.read('CHECKSUMS.sha256').decode().splitlines():
            expected,name=line.split('  ',1)
            assert hashlib.sha256(archive.read(name)).hexdigest()==expected,name
    print(f'{OUTPUT}: {len(files)} files, {OUTPUT.stat().st_size:,} bytes; all checksums verified')


if __name__=='__main__':
    main()
