"""One-pass CPU spectral audit and four filtering experiments, without model training.

Reuse the saved molecular manifest and the model's shared peak preparation. Write
row-level flags, replayable corrections, quarantines, duplicate groups and tables.
The original Parquet file and molecular/split artifacts are never overwritten.
"""
import os
for _thread_variable in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_thread_variable] = '1'

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import pickle
import platform
import resource
import sqlite3
import sys
import tempfile
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from molecular_preprocessing import POLICY, RDKIT_VERSION, apply_preprocessing, load_preprocessing
from spectral_preprocessing import clean_fragment_pairs

VARIANTS = [('no_cutoff_127', 0., 127), ('one_percent_127', .01, 127),
            ('no_cutoff_255', 0., 255), ('one_percent_255', .01, 255)]
MISSING_CATEGORIES = ('adduct', 'ionization_mode', 'instrument_type')
UNKNOWN = {'unknown', 'none', 'null', 'nan', 'n/a', 'na', 'unavailable'}
SPLITS = ('Training', 'Validation', 'Ranker holdout')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda: stream.read(4*1024*1024), b''):
            digest.update(data)
    return digest.hexdigest()


def missing(value):
    return value is None or (isinstance(value, str) and not value.strip()) or (
        isinstance(value, (float, np.floating)) and np.isnan(value))


def energy_values(value):
    if value is None:
        return None, 0
    try:
        values = np.asarray(value, dtype=np.float32).reshape(-1)
    except (ValueError, TypeError):
        return None, 1
    valid = np.isfinite(values) & (values >= 0)
    return values[valid].tolist(), int((~valid).sum())


def measurement_hash(row, mz, intensity, energy):
    # Identity and ingest provenance are deliberately excluded. Keep the full
    # cleaned multiset (not just top-127); sort to catch reordered measurements.
    order = np.lexsort((intensity, mz))
    digest = hashlib.blake2b(digest_size=16)
    digest.update(mz[order].astype('<f4', copy=False).tobytes())
    digest.update(intensity[order].astype('<f4', copy=False).tobytes())
    context = (float(row['precursor_mz']), *(row.get(c) for c in MISSING_CATEGORIES),
               tuple(sorted(energy)) if energy else None, len(mz))
    digest.update(pickle.dumps(context, protocol=5))
    return digest.digest()


class DuplicateIndex:
    """Disk-backed global index; memory does not grow with the number of spectra."""
    def __init__(self, path):
        self.db = sqlite3.connect(path)
        self.db.execute('PRAGMA journal_mode=OFF')
        self.db.execute('PRAGMA synchronous=OFF')
        self.db.execute('PRAGMA cache_size=-32768')
        self.db.execute('''CREATE TABLE seen (
          kind INTEGER, digest BLOB, first_row INTEGER, first_identity TEXT,
          first_split TEXT, n INTEGER, eligible_n INTEGER,
          identity_conflict INTEGER, split_conflict INTEGER,
          PRIMARY KEY(kind,digest)) WITHOUT ROWID''')

    def add(self, kind, digest, source_row, identity, split, eligible):
        try:
            self.db.execute('INSERT INTO seen VALUES (?,?,?,?,?,1,?,0,0)',
                            (kind,digest,source_row,identity,split,int(eligible)))
            return None
        except sqlite3.IntegrityError:
            first, first_identity, first_split = self.db.execute(
                'SELECT first_row,first_identity,first_split FROM seen WHERE kind=? AND digest=?',
                (kind,digest)).fetchone()
            self.db.execute('''UPDATE seen SET n=n+1,eligible_n=eligible_n+?,
                identity_conflict=MAX(identity_conflict,?),split_conflict=MAX(split_conflict,?)
                WHERE kind=? AND digest=?''',
                (int(eligible),int(identity!=first_identity),int(split!=first_split),kind,digest))
            return first

    def group_counts(self, kind):
        result = self.db.execute('''SELECT COUNT(*),COALESCE(SUM(n),0),COALESCE(SUM(n-1),0),
             COALESCE(SUM(CASE WHEN eligible_n>1 THEN eligible_n ELSE 0 END),0),
             COALESCE(SUM(MAX(eligible_n-1,0)),0),
             COALESCE(SUM(identity_conflict),0),COALESCE(SUM(split_conflict),0)
             FROM seen WHERE kind=? AND n>1''',(kind,)).fetchone()
        keys = ('groups','member_rows','excess_rows','eligible_member_rows','eligible_excess_rows',
                'groups_with_identity_conflict','groups_with_split_conflict')
        return dict(zip(keys,map(int,result)))

    def close(self):
        self.db.close()


def evaluate_row(row, sparse_below=5):
    """Return audit fields and cleaned arrays; never invent missing measurements."""
    flags = {}
    for field in MISSING_CATEGORIES:
        value = row.get(field)
        flags[f'missing_{field}'] = missing(value)
        flags[f'unknown_{field}'] = isinstance(value,str) and value.strip().lower() in UNKNOWN
    energy, bad_energy = energy_values(row.get('collision_energy_ev'))
    flags.update(missing_collision_energy_ev=not energy, invalid_collision_energy_values=bad_energy)
    reasons = []
    try:
        precursor = float(row['precursor_mz'])
        precursor_valid = bool(np.isfinite(precursor) and 0 < precursor <= np.finfo(np.float32).max)
    except (ValueError, TypeError, KeyError):
        precursor, precursor_valid = None, False
    flags['invalid_precursor'] = not precursor_valid
    if not precursor_valid:
        reasons.append('invalid_precursor')
    try:
        mz, intensity, keep = clean_fragment_pairs(row.get('ms2_mzs'), row.get('ms2_normalized_intensities'))
        with np.errstate(over='ignore',invalid='ignore'):
            raw_mz = np.asarray(row['ms2_mzs'],dtype=np.float32)
            raw_intensity = np.asarray(row['ms2_normalized_intensities'],dtype=np.float32)
        finite = np.isfinite(raw_mz)&np.isfinite(raw_intensity)
        invalid = ~finite | (raw_mz<=0) | (raw_intensity<0)
        flags.update(unusable_fragment_arrays=False,raw_fragment_entries=len(keep),
                     clean_positive_fragments=len(mz),invalid_fragment_pairs=int(invalid.sum()),
                     removed_nonpositive_or_nonfinite_pairs=int((~keep).sum()),
                     nonfinite_pairs=int((~finite).sum()),nonpositive_mz_pairs=int((raw_mz<=0).sum()),
                     negative_intensity_pairs=int((raw_intensity<0).sum()),
                     zero_intensity_pairs=int((finite&(raw_mz>0)&(raw_intensity==0)).sum()),
                     sparse_before=len(keep)<sparse_below,sparse_after=len(mz)<sparse_below,
                     empty_after=len(mz)==0, duplicate_mz_excess=len(mz)-len(np.unique(mz)),
                     above_precursor_peaks=int((mz>precursor).sum()) if precursor_valid else None,
                     intensity_above_one=bool((intensity>1).any()))
    except (ValueError,TypeError):
        mz = intensity = np.array([],dtype=np.float32)
        flags.update(unusable_fragment_arrays=True,raw_fragment_entries=None,clean_positive_fragments=None,
                     invalid_fragment_pairs=None,removed_nonpositive_or_nonfinite_pairs=None,
                     nonfinite_pairs=None,nonpositive_mz_pairs=None,negative_intensity_pairs=None,
                     zero_intensity_pairs=None,sparse_before=None,sparse_after=None,empty_after=None,
                     duplicate_mz_excess=None,above_precursor_peaks=None,intensity_above_one=None)
        reasons.append('unusable_fragment_arrays')
    return flags, reasons, mz, intensity, energy


def filtering_metrics(mz, intensity):
    """Four descriptive arms relative to valid positive, untruncated fragments."""
    order = np.argsort(-intensity)
    values = intensity[order]
    total = float(values.sum(dtype=np.float64))
    prefix = np.r_[0., np.cumsum(values,dtype=np.float64)]
    results = {}
    for name, cutoff, cap in VARIANTS:
        after_cutoff = int((values >= cutoff*float(values[0])).sum()) if len(values) else 0
        kept = min(after_cutoff,cap)
        retained = float(prefix[kept])
        results[name] = dict(kept_peaks=kept,cutoff_removed=len(values)-after_cutoff,
                             cap_removed=after_cutoff-kept,total_removed=len(values)-kept,
                             retained_intensity=retained,
                             retained_fraction=retained/total if total>0 else None)
    return results, total


def audit_schema():
    fields = [('source_row',pa.int64()),('competition_identity',pa.string()),('split',pa.string()),
              ('baseline_usable',pa.bool_()),('quarantine_reason',pa.string()),
              ('strict_record_hash',pa.binary(16)),('measurement_hash',pa.binary(16)),
              ('duplicate_record_of',pa.int64()),('duplicate_measurement_of',pa.int64())]
    for field in MISSING_CATEGORIES:
        fields.extend([(f'missing_{field}',pa.bool_()),(f'unknown_{field}',pa.bool_())])
    fields += [('missing_collision_energy_ev',pa.bool_()),('invalid_collision_energy_values',pa.int32()),
               ('invalid_precursor',pa.bool_()),('unusable_fragment_arrays',pa.bool_())]
    for name in ('raw_fragment_entries','clean_positive_fragments','invalid_fragment_pairs',
                 'removed_nonpositive_or_nonfinite_pairs','nonfinite_pairs','nonpositive_mz_pairs',
                 'negative_intensity_pairs','zero_intensity_pairs','duplicate_mz_excess','above_precursor_peaks'):
        fields.append((name,pa.int32()))
    for name in ('sparse_before','sparse_after','empty_after','intensity_above_one'):
        fields.append((name,pa.bool_()))
    for name,_,_ in VARIANTS:
        for metric in ('kept_peaks','cutoff_removed','cap_removed','total_removed'):
            fields.append((f'{name}_{metric}',pa.int32()))
        fields.extend([(f'{name}_retained_intensity',pa.float64()),(f'{name}_retained_fraction',pa.float64())])
    return pa.schema(fields)


def correction_schema():
    return pa.schema([('source_row',pa.int64()),('competition_identity',pa.string()),('split',pa.string()),
                      ('normalized_smiles',pa.string()),('ms2_mzs',pa.list_(pa.float32())),
                      ('ms2_normalized_intensities',pa.list_(pa.float32())),
                      ('collision_energy_ev',pa.list_(pa.float32())),('removed_fragment_entries',pa.int32()),
                      ('removed_collision_energy_values',pa.int32())])


def summarize_duplicates(index, output):
    schema = pa.schema([('kind',pa.string()),('fingerprint',pa.binary(16)),('first_source_row',pa.int64()),
                        ('first_identity',pa.string()),('first_split',pa.string()),('rows',pa.int64()),
                        ('eligible_rows',pa.int64()),('identity_conflict',pa.bool_()),('split_conflict',pa.bool_())])
    memberships = {0:set(),1:set()}
    conflicts = {name:set() for name in ('identity','split')}
    cursor = index.db.execute('SELECT * FROM seen WHERE n>1 ORDER BY kind,digest')
    with pq.ParquetWriter(output/'duplicate_groups.parquet',schema,compression='zstd') as writer:
        while True:
            rows=cursor.fetchmany(4096)
            if not rows:break
            records=[]
            for kind,digest,first,identity,split,count,eligible,identity_conflict,split_conflict in rows:
                memberships[kind].add(digest)
                if kind==1 and identity_conflict:conflicts['identity'].add(digest)
                if kind==1 and split_conflict:conflicts['split'].add(digest)
                records.append(dict(kind='exact_record' if kind==0 else 'cleaned_measurement',fingerprint=digest,
                                    first_source_row=first,first_identity=identity,first_split=split,rows=count,
                                    eligible_rows=eligible,identity_conflict=bool(identity_conflict),split_conflict=bool(split_conflict)))
            writer.write_table(pa.Table.from_pylist(records,schema=schema))
    return memberships,conflicts


def finalize_rows(staged, output, memberships, conflicts):
    source=pq.ParquetFile(staged)
    extras=['duplicate_record_member','duplicate_measurement_member','measurement_identity_conflict','measurement_split_conflict']
    schema=source.schema_arrow.append(pa.field(extras[0],pa.bool_()))
    for name in extras[1:]:schema=schema.append(pa.field(name,pa.bool_()))
    with pq.ParquetWriter(output/'audit_rows.parquet',schema,compression='zstd') as writer:
        for batch in source.iter_batches(batch_size=8192,use_threads=False):
            table=pa.Table.from_batches([batch])
            strict=table['strict_record_hash'].to_pylist();measurement=table['measurement_hash'].to_pylist()
            values=[[v in memberships[0] for v in strict], [v in memberships[1] for v in measurement],
                    [v in conflicts['identity'] for v in measurement],[v in conflicts['split'] for v in measurement]]
            for name,flags in zip(extras,values):table=table.append_column(name,pa.array(flags,type=pa.bool_()))
            writer.write_table(table)


def experiment_table(aggregates, output):
    records=[]
    for split in ('All',)+SPLITS:
        for name,cutoff,cap in VARIANTS:
            count=aggregates[(split,name)]
            records.append(dict(split=split,variant=name,relative_cutoff=cutoff,fragment_cap=cap,
                transformer_encoder_cap=cap+2,eligible_spectra=count['spectra'],
                spectra_affected=count['affected'],spectra_affected_by_cutoff=count['affected_cutoff'],
                spectra_affected_by_cap=count['affected_cap'],input_clean_peaks=count['input_peaks'],
                removed_by_cutoff=count['cutoff_removed'],removed_by_cap=count['cap_removed'],
                total_removed_peaks=count['total_removed'],retained_peaks=count['kept_peaks'],
                positive_intensity_spectra=count['positive_spectra'],zero_intensity_spectra=count['spectra']-count['positive_spectra'],
                spectra_below_sparse_flag=count['sparse'],retained_intensity=count['retained_intensity'],
                input_clean_intensity=count['input_intensity'],
                weighted_retained_intensity_pct=100*count['retained_intensity']/count['input_intensity'] if count['input_intensity'] else None,
                mean_retained_intensity_pct=100*count['ratio_sum']/count['positive_spectra'] if count['positive_spectra'] else None))
    frame=pd.DataFrame(records)
    frame.to_csv(output/'filtering_experiments.csv',index=False)
    return records


def before_after_table(counts, duplicate_counts, output):
    records=[]
    def add(metric,before,after,status):
        records.append(dict(metric=metric,before=int(before),after_baseline=int(after),action=status))
    add('Source records (immutable)',counts['rows'],counts['rows'],'Original file preserved; all source rows audited')
    add('Records available in cleaned baseline view',counts['rows'],counts['eligible'],'Exclude only quarantined records from the usable view')
    add('Invalid molecular-target rows',counts['invalid_molecule'],0,'Reuse saved invalid status; quarantine')
    add('Invalid precursor rows',counts['all_invalid_precursor'],0,'Quarantine; do not infer precursor')
    add('Unusable/misaligned fragment-array rows',counts['all_unusable_fragment_arrays'],0,'Quarantine; do not guess peak alignment')
    add('Invalid fragment pairs (usable cohort)',counts['eligible_invalid_fragment_pairs'],0,'Remove nonfinite, nonpositive-m/z or negative-intensity pairs')
    add('Zero-intensity fragment pairs (usable cohort)',counts['eligible_zero_intensity_pairs'],0,'Remove noninformative zero pairs, matching existing input preparation')
    add('Exact record duplicate excess rows',duplicate_counts['exact_record']['excess_rows'],duplicate_counts['exact_record']['eligible_excess_rows'],'Retain and flag; repeated molecules alone are not duplicates')
    for field in MISSING_CATEGORIES:
        add(f'Missing {field} rows',counts[f'all_missing_{field}'],counts[f'eligible_missing_{field}'],'Retain and flag; no imputation')
        add(f'Unknown-placeholder {field} rows',counts[f'all_unknown_{field}'],counts[f'eligible_unknown_{field}'],'Retain original value and flag separately from missing')
    add('Missing usable collision-energy rows',counts['all_missing_collision_energy_ev'],counts['eligible_missing_collision_energy_ev'],'Retain and flag missingness')
    add('Invalid collision-energy values (usable cohort)',counts['eligible_invalid_collision_energy_values'],0,'Ignore nonfinite/negative components, matching existing metadata preparation')
    add('Spectra below sparse flag (aligned usable cohort)',counts['eligible_sparse_before'],counts['eligible_sparse_after'],'Retain and flag; before counts supplied entries, after counts positive valid fragments')
    add('Precursor-only usable spectra',counts['eligible_empty_after'],counts['eligible_empty_after'],'Retain and flag; no positive fragment signal')
    add('Usable spectra with repeated m/z values',counts['eligible_duplicate_mz_rows'],counts['eligible_duplicate_mz_rows'],'Retain and flag; no peak merging')
    add('Usable spectra with above-precursor fragments',counts['eligible_above_precursor_rows'],counts['eligible_above_precursor_rows'],'Retain and flag; no above-precursor deletion rule')
    pd.DataFrame(records).to_csv(output/'before_after_audit.csv',index=False)
    return records


def write_report(report, before_after, experiments, output):
    lines=['# Phase 3 CPU preprocessing audit','',
           f'Full source scan: **{report["counts"]["rows"]:,} training-file records**, with saved generator-training, validation and ranker-holdout identities. No model training or prediction-accuracy comparison was performed.','',
           '## Before/after audit','',
           'Before record-level counts cover all source rows; after counts cover the baseline-usable view, except the immutable source-row total. Fragment-pair counts use the same usable cohort before and after. Reasons can overlap and must not be summed as disjoint record counts.','',
           '| Measure | Before | After baseline cleaning | Action / status |','|---|---:|---:|---|']
    lines += [f'| {r["metric"]} | {r["before"]:,} | {r["after_baseline"]:,} | {r["action"]} |' for r in before_after]
    lines += ['',f'Quarantined records: **{report["counts"]["quarantined"]:,}**. Cleaned correction patches: **{report["counts"]["correction_rows"]:,}**. '
              'Quarantined source records remain recoverable from the immutable source file and audit row index; no ambiguous record is silently repaired. The baseline uses no relative-intensity cutoff.','',
              '## Filtering experiments: four arms','',
              'All four arms start from the same baseline-usable cohort and the same finite, positive, untruncated fragment pairs. '
              'The 1% arm removes values strictly below 1% of that spectrum\'s base peak, before applying the fragment cap; values exactly at 1% remain. '
              '127 and 255 are **fragment counts**, corresponding to 129 and 257 total Transformer positions after precursor and metadata. These are experiments, not adopted cleaning rules.','',
              '| Cutoff | Fragment cap | Eligible spectra | Affected spectra | Peaks removed by cutoff | Peaks removed by cap | Total peaks removed | Mean retained intensity | Weighted retained intensity |','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for row in experiments:
        if row['split']!='All':continue
        fmt=lambda x:'Undefined' if x is None else f'{x:.4f}%'
        lines.append(f'| {row["relative_cutoff"]:.0%} | {row["fragment_cap"]} | {row["eligible_spectra"]:,} | {row["spectra_affected"]:,} | {row["removed_by_cutoff"]:,} | {row["removed_by_cap"]:,} | {row["total_removed_peaks"]:,} | {fmt(row["mean_retained_intensity_pct"])} | {fmt(row["weighted_retained_intensity_pct"])} |')
    lines += ['', 'Counts affected by cutoff and cap are also recorded separately in filtering_experiments.csv, overall and by the three saved partitions. '
              'Mean retention averages per-spectrum ratios; weighted retention divides summed retained intensity by summed valid input intensity. '
              'The artificial precursor and invalid/zero pairs are excluded from both intensity denominators. Precursor-only spectra remain in record/peak counts but have undefined intensity retention. '
              'Retained intensity is not retained chemical information: weak diagnostic fragments may matter.','',
              '## Duplicates and uncertain cases','',
              'Exact-record fingerprints include all supplied columns in schema order. Cleaned-measurement fingerprints include the full cleaned fragment multiset, precursor and conditioning metadata, while ignoring molecular labels and ingest provenance. '
              'The latter catches reordered peak arrays and identical inputs assigned to different identities; those can be ambiguous measurements, not proof of an incorrect label or split. No duplicate is deleted and no identity is moved.','',
              f'- Exact-record duplicate groups: {report["duplicates"]["exact_record"]["groups"]:,}; excess rows: {report["duplicates"]["exact_record"]["excess_rows"]:,}.',
              f'- Cleaned-measurement duplicate groups: {report["duplicates"]["cleaned_measurement"]["groups"]:,}; excess rows: {report["duplicates"]["cleaned_measurement"]["excess_rows"]:,}.',
              f'- Measurement groups spanning identities: {report["duplicates"]["cleaned_measurement"]["groups_with_identity_conflict"]:,}; spanning saved partitions: {report["duplicates"]["cleaned_measurement"]["groups_with_split_conflict"]:,}.','',
              f'The sparse-spectrum flag means fewer than **{report["policy"]["sparse_below"]}** positive valid fragments. It is diagnostic only. Missing/unknown metadata, repeated m/z values and fragments above precursor m/z are retained and flagged; no mass tolerance, imputation or merging rule is adopted.','',
              '## Implemented versus planned','',
              '| Step | Status | Evidence / next step |','|---|---|---|',
              '| Molecular normalization and identity assignment | Reused, implemented | Existing pinned RDKit manifest; no renormalization or resplitting |',
              '| Spectral audit and unambiguous cleaning | Implemented on CPU | Complete row flags, quarantine reasons and replayable cleaned patches |',
              '| No-cutoff/1% and 127/255 comparisons | Measured on CPU | Full-scan counts and intensity retention; experimental thresholds only |',
              '| Missing metadata, sparse spectra, duplicates | Retained and flagged | No automatic deletion or identity reassignment |',
              '| Training preprocessing variants | Planned modeling stage | Compare candidate recall and validation MRR@25 on unchanged identities |',
              '| Select a cutoff/cap based on prediction accuracy | Not established | Requires model training/validation; intensity retention alone is insufficient |','',
              '## Reproduce and use the baseline view','',
              '```bash',
              'python casmi_generative_ranker/audit_spectra.py \\',
              '  --train dataset/train.parquet \\',
              '  --preprocessing casmi_generative_ranker/preprocessing_results \\',
              '  --output output/cpu_preprocessing_audit',
              '```','',
              'The script is single-process, reads Parquet batches without Arrow worker threads, and uses a disk-backed duplicate index. '
              'audit_rows.parquet preserves every source_row and its original identity/split. cleaned_corrections.parquet contains only usable rows needing a measurement patch; it is not a complete copy of the dataset. '
              'quarantined_rows.parquet lists unusable rows and duplicate_groups.parquet identifies all duplicate members, including each group\'s first row. '
              'Replay the usable view with apply_audit_to_frame() or pipeline.py train --spectral-audit; only the modeling stage applies the current top-127 cap.','',
              f'Runtime: {report["runtime"]["seconds"]:.1f} seconds; process peak RSS: {report["runtime"]["peak_rss_gib"]:.3f} GiB. '
              'This bounded-memory audit is not evidence that the existing full-pandas training loader fits the same RAM budget. '
              'audit_report.json records source/manifest/code hashes, versions, policies, row counts and partition counts.']
    (output/'report.md').write_text('\n'.join(lines)+'\n')


def apply_audit_to_frame(frame, directory, source_path):
    """Replay baseline flags/patches before existing molecular preprocessing."""
    directory=Path(directory)
    report=json.loads((directory/'audit_report.json').read_text())
    if report['normalization_policy']!=POLICY or report['rdkit_version']!=RDKIT_VERSION:
        raise ValueError('Spectral audit normalization policy/version mismatch')
    if len(frame)!=report['counts']['rows'] or sha256(source_path)!=report['source']['sha256']:
        raise ValueError('Spectral audit belongs to different source data')
    if not frame.index.equals(pd.RangeIndex(len(frame))):
        raise ValueError('Replay requires the original zero-based source-row order')
    lookup=pd.read_parquet(directory/'audit_rows.parquet',columns=['source_row','baseline_usable'])
    if lookup.source_row.tolist()!=list(range(len(frame))):raise ValueError('Incomplete spectral audit row index')
    result=frame.loc[lookup.baseline_usable.to_numpy()].copy()
    patches=pd.read_parquet(directory/'cleaned_corrections.parquet')
    if 'collision_energy_ev' not in result:
        result['collision_energy_ev']=pd.Series(None,index=result.index,dtype=object)
    for row in patches.itertuples(index=False):
        for key in ('ms2_mzs','ms2_normalized_intensities','collision_energy_ev'):
            value=getattr(row,key)
            # Pandas may unbox a length-one ndarray into a zero-dimensional
            # object during scalar assignment. Preserve one-dimensional lists.
            result.at[row.source_row,key]=value.tolist() if isinstance(value,np.ndarray) else value
    return result


def run(args):
    started=time.perf_counter()
    pa.set_cpu_count(1);pa.set_io_thread_count(1)
    source_path=Path(args.train).resolve();output=Path(args.output)
    if args.batch_size<1 or args.sparse_below<1:raise ValueError('Batch size and sparse flag must be positive')
    output.mkdir(parents=True,exist_ok=True)
    known=['audit_rows.parquet','cleaned_corrections.parquet','quarantined_rows.parquet',
           'duplicate_groups.parquet','before_after_audit.csv','filtering_experiments.csv','audit_report.json','report.md']
    if any((output/name).exists() for name in known) and not args.overwrite:
        raise FileExistsError('Audit output already exists; use --overwrite to replace this script\'s artifacts')
    if args.overwrite and (output/'audit_report.json').exists():(output/'audit_report.json').unlink()
    input_stat=source_path.stat();input_hash=sha256(source_path)
    mapping,molecular_report=load_preprocessing(args.preprocessing)
    valid=mapping[mapping.status=='valid']
    if (valid.groupby('competition_identity').split.nunique()>1).any() or (valid.groupby('competition_identity').target_smiles.nunique()>1).any():
        raise ValueError('Existing molecular manifest contains conflicting partitions or targets')
    indexed=mapping.drop_duplicates('input_smiles').set_index('input_smiles',drop=False)
    counts=Counter(rows=0,eligible=0,quarantined=0,correction_rows=0,invalid_molecule=0)
    aggregates={(split,name):Counter() for split in ('All',)+SPLITS for name,_,_ in VARIANTS}
    partitions={split:Counter() for split in SPLITS};last_print=time.perf_counter()
    schema=audit_schema();correction=correction_schema()
    with tempfile.TemporaryDirectory(prefix='.audit-',dir=output) as scratch:
        index=DuplicateIndex(Path(scratch)/'duplicates.sqlite')
        try:
            with pq.ParquetFile(source_path,pre_buffer=False) as source, \
                 pq.ParquetWriter(Path(scratch)/'rows.parquet',schema,compression='zstd') as writer, \
                 pq.ParquetWriter(output/'cleaned_corrections.parquet',correction,compression='zstd') as patch_writer, \
                 pq.ParquetWriter(output/'quarantined_rows.parquet',schema,compression='zstd') as quarantine_writer:
                if source.metadata.num_rows!=molecular_report['rows_scanned']:raise ValueError('Source row count differs from molecular manifest')
                required={'normalized_smiles','ms2_mzs','ms2_normalized_intensities','precursor_mz'}
                if not required<=set(source.schema_arrow.names):raise ValueError(f'Missing source columns: {sorted(required-set(source.schema_arrow.names))}')
                columns=source.schema_arrow.names
                for batch in source.iter_batches(batch_size=args.batch_size,use_threads=False):
                    rows=batch.to_pylist()
                    molecular=pd.DataFrame({'normalized_smiles':[row['normalized_smiles'] for row in rows]})
                    selected=indexed.reindex(pd.unique(molecular.normalized_smiles)).reset_index(drop=True)
                    normalized=apply_preprocessing(molecular,selected)
                    contexts={int(i):(row.competition_identity,row.split,row.normalized_smiles)
                              for i,row in normalized.iterrows()}
                    audit_records=[];patches=[];quarantines=[]
                    for local,row in enumerate(rows):
                        source_row=counts['rows'];counts['rows']+=1
                        identity,split,target=contexts.get(local,(None,None,None))
                        flags,reasons,mz,intensity,energy=evaluate_row(row,args.sparse_below)
                        if identity is None:reasons.append('invalid_molecular_target');counts['invalid_molecule']+=1
                        usable=not reasons
                        strict=hashlib.blake2b(pickle.dumps(tuple(row.get(c) for c in columns),protocol=5),digest_size=16).digest()
                        repeated=index.add(0,strict,source_row,identity,split,usable)
                        measurement=measurement_hash(row,mz,intensity,energy) if usable else None
                        repeated_measurement=index.add(1,measurement,source_row,identity,split,True) if usable else None
                        record=dict(source_row=source_row,competition_identity=identity,split=split,
                            baseline_usable=usable,quarantine_reason=';'.join(reasons) or None,
                            strict_record_hash=strict,measurement_hash=measurement,
                            duplicate_record_of=repeated,duplicate_measurement_of=repeated_measurement,**flags)
                        for key,value in flags.items():
                            if value is not None:counts['all_'+key]+=int(value)
                        if split in partitions:
                            partitions[split]['source_rows']+=1
                            partitions[split]['baseline_usable' if usable else 'quarantined']+=1
                        if usable:
                            counts['eligible']+=1
                            for key,value in flags.items():
                                if value is not None:counts['eligible_'+key]+=int(value)
                            counts['eligible_duplicate_mz_rows']+=int(flags['duplicate_mz_excess']>0)
                            counts['eligible_above_precursor_rows']+=int(flags['above_precursor_peaks']>0)
                            metrics,total=filtering_metrics(mz,intensity)
                            for name,result in metrics.items():
                                record.update({f'{name}_{key}':value for key,value in result.items()})
                                for bucket in ('All',split):
                                    summary=aggregates[(bucket,name)]
                                    summary.update(spectra=1,input_peaks=len(mz),input_intensity=total,
                                        affected=int(result['total_removed']>0),affected_cutoff=int(result['cutoff_removed']>0),
                                        affected_cap=int(result['cap_removed']>0),positive_spectra=int(total>0),
                                        sparse=int(result['kept_peaks']<args.sparse_below),
                                        ratio_sum=result['retained_fraction'] or 0.)
                                    for key in ('kept_peaks','cutoff_removed','cap_removed','total_removed','retained_intensity'):
                                        summary[key]+=result[key]
                            removed=flags['removed_nonpositive_or_nonfinite_pairs']
                            if removed or flags['invalid_collision_energy_values']:
                                patches.append(dict(source_row=source_row,competition_identity=identity,split=split,
                                    normalized_smiles=target,ms2_mzs=mz.tolist(),ms2_normalized_intensities=intensity.tolist(),
                                    collision_energy_ev=energy,removed_fragment_entries=removed,
                                    removed_collision_energy_values=flags['invalid_collision_energy_values']))
                                counts['correction_rows']+=1
                        else:
                            counts['quarantined']+=1;quarantines.append(record)
                        audit_records.append(record)
                    writer.write_table(pa.Table.from_pylist(audit_records,schema=schema))
                    if patches:patch_writer.write_table(pa.Table.from_pylist(patches,schema=correction))
                    if quarantines:quarantine_writer.write_table(pa.Table.from_pylist(quarantines,schema=schema))
                    index.db.commit()
                    now=time.perf_counter()
                    if now-last_print>=15:
                        print(f'Audited {counts["rows"]:,}/{source.metadata.num_rows:,} rows; {counts["quarantined"]:,} quarantined; elapsed {now-started:.0f}s',flush=True)
                        last_print=now
            if source_path.stat().st_mtime_ns!=input_stat.st_mtime_ns or source_path.stat().st_size!=input_stat.st_size:
                raise RuntimeError('Input changed during the audit')
            duplicates={name:index.group_counts(kind) for kind,name in [(0,'exact_record'),(1,'cleaned_measurement')]}
            memberships,conflicts=summarize_duplicates(index,output)
            finalize_rows(Path(scratch)/'rows.parquet',output,memberships,conflicts)
        finally:
            index.close()
    experiments=experiment_table(aggregates,output)
    before_after=before_after_table(counts,duplicates,output)
    peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_bytes=peak if sys.platform=='darwin' else peak*1024
    report=dict(status='complete',source=dict(path=str(source_path),size_bytes=input_stat.st_size,
        mtime_ns=input_stat.st_mtime_ns,sha256=input_hash),counts=dict(counts),duplicates=duplicates,
        normalization_policy=POLICY,rdkit_version=RDKIT_VERSION,
        molecular_manifest=dict(path=str(Path(args.preprocessing).resolve()),
            report_sha256=sha256(Path(args.preprocessing)/'preprocessing_report.json'),
            mapping_sha256=sha256(Path(args.preprocessing)/'molecule_mapping.parquet')),
        partitions={k:dict(v) for k,v in partitions.items()},
        policy=dict(sparse_below=args.sparse_below,base_relative_cutoff=0.,production_fragment_cap=127,
            fingerprint='BLAKE2b-128 over protocol-5 source-column tuple; cleaned-measurement fingerprint sorts full valid float32 pairs and ignores labels/provenance',
            uncertain_records='retained and flagged; duplicates not deleted; identities/targets/splits unchanged',
            empty_intensity='retained as precursor-only; excluded from intensity-ratio denominators',
            experiments=[dict(name=name,relative_cutoff=cutoff,fragment_cap=cap) for name,cutoff,cap in VARIANTS]),
        versions=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,pyarrow=pa.__version__),
        runtime=dict(seconds=time.perf_counter()-started,peak_rss_gib=peak_bytes/2**30,batch_size=args.batch_size,
                     worker_processes=1,arrow_threads=1),
        script_sha256=sha256(__file__),shared_preparation_sha256=sha256(Path(__file__).with_name('spectral_preprocessing.py')),
        modeling_variants_trained=False,thresholds_adopted=False)
    write_report(report,before_after,experiments,output)
    (output/'audit_report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(rows=counts['rows'],baseline_usable=counts['eligible'],quarantined=counts['quarantined'],
                         correction_rows=counts['correction_rows'],seconds=report['runtime']['seconds'],
                         peak_rss_gib=report['runtime']['peak_rss_gib']),indent=2),flush=True)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train',default='dataset/train.parquet')
    parser.add_argument('--preprocessing',default='casmi_generative_ranker/preprocessing_results')
    parser.add_argument('--output',default='output/cpu_preprocessing_audit')
    parser.add_argument('--batch-size',type=int,default=4096)
    parser.add_argument('--sparse-below',type=int,default=5,help='Diagnostic flag only; do not remove sparse spectra')
    parser.add_argument('--overwrite',action='store_true')
    run(parser.parse_args())


if __name__=='__main__':main()
