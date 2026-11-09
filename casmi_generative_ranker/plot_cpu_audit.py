"""Plot measured filtering retention and three reproducibly selected real spectra.

No training, resplitting, new filtering policy, or changes to source measurements.
"""
import os
for _name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '1'

import argparse
import hashlib
import json
from pathlib import Path
import tempfile

os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir())/'casmi_phase3_matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from spectral_preprocessing import clean_fragment_pairs

VARIANTS = ['no_cutoff_127', 'one_percent_127', 'no_cutoff_255', 'one_percent_255']
BLUE, GOLD, RED = '#28638D', '#D69B31', '#B74343'


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4*1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def save_figure(fig, output, stem):
    for suffix in ('png', 'pdf', 'svg'):
        fig.savefig(output/f'{stem}.{suffix}', dpi=220, facecolor='white')
    plt.close(fig)


def retention_plot(table, output):
    rows = table[table.split == 'All'].set_index('variant').loc[VARIANTS]
    values = rows[['mean_retained_intensity_pct', 'weighted_retained_intensity_pct']].to_numpy()
    if not np.isfinite(values).all() or not ((values >= 0) & (values <= 100)).all():
        raise ValueError('Expected finite intensity-retention percentages in [0,100]')
    fig, ax = plt.subplots(figsize=(10.2, 5.5))
    fig.subplots_adjust(left=.085, right=.985, bottom=.21, top=.77)
    x = np.arange(4); width = .34
    for j, (label, color) in enumerate([('Mean retained intensity', BLUE), ('Weighted retained intensity', GOLD)]):
        bars = ax.bar(x+(j-.5)*width, values[:, j], width, label=label, color=color, zorder=3)
        for bar, value in zip(bars, values[:, j]):
            ax.text(bar.get_x()+bar.get_width()/2, value-4.8, f'{value:.2f}%',
                    ha='center', va='center', color='white' if j == 0 else '#262626',
                    fontsize=10, fontweight='bold')
    ax.set(ylim=(0, 100), ylabel='Retained intensity (%)', xticks=x,
           xticklabels=['No cutoff\n127 fragment peaks', '1% cutoff\n127 fragment peaks',
                        'No cutoff\n255 fragment peaks', '1% cutoff\n255 fragment peaks'])
    ax.set_yticks(np.arange(0, 101, 20)); ax.grid(axis='y', color='#DBE1E6', linewidth=.7, zorder=0)
    ax.spines[['top', 'right']].set_visible(False)
    fig.suptitle('Intensity retention under filtering choices', x=.535, y=.97, fontsize=17, fontweight='bold')
    fig.text(.535, .905, f'{int(rows.iloc[0].eligible_spectra):,} spectra; valid, positive fragments before truncation',
             ha='center', fontsize=10, color='#48535C')
    fig.legend(*ax.get_legend_handles_labels(), loc='upper center', bbox_to_anchor=(.535, .865),
               ncol=2, frameon=False, fontsize=10)
    fig.text(.535, .075, 'Mean: average of per-spectrum ratios. Weighted: summed retained / summed input intensity.',
             ha='center', fontsize=9, color='#48535C')
    fig.text(.535, .03, 'Intensity retention only — identification accuracy has not been measured.',
             ha='center', fontsize=10, fontweight='bold')
    save_figure(fig, output, 'filtering_intensity_retention')


def select_examples(train, audit_dir, rare_adduct):
    """Use light columns only; the three fragment arrays are fetched separately."""
    source = pq.read_table(train, columns=['ingest_lib', 'instrument_type', 'adduct', 'base_peak_intensity'],
                           use_threads=False, pre_buffer=False).to_pandas(use_threads=False)
    flags = pq.read_table(audit_dir/'audit_rows.parquet',
                         columns=['source_row', 'split', 'baseline_usable', 'clean_positive_fragments',
                                  'measurement_split_conflict'], use_threads=False,
                         pre_buffer=False).to_pandas(use_threads=False)
    if len(source) != len(flags) or not np.array_equal(flags.source_row, np.arange(len(source))):
        raise ValueError('Source/audit row alignment differs')
    counts = source.adduct.value_counts(dropna=False)
    n = flags.clean_positive_fragments.to_numpy()
    usable = flags.baseline_usable.to_numpy()
    median = float(np.median(n[usable]))
    # Illustrative examples use training rows, avoiding held-out sample inspection
    # and the already flagged cross-partition duplicate measurements.
    eligible = usable & (flags.split == 'Training').to_numpy() & ~flags.measurement_split_conflict.to_numpy()
    bpi = source.base_peak_intensity.to_numpy()
    measured = eligible & np.isfinite(bpi) & (bpi > 0)
    groups = source.loc[measured].groupby(['ingest_lib', 'instrument_type', 'adduct'], dropna=True).size()
    if groups.empty:
        raise ValueError('No cohort has recorded positive base-peak intensity')
    cohort = groups.sort_values(ascending=False, kind='stable').index[0]
    matched = measured.copy()
    for column, value in zip(['ingest_lib', 'instrument_type', 'adduct'], cohort):
        matched &= (source[column] == value).to_numpy()
    indices = np.flatnonzero(matched)
    cohort_values = bpi[indices]
    middle = float(np.median(cohort_values))
    low = float(np.quantile(cohort_values, .01))
    typical = int(indices[np.lexsort((indices, np.abs(bpi[indices]-middle), np.abs(n[indices]-median)))[0]])
    # Exclude ties at the boundary so the empirical percentile cannot exceed 1%.
    low_candidates = indices[(bpi[indices] < low) & (indices != typical)]
    if not len(low_candidates):
        raise ValueError('No distinct example within the bottom 1% of recorded intensity')
    weak = int(low_candidates[np.lexsort((low_candidates, np.abs(n[low_candidates]-median),
                                          np.abs(bpi[low_candidates]-low)))[0]])
    rare_candidates = np.flatnonzero(eligible & (source.adduct == rare_adduct).to_numpy() & (n > 0))
    rare_candidates = rare_candidates[~np.isin(rare_candidates, [typical, weak])]
    if not len(rare_candidates) or counts.get(rare_adduct, 0)/len(source) >= .01:
        raise ValueError('Uncommon-adduct example must have eligible rows and frequency below 1%')
    rare = int(rare_candidates[np.lexsort((rare_candidates, np.abs(n[rare_candidates]-median)))[0]])
    selection = dict(global_median_positive_fragments=median, source_spectra=len(source),
                     scope='Training partition; baseline usable; no flagged cross-partition measurement duplicate',
                     matched_cohort=dict(zip(['ingest_lib', 'instrument_type', 'adduct'], cohort)),
                     matched_cohort_spectra=len(indices), cohort_median_recorded_base_peak_intensity=middle,
                     cohort_first_percentile_recorded_base_peak_intensity=low,
                     uncommon_adduct=rare_adduct, uncommon_adduct_spectra=int(counts[rare_adduct]),
                     uncommon_adduct_pct=100*int(counts[rare_adduct])/len(source),
                     rules=dict(typical='Nearest global median fragment count in most frequent positive-intensity acquisition cohort; then nearest cohort median recorded base-peak intensity',
                                low_signal='Strictly below 1st-percentile recorded base-peak intensity in the same cohort, excluding boundary ties; nearest boundary, then nearest global median fragment count',
                                uncommon_adduct='Supplied [M+Li]+ by default (CLI override); nearest global median fragment count among eligible rows',
                                tie_break='Lowest zero-based source_row'))
    chosen = dict(typical=typical, low_signal=weak, uncommon_adduct=rare)
    contexts = {key:dict(source_row=row, split=str(flags.loc[row, 'split']),
                         positive_fragments=int(n[row]),
                         recorded_base_peak_intensity_percentile_in_matched_cohort=
                             float(100*np.mean(cohort_values <= bpi[row])) if key != 'uncommon_adduct' else None)
                for key, row in chosen.items()}
    return chosen, contexts, selection


def fetch_rows(train, chosen):
    """Read only required row groups, in small batches, with one Arrow worker."""
    result = {}; reverse = {row:key for key, row in chosen.items()}
    columns = ['normalized_smiles', 'precursor_mz', 'adduct', 'ingest_lib', 'instrument_type',
               'ionization_mode', 'base_peak_intensity', 'collision_energy_ev',
               'ms2_mzs', 'ms2_normalized_intensities']
    with pq.ParquetFile(train, pre_buffer=False) as source:
        offset = 0
        for group in range(source.num_row_groups):
            end = offset+source.metadata.row_group(group).num_rows
            wanted = {i for i in reverse if offset <= i < end}
            if wanted:
                batch_start = offset
                for batch in source.iter_batches(row_groups=[group], columns=columns, batch_size=1024, use_threads=False):
                    for row in sorted(wanted):
                        if batch_start <= row < batch_start+len(batch):
                            result[reverse[row]] = batch.slice(row-batch_start, 1).to_pylist()[0]
                    batch_start += len(batch)
                    if batch_start > max(wanted):
                        break
            offset = end
    if set(result) != set(chosen):
        raise ValueError('Could not retrieve all selected source rows')
    return result


def spectra_plot(examples, selection, output):
    fig, axes = plt.subplots(3, 1, figsize=(11, 11.7))
    fig.subplots_adjust(left=.10, right=.97, top=.85, bottom=.17, hspace=.90)
    labels = [('typical', 'A  Typical peak count'), ('low_signal', 'B  Low recorded signal'),
              ('uncommon_adduct', 'C  Uncommon supplied adduct')]
    for ax, (key, title) in zip(axes, labels):
        row = examples[key]
        mz, intensity, _ = clean_fragment_pairs(row['ms2_mzs'], row['ms2_normalized_intensities'])
        values = 100*intensity/float(intensity.max())
        precursor = row['precursor_mz']
        ax.vlines(mz, 0, values, color=BLUE, linewidth=1.15, zorder=3)
        ax.axvline(precursor, color=RED, linestyle='--', linewidth=1., zorder=2)
        ax.set(xlim=(0, max(float(mz.max()), precursor)*1.04), ylim=(0, 108),
               ylabel='Relative intensity (%)', xlabel='m/z')
        ax.set_yticks([0, 25, 50, 75, 100]); ax.grid(axis='y', color='#E5EAED', linewidth=.6, zorder=0)
        ax.spines[['top', 'right']].set_visible(False)
        bpi = row['base_peak_intensity']
        recorded = f'{bpi:,.0f}' if bpi is not None and np.isfinite(bpi) else 'unavailable'
        detail = f'row {row["source_row"]:,} | {len(mz)} fragments | {row["adduct"]} | precursor {precursor:.4f}'
        if key == 'typical':
            note = f'Global median: {selection["global_median_positive_fragments"]:g} fragments; recorded base peak: {recorded}'
        elif key == 'low_signal':
            note = f'Recorded base peak: {recorded}; bottom 1% of matched source / instrument / adduct cohort'
        else:
            note = f'Adduct observed in {selection["uncommon_adduct_spectra"]:,} / {selection["source_spectra"]:,} spectra ({selection["uncommon_adduct_pct"]:.5f}%)'
        # Keep explanatory text outside the data area so it cannot hide peaks.
        ax.set_title(title+'\n'+detail+'\n'+note, loc='left', fontsize=10, linespacing=1.4, pad=10)
    fig.suptitle('Three actual training spectra: sample inspection', y=.97, fontsize=17, fontweight='bold')
    fig.text(.535, .925, 'Full valid fragment arrays; no relative cutoff or peak cap applied', ha='center', fontsize=10, color='#48535C')
    fig.legend(handles=[Line2D([0], [0], color=BLUE, label='Observed fragments'),
                        Line2D([0], [0], color=RED, linestyle='--', label='Precursor m/z reference')],
               loc='lower center', bbox_to_anchor=(.535, .073), ncol=2, frameon=False, fontsize=10)
    fig.text(.535, .045, 'Each spectrum is normalized to its own base peak; normalized heights do not show absolute signal differences.',
             ha='center', fontsize=9)
    fig.text(.535, .022, 'Low recorded intensity is a cohort-relative observation, not an SNR measurement or an identification-accuracy result.',
             ha='center', fontsize=9)
    save_figure(fig, output, 'sample_spectra')


def insert_section(text, marker, content, anchor):
    start, end = f'<!-- {marker}:start -->', f'<!-- {marker}:end -->'
    block = start+'\n'+content.strip()+'\n'+end+'\n\n'
    if start in text:
        before, remainder = text.split(start, 1)
        _, after = remainder.split(end, 1)
        return before+block+after.lstrip('\n')
    if anchor not in text:
        raise ValueError(f'Missing report insertion anchor: {anchor}')
    return text.replace(anchor, block+anchor, 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train', default='dataset/train.parquet')
    parser.add_argument('--audit', default='output/cpu_preprocessing_audit')
    parser.add_argument('--uncommon-adduct', default='[M+Li]+')
    args = parser.parse_args()
    pa.set_cpu_count(1); pa.set_io_thread_count(1)
    audit_dir = Path(args.audit); train = Path(args.train); output = audit_dir/'figures'
    report = json.loads((audit_dir/'audit_report.json').read_text())
    if report['status'] != 'complete' or sha256(train) != report['source']['sha256']:
        raise ValueError('A completed audit matching this source file is required')
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':10, 'pdf.fonttype':42,
                         'svg.fonttype':'none', 'axes.labelcolor':'#26343F', 'text.color':'#26343F'})
    retention_plot(pd.read_csv(audit_dir/'filtering_experiments.csv'), output)
    chosen, contexts, selection = select_examples(train, audit_dir, args.uncommon_adduct)
    rows = fetch_rows(train, chosen)
    examples = {key:dict(**rows[key], **contexts[key]) for key in chosen}
    spectra_plot(examples, selection, output)
    (output/'sample_spectra.json').write_text(json.dumps(examples, indent=2, allow_nan=False)+'\n')
    manifest = dict(source_sha256=report['source']['sha256'], audit_report_sha256=sha256(audit_dir/'audit_report.json'),
                    filtering_table_sha256=sha256(audit_dir/'filtering_experiments.csv'), script_sha256=sha256(__file__),
                    matplotlib_version=matplotlib.__version__, numpy_version=np.__version__, selection=selection,
                    selected_source_rows=chosen, worker_processes=1, arrow_threads=1,
                    figures_measure='intensity retention and observed spectra; not identification accuracy')
    (output/'plot_manifest.json').write_text(json.dumps(manifest, indent=2, allow_nan=False)+'\n')
    text = (audit_dir/'report.md').read_text()
    text = insert_section(text, 'retention-figure', '''
![Grouped bars showing mean and weighted intensity retention for the four filtering experiments](figures/filtering_intensity_retention.png)

**Intensity retention, not identification accuracy.** The 0–100% axis is shared by both series. Mean retention averages per-spectrum ratios; weighted retention uses summed intensities. Both series use the same full audit cohort and exclude the artificial precursor. [Vector PDF](figures/filtering_intensity_retention.pdf).
''', '## Duplicates and uncertain cases')
    typical, weak, rare = (examples[key] for key in ['typical', 'low_signal', 'uncommon_adduct'])
    cohort = selection['matched_cohort']
    section = f'''## Inspection of three actual spectra

![Three actual spectra illustrating typical peak count, low recorded signal, and an uncommon supplied adduct](figures/sample_spectra.png)

| Example | Source row (zero-based) | Selection and interpretation |
|---|---:|---|
| Typical peak count | {typical['source_row']:,} | {typical['positive_fragments']} valid fragments, matching the global median ({selection['global_median_positive_fragments']:g}); common {typical['adduct']} adduct. Among these rows, choose recorded base-peak intensity nearest the matched-cohort median. |
| Low recorded signal | {weak['source_row']:,} | Recorded base-peak intensity {weak['base_peak_intensity']:,.0f}, at percentile {weak['recorded_base_peak_intensity_percentile_in_matched_cohort']:.3f} within {selection['matched_cohort_spectra']:,} training spectra from {cohort['ingest_lib']} / {cohort['instrument_type']} / {cohort['adduct']}. Choose within the bottom 1%, nearest its upper boundary. This is a relative intensity observation, not a measured signal-to-noise ratio. |
| Uncommon supplied adduct | {rare['source_row']:,} | {rare['adduct']} occurs in {selection['uncommon_adduct_spectra']} of {selection['source_spectra']:,} source spectra ({selection['uncommon_adduct_pct']:.5f}%); choose the eligible example nearest the global median peak count. The supplied assignment is preserved. |

These examples use only baseline-usable **training** rows and exclude flagged cross-partition duplicate measurements. Selection is deterministic, with source-row order breaking ties; the examples do not establish population-wide quality. Each panel shows all positive valid fragments, without a relative cutoff or peak cap. Dashed precursor lines are position references, not added intensity peaks. Each spectrum is normalized to its own base peak, so panel B's low recorded absolute intensity cannot be inferred from normalized heights alone. Acquisition intensities are compared only within the matched source/instrument/adduct cohort, not across libraries.

[Vector PDF](figures/sample_spectra.pdf), [actual selected records](figures/sample_spectra.json), and [selection rules and provenance](figures/plot_manifest.json) accompany the figures. SVG exports are also saved. Missingness and duplicate counts remain in the existing tables.

Regenerate both figures after the CPU audit, using one CPU worker:

```bash
python casmi_generative_ranker/plot_cpu_audit.py --train dataset/train.parquet --audit output/cpu_preprocessing_audit
```
'''
    text = insert_section(text, 'sample-spectra', section, '## Implemented versus planned')
    (audit_dir/'report.md').write_text(text)
    print(json.dumps(dict(output=str(output), selected_source_rows=chosen, selection=selection), indent=2))


if __name__ == '__main__':
    main()
