"""Exercise corrupted records, experimental thresholds and baseline replay."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import audit_spectra as audit
import molecular_preprocessing as molecular
from spectral_preprocessing import clean_fragment_pairs, peaks, select_fragments


class SpectralAuditTests(unittest.TestCase):
    def test_cleaning_and_existing_input_parity(self):
        row = dict(precursor_mz=150., ms2_mzs=[10., 20., 30., -1., np.nan, 40., 50.],
                   ms2_normalized_intensities=[.1, 1., .01, 2., 1., 0., -.1])
        # Reference the original production operations independently.
        mz = np.asarray(row['ms2_mzs'], dtype=np.float32)
        values = np.asarray(row['ms2_normalized_intensities'], dtype=np.float32)
        keep = np.isfinite(mz) & np.isfinite(values) & (mz > 0) & (values > 0)
        mz, values = mz[keep], values[keep]
        order = np.argsort(-values)[:2]
        expected_mz = np.r_[150., mz[order]].astype(np.float32)
        expected_i = np.r_[2., values[order]/values[order].max()].astype(np.float32)
        actual_mz, actual_i = peaks(row, 3)
        np.testing.assert_array_equal(actual_mz, expected_mz)
        np.testing.assert_array_equal(actual_i, expected_i)
        flags, reasons, _, _, _ = audit.evaluate_row(row)
        self.assertEqual(reasons, [])
        self.assertEqual(flags['invalid_fragment_pairs'], 3)
        self.assertEqual(flags['zero_intensity_pairs'], 1)
        with self.assertRaisesRegex(ValueError, 'aligned'):
            clean_fragment_pairs([10., 20.], [1.])
        with self.assertRaisesRegex(ValueError, 'Precursor'):
            peaks(dict(row, precursor_mz=np.inf), 128)

    def test_cutoff_boundary_cap_and_empty_denominator(self):
        mz = np.arange(1, 301, dtype=np.float32)
        values = np.arange(300, 0, -1, dtype=np.float32)
        metrics, total = audit.filtering_metrics(mz, values)
        self.assertEqual(total, 45150.)
        self.assertEqual(metrics['no_cutoff_127']['total_removed'], 173)
        self.assertEqual(metrics['one_percent_255']['cutoff_removed'], 2)
        self.assertEqual(metrics['one_percent_255']['cap_removed'], 43)
        self.assertEqual(metrics['one_percent_255']['total_removed'], 45)
        for name, cutoff, cap in audit.VARIANTS:
            selected, retained = select_fragments(mz, values, cap, cutoff)
            self.assertEqual(len(selected), metrics[name]['kept_peaks'])
            self.assertAlmostEqual(retained.sum(dtype=np.float64), metrics[name]['retained_intensity'])
        # Exactly 1% is retained; only the value strictly below it is removed.
        selected, _ = select_fragments(np.arange(3), np.array([100., 1., .99]), 255, .01)
        self.assertEqual(selected.tolist(), [0, 1])
        metrics, total = audit.filtering_metrics(np.array([]), np.array([]))
        self.assertEqual(total, 0)
        self.assertIsNone(metrics['one_percent_127']['retained_fraction'])

    def test_full_audit_replay_flags_duplicates_and_identity_preservation(self):
        row = dict(normalized_smiles='CCO', precursor_mz=100.,
                   ms2_mzs=[10., 20., 30., 40., 50., np.nan, -1.],
                   ms2_normalized_intensities=[100., 1., .5, 0., .001, 2., 3.],
                   adduct='[M+H]+', ionization_mode='positive', instrument_type='Orbitrap',
                   collision_energy_ev=[20.], ingest_lib='fixture')
        rows = [row.copy(), row.copy(),
                dict(row, normalized_smiles='C', precursor_mz=50., ms2_mzs=[],
                     ms2_normalized_intensities=[], adduct=None, instrument_type='unknown', collision_energy_ev=None),
                dict(row, normalized_smiles='CCN', ms2_mzs=[10., 20.], ms2_normalized_intensities=[1.]),
                dict(row, normalized_smiles='CCCC', precursor_mz=np.nan),
                dict(row, normalized_smiles='bad('),
                dict(row, precursor_mz=500., ms2_mzs=list(range(1, 301)),
                     ms2_normalized_intensities=list(range(300, 0, -1))),
                dict(row, precursor_mz=15., ms2_mzs=[10., 10., 20.],
                     ms2_normalized_intensities=[1., .5, .01], collision_energy_ev=[10., np.inf, -2.]),
                dict(row, ms2_mzs=row['ms2_mzs'][::-1],
                     ms2_normalized_intensities=row['ms2_normalized_intensities'][::-1]),
                dict(row, normalized_smiles='CCN')]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); source = root/'source.parquet'; prep = root/'prep'; output = root/'audit'
            pd.DataFrame(rows).to_parquet(source, index=False)
            molecular.build_preprocessing(source, prep, holdout=.2, val_frac=.2, workers=1)
            before_hash = audit.sha256(source)
            mapping_hash = audit.sha256(prep/'molecule_mapping.parquet')
            args = SimpleNamespace(train=source, preprocessing=prep, output=output,
                                   batch_size=3, sparse_below=5, overwrite=False)
            report = audit.run(args)
            self.assertEqual(report['counts']['rows'], 10)
            self.assertEqual(report['counts']['eligible'], 7)
            self.assertEqual(report['counts']['quarantined'], 3)
            self.assertEqual(report['counts']['correction_rows'], 5)
            self.assertEqual(report['duplicates']['exact_record']['excess_rows'], 1)
            self.assertEqual(report['duplicates']['cleaned_measurement']['excess_rows'], 3)
            self.assertEqual(report['duplicates']['cleaned_measurement']['groups_with_identity_conflict'], 1)
            flags = pd.read_parquet(output/'audit_rows.parquet').set_index('source_row')
            self.assertTrue(flags.loc[0, 'duplicate_record_member'])  # first member also flagged
            self.assertTrue(flags.loc[[0, 1, 8, 9], 'measurement_identity_conflict'].all())
            self.assertTrue(flags.loc[2, 'baseline_usable'])
            self.assertTrue(flags.loc[2, 'empty_after'])
            self.assertTrue(flags.loc[2, 'missing_adduct'])
            self.assertTrue(flags.loc[2, 'unknown_instrument_type'])
            self.assertTrue(pd.isna(flags.loc[2, 'no_cutoff_127_retained_fraction']))
            self.assertEqual(flags.loc[7, 'duplicate_mz_excess'], 1)
            self.assertEqual(flags.loc[7, 'above_precursor_peaks'], 1)
            self.assertEqual(pq.ParquetFile(output/'quarantined_rows.parquet').metadata.num_rows, 3)
            self.assertEqual(before_hash, audit.sha256(source))
            self.assertEqual(mapping_hash, audit.sha256(prep/'molecule_mapping.parquet'))
            raw = pd.read_parquet(source)
            cleaned = audit.apply_audit_to_frame(raw, output, source)
            self.assertEqual(cleaned.index.tolist(), [0, 1, 2, 6, 7, 8, 9])
            self.assertEqual(list(cleaned.loc[0, 'ms2_mzs']), [10., 20., 30., 50.])
            self.assertEqual(list(cleaned.loc[7, 'collision_energy_ev']), [10.])
            self.assertEqual(len(cleaned.loc[6, 'ms2_mzs']), 300)  # experimental caps never adopted
            mapping, _ = molecular.load_preprocessing(prep)
            original = molecular.apply_preprocessing(raw, mapping).loc[cleaned.index]
            normalized = molecular.apply_preprocessing(cleaned, mapping)
            pd.testing.assert_frame_equal(original[['competition_identity', 'split', 'normalized_smiles']],
                                          normalized[['competition_identity', 'split', 'normalized_smiles']])
            summary = pd.read_csv(output/'filtering_experiments.csv')
            self.assertEqual(len(summary), 16)
            self.assertEqual(summary.loc[summary.split == 'All', 'eligible_spectra'].tolist(), [7]*4)
            self.assertFalse(report['modeling_variants_trained'])
            self.assertFalse(report['thresholds_adopted'])
            self.assertEqual(json.loads((output/'audit_report.json').read_text())['status'], 'complete')
            with self.assertRaises(FileExistsError):
                audit.run(args)
            changed = root/'changed.parquet'; raw.iloc[::-1].to_parquet(changed)
            with self.assertRaisesRegex(ValueError, 'different source'):
                audit.apply_audit_to_frame(raw, output, changed)


if __name__ == '__main__':
    unittest.main()
