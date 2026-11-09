"""Check workload units and budget decisions against the recorded GPU experiment."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest

import benchmark


class BenchmarkReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not (Path(__file__).parent/'benchmark_results'/'hardware.json').exists():
            raise unittest.SkipTest('Recorded GPU report fixtures are not included in this staging bundle')

    def generate(self, folder, budget=None):
        source = Path(__file__).parent/'benchmark_results'
        for name in ('hardware.json', 'cnn_gru.json', 'cnn_hardware.json', 'pooled_gru.json', 'transformer.json', 'ranker_cpu.json', 'run_context.json'):
            shutil.copyfile(source/name, folder/name)
        args = SimpleNamespace(output=str(folder), epochs=10, eval_every=2000, val_max=5000,
                               loading_allowance=.1, prediction_spectra=1213, ranker_structures=2000,
                               spectra_per_structure=5, gpu_budget_hours=budget,
                               proposed_gpu_budget_hours=12., evaluation_retry_reserve_hours=2.,
                               spent_gpu_hours=216/3600, benchmark_budget_hours=1.5,
                               infer_batch=8, sample_chunk=100)
        with redirect_stdout(io.StringIO()):
            benchmark.report(args)
        return json.loads((folder/'projections.json').read_text()), (folder/'report.md').read_text()

    def test_subset_and_full_partition_costs_are_distinct(self):
        with tempfile.TemporaryDirectory() as directory:
            projection, report = self.generate(Path(directory))
        workloads = projection['workloads']
        self.assertEqual(workloads['ranker_subset_spectra_cap'], 10000)
        self.assertEqual(workloads['ranker_full_holdout_spectra'], 247342)
        self.assertEqual(workloads['subset_beam_validation_spectra'], 553)
        self.assertEqual(workloads['validation_partition_spectra'], 49951)
        self.assertEqual(workloads['teacher_forced_validation_spectra_per_event'], 5000)
        self.assertAlmostEqual(projection['ranker_gpu_hours'], .4047, places=4)
        self.assertAlmostEqual(projection['full_holdout_gpu_compute_hours'], 10.0101, places=4)
        self.assertAlmostEqual(projection['full_validation_beam_hours'], .2972, places=4)
        self.assertAlmostEqual(projection['estimates']['transformer']['total_hours'], 7.1559, places=4)
        for count in ('10,000', '247,342', '49,951', '553'):
            self.assertIn(count, report)

    def test_unknown_allocation_is_not_replaced_by_proposed_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            projection, report = self.generate(Path(directory))
        self.assertIsNone(projection['gpu_budget_hours'])
        self.assertIsNone(projection['remaining_gpu_hours_after_base_plan'])
        self.assertFalse(projection['ranker_within_budget'])
        self.assertFalse(projection['ranker_within_proposed_budget'])
        self.assertAlmostEqual(projection['proposed_remaining_gpu_hours_after_base_and_reserve'], .0105, places=4)
        self.assertIn('not specified or approved', report)

    def test_inline_cpu_and_retry_reserve_affect_gpu_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            projection, _ = self.generate(Path(directory), budget=12.45)
        remaining = projection['remaining_gpu_hours_after_base_plan']
        self.assertGreater(remaining, projection['ranker_gpu_hours'])
        self.assertLess(remaining, projection['ranker_subset_inline_allocation_hours_including_fit_proxy'])
        self.assertFalse(projection['ranker_within_budget'])
        self.assertEqual(projection['evaluation_retry_reserve_hours'], 2.)
        with tempfile.TemporaryDirectory() as directory:
            projection, _ = self.generate(Path(directory), budget=12.6)
        self.assertTrue(projection['ranker_within_budget'])

    def test_report_names_baseline_and_is_self_contained(self):
        with tempfile.TemporaryDirectory() as directory:
            _, report = self.generate(Path(directory))
        for text in ('Pooled peak embeddings + GRU is auxiliary only', 'The planned baseline is **CNN + GRU**',
                     'Prediction quality remains untested', '384', '1536', 'BF16',
                     '19,200', '150 timed complete updates', '32', '128',
                     '2.00 GPU-hours', '1.50 hours', '216 seconds', '129 total encoder positions',
                     '160 SMILES tokens', 'support preliminary compute-feasibility estimates',
                     'Full-dataset loading and host-memory feasibility have not been demonstrated'):
            self.assertIn(text, report)

    def test_auxiliary_gru_is_excluded_from_planned_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            projection, report = self.generate(Path(directory))
        self.assertEqual(projection['base_plan_models'], ['cnn_gru', 'transformer'])
        self.assertFalse(projection['estimates']['pooled_gru']['included_in_base_plan'])
        self.assertAlmostEqual(projection['base_plan_gpu_hours'], 9.9295, places=4)
        self.assertAlmostEqual(projection['minimum_total_budget_with_reserve_and_prior_spend'], 11.9895, places=4)
        auxiliary_row = next(line for line in report.splitlines() if line.startswith('| Pooled'))
        self.assertIn('Auxiliary only', auxiliary_row)
        self.assertNotIn('Required', auxiliary_row)

    def test_missing_cnn_measurement_cannot_reuse_pooled_gru_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            self.generate(folder)
            (folder/'cnn_gru.json').unlink()
            with self.assertRaisesRegex(ValueError, 'pooled-GRU cannot supply its budget'):
                benchmark.report(SimpleNamespace(output=str(folder)))


if __name__ == '__main__':
    unittest.main()
