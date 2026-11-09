"""Model correctness checks independent of training quality or GPU performance."""
import copy
import tempfile
from pathlib import Path
from types import MethodType, SimpleNamespace
import unittest

import numpy as np
import pandas as pd
import torch

import pipeline as p


class PipelineTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        torch.set_num_threads(1)
        self.batch = p.collate([
            dict(mz=np.array([50., 100.], dtype=np.float32),
                 int=np.array([2., 1.], dtype=np.float32), smiles='CCO', id='a',
                 adduct='[M+H]+', collision_energy_ev=[20., 40.]),
            dict(mz=np.array([70., 120., 180.], dtype=np.float32),
                 int=np.array([2., 1., .5], dtype=np.float32), smiles='CCN', id='b'),
        ])
        self.batch['tokens'] = torch.tensor([[p.BOS, 4, p.EOS, p.PAD, p.PAD],
                                             [p.BOS, 5, 6, 4, p.EOS]])

    def model(self, architecture='transformer'):
        return p.Generator(10, d=16, heads=2, enc_layers=1, dec_layers=1,
                           ff=32, dropout=0, max_len=8, architecture=architecture,
                           cnn_max_mz=256., cnn_bin_width=1., cnn_pool_bins=4)

    def test_causality_padding_and_metadata_conditioning(self):
        for architecture in ('transformer', 'pooled_gru', 'cnn_gru'):
            model = self.model(architecture).eval()
            tokens = self.batch['tokens'][:, :-1]
            memory = model.encode(self.batch)
            logits = model.decode(tokens, memory, self.batch['mask'])
            changed = tokens.clone()
            changed[:, -1] = 7
            future = model.decode(changed, memory, self.batch['mask'])
            torch.testing.assert_close(logits[:, :-1], future[:, :-1])
            padded = copy.deepcopy(self.batch)
            padded['mz'][0, -1] = 999
            padded['ints'][0, -1] = 999
            torch.testing.assert_close(logits, model.decode(tokens, model.encode(padded), padded['mask']))
            metadata = copy.deepcopy(self.batch)
            metadata['metadata_categories'][0, 0] = (metadata['metadata_categories'][0, 0] + 1) % 128
            conditioned = model.decode(tokens, model.encode(metadata), metadata['mask'])
            self.assertFalse(torch.allclose(logits[0], conditioned[0]))
            model(self.batch).backward()
            self.assertTrue(all(torch.isfinite(x.grad).all() for x in model.parameters() if x.grad is not None))

    def test_cnn_grid_has_mass_axis_and_separate_precursor(self):
        model = self.model('cnn_gru').eval()
        grid = model.binned_spectrum(self.batch)
        self.assertEqual(grid.shape, (2, 2, 256))
        self.assertEqual(float(grid[0, 1, 50]), 1.)
        self.assertEqual(float(grid[0, 0, 100]), 1.)
        self.assertEqual(float(grid[0, 0, 50]), 0.)
        permuted = copy.deepcopy(self.batch)
        for key in ('mz', 'ints', 'mask'):
            permuted[key] = permuted[key][:, [0, 2, 1]]
        torch.testing.assert_close(model.encode(self.batch), model.encode(permuted))
        invalid = copy.deepcopy(self.batch)
        invalid['mz'][0, 0] = 256.
        with self.assertRaisesRegex(ValueError, 'increase --cnn-max-mz'):
            model.encode(invalid)

    def test_transformer_uses_129_total_encoder_positions_and_160_tokens(self):
        model = p.Generator(10, d=16, heads=2, enc_layers=1, dec_layers=1, ff=32).eval()
        batch = dict(mz=torch.ones(1, 128), ints=torch.ones(1, 128),
                     mask=torch.ones(1, 128, dtype=torch.bool))
        self.assertEqual(model.encode(batch).shape[1], 129)
        self.assertEqual(model.max_len, 160)

    def test_unequal_microbatches_match_full_batch_update(self):
        full = self.model()
        accumulated = copy.deepcopy(full)
        opt_full = torch.optim.SGD(full.parameters(), lr=.01)
        opt_accum = torch.optim.SGD(accumulated.parameters(), lr=.01)
        scaler_full = torch.amp.GradScaler('cuda', enabled=False)
        scaler_accum = torch.amp.GradScaler('cuda', enabled=False)
        count = int((self.batch['tokens'][:, 1:] != p.PAD).sum())
        full(self.batch, reduction='sum').backward()
        p.optimizer_update(full, opt_full, scaler_full, count)
        # Two microbatches in an intended group of four, with unequal target lengths.
        for i in range(2):
            micro = {k: v[i:i+1] if torch.is_tensor(v) else v[i:i+1]
                     for k, v in self.batch.items()}
            (accumulated(micro, reduction='sum') / 4).backward()
        p.optimizer_update(accumulated, opt_accum, scaler_accum, count, loss_divisor=4)
        for a, b in zip(full.parameters(), accumulated.parameters()):
            torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-7)

    def test_beam_preserves_scores_and_completion(self):
        model = self.model().eval()
        def decode(this, tokens, memory, mask):
            logits = torch.full((len(tokens), tokens.shape[1], 10), -100.)
            logits[:, -1, 4] = 2. if tokens.shape[1] == 1 else 0.
            logits[:, -1, p.EOS] = 0. if tokens.shape[1] == 1 else 3.
            return logits
        model.decode = MethodType(decode, model)
        beams = model.beam(self.batch, beam_size=2, max_tokens=4, length_penalty=0)
        for group in beams:
            self.assertEqual(group[0][0][:3], [p.BOS, 4, p.EOS])
            self.assertEqual(group[0][2], 2)
            self.assertGreater(group[0][1], group[1][1])
            self.assertTrue(all(p.EOS in ids for ids, _, _ in group))

    def test_unterminated_sequences_are_not_candidates(self):
        with tempfile.TemporaryDirectory() as folder:
            tk = p.train_tokenizer(pd.DataFrame({'normalized_smiles': ['CCO']}), Path(folder)/'tk.json', 16)
            ids = tk.encode('CCO').ids
            self.assertIsNone(p.decode_smiles(tk, ids[:-1]))
            class Sampler:
                def sample(self, batch, *args):
                    return [[(ids[:-1], -1., len(ids)-2), (ids, -2., len(ids)-1)]]
            frame = pd.DataFrame({'molecule_id': ['a'], 'ms2_mzs': [[50.]],
                                  'ms2_normalized_intensities': [[1.]], 'precursor_mz': [100.]})
            args = SimpleNamespace(infer_batch=1, workers=0, device='cpu', bf16=False,
                                   samples=2, max_tokens=8, temperature=1., sample_chunk=2, progress_every=0)
            groups = p.build_candidates(frame, Sampler(), tk, 8, args)
            self.assertEqual(len(groups['a']['candidates']), 1)
            self.assertEqual(len(next(iter(groups['a']['candidates'].values()))['scores']), 1)

    def test_submission_format_and_missing_coverage(self):
        sample = pd.DataFrame({'molecule_id': ['b', 'a'], 'smiles': ['', '']})
        frame = pd.DataFrame({'molecule_id': ['a', 'a', 'b'], 'smiles': ['CCO', 'CCN', 'C']})
        result = p.submission_frame(sample, frame)
        self.assertEqual(result.smiles.tolist(), ['C', 'CCO;CCN'])
        with self.assertRaisesRegex(ValueError, 'No completed valid candidates'):
            p.submission_frame(sample, frame[frame.molecule_id == 'a'])


if __name__ == '__main__':
    unittest.main()
