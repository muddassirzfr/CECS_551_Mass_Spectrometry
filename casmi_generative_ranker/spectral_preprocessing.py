"""Shared CPU peak preparation for the model and descriptive spectral audit.

The production default remains no relative cutoff and 127 fragment peaks.
Alternative cutoffs/caps are explicit experiments, not new cleaning defaults.
"""
import numpy as np


def clean_fragment_pairs(mzs, intensities):
    """Apply the model's existing finite/positive pair mask, without guessing alignment."""
    if mzs is None or intensities is None:
        raise ValueError('Missing fragment array')
    with np.errstate(over='ignore', invalid='ignore'):
        mz = np.asarray(mzs, dtype=np.float32)
        values = np.asarray(intensities, dtype=np.float32)
    if mz.ndim != 1 or values.ndim != 1 or len(mz) != len(values):
        raise ValueError('Fragment arrays must be aligned one-dimensional arrays')
    keep = np.isfinite(mz) & np.isfinite(values) & (mz > 0) & (values > 0)
    return mz[keep], values[keep], keep


def select_fragments(mz, values, fragment_cap, relative_cutoff=0.):
    """Select already-cleaned pairs; keep the exact cutoff boundary.

    Intensities are returned in their supplied units. Normalization and the added
    precursor are handled by peaks(). The sort matches the existing model path.
    """
    if fragment_cap < 0 or not 0 <= relative_cutoff <= 1:
        raise ValueError('Invalid fragment cap or relative cutoff')
    if len(values) == 0:
        return mz, values
    indices = np.argsort(-values)
    if relative_cutoff:
        indices = indices[values[indices] >= relative_cutoff*float(values.max())]
    indices = indices[:fragment_cap]
    return mz[indices], values[indices]


def peaks(row, max_peaks):
    """Production input: finite positive fragments, strongest first, then prepend precursor.

    max_peaks includes the precursor, but excludes the Transformer's metadata token.
    No 1% cutoff, duplicate merging or above-precursor deletion is applied.
    """
    if max_peaks < 1:
        raise ValueError('max_peaks must include at least the precursor position')
    precursor = float(row['precursor_mz'])
    if not np.isfinite(precursor) or precursor <= 0 or precursor > np.finfo(np.float32).max:
        raise ValueError('Precursor m/z must be positive and representable as finite float32')
    mz, values, _ = clean_fragment_pairs(row['ms2_mzs'], row['ms2_normalized_intensities'])
    mz, values = select_fragments(mz, values, max_peaks-1)
    if len(values):
        values = values/max(float(values.max()), 1e-9)
    return (np.r_[precursor, mz].astype(np.float32),
            np.r_[2., values].astype(np.float32))
