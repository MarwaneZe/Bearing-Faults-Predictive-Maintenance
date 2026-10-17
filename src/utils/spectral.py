"""Frequency-domain & MCSA exploration. Detect-only.

Option B is honored implicitly: spectra are computed with Welch averaging over
fixed nperseg windows, so variable recording length does not affect results.
"""

import numpy as np
from scipy import signal

# Default bearing geometry (6203 deep-groove ball bearing, Paderborn rig).
# Tune from the official bearing sheet if needed.
GEOMETRY = {
    "z": 8,             # number of rolling elements
    "ball_d": 6.75,     # ball diameter (mm)
    "pitch_d": 28.5,    # pitch diameter (mm)
    "contact_angle": 0.0,   # degrees
}

RATE_VIB = 64_000
RATE_CUR = 64_000
LINE_FREQ = 50.0       # mains / supply frequency (Hz), for MCSA sidebands

NPERSEG = 4096


def fault_frequencies(shaft_hz, geometry=None):
    """Characteristic fault frequencies (FTF/BPFO/BPFI/BSF) at shaft_hz."""
    g = geometry or GEOMETRY
    z, bd, pd = g["z"], g["ball_d"], g["pitch_d"]
    ratio = (bd / pd) * np.cos(np.deg2rad(g["contact_angle"]))
    return {
        "shaft": shaft_hz,
        "ftf": shaft_hz / 2 * (1 - ratio),
        "bpfo": z / 2 * shaft_hz * (1 - ratio),
        "bpfi": z / 2 * shaft_hz * (1 + ratio),
        "bsf": (pd / (2 * bd)) * shaft_hz * (1 - ratio ** 2),
    }


def psd(array, rate, nperseg=NPERSEG):
    """Averaged power spectrum (Welch). Length-independent under Option B."""
    f, p = signal.welch(np.asarray(array, float), fs=rate, nperseg=nperseg)
    return f, p


def envelope_spectrum(array, rate, low=1000.0, high=16000.0, nperseg=NPERSEG):
    """Envelope (Hilbert) spectrum after band-pass -- targets bearing impulses
    that time-domain stats missed (Block 7 finding)."""
    x = np.asarray(array, float)
    sos = signal.butter(4, [low, high], btype="band", fs=rate, output="sos")
    x = signal.sosfilt(sos, x)
    env = np.abs(signal.hilbert(x))
    f, p = signal.welch(env, fs=rate, nperseg=nperseg)
    return f, p


def fault_band_ratio(freqs, power, fault_hz, bw=2.0, ref_lo=20.0, ref_hi=1000.0):
    """PSD peak ratio in a fault band vs median power in a reference band.

    A >1 ratio means spectral energy concentrates near the fault frequency.
    Falls back to the nearest FFT bin when the band is narrower than the bin
    spacing (resolution = rate / nperseg). Detection-only measure.
    """
    freqs = np.asarray(freqs)
    power = np.asarray(power)
    sel = (freqs >= fault_hz - bw) & (freqs <= fault_hz + bw)
    if not sel.any():
        sel = np.abs(freqs - fault_hz) == np.min(np.abs(freqs - fault_hz))
    peak = float(np.max(power[sel]))
    ref = float(np.median(power[(freqs >= ref_lo) & (freqs <= ref_hi)]))
    return peak / ref if ref > 0 else np.nan


def sideband_ratio(freqs, power, fault_hz, line_freq=LINE_FREQ, k=1, bw=2.0):
    """MCSA sideband ratios around the supply line, for one current spectrum.

    Bearing damage amplitude-modulates the supply current; the signature lives
    at line_freq +/- k * fault_hz. Returns the ratio at the lower (folded) and
    the upper sideband, e.g. at 50-76=26 Hz and 50+76=126 Hz for BPFO.
    """
    low = abs(line_freq - k * fault_hz)
    high = line_freq + k * fault_hz
    lo = fault_band_ratio(freqs, power, low, bw=bw)
    hi = fault_band_ratio(freqs, power, high, bw=bw)
    return lo, hi


def fault_energy_share(freqs, power, fault_hz, bw=8.0):
    """Energy share of one PSD inside a narrow band around fault_hz.

    Scale-free counterpart of fault_band_ratio: fraction of total spectral
    power concentrated at the fault frequency (0..1).
    """
    freqs = np.asarray(freqs)
    power = np.asarray(power)
    sel = np.abs(freqs - fault_hz) <= bw
    if not sel.any():
        sel = np.abs(freqs - fault_hz) == np.min(np.abs(freqs - fault_hz))
    total = float(np.sum(power))
    return float(np.sum(power[sel])) / total if total > 0 else np.nan