from __future__ import annotations

import numpy as np
from scipy import signal


EPS = np.finfo(float).eps


def bandpass_filter(data: np.ndarray, fs: float, low: float, high: float, order: int = 4) -> np.ndarray:
    nyq = fs / 2.0
    low = max(low, 0.001)
    high = min(high, nyq * 0.95)
    if high <= low:
        return data - np.mean(data, axis=0, keepdims=True)
    sos = signal.butter(order, [low, high], btype="bandpass", fs=fs, output="sos")
    padlen = min(data.shape[0] - 1, 3 * (2 * len(sos) + 1))
    if padlen < 1:
        return data
    return signal.sosfiltfilt(sos, data, axis=0, padlen=padlen)


def lowpass_decimate(data: np.ndarray, fs: float, target_fs: float = 500.0) -> tuple[np.ndarray, float]:
    if fs <= target_fs * 1.25:
        return data, fs
    q = max(1, int(round(fs / target_fs)))
    return signal.decimate(data, q, axis=0, zero_phase=True), fs / q


def pairwise_emg_from_lfp(data: np.ndarray, fs: float, epoch_sec: float) -> np.ndarray:
    """Mean pairwise zero-lag correlation of high-frequency LFP per epoch."""
    if data.shape[1] < 2:
        return rms_epochs(data[:, 0], fs, epoch_sec)
    high = min(600.0, fs / 2.0 * 0.8)
    low = min(300.0, high * 0.5)
    filt = bandpass_filter(data, fs, low, high)
    n = int(round(epoch_sec * fs))
    n_epochs = filt.shape[0] // n
    out = np.full(n_epochs, np.nan)
    for idx in range(n_epochs):
        chunk = filt[idx * n : (idx + 1) * n, :]
        corr = np.corrcoef(chunk, rowvar=False)
        vals = corr[np.triu_indices(corr.shape[0], 1)]
        out[idx] = np.nanmean(vals)
    return out


def rms_epochs(x: np.ndarray, fs: float, epoch_sec: float) -> np.ndarray:
    n = int(round(epoch_sec * fs))
    n_epochs = x.shape[0] // n
    x = x[: n_epochs * n]
    if n_epochs == 0:
        return np.zeros(0)
    return np.sqrt(np.mean(x.reshape(n_epochs, n) ** 2, axis=1))


def bandpower_epochs(x: np.ndarray, fs: float, epoch_sec: float, band: tuple[float, float]) -> np.ndarray:
    n = int(round(epoch_sec * fs))
    n_epochs = x.shape[0] // n
    if n_epochs == 0:
        return np.zeros(0)
    x = x[: n_epochs * n].reshape(n_epochs, n)
    nperseg = min(n, int(round(2.0 * fs)))
    freqs, psd = signal.welch(x, fs=fs, nperseg=nperseg, axis=1)
    mask = (freqs >= band[0]) & (freqs <= band[1])
    if not np.any(mask):
        return np.zeros(n_epochs)
    return np.trapezoid(psd[:, mask], freqs[mask], axis=1)


def sleep_indices(lfp: np.ndarray, fs: float, epoch_sec: float) -> dict[str, np.ndarray]:
    x = lfp - np.nanmedian(lfp)
    delta = bandpower_epochs(x, fs, epoch_sec, (0.5, 4.0))
    theta = bandpower_epochs(x, fs, epoch_sec, (6.0, 10.0))
    alpha = bandpower_epochs(x, fs, epoch_sec, (10.0, 15.0))
    broad = bandpower_epochs(x, fs, epoch_sec, (0.5, 30.0))
    return {
        "delta_power": delta,
        "theta_power": theta,
        "alpha_power": alpha,
        "nrem_sw_index": delta / (broad + EPS),
        "theta_delta_ratio": theta / (delta + EPS),
        "theta_rem_index": (theta * theta) / ((delta + EPS) * (alpha + EPS)),
    }


def robust_z(x: np.ndarray) -> np.ndarray:
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    scale = 1.4826 * mad if mad > 0 else np.nanstd(x)
    if not np.isfinite(scale) or scale == 0:
        scale = 1.0
    return (x - med) / scale

