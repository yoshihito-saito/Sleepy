from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

import numpy as np

from .features import lowpass_decimate, pairwise_emg_from_lfp, sleep_indices
from .intan import IntanSession, read_channels


@dataclass
class ChannelProfile:
    version: int
    created_at: str
    source_basepath: str
    sample_rate: float
    n_channels: int
    emg_from_lfp_channels: list[int]
    nrem_sw_channel: int
    rem_theta_channel: int
    bad_channels: list[int]
    calibration_minutes: float
    quality_metrics: dict

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")


def load_profile(path: str | Path) -> ChannelProfile:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return ChannelProfile(**data)


def default_profile_path(session: IntanSession) -> Path:
    return session.basepath / f"{session.basename}.online_sleep_channel_profile.json"


def find_profile(path: str | Path) -> Path | None:
    p = Path(path)
    if p.is_file():
        return p
    if p.is_dir():
        matches = sorted(p.glob("*.online_sleep_channel_profile.json"))
        if matches:
            return matches[0]
        matches = sorted(p.glob("*channel_profile*.json"))
        if matches:
            return matches[0]
    return None


def estimate_channel_profile(
    session: IntanSession,
    calibration_minutes: float = 10.0,
    epoch_sec: float = 4.0,
    target_fs: float = 1250.0,
) -> ChannelProfile:
    usable_by_group = [group.usable for group in session.groups if group.usable]
    usable = session.usable_channels
    if not usable:
        raise ValueError("No usable channels found in amplifier.xml")

    emg_channels = _deterministic_emg_channels(usable_by_group, usable)
    candidates = _evenly_spaced(usable, max_count=64)
    n_samples = min(session.total_samples, int(round(calibration_minutes * 60.0 * session.sample_rate)))
    if n_samples < int(round(epoch_sec * session.sample_rate)):
        raise ValueError("Recording is shorter than one scoring epoch")

    delta_scores: list[float] = []
    theta_scores: list[float] = []
    for start in range(0, len(candidates), 8):
        batch = candidates[start : start + 8]
        raw = read_channels(session, batch, 0, n_samples)
        lfp, lfp_fs = lowpass_decimate(raw, session.sample_rate, target_fs=target_fs)
        for cidx in range(lfp.shape[1]):
            idx = sleep_indices(lfp[:, cidx], lfp_fs, epoch_sec)
            delta_scores.append(_separation_score(idx["nrem_sw_index"]))
            theta_scores.append(_separation_score(idx["theta_rem_index"]))

    nrem_idx = int(np.nanargmax(delta_scores))
    theta_idx = int(np.nanargmax(theta_scores))
    if theta_idx == nrem_idx and len(theta_scores) > 1:
        theta_order = np.argsort(theta_scores)[::-1]
        for candidate_idx in theta_order:
            if int(candidate_idx) != nrem_idx:
                theta_idx = int(candidate_idx)
                break

    emg_raw = read_channels(session, emg_channels, 0, n_samples)
    emg_lfp, emg_fs = lowpass_decimate(emg_raw, session.sample_rate, target_fs=target_fs)
    emg = pairwise_emg_from_lfp(emg_lfp, emg_fs, epoch_sec)
    emg_center = float(np.nanmedian(emg))
    emg_mad = float(np.nanmedian(np.abs(emg - emg_center)))
    emg_scale = float(1.4826 * emg_mad) if emg_mad > 0 else float(np.nanstd(emg))
    if not np.isfinite(emg_scale) or emg_scale <= 0:
        emg_scale = 1.0
    emg_z = (emg - emg_center) / emg_scale
    suggested_emg_threshold = _estimate_emg_threshold(emg_z)
    suggested_emg_raw_threshold = emg_center + suggested_emg_threshold * emg_scale

    theta_raw = read_channels(session, [int(candidates[theta_idx])], 0, n_samples)
    theta_lfp, theta_lfp_fs = lowpass_decimate(theta_raw, session.sample_rate, target_fs=target_fs)
    theta_idx_features = sleep_indices(theta_lfp[:, 0], theta_lfp_fs, epoch_sec)
    log_delta_theta = -np.log10(np.maximum(theta_idx_features["theta_delta_ratio"], np.finfo(float).eps))
    low_emg_log_delta_theta = log_delta_theta[emg_z <= suggested_emg_threshold]
    suggested_delta_theta_threshold = _estimate_delta_theta_threshold(low_emg_log_delta_theta)

    quality = {
        "candidate_channels": candidates,
        "emg_epoch_median": emg_center,
        "emg_epoch_mad": emg_mad,
        "emg_z_center": emg_center,
        "emg_z_scale": emg_scale,
        "emg_epoch_iqr": _iqr(emg),
        "suggested_emg_z_threshold": suggested_emg_threshold,
        "suggested_emg_raw_threshold": float(suggested_emg_raw_threshold),
        "suggested_log_delta_theta_threshold": suggested_delta_theta_threshold,
        "target_fs": float(target_fs),
        "nrem_sw_scores": [float(x) for x in delta_scores],
        "theta_scores": [float(x) for x in theta_scores],
        "selected_nrem_sw_score": float(delta_scores[nrem_idx]),
        "selected_theta_score": float(theta_scores[theta_idx]),
    }
    return ChannelProfile(
        version=1,
        created_at=datetime.now().isoformat(timespec="seconds"),
        source_basepath=str(session.basepath),
        sample_rate=session.sample_rate,
        n_channels=session.n_channels,
        emg_from_lfp_channels=emg_channels,
        nrem_sw_channel=int(candidates[nrem_idx]),
        rem_theta_channel=int(candidates[theta_idx]),
        bad_channels=session.bad_channels,
        calibration_minutes=float(calibration_minutes),
        quality_metrics=quality,
    )


def estimate_channel_profile_from_blocks(
    session: IntanSession,
    epoch_sec: float = 4.0,
    n_blocks: int = 12,
    block_sec: float = 20.0,
    target_fs: float = 1250.0,
    max_candidates: int = 32,
    n_jobs: int = 4,
) -> ChannelProfile:
    """Estimate parameters from blocks spread across a completed session.

    Slow-wave/theta and sampled EMGFromLFP metrics are computed after
    downsampling to ``target_fs``. With Intan data this is usually close to
    the conventional LFP rate and still leaves enough bandwidth for a
    250-500 Hz intracranial-EMG proxy.
    """
    usable_by_group = [group.usable for group in session.groups if group.usable]
    usable = session.usable_channels
    if not usable:
        raise ValueError("No usable channels found in amplifier.xml")

    emg_channels = _deterministic_emg_channels(usable_by_group, usable)
    candidates = _evenly_spaced(usable, max_count=max_candidates)
    block_samples = int(round(block_sec * session.sample_rate))
    if block_samples < int(round(epoch_sec * session.sample_rate)):
        block_samples = int(round(epoch_sec * session.sample_rate))
    starts = _spread_block_starts(session.total_samples, block_samples, n_blocks)
    if not starts:
        raise ValueError("Recording is shorter than one estimation block")

    batches = [candidates[start : start + 8] for start in range(0, len(candidates), 8)]
    worker_count = max(1, min(int(n_jobs), len(batches)))
    if worker_count == 1:
        scored_batches = [
            _score_candidate_batch(session, batch, starts, block_samples, target_fs, epoch_sec)
            for batch in batches
        ]
    else:
        with ThreadPoolExecutor(max_workers=worker_count) as pool:
            scored_batches = list(
                pool.map(
                    lambda batch: _score_candidate_batch(
                        session, batch, starts, block_samples, target_fs, epoch_sec
                    ),
                    batches,
                )
            )

    scored_candidates: list[int] = []
    delta_scores: list[float] = []
    theta_scores: list[float] = []
    for batch_channels, batch_delta, batch_theta in scored_batches:
        scored_candidates.extend(batch_channels)
        delta_scores.extend(batch_delta)
        theta_scores.extend(batch_theta)

    nrem_idx = int(np.nanargmax(delta_scores))
    theta_idx = int(np.nanargmax(theta_scores))
    if theta_idx == nrem_idx and len(theta_scores) > 1:
        theta_order = np.argsort(theta_scores)[::-1]
        for candidate_idx in theta_order:
            if int(candidate_idx) != nrem_idx:
                theta_idx = int(candidate_idx)
                break

    emg_chunks = []
    for sample_start in starts:
        emg_raw = read_channels(session, emg_channels, sample_start, block_samples)
        emg_lfp, emg_fs = lowpass_decimate(emg_raw, session.sample_rate, target_fs=target_fs)
        emg_chunks.append(pairwise_emg_from_lfp(emg_lfp, emg_fs, epoch_sec))
    emg = np.concatenate(emg_chunks)
    emg_center = float(np.nanmedian(emg))
    emg_mad = float(np.nanmedian(np.abs(emg - emg_center)))
    emg_scale = float(1.4826 * emg_mad) if emg_mad > 0 else float(np.nanstd(emg))
    if not np.isfinite(emg_scale) or emg_scale <= 0:
        emg_scale = 1.0
    emg_z = (emg - emg_center) / emg_scale
    suggested_emg_threshold = _estimate_emg_threshold(emg_z)
    suggested_emg_raw_threshold = emg_center + suggested_emg_threshold * emg_scale

    theta_blocks = []
    theta_fs = target_fs
    for sample_start in starts:
        theta_raw = read_channels(session, [int(scored_candidates[theta_idx])], sample_start, block_samples)
        theta_lfp, theta_fs = lowpass_decimate(theta_raw, session.sample_rate, target_fs=target_fs)
        theta_blocks.append(theta_lfp)
    theta_all = np.vstack(theta_blocks)
    theta_idx_features = sleep_indices(theta_all[:, 0], theta_fs, epoch_sec)
    log_delta_theta = -np.log10(np.maximum(theta_idx_features["theta_delta_ratio"], np.finfo(float).eps))
    low_emg_log_delta_theta = log_delta_theta[emg_z[: log_delta_theta.size] <= suggested_emg_threshold]
    suggested_delta_theta_threshold = _estimate_delta_theta_threshold(low_emg_log_delta_theta)

    quality = {
        "profile_mode": "sampled_full_recording",
        "candidate_channels": candidates,
        "sampled_block_starts_sec": [float(x / session.sample_rate) for x in starts],
        "sampled_block_sec": float(block_sec),
        "target_fs": float(target_fs),
        "emg_epoch_median": emg_center,
        "emg_epoch_mad": emg_mad,
        "emg_z_center": emg_center,
        "emg_z_scale": emg_scale,
        "emg_epoch_iqr": _iqr(emg),
        "suggested_emg_z_threshold": suggested_emg_threshold,
        "suggested_emg_raw_threshold": float(suggested_emg_raw_threshold),
        "suggested_log_delta_theta_threshold": suggested_delta_theta_threshold,
        "nrem_sw_scores": [float(x) for x in delta_scores],
        "theta_scores": [float(x) for x in theta_scores],
        "selected_nrem_sw_score": float(delta_scores[nrem_idx]),
        "selected_theta_score": float(theta_scores[theta_idx]),
    }
    sampled_minutes = len(starts) * block_sec / 60.0
    return ChannelProfile(
        version=1,
        created_at=datetime.now().isoformat(timespec="seconds"),
        source_basepath=str(session.basepath),
        sample_rate=session.sample_rate,
        n_channels=session.n_channels,
        emg_from_lfp_channels=emg_channels,
        nrem_sw_channel=int(scored_candidates[nrem_idx]),
        rem_theta_channel=int(scored_candidates[theta_idx]),
        bad_channels=session.bad_channels,
        calibration_minutes=float(sampled_minutes),
        quality_metrics=quality,
    )


def make_manual_channel_profile(session: IntanSession) -> ChannelProfile:
    """Create a no-calibration profile from amplifier.xml channel groups only."""
    usable_by_group = [group.usable for group in session.groups if group.usable]
    usable = session.usable_channels
    if not usable:
        raise ValueError("No usable channels found in amplifier.xml")

    emg_channels = _deterministic_emg_channels(usable_by_group, usable)
    nrem_sw_channel = usable[len(usable) // 3]
    rem_theta_channel = usable[(2 * len(usable)) // 3]
    if rem_theta_channel == nrem_sw_channel and len(usable) > 1:
        rem_theta_channel = usable[-1]

    quality = {
        "profile_mode": "manual_no_calibration",
        "candidate_channels": usable,
        "suggested_emg_raw_threshold": float("nan"),
        "suggested_log_delta_theta_threshold": float("nan"),
        "emg_z_center": float("nan"),
        "emg_z_scale": float("nan"),
    }
    return ChannelProfile(
        version=1,
        created_at=datetime.now().isoformat(timespec="seconds"),
        source_basepath=str(session.basepath),
        sample_rate=session.sample_rate,
        n_channels=session.n_channels,
        emg_from_lfp_channels=emg_channels,
        nrem_sw_channel=int(nrem_sw_channel),
        rem_theta_channel=int(rem_theta_channel),
        bad_channels=session.bad_channels,
        calibration_minutes=0.0,
        quality_metrics=quality,
    )


def _deterministic_emg_channels(usable_by_group: list[list[int]], usable: list[int]) -> list[int]:
    if len(usable_by_group) > 1:
        selected = [group[len(group) // 2] for group in usable_by_group if group]
    else:
        selected = _evenly_spaced(usable, max_count=min(5, len(usable)))
    if len(selected) < 2 and len(usable) >= 2:
        selected = _evenly_spaced(usable, max_count=2)
    return sorted(set(map(int, selected)))


def _evenly_spaced(values: list[int], max_count: int) -> list[int]:
    vals = sorted(set(map(int, values)))
    if len(vals) <= max_count:
        return vals
    idx = np.linspace(0, len(vals) - 1, max_count).round().astype(int)
    return [vals[i] for i in sorted(set(idx))]


def _spread_block_starts(total_samples: int, block_samples: int, n_blocks: int) -> list[int]:
    if total_samples < block_samples:
        return []
    max_start = total_samples - block_samples
    if max_start <= 0 or n_blocks <= 1:
        return [0]
    starts = np.linspace(0, max_start, n_blocks).round().astype(int)
    return sorted(set(map(int, starts)))


def _score_candidate_batch(
    session: IntanSession,
    batch: list[int],
    starts: list[int],
    block_samples: int,
    target_fs: float,
    epoch_sec: float,
) -> tuple[list[int], list[float], list[float]]:
    spectral_blocks = []
    spectral_fs = target_fs
    for sample_start in starts:
        raw = read_channels(session, batch, sample_start, block_samples)
        lfp, spectral_fs = lowpass_decimate(raw, session.sample_rate, target_fs=target_fs)
        spectral_blocks.append(lfp)
    lfp_all = np.vstack(spectral_blocks)
    delta_scores: list[float] = []
    theta_scores: list[float] = []
    for cidx in range(lfp_all.shape[1]):
        idx = sleep_indices(lfp_all[:, cidx], spectral_fs, epoch_sec)
        delta_scores.append(_separation_score(idx["nrem_sw_index"]))
        theta_scores.append(_separation_score(idx["theta_rem_index"]))
    return list(batch), delta_scores, theta_scores


def _iqr(x: np.ndarray) -> float:
    q25, q75 = np.nanpercentile(x, [25, 75])
    return float(q75 - q25)


def _separation_score(x: np.ndarray) -> float:
    if x.size < 3:
        return 0.0
    lo, hi = np.nanpercentile(x, [25, 75])
    iqr = hi - lo
    scale = iqr if iqr > 0 else np.nanstd(x)
    if not np.isfinite(scale) or scale == 0:
        return 0.0
    return float((hi - lo) / scale * np.log1p(max(0.0, hi)))


def _estimate_emg_threshold(emg_z: np.ndarray) -> float:
    x = np.asarray(emg_z, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 8:
        return 1.0
    x = np.clip(x, np.nanpercentile(x, 1), np.nanpercentile(x, 99))
    c1, c2 = np.nanpercentile(x, [35, 85])
    for _ in range(20):
        d1 = np.abs(x - c1)
        d2 = np.abs(x - c2)
        low = x[d1 <= d2]
        high = x[d2 < d1]
        if low.size == 0 or high.size == 0:
            break
        c1_new = float(np.nanmean(low))
        c2_new = float(np.nanmean(high))
        if abs(c1_new - c1) + abs(c2_new - c2) < 1e-6:
            break
        c1, c2 = c1_new, c2_new
    threshold = (min(c1, c2) + max(c1, c2)) / 2.0
    if not np.isfinite(threshold):
        threshold = 1.0
    return float(np.clip(threshold, 0.5, 2.5))


def _estimate_delta_theta_threshold(log_delta_theta: np.ndarray) -> float:
    x = np.asarray(log_delta_theta, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 8:
        return 0.0
    x = np.clip(x, np.nanpercentile(x, 2), np.nanpercentile(x, 98))
    c1, c2 = np.nanpercentile(x, [25, 75])
    for _ in range(20):
        d1 = np.abs(x - c1)
        d2 = np.abs(x - c2)
        low = x[d1 <= d2]
        high = x[d2 < d1]
        if low.size == 0 or high.size == 0:
            break
        c1_new = float(np.nanmean(low))
        c2_new = float(np.nanmean(high))
        if abs(c1_new - c1) + abs(c2_new - c2) < 1e-6:
            break
        c1, c2 = c1_new, c2_new
    threshold = (min(c1, c2) + max(c1, c2)) / 2.0
    if not np.isfinite(threshold):
        threshold = float(np.nanmedian(x))
    return float(threshold)
