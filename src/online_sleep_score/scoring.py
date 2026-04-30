from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from .features import lowpass_decimate, pairwise_emg_from_lfp, robust_z, sleep_indices
from .intan import IntanSession, read_channels
from .profile import ChannelProfile


STATE_CODES = {"Wake": 1, "NREM": 3, "REM": 5}
CODE_STATES = {v: k for k, v in STATE_CODES.items()}


@dataclass
class ScoringParams:
    epoch_sec: float = 4.0
    confirmation_count: int = 3
    emg_threshold_mode: str = "manual"
    emg_threshold: float = 0.3
    emg_threshold_actual: float = float("nan")
    emg_z_center: float = float("nan")
    emg_z_scale: float = float("nan")
    delta_theta_threshold_mode: str = "manual"
    delta_theta_threshold: float = 0.0
    delta_theta_threshold_actual: float = float("nan")
    wake_to_rem_block: bool = True
    online: bool = True
    auto_zscore_features: bool = True
    processing_target_fs: float = 1250.0


@dataclass
class ScoringResult:
    timestamps_sec: np.ndarray
    estimated_emg: np.ndarray
    nrem_sw_index: np.ndarray
    theta_rem_index: np.ndarray
    theta_delta_ratio: np.ndarray
    estimated_emg_z: np.ndarray
    log_delta_theta_ratio: np.ndarray
    emg_threshold_history: np.ndarray
    delta_theta_threshold_history: np.ndarray
    raw_state: list[str]
    confirmed_state: list[str]
    reason_code: list[str]
    profile: ChannelProfile
    params: ScoringParams

    @property
    def raw_state_codes(self) -> np.ndarray:
        return np.array([STATE_CODES[x] for x in self.raw_state], dtype=np.int16)

    @property
    def confirmed_state_codes(self) -> np.ndarray:
        return np.array([STATE_CODES[x] for x in self.confirmed_state], dtype=np.int16)


@dataclass
class ScoringAccumulator:
    profile: ChannelProfile
    params: ScoringParams
    timestamps_sec: list[float] = field(default_factory=list)
    estimated_emg: list[float] = field(default_factory=list)
    nrem_sw_index: list[float] = field(default_factory=list)
    theta_rem_index: list[float] = field(default_factory=list)
    theta_delta_ratio: list[float] = field(default_factory=list)
    estimated_emg_z: list[float] = field(default_factory=list)
    log_delta_theta_ratio: list[float] = field(default_factory=list)
    emg_threshold_history: list[float] = field(default_factory=list)
    delta_theta_threshold_history: list[float] = field(default_factory=list)
    raw_state: list[str] = field(default_factory=list)
    confirmed_state: list[str] = field(default_factory=list)
    reason_code: list[str] = field(default_factory=list)

    def append(self, part: ScoringResult) -> None:
        self.timestamps_sec.extend(map(float, part.timestamps_sec))
        self.estimated_emg.extend(map(float, part.estimated_emg))
        self.nrem_sw_index.extend(map(float, part.nrem_sw_index))
        self.theta_rem_index.extend(map(float, part.theta_rem_index))
        self.theta_delta_ratio.extend(map(float, part.theta_delta_ratio))
        self.estimated_emg_z.extend(map(float, part.estimated_emg_z))
        self.log_delta_theta_ratio.extend(map(float, part.log_delta_theta_ratio))
        self.emg_threshold_history.extend(map(float, part.emg_threshold_history))
        self.delta_theta_threshold_history.extend(map(float, part.delta_theta_threshold_history))
        self.raw_state.extend(part.raw_state)
        self.confirmed_state.extend(part.confirmed_state)
        self.reason_code.extend(part.reason_code)

    def __len__(self) -> int:
        return len(self.timestamps_sec)

    def to_result(self) -> ScoringResult:
        return ScoringResult(
            timestamps_sec=np.asarray(self.timestamps_sec, dtype=float),
            estimated_emg=np.asarray(self.estimated_emg, dtype=float),
            nrem_sw_index=np.asarray(self.nrem_sw_index, dtype=float),
            theta_rem_index=np.asarray(self.theta_rem_index, dtype=float),
            theta_delta_ratio=np.asarray(self.theta_delta_ratio, dtype=float),
            estimated_emg_z=np.asarray(self.estimated_emg_z, dtype=float),
            log_delta_theta_ratio=np.asarray(self.log_delta_theta_ratio, dtype=float),
            emg_threshold_history=np.asarray(self.emg_threshold_history, dtype=float),
            delta_theta_threshold_history=np.asarray(self.delta_theta_threshold_history, dtype=float),
            raw_state=list(self.raw_state),
            confirmed_state=list(self.confirmed_state),
            reason_code=list(self.reason_code),
            profile=self.profile,
            params=self.params,
        )


@dataclass
class ScoringState:
    raw_history: list[str] = field(default_factory=list)
    confirmed_state: str = "Wake"

    def step(self, raw: str, params: ScoringParams) -> str:
        self.raw_history.append(raw)
        n = params.confirmation_count
        if len(self.raw_history) >= n and len(set(self.raw_history[-n:])) == 1:
            self.confirmed_state = raw
        return self.confirmed_state


@dataclass
class OnlineThresholdState:
    estimated_emg: list[float] = field(default_factory=list)
    log_delta_theta_ratio: list[float] = field(default_factory=list)
    low_emg_log_delta_theta: list[float] = field(default_factory=list)

    def update_and_classify(
        self,
        estimated_emg: float,
        theta_delta_ratio: float,
        scoring_state: ScoringState,
        params: ScoringParams,
    ) -> tuple[float, float, float, float, str, str, str]:
        self.estimated_emg.append(float(estimated_emg))
        log_dt = float(-np.log10(max(theta_delta_ratio, np.finfo(float).eps)))
        self.log_delta_theta_ratio.append(log_dt)

        if np.isfinite(params.emg_z_center) and np.isfinite(params.emg_z_scale) and params.emg_z_scale > 0:
            emg_z = float((estimated_emg - params.emg_z_center) / params.emg_z_scale)
        else:
            emg_z = float(robust_z(np.asarray(self.estimated_emg, dtype=float))[-1])

        emg_threshold = params.emg_threshold_actual
        if not np.isfinite(emg_threshold):
            emg_threshold = float(params.emg_threshold)
            params.emg_threshold_actual = emg_threshold

        if estimated_emg <= emg_threshold:
            self.low_emg_log_delta_theta.append(log_dt)
        delta_theta_threshold = params.delta_theta_threshold_actual
        if not np.isfinite(delta_theta_threshold):
            delta_theta_threshold = float(params.delta_theta_threshold)
            params.delta_theta_threshold_actual = delta_theta_threshold

        raw_label, reason = classify_epoch(
            float(estimated_emg),
            log_dt,
            scoring_state.confirmed_state,
            params,
        )
        confirmed = scoring_state.step(raw_label, params)
        return emg_z, log_dt, emg_threshold, delta_theta_threshold, raw_label, confirmed, reason


ProgressCallback = Callable[[str], None]


def score_recording(
    session: IntanSession,
    profile: ChannelProfile,
    params: ScoringParams,
    start_sample: int = 0,
    stop_sample: int | None = None,
    progress: ProgressCallback | None = None,
) -> ScoringResult:
    fs = session.sample_rate
    epoch_samples = int(round(params.epoch_sec * fs))
    stop_sample = session.total_samples if stop_sample is None else min(stop_sample, session.total_samples)
    total = max(0, stop_sample - start_sample)
    total_epochs = total // epoch_samples
    if total_epochs <= 0:
        raise ValueError("No complete epochs available for scoring")

    emg_all: list[np.ndarray] = []
    nrem_all: list[np.ndarray] = []
    theta_all: list[np.ndarray] = []
    theta_delta_all: list[np.ndarray] = []
    times_all: list[np.ndarray] = []

    channels = sorted(set(profile.emg_from_lfp_channels + [profile.nrem_sw_channel, profile.rem_theta_channel]))
    emg_idx = [channels.index(ch) for ch in profile.emg_from_lfp_channels]
    nrem_idx = channels.index(profile.nrem_sw_channel)
    theta_idx = channels.index(profile.rem_theta_channel)

    epochs_per_block = max(1, int(round(60.0 / params.epoch_sec)))
    block_samples = epochs_per_block * epoch_samples
    done_epochs = 0
    while done_epochs < total_epochs:
        read_epochs = min(epochs_per_block, total_epochs - done_epochs)
        read_start = start_sample + done_epochs * epoch_samples
        n_samples = read_epochs * epoch_samples
        raw = read_channels(session, channels, read_start, n_samples)

        proc, proc_fs = lowpass_decimate(raw, fs, target_fs=params.processing_target_fs)
        emg_all.append(pairwise_emg_from_lfp(proc[:, emg_idx], proc_fs, params.epoch_sec))
        nrem_features = sleep_indices(proc[:, nrem_idx], proc_fs, params.epoch_sec)
        theta_features = sleep_indices(proc[:, theta_idx], proc_fs, params.epoch_sec)
        nrem_all.append(nrem_features["nrem_sw_index"])
        theta_all.append(theta_features["theta_rem_index"])
        theta_delta_all.append(theta_features["theta_delta_ratio"])
        times_all.append((read_start / fs) + np.arange(read_epochs) * params.epoch_sec)
        done_epochs += read_epochs
        if progress:
            progress(f"Scored {done_epochs}/{total_epochs} epochs")

    estimated_emg = np.concatenate(emg_all)
    nrem_sw_index = np.concatenate(nrem_all)
    theta_rem_index = np.concatenate(theta_all)
    theta_delta_ratio = np.concatenate(theta_delta_all)
    timestamps_sec = np.concatenate(times_all)

    raw_state: list[str] = []
    confirmed_state: list[str] = []
    reasons: list[str] = []
    emg_z: list[float] = []
    log_dt: list[float] = []
    emg_thresholds: list[float] = []
    delta_theta_thresholds: list[float] = []
    state_machine = ScoringState()
    threshold_state = OnlineThresholdState()
    freeze_thresholds_from_profile(profile, params)
    for idx in range(len(timestamps_sec)):
        ez, ldt, eth, dth, raw_label, confirmed, reason = threshold_state.update_and_classify(
            estimated_emg[idx],
            theta_delta_ratio[idx],
            state_machine,
            params,
        )
        emg_z.append(ez)
        log_dt.append(ldt)
        emg_thresholds.append(eth)
        delta_theta_thresholds.append(dth)
        raw_state.append(raw_label)
        confirmed_state.append(confirmed)
        reasons.append(reason)

    return ScoringResult(
        timestamps_sec=timestamps_sec,
        estimated_emg=estimated_emg,
        nrem_sw_index=nrem_sw_index,
        theta_rem_index=theta_rem_index,
        theta_delta_ratio=theta_delta_ratio,
        estimated_emg_z=np.asarray(emg_z, dtype=float),
        log_delta_theta_ratio=np.asarray(log_dt, dtype=float),
        emg_threshold_history=np.asarray(emg_thresholds, dtype=float),
        delta_theta_threshold_history=np.asarray(delta_theta_thresholds, dtype=float),
        raw_state=raw_state,
        confirmed_state=confirmed_state,
        reason_code=reasons,
        profile=profile,
        params=params,
    )


def extract_feature_epochs(
    session: IntanSession,
    profile: ChannelProfile,
    params: ScoringParams,
    start_sample: int,
    n_epochs: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    fs = session.sample_rate
    epoch_samples = int(round(params.epoch_sec * fs))
    channels = sorted(set(profile.emg_from_lfp_channels + [profile.nrem_sw_channel, profile.rem_theta_channel]))
    emg_idx = [channels.index(ch) for ch in profile.emg_from_lfp_channels]
    nrem_idx = channels.index(profile.nrem_sw_channel)
    theta_idx = channels.index(profile.rem_theta_channel)
    raw = read_channels(session, channels, start_sample, n_epochs * epoch_samples)
    actual_epochs = raw.shape[0] // epoch_samples
    raw = raw[: actual_epochs * epoch_samples]
    proc, proc_fs = lowpass_decimate(raw, fs, target_fs=params.processing_target_fs)
    estimated_emg = pairwise_emg_from_lfp(proc[:, emg_idx], proc_fs, params.epoch_sec)
    nrem_features = sleep_indices(proc[:, nrem_idx], proc_fs, params.epoch_sec)
    theta_features = sleep_indices(proc[:, theta_idx], proc_fs, params.epoch_sec)
    timestamps_sec = (start_sample / fs) + np.arange(actual_epochs) * params.epoch_sec
    return (
        timestamps_sec,
        estimated_emg,
        nrem_features["nrem_sw_index"],
        theta_features["theta_rem_index"],
        theta_features["theta_delta_ratio"],
    )


def classify_feature_epochs(
    timestamps_sec: np.ndarray,
    estimated_emg: np.ndarray,
    nrem_sw_index: np.ndarray,
    theta_rem_index: np.ndarray,
    theta_delta_ratio: np.ndarray,
    profile: ChannelProfile,
    params: ScoringParams,
    state_machine: ScoringState | None = None,
    threshold_state: OnlineThresholdState | None = None,
) -> ScoringResult:
    state_machine = state_machine or ScoringState()
    threshold_state = threshold_state or OnlineThresholdState()
    freeze_thresholds_from_profile(profile, params)

    raw_state: list[str] = []
    confirmed_state: list[str] = []
    reasons: list[str] = []
    emg_z: list[float] = []
    log_dt: list[float] = []
    emg_thresholds: list[float] = []
    delta_theta_thresholds: list[float] = []
    for idx in range(len(timestamps_sec)):
        ez, ldt, eth, dth, raw_label, confirmed, reason = threshold_state.update_and_classify(
            estimated_emg[idx],
            theta_delta_ratio[idx],
            state_machine,
            params,
        )
        emg_z.append(ez)
        log_dt.append(ldt)
        emg_thresholds.append(eth)
        delta_theta_thresholds.append(dth)
        raw_state.append(raw_label)
        confirmed_state.append(confirmed)
        reasons.append(reason)

    return ScoringResult(
        timestamps_sec=timestamps_sec,
        estimated_emg=estimated_emg,
        nrem_sw_index=nrem_sw_index,
        theta_rem_index=theta_rem_index,
        theta_delta_ratio=theta_delta_ratio,
        estimated_emg_z=np.asarray(emg_z, dtype=float),
        log_delta_theta_ratio=np.asarray(log_dt, dtype=float),
        emg_threshold_history=np.asarray(emg_thresholds, dtype=float),
        delta_theta_threshold_history=np.asarray(delta_theta_thresholds, dtype=float),
        raw_state=raw_state,
        confirmed_state=confirmed_state,
        reason_code=reasons,
        profile=profile,
        params=params,
    )


def concatenate_results(parts: list[ScoringResult]) -> ScoringResult:
    if not parts:
        raise ValueError("No result parts to concatenate")
    first = parts[0]
    return ScoringResult(
        timestamps_sec=np.concatenate([p.timestamps_sec for p in parts]),
        estimated_emg=np.concatenate([p.estimated_emg for p in parts]),
        nrem_sw_index=np.concatenate([p.nrem_sw_index for p in parts]),
        theta_rem_index=np.concatenate([p.theta_rem_index for p in parts]),
        theta_delta_ratio=np.concatenate([p.theta_delta_ratio for p in parts]),
        estimated_emg_z=np.concatenate([p.estimated_emg_z for p in parts]),
        log_delta_theta_ratio=np.concatenate([p.log_delta_theta_ratio for p in parts]),
        emg_threshold_history=np.concatenate([p.emg_threshold_history for p in parts]),
        delta_theta_threshold_history=np.concatenate([p.delta_theta_threshold_history for p in parts]),
        raw_state=[x for p in parts for x in p.raw_state],
        confirmed_state=[x for p in parts for x in p.confirmed_state],
        reason_code=[x for p in parts for x in p.reason_code],
        profile=first.profile,
        params=first.params,
    )


def classify_epoch(
    estimated_emg: float,
    log_delta_theta: float,
    previous_confirmed: str,
    params: ScoringParams,
) -> tuple[str, str]:
    emg_threshold = params.emg_threshold_actual
    if not np.isfinite(emg_threshold):
        emg_threshold = params.emg_threshold
    if estimated_emg > emg_threshold:
        return "Wake", "high_emg"
    delta_theta_threshold = params.delta_theta_threshold_actual
    if not np.isfinite(delta_theta_threshold):
        delta_theta_threshold = params.delta_theta_threshold
    if log_delta_theta >= delta_theta_threshold:
        return "NREM", "low_emg_high_delta_theta"
    if params.wake_to_rem_block and previous_confirmed == "Wake":
        return "Wake", "wake_to_rem_blocked_low_delta_theta"
    return "REM", "low_emg_low_delta_theta"


def prepare_rule_features(
    estimated_emg: np.ndarray,
    nrem_sw_index: np.ndarray,
    theta_rem_index: np.ndarray,
    params: ScoringParams,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if params.auto_zscore_features:
        emg_for_rules = robust_z(estimated_emg)
        nrem_for_rules = robust_z(nrem_sw_index)
        theta_for_rules = robust_z(theta_rem_index)
    else:
        emg_for_rules = estimated_emg
        nrem_for_rules = nrem_sw_index
        theta_for_rules = theta_rem_index

    if params.emg_threshold_mode == "auto":
        params.emg_threshold_actual = estimate_emg_threshold(emg_for_rules)
    else:
        params.emg_threshold_actual = params.emg_threshold
    return emg_for_rules, nrem_for_rules, theta_for_rules


def estimate_emg_threshold(emg_for_rules: np.ndarray) -> float:
    x = np.asarray(emg_for_rules, dtype=float)
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
        new_c1 = float(np.nanmean(low))
        new_c2 = float(np.nanmean(high))
        if abs(new_c1 - c1) + abs(new_c2 - c2) < 1e-6:
            break
        c1, c2 = new_c1, new_c2
    threshold = (min(c1, c2) + max(c1, c2)) / 2.0
    fallback = 1.0
    if not np.isfinite(threshold):
        threshold = fallback
    return float(np.clip(threshold, 0.5, 2.5))


def estimate_delta_theta_threshold(log_delta_theta: np.ndarray) -> float:
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
        new_c1 = float(np.nanmean(low))
        new_c2 = float(np.nanmean(high))
        if abs(new_c1 - c1) + abs(new_c2 - c2) < 1e-6:
            break
        c1, c2 = new_c1, new_c2
    threshold = (min(c1, c2) + max(c1, c2)) / 2.0
    if not np.isfinite(threshold):
        threshold = float(np.nanmedian(x))
    return float(threshold)


def freeze_thresholds_from_profile(profile: ChannelProfile, params: ScoringParams) -> None:
    q = profile.quality_metrics or {}
    if not np.isfinite(params.emg_z_center):
        params.emg_z_center = _float_or_nan(q.get("emg_z_center", q.get("emg_epoch_median")))
    if not np.isfinite(params.emg_z_scale) or params.emg_z_scale <= 0:
        params.emg_z_scale = _float_or_nan(q.get("emg_z_scale"))

    if params.emg_threshold_mode == "auto":
        params.emg_threshold_actual = _float_or_default(q.get("suggested_emg_raw_threshold"), params.emg_threshold)
    else:
        params.emg_threshold_actual = float(params.emg_threshold)

    if params.delta_theta_threshold_mode == "auto":
        params.delta_theta_threshold_actual = _float_or_default(
            q.get("suggested_log_delta_theta_threshold"),
            params.delta_theta_threshold,
        )
    else:
        params.delta_theta_threshold_actual = float(params.delta_theta_threshold)


def _float_or_nan(value) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if np.isfinite(out) else float("nan")


def _float_or_default(value, default: float) -> float:
    out = _float_or_nan(value)
    return float(default) if not np.isfinite(out) else out


def result_to_dict(result: ScoringResult) -> dict:
    return {
        "timestamps_sec": result.timestamps_sec,
        "estimated_emg": result.estimated_emg,
        "nrem_sw_index": result.nrem_sw_index,
        "theta_rem_index": result.theta_rem_index,
        "theta_delta_ratio": result.theta_delta_ratio,
        "estimated_emg_z": result.estimated_emg_z,
        "log_delta_theta_ratio": result.log_delta_theta_ratio,
        "emg_threshold_history": result.emg_threshold_history,
        "delta_theta_threshold_history": result.delta_theta_threshold_history,
        "raw_state": np.array(result.raw_state, dtype=object),
        "confirmed_state": np.array(result.confirmed_state, dtype=object),
        "raw_state_codes": result.raw_state_codes,
        "confirmed_state_codes": result.confirmed_state_codes,
        "reason_code": np.array(result.reason_code, dtype=object),
        "profile": asdict(result.profile),
        "params": asdict(result.params),
    }
