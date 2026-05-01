from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np
from scipy.io import savemat

from .scoring import ScoringResult, STATE_CODES, result_to_dict


def save_states_mat(result: ScoringResult, basepath: str | Path) -> Path:
    base = Path(basepath)
    out = base / f"{base.name}.OnlineSleepState.states.mat"
    timestamps = result.timestamps_sec.reshape(-1, 1)
    states = result.confirmed_state_codes.reshape(-1, 1)

    sleep_state = {
        "idx": {
            "timestamps": timestamps,
            "states": states,
            "states_raw": result.raw_state_codes.reshape(-1, 1),
        },
        "ints": _intervals_by_state(result.timestamps_sec, result.confirmed_state_codes, result.params.epoch_sec),
        "detectorinfo": {
            "detectorname": "sleepy_python",
            "detectiondate": "",
            "detectionparms": asdict(result.params),
            "state_codes": STATE_CODES,
        },
    }

    online_features = result_to_dict(result)
    savemat(
        out,
        {"SleepState": sleep_state, "OnlineSleepFeatures": online_features},
        do_compression=True,
        long_field_names=True,
    )
    return out


def _intervals_by_state(timestamps: np.ndarray, codes: np.ndarray, epoch_sec: float) -> dict:
    out = {}
    for name, code in STATE_CODES.items():
        mask = codes == code
        intervals = []
        if mask.size:
            starts = np.flatnonzero(np.diff(np.r_[False, mask].astype(int)) == 1)
            ends = np.flatnonzero(np.diff(np.r_[mask, False].astype(int)) == -1) - 1
            for s, e in zip(starts, ends):
                intervals.append([timestamps[s], timestamps[e] + epoch_sec])
        out[name] = np.asarray(intervals, dtype=float).reshape((-1, 2))
    return out
