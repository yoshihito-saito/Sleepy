from __future__ import annotations

from pathlib import Path
import queue
import sys
import threading
import time

import matplotlib

matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np
from PySide6.QtCore import QTimer
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from .intan import load_session
from .matlab_save import save_states_mat
from .profile import default_profile_path, estimate_channel_profile, estimate_channel_profile_from_blocks, load_profile
from .scoring import (
    OnlineThresholdState,
    ScoringAccumulator,
    ScoringParams,
    ScoringResult,
    ScoringState,
    STATE_CODES,
    classify_feature_epochs,
    extract_feature_epochs,
)


class SleepScoreApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("sleepy")
        icon_path = _resource_path("logo/logo.ico")
        if icon_path.exists():
            self.setWindowIcon(QIcon(str(icon_path)))
        self.resize(1220, 820)
        self.queue: queue.Queue = queue.Queue()
        self.worker: threading.Thread | None = None
        self.stop_event = threading.Event()
        self.result: ScoringResult | None = None
        self.estimated_profile = None
        self._build_ui()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._poll_queue)
        self.timer.start(200)

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        controls = QWidget()
        controls.setFixedWidth(345)
        controls_layout = QVBoxLayout(controls)
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.setSpacing(8)
        root.addWidget(controls)

        plots = QWidget()
        plots_layout = QVBoxLayout(plots)
        plots_layout.setContentsMargins(0, 0, 0, 0)
        root.addWidget(plots, 1)

        self.basepath_edit = self._path_row(controls_layout, "Basepath", self._browse_basepath)
        self.previous_path_edit = self._path_row(controls_layout, "Previous datapath", self._browse_previous_path)

        estimate_button = QPushButton("Estimate parameters from previous session")
        estimate_button.clicked.connect(self._start_previous_parameter_estimation)
        controls_layout.addWidget(estimate_button)

        params = QGroupBox("Parameters")
        params_layout = QGridLayout(params)
        params_layout.setContentsMargins(8, 8, 8, 8)
        params_layout.setHorizontalSpacing(8)
        params_layout.setVerticalSpacing(6)
        controls_layout.addWidget(params)

        self.estimation_minutes = self._double_spin(params_layout, 0, "Estimation min", 1.0, 0.25, 30, 0.25)
        self.epoch_sec = self._double_spin(params_layout, 1, "Epoch sec", 4.0, 1, 30, 1)
        self.confirmation_count = self._int_spin(params_layout, 2, "Confirm count", 3, 1, 10)
        self.emg_threshold = self._line_value(params_layout, 3, "EMG threshold", "0.3")
        self.delta_theta_threshold = self._line_value(params_layout, 4, "log10 D/T threshold", "0")
        self.wake_to_rem_block = QCheckBox("Block Wake -> REM")
        self.wake_to_rem_block.setChecked(True)
        params_layout.addWidget(self.wake_to_rem_block, 5, 0, 1, 2)

        actions = QHBoxLayout()
        run_button = QPushButton("Run scoring")
        run_button.clicked.connect(self._start_scoring)
        stop_button = QPushButton("Stop scoring")
        stop_button.clicked.connect(self._stop_online)
        actions.addWidget(run_button)
        actions.addWidget(stop_button)
        controls_layout.addLayout(actions)

        state_box = QGroupBox("Current Output")
        state_layout = QGridLayout(state_box)
        state_layout.setContentsMargins(8, 8, 8, 8)
        controls_layout.addWidget(state_box)
        self.confirmed_state = QLabel("-")
        self.emg_threshold_label = QLabel("threshold: -")
        self.delta_theta_threshold_label = QLabel("log10 D/T threshold: -")
        self._label_value(state_layout, 0, "Confirmed", self.confirmed_state)
        self._label_value(state_layout, 1, "EMG", self.emg_threshold_label)
        self._label_value(state_layout, 2, "Delta/Theta", self.delta_theta_threshold_label)

        self.status = QLabel("Select an Intan basepath.")
        self.status.setWordWrap(True)
        controls_layout.addWidget(self.status)
        controls_layout.addStretch(1)

        fig = Figure(figsize=(8, 6), dpi=100)
        gs = fig.add_gridspec(4, 3, height_ratios=[1.0, 1.0, 1.0, 0.95])
        self.axes = [
            fig.add_subplot(gs[0, :]),
            fig.add_subplot(gs[1, :]),
            fig.add_subplot(gs[2, :]),
        ]
        self.duration_axes = [fig.add_subplot(gs[3, idx]) for idx in range(3)]
        self.axes[0].set_ylabel("EMG")
        self.axes[1].set_ylabel("log10 D/T")
        self.axes[2].set_ylabel("Hypnogram")
        self.duration_axes[1].set_xlabel("Time (min)")
        fig.tight_layout()
        self.canvas = FigureCanvas(fig)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        plots_layout.addWidget(self.canvas)

    def _path_row(self, parent: QVBoxLayout, label: str, command) -> QLineEdit:
        frame = QFrame()
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        layout.addWidget(QLabel(label))
        row = QHBoxLayout()
        edit = QLineEdit()
        button = QPushButton("Browse")
        button.clicked.connect(command)
        row.addWidget(edit, 1)
        row.addWidget(button)
        layout.addLayout(row)
        parent.addWidget(frame)
        return edit

    def _double_spin(self, parent: QGridLayout, row: int, label: str, value: float, start: float, stop: float, inc: float) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(start, stop)
        spin.setSingleStep(inc)
        spin.setValue(value)
        spin.setDecimals(2)
        parent.addWidget(QLabel(label), row, 0)
        parent.addWidget(spin, row, 1)
        return spin

    def _int_spin(self, parent: QGridLayout, row: int, label: str, value: int, start: int, stop: int) -> QSpinBox:
        spin = QSpinBox()
        spin.setRange(start, stop)
        spin.setValue(value)
        parent.addWidget(QLabel(label), row, 0)
        parent.addWidget(spin, row, 1)
        return spin

    def _line_value(self, parent: QGridLayout, row: int, label: str, value: str) -> QLineEdit:
        edit = QLineEdit(value)
        parent.addWidget(QLabel(label), row, 0)
        parent.addWidget(edit, row, 1)
        return edit

    def _label_value(self, parent: QGridLayout, row: int, label: str, value: QLabel) -> None:
        parent.addWidget(QLabel(label), row, 0)
        parent.addWidget(value, row, 1)

    def _browse_basepath(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Choose Intan basepath")
        if path:
            self.basepath_edit.setText(path)

    def _browse_previous_path(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Choose previous datapath")
        if path:
            self.previous_path_edit.setText(path)

    def _start_scoring(self) -> None:
        if self.worker and self.worker.is_alive():
            QMessageBox.information(self, "Scoring", "Scoring is already running.")
            return
        basepath = self.basepath_edit.text().strip()
        if not basepath:
            QMessageBox.critical(self, "Missing basepath", "Choose a folder containing amplifier.dat and amplifier.xml.")
            return
        if not self.emg_threshold.text().strip() or not self.delta_theta_threshold.text().strip():
            QMessageBox.critical(self, "Missing thresholds", "Estimate parameters or enter EMG and log10 D/T thresholds before scoring.")
            return
        self.status.setText("Starting...")
        self.stop_event.clear()
        self.worker = threading.Thread(target=self._run_scoring_worker, daemon=True)
        self.worker.start()

    def _start_previous_parameter_estimation(self) -> None:
        if self.worker and self.worker.is_alive():
            QMessageBox.information(self, "Scoring", "A job is already running.")
            return
        target_path = self.previous_path_edit.text().strip()
        if not target_path:
            QMessageBox.critical(self, "Missing previous datapath", "Choose a previous session folder first.")
            return
        self.status.setText("Estimating parameters...")
        self.stop_event.clear()
        self.worker = threading.Thread(
            target=self._estimate_parameters_worker,
            args=(target_path,),
            daemon=True,
        )
        self.worker.start()

    def _estimate_parameters_worker(self, target_path: str) -> None:
        try:
            session = load_session(target_path)
            use_full_session_blocks = bool(self.previous_path_edit.text().strip()) and Path(target_path).resolve() == Path(self.previous_path_edit.text().strip()).resolve()
            if use_full_session_blocks:
                profile = estimate_channel_profile_from_blocks(session, epoch_sec=float(self.epoch_sec.value()))
            else:
                profile = estimate_channel_profile(
                    session,
                    calibration_minutes=float(self.estimation_minutes.value()),
                    epoch_sec=float(self.epoch_sec.value()),
                )
            out_profile = default_profile_path(session)
            profile.save(out_profile)
            self.estimated_profile = profile
            self._queue_profile_threshold_fill(profile)
            self.queue.put(("status", f"Estimated parameters and saved {out_profile}"))
        except Exception as exc:
            self.queue.put(("error", str(exc)))

    def _stop_online(self) -> None:
        self.stop_event.set()
        self.status.setText("Stopping online scoring...")

    def closeEvent(self, event) -> None:
        self.stop_event.set()
        event.accept()

    def _run_scoring_worker(self) -> None:
        try:
            session = load_session(self.basepath_edit.text().strip())
            emg_threshold_value = float(self.emg_threshold.text())
            delta_theta_threshold_value = float(self.delta_theta_threshold.text())
            profile = self.estimated_profile
            if profile is not None:
                self._validate_profile_for_session(profile, session)
                self.queue.put(("status", "Using parameters estimated from previous session."))
            else:
                estimation_sec = float(self.estimation_minutes.value()) * 60.0
                required_samples = int(round(estimation_sec * session.sample_rate))
                while session.total_samples < required_samples and not self.stop_event.is_set():
                    available_sec = session.total_samples / session.sample_rate
                    self.queue.put((
                        "status",
                        f"Waiting for estimation window: {available_sec:.0f}/{estimation_sec:.0f} sec",
                    ))
                    time.sleep(1.0)
                if self.stop_event.is_set():
                    self.queue.put(("status", "Scoring stopped before estimation."))
                    return
                self.queue.put(("status", "Estimating current-session channels from initial window..."))
                profile = estimate_channel_profile(
                    session,
                    calibration_minutes=float(self.estimation_minutes.value()),
                    epoch_sec=float(self.epoch_sec.value()),
                )
            out_profile = default_profile_path(session)
            profile.save(out_profile)
            score_start_sample = 0
            self.queue.put(("status", f"Saved scoring parameters: {out_profile}"))

            params = ScoringParams(
                epoch_sec=float(self.epoch_sec.value()),
                confirmation_count=int(self.confirmation_count.value()),
                emg_threshold_mode="manual",
                emg_threshold=emg_threshold_value,
                delta_theta_threshold_mode="manual",
                delta_theta_threshold=delta_theta_threshold_value,
                wake_to_rem_block=bool(self.wake_to_rem_block.isChecked()),
                online=True,
            )
            self._run_online_loop(session, profile, params, score_start_sample)
        except Exception as exc:
            self.queue.put(("error", str(exc)))

    def _run_online_loop(self, session, profile, params: ScoringParams, score_start_sample: int) -> None:
        epoch_samples = int(round(params.epoch_sec * session.sample_rate))
        next_epoch = int(np.ceil(score_start_sample / epoch_samples))
        accumulator = ScoringAccumulator(profile=profile, params=params)
        state_machine = ScoringState()
        threshold_state = OnlineThresholdState()
        last_save_time = 0.0
        last_plot_time = 0.0
        catch_up_epochs = max(1, int(round(120.0 / params.epoch_sec)))
        while not self.stop_event.is_set():
            available_epochs = session.total_samples // epoch_samples
            new_epochs = available_epochs - next_epoch
            if new_epochs > 0:
                read_epochs = catch_up_epochs if new_epochs > catch_up_epochs else 1
                read_epochs = min(read_epochs, new_epochs)
                start_sample = next_epoch * epoch_samples
                features = extract_feature_epochs(session, profile, params, start_sample, read_epochs)
                part = classify_feature_epochs(
                    *features,
                    profile=profile,
                    params=params,
                    state_machine=state_machine,
                    threshold_state=threshold_state,
                )
                accumulator.append(part)
                next_epoch += len(part.timestamps_sec)
                result = accumulator.to_result()
                now = time.time()
                backlog = (session.total_samples // epoch_samples) - next_epoch
                out = None
                if backlog == 0 or now - last_save_time >= 5.0:
                    out = save_states_mat(result, session.basepath)
                    last_save_time = now
                if backlog == 0 or now - last_plot_time >= 2.0:
                    self.queue.put(("result", result, out))
                    last_plot_time = now
                self.queue.put(("status", f"Scoring: {next_epoch} epochs processed"))
            else:
                self.queue.put(("status", f"Caught up: {next_epoch} epochs processed. Waiting for next epoch..."))
                time.sleep(1.0)
        if len(accumulator):
            result = accumulator.to_result()
            out = save_states_mat(result, session.basepath)
            self.queue.put(("result", result, out))
        self.queue.put(("status", "Online scoring stopped."))

    def _validate_profile_for_session(self, profile, session) -> None:
        if int(profile.n_channels) != int(session.n_channels):
            raise ValueError(
                "Previous-session parameters cannot be used because n_channels differs: "
                f"profile={profile.n_channels}, current={session.n_channels}"
            )
        used_channels = list(profile.emg_from_lfp_channels) + [
            int(profile.nrem_sw_channel),
            int(profile.rem_theta_channel),
        ]
        bad = [ch for ch in used_channels if int(ch) < 0 or int(ch) >= session.n_channels]
        if bad:
            raise ValueError(
                "Previous-session parameters contain channels outside the current recording: "
                + ", ".join(map(str, sorted(set(bad))))
            )

    def _poll_queue(self) -> None:
        try:
            while True:
                item = self.queue.get_nowait()
                if item[0] == "status":
                    self.status.setText(item[1])
                elif item[0] == "error":
                    self.status.setText("Error")
                    QMessageBox.critical(self, "Scoring failed", item[1])
                elif item[0] == "result":
                    _, result, out = item
                    self.result = result
                    self._render_result(result)
                    if out is not None:
                        self.status.setText(f"Saved {out}")
                elif item[0] == "fill_thresholds":
                    _, emg_threshold, delta_theta_threshold = item
                    if np.isfinite(emg_threshold):
                        self.emg_threshold.setText(f"{emg_threshold:.6g}")
                    if np.isfinite(delta_theta_threshold):
                        self.delta_theta_threshold.setText(f"{delta_theta_threshold:.6g}")
        except queue.Empty:
            pass

    def _render_result(self, result: ScoringResult) -> None:
        t = result.timestamps_sec / 60.0
        plot_idx = self._plot_indices(len(t), max_points=5000)
        tp = t[plot_idx]
        for ax in self.axes:
            ax.clear()
        for ax in self.duration_axes:
            ax.clear()
        self.axes[0].plot(tp, result.estimated_emg[plot_idx], color="#2f6f73", lw=1)
        if result.emg_threshold_history.size:
            self.axes[0].plot(tp, result.emg_threshold_history[plot_idx], color="#a23b3b", lw=0.9, ls="--")
            display_threshold = result.emg_threshold_history[-1]
            self.emg_threshold_label.setText(f"threshold: {display_threshold:.3g}")
        self.axes[0].set_ylabel("EMG")
        self.axes[1].plot(tp, result.log_delta_theta_ratio[plot_idx], color="#855c1b", lw=1)
        if result.delta_theta_threshold_history.size:
            self.axes[1].plot(tp, result.delta_theta_threshold_history[plot_idx], color="#a23b3b", lw=0.9, ls="--")
            dt_threshold = result.delta_theta_threshold_history[-1]
            self.delta_theta_threshold_label.setText(f"log10 D/T threshold: {dt_threshold:.3g}")
        self.axes[1].set_ylabel("log10 D/T")
        state_y = np.array([STATE_CODES[s] for s in result.confirmed_state])
        self.axes[2].step(tp, state_y[plot_idx], where="post", color="#111111", lw=1.5)
        self.axes[2].set_yticks([1, 3, 5], ["Wake", "NREM", "REM"])
        self.axes[2].set_xlabel("Time (min)")
        self._render_duration_panels(result, t)
        self.canvas.figure.tight_layout()
        self.canvas.draw_idle()
        self.confirmed_state.setText(result.confirmed_state[-1])

    def _render_duration_panels(self, result: ScoringResult, t: np.ndarray) -> None:
        colors = {"Wake": "#4c72b0", "NREM": "#c44e52", "REM": "#d9a51d"}
        labels = ["Wake", "NREM", "REM"]
        states = np.array(result.confirmed_state)
        plot_idx = self._plot_indices(len(t), max_points=2000)
        for ax, label in zip(self.duration_axes, labels):
            cumulative = np.cumsum(states == label) * result.params.epoch_sec / 60.0
            ax.plot(t[plot_idx], cumulative[plot_idx], color=colors[label], lw=1.5)
            ax.set_title(f"{label}: {cumulative[-1]:.1f} min")
            ax.set_ylabel("min")
            ax.grid(True, alpha=0.25)

    def _plot_indices(self, n_points: int, max_points: int) -> np.ndarray:
        if n_points <= max_points:
            return np.arange(n_points)
        step = int(np.ceil(n_points / max_points))
        idx = np.arange(0, n_points, step)
        if idx[-1] != n_points - 1:
            idx = np.r_[idx, n_points - 1]
        return idx

    def _save_current(self) -> None:
        if self.result is None:
            QMessageBox.information(self, "No result", "Run scoring first.")
            return
        out = save_states_mat(self.result, self.basepath_edit.text().strip())
        self.status.setText(f"Saved {out}")

    def _fill_thresholds_from_profile(self, profile_path: str) -> None:
        try:
            profile = load_profile(profile_path)
        except Exception as exc:
            self.status.setText(f"Could not read profile thresholds: {exc}")
            return
        emg_threshold, delta_theta_threshold = self._profile_suggested_thresholds(profile)
        if np.isfinite(emg_threshold):
            self.emg_threshold.setText(f"{emg_threshold:.6g}")
        if np.isfinite(delta_theta_threshold):
            self.delta_theta_threshold.setText(f"{delta_theta_threshold:.6g}")
        self.status.setText("Loaded suggested thresholds from profile.")

    def _queue_profile_threshold_fill(self, profile) -> None:
        emg_threshold, delta_theta_threshold = self._profile_suggested_thresholds(profile)
        self.queue.put(("fill_thresholds", emg_threshold, delta_theta_threshold))

    def _profile_suggested_thresholds(self, profile) -> tuple[float, float]:
        q = profile.quality_metrics or {}
        return (
            self._suggested_emg_threshold(q),
            self._finite_float(q.get("suggested_log_delta_theta_threshold")),
        )

    def _suggested_emg_threshold(self, quality_metrics: dict) -> float:
        raw = self._finite_float(quality_metrics.get("suggested_emg_raw_threshold"))
        if np.isfinite(raw):
            return raw
        z_threshold = self._finite_float(quality_metrics.get("suggested_emg_z_threshold"))
        center = self._finite_float(quality_metrics.get("emg_z_center", quality_metrics.get("emg_epoch_median")))
        scale = self._finite_float(quality_metrics.get("emg_z_scale"))
        if np.isfinite(z_threshold) and np.isfinite(center) and np.isfinite(scale):
            return center + z_threshold * scale
        return float("nan")

    def _finite_float(self, value) -> float:
        try:
            out = float(value)
        except (TypeError, ValueError):
            return float("nan")
        return out if np.isfinite(out) else float("nan")


def main() -> None:
    app = QApplication.instance() or QApplication(sys.argv)
    icon_path = _resource_path("logo/logo.ico")
    if icon_path.exists():
        app.setWindowIcon(QIcon(str(icon_path)))
    window = SleepScoreApp()
    window.show()
    app.exec()


def _resource_path(relative_path: str) -> Path:
    base = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parents[2]))
    return base / relative_path
