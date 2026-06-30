from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET

import numpy as np


@dataclass(frozen=True)
class ChannelGroup:
    channels: list[int]
    skip: list[bool]

    @property
    def usable(self) -> list[int]:
        return [ch for ch, skipped in zip(self.channels, self.skip) if not skipped]


@dataclass(frozen=True)
class IntanSession:
    basepath: Path
    dat_path: Path
    xml_path: Path
    basename: str
    sample_rate: float
    n_channels: int
    file_n_channels: int
    lfp_sample_rate: float | None
    groups: list[ChannelGroup]
    recording_format: str = "intan"

    @property
    def total_samples(self) -> int:
        return self.dat_path.stat().st_size // (2 * self.file_n_channels)

    @property
    def duration_sec(self) -> float:
        return self.total_samples / self.sample_rate

    @property
    def bad_channels(self) -> list[int]:
        out: list[int] = []
        for group in self.groups:
            out.extend(
                ch
                for ch, skipped in zip(group.channels, group.skip)
                if skipped and 0 <= ch < self.n_channels
            )
        return sorted(set(out))

    @property
    def usable_channels(self) -> list[int]:
        if self.groups:
            usable: list[int] = []
            for group in self.groups:
                usable.extend(ch for ch in group.usable if 0 <= ch < self.n_channels)
            return sorted(set(usable))
        return list(range(self.n_channels))

    @property
    def excluded_file_channels(self) -> list[int]:
        return list(range(self.n_channels, self.file_n_channels))


def load_session(basepath: str | Path) -> IntanSession:
    selected = Path(basepath).expanduser().resolve()
    dat_path = _resolve_recording_dat_path(selected)
    base = selected if selected.is_dir() else dat_path.parent
    xml_path = _resolve_xml_path(selected, dat_path)
    if xml_path is None:
        raise FileNotFoundError(
            f"Missing amplifier.xml, <basename>.xml, or adjacent continuous.xml for {selected}"
        )

    tree = ET.parse(xml_path)
    root = tree.getroot()

    acq = root.find(".//acquisitionSystem")
    n_channels = _xml_number(acq, "nChannels", int, default=128)
    sample_rate = _xml_number(acq, "samplingRate", float, default=20000.0)
    lfp_node = root.find(".//fieldPotentials")
    lfp_sample_rate = _xml_number(lfp_node, "lfpSamplingRate", float, default=None)
    file_n_channels = _infer_file_n_channels(dat_path, n_channels)

    groups: list[ChannelGroup] = []
    for group_node in root.findall(".//anatomicalDescription/channelGroups/group"):
        channels: list[int] = []
        skip: list[bool] = []
        for ch_node in group_node.findall("channel"):
            if ch_node.text is None:
                continue
            channel = int(ch_node.text.strip())
            channels.append(channel)
            skip.append(ch_node.attrib.get("skip", "0") == "1" or channel < 0 or channel >= n_channels)
        if channels:
            groups.append(ChannelGroup(channels=channels, skip=skip))

    return IntanSession(
        basepath=base,
        dat_path=dat_path,
        xml_path=xml_path,
        basename=base.name,
        sample_rate=sample_rate,
        n_channels=n_channels,
        file_n_channels=file_n_channels,
        lfp_sample_rate=lfp_sample_rate,
        groups=groups,
        recording_format="open_ephys" if dat_path.name.lower() in {"continuous.dat", "continous.dat"} else "intan",
    )


def _xml_number(node: ET.Element | None, tag: str, cast, default):
    if node is None:
        return default
    child = node.find(tag)
    if child is None or child.text is None or child.text.strip() == "":
        return default
    return cast(child.text.strip())


def _resolve_recording_dat_path(selected: Path) -> Path:
    if selected.is_file():
        if selected.suffix.lower() != ".dat":
            raise FileNotFoundError(f"Expected a .dat file, got: {selected.name}")
        return selected
    if not selected.exists():
        raise FileNotFoundError(f"Path does not exist: {selected}")
    if not selected.is_dir():
        raise FileNotFoundError(f"Unsupported path: {selected}")

    open_ephys_matches = _find_open_ephys_continuous_dat_paths(selected)
    if open_ephys_matches:
        return open_ephys_matches[-1]

    direct_dat = selected / "amplifier.dat"
    if direct_dat.exists():
        return direct_dat
    basename_dat = selected / f"{selected.name}.dat"
    if basename_dat.exists():
        return basename_dat

    children = sorted(selected.iterdir(), key=lambda path: path.name.lower())
    matches = [
        candidate
        for child in children
        if child.is_dir()
        for candidate in (child / "amplifier.dat", child / f"{child.name}.dat")
        if candidate.is_file()
    ]
    if matches:
        return sorted(matches, key=_recording_session_sort_key)[-1]
    raise FileNotFoundError(f"No amplifier.dat, basename.dat, or continuous.dat found under: {selected}")


def _find_open_ephys_continuous_dat_paths(selected: Path) -> list[Path]:
    matches: list[Path] = []
    stack: list[tuple[Path, int]] = [(selected, 0)]
    skip_names = {
        ".git",
        "__pycache__",
        "analysis",
        "kilosort",
        "kilosort2",
        "kilosort3",
        "phy",
        "original_dat",
    }
    while stack:
        folder, depth = stack.pop()
        for filename in ("continuous.dat", "continous.dat"):
            candidate = folder / filename
            if candidate.is_file():
                matches.append(candidate)
                break
        else:
            if depth >= 8:
                continue
            try:
                children = sorted(folder.iterdir(), key=lambda path: path.name.lower())
            except OSError:
                continue
            for child in reversed(children):
                if child.is_dir() and child.name.lower() not in skip_names:
                    stack.append((child, depth + 1))
            continue
        continue
    return _unique_paths(_filter_open_ephys_primary_streams(matches))


def _filter_open_ephys_primary_streams(paths: list[Path]) -> list[Path]:
    if len(paths) <= 1:
        return sorted(paths, key=_recording_session_sort_key)
    grouped: dict[Path, list[Path]] = {}
    for path in paths:
        grouped.setdefault(_open_ephys_recording_key(path), []).append(path)
    return sorted((_best_open_ephys_stream(candidates) for candidates in grouped.values()), key=_recording_session_sort_key)


def _open_ephys_recording_key(path: Path) -> Path:
    for parent in path.parents:
        if parent.name.lower() == "continuous":
            return parent.parent
    return path.parent


def _best_open_ephys_stream(paths: list[Path]) -> Path:
    return max(paths, key=_open_ephys_stream_score)


def _open_ephys_stream_score(path: Path) -> tuple[int, int, str]:
    stream_name = path.parent.name.lower()
    auxiliary_tokens = ("adc", "analog", "aux", "digital", "event", "ttl")
    primary = 0 if any(token in stream_name for token in auxiliary_tokens) else 1
    try:
        file_size = int(path.stat().st_size)
    except OSError:
        file_size = 0
    return primary, file_size, str(path).lower()


def _unique_paths(paths: list[Path]) -> list[Path]:
    unique: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        key = str(path.resolve()) if path.exists() else str(path)
        if key not in seen:
            seen.add(key)
            unique.append(path)
    return unique


def _recording_session_sort_key(path: Path) -> tuple[int, str]:
    timestamp = _extract_recording_folder_timestamp(path)
    return (timestamp or -1, str(path).lower())


def _extract_recording_folder_timestamp(path: Path) -> int | None:
    pattern = re.compile(r"(\d{4})-(\d{2})-(\d{2})[_-](\d{2})-(\d{2})-(\d{2})")
    for part in path.parts:
        match = pattern.search(part)
        if match:
            return int("".join(match.groups()))
    return None


def _resolve_xml_path(selected: Path, dat_path: Path) -> Path | None:
    candidates: list[Path] = []
    if selected.is_dir():
        candidates.append(selected / f"{selected.name}.xml")
        candidates.append(selected / "amplifier.xml")
    else:
        candidates.append(selected.with_suffix(".xml"))
    candidates.append(dat_path.with_suffix(".xml"))
    if dat_path.name.lower() in {"continuous.dat", "continous.dat"}:
        seen: set[str] = {str(path) for path in candidates}
        for parent in dat_path.parents:
            for candidate in (parent / f"{parent.name}.xml", parent / "amplifier.xml"):
                key = str(candidate)
                if key not in seen:
                    seen.add(key)
                    candidates.append(candidate)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _infer_file_n_channels(dat_path: Path, xml_n_channels: int) -> int:
    from_oebin = _infer_file_n_channels_from_oebin(dat_path)
    if from_oebin is not None and from_oebin >= xml_n_channels:
        return from_oebin

    from_timestamps = _infer_file_n_channels_from_frame_count(dat_path)
    if from_timestamps is not None and from_timestamps >= xml_n_channels:
        return from_timestamps

    file_size = dat_path.stat().st_size
    for candidate in range(xml_n_channels, xml_n_channels + 65):
        if file_size % (2 * candidate) == 0:
            return candidate
    return xml_n_channels


def _infer_file_n_channels_from_frame_count(dat_path: Path) -> int | None:
    total_values = dat_path.stat().st_size // np.dtype(np.int16).itemsize
    for filename in ("timestamps.npy", "sample_numbers.npy"):
        npy_path = dat_path.parent / filename
        if not npy_path.exists():
            continue
        try:
            frames = int(np.load(npy_path, mmap_mode="r").shape[0])
        except Exception:
            continue
        if frames > 0 and total_values % frames == 0:
            return total_values // frames
    return None


def _infer_file_n_channels_from_oebin(dat_path: Path) -> int | None:
    for parent in dat_path.parents:
        oebin_path = parent / "structure.oebin"
        if not oebin_path.exists():
            continue
        try:
            data = json.loads(oebin_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        stream_count = _match_oebin_stream_channel_count(data, dat_path.parent.name)
        if stream_count is not None:
            return stream_count
    return None


def _match_oebin_stream_channel_count(data, stream_folder_name: str) -> int | None:
    entries = [entry for entry in _walk_json_dicts(data) if _oebin_channel_count(entry) is not None]
    if not entries:
        return None
    for entry in entries:
        folder = str(entry.get("folder_name", entry.get("folder", entry.get("name", ""))))
        if folder == stream_folder_name:
            return _oebin_channel_count(entry)
    if len(entries) == 1:
        return _oebin_channel_count(entries[0])
    primary = [entry for entry in entries if _oebin_stream_is_primary(entry)]
    if len(primary) == 1:
        return _oebin_channel_count(primary[0])
    return None


def _walk_json_dicts(value) -> list[dict]:
    out: list[dict] = []
    if isinstance(value, dict):
        out.append(value)
        for child in value.values():
            out.extend(_walk_json_dicts(child))
    elif isinstance(value, list):
        for child in value:
            out.extend(_walk_json_dicts(child))
    return out


def _oebin_channel_count(entry: dict) -> int | None:
    for key in ("num_channels", "n_channels", "channel_count"):
        value = entry.get(key)
        if value is not None:
            try:
                return int(value)
            except (TypeError, ValueError):
                pass
    channels = entry.get("channels")
    if isinstance(channels, list) and channels:
        return len(channels)
    return None


def _oebin_stream_is_primary(entry: dict) -> bool:
    text = " ".join(str(entry.get(key, "")) for key in ("folder_name", "name", "source_processor_name")).lower()
    auxiliary_tokens = ("adc", "analog", "aux", "digital", "event", "ttl")
    return not any(token in text for token in auxiliary_tokens)


def read_channels(
    session: IntanSession,
    channels: list[int],
    start_sample: int,
    n_samples: int,
) -> np.ndarray:
    """Read selected int16 amplifier channels as samples x channels float64."""
    if not channels:
        raise ValueError("channels is empty")
    start_sample = max(0, int(start_sample))
    n_samples = max(0, int(n_samples))
    available = max(0, session.total_samples - start_sample)
    n_samples = min(n_samples, available)
    if n_samples == 0:
        return np.zeros((0, len(channels)), dtype=np.float64)
    invalid = [ch for ch in channels if ch < 0 or ch >= session.n_channels]
    if invalid:
        raise ValueError(
            "Requested channels outside neural channel range 0.."
            f"{session.n_channels - 1}: {sorted(set(invalid))}"
        )

    raw = np.memmap(session.dat_path, dtype=np.int16, mode="r")
    usable_values = (raw.size // session.file_n_channels) * session.file_n_channels
    shaped = raw[:usable_values].reshape((-1, session.file_n_channels))
    return np.asarray(shaped[start_sample : start_sample + n_samples, channels], dtype=np.float64)
