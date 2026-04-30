from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
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
    lfp_sample_rate: float | None
    groups: list[ChannelGroup]

    @property
    def total_samples(self) -> int:
        return self.dat_path.stat().st_size // (2 * self.n_channels)

    @property
    def duration_sec(self) -> float:
        return self.total_samples / self.sample_rate

    @property
    def bad_channels(self) -> list[int]:
        out: list[int] = []
        for group in self.groups:
            out.extend(ch for ch, skipped in zip(group.channels, group.skip) if skipped)
        return sorted(set(out))

    @property
    def usable_channels(self) -> list[int]:
        if self.groups:
            usable: list[int] = []
            for group in self.groups:
                usable.extend(group.usable)
            return sorted(set(usable))
        return list(range(self.n_channels))


def load_session(basepath: str | Path) -> IntanSession:
    base = Path(basepath).expanduser().resolve()
    dat_path = base / "amplifier.dat"
    xml_path = base / "amplifier.xml"
    if not dat_path.exists():
        raise FileNotFoundError(f"Missing amplifier.dat in {base}")
    if not xml_path.exists():
        raise FileNotFoundError(f"Missing amplifier.xml in {base}")

    tree = ET.parse(xml_path)
    root = tree.getroot()

    acq = root.find(".//acquisitionSystem")
    n_channels = _xml_number(acq, "nChannels", int, default=128)
    sample_rate = _xml_number(acq, "samplingRate", float, default=20000.0)
    lfp_node = root.find(".//fieldPotentials")
    lfp_sample_rate = _xml_number(lfp_node, "lfpSamplingRate", float, default=None)

    groups: list[ChannelGroup] = []
    for group_node in root.findall(".//anatomicalDescription/channelGroups/group"):
        channels: list[int] = []
        skip: list[bool] = []
        for ch_node in group_node.findall("channel"):
            if ch_node.text is None:
                continue
            channels.append(int(ch_node.text.strip()))
            skip.append(ch_node.attrib.get("skip", "0") == "1")
        if channels:
            groups.append(ChannelGroup(channels=channels, skip=skip))

    return IntanSession(
        basepath=base,
        dat_path=dat_path,
        xml_path=xml_path,
        basename=base.name,
        sample_rate=sample_rate,
        n_channels=n_channels,
        lfp_sample_rate=lfp_sample_rate,
        groups=groups,
    )


def _xml_number(node: ET.Element | None, tag: str, cast, default):
    if node is None:
        return default
    child = node.find(tag)
    if child is None or child.text is None or child.text.strip() == "":
        return default
    return cast(child.text.strip())


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

    raw = np.memmap(session.dat_path, dtype=np.int16, mode="r")
    shaped = raw.reshape((-1, session.n_channels))
    return np.asarray(shaped[start_sample : start_sample + n_samples, channels], dtype=np.float64)

