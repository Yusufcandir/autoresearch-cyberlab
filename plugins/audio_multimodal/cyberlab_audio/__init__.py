"""Audio capability plugin (AT-16): WAV validation, speaker-aware splits, acoustic features.

Uses only the standard library `wave` module and numpy: no codecs are downloaded and no
external tools are run. Transcription is deliberately not offered (no licensed ASR model is
installed), so asking for it raises UNAVAILABLE_FEATURE_VIEW instead of inventing text.
"""

from __future__ import annotations

import wave
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from cyberlab_schemas.plugins import (
    SCHEMA_INVALID,
    Capability,
    PluginError,
    ValidationReport,
    group_split,
    require_view,
)

CAPABILITY = Capability(
    name="audio",
    version="1",
    supported_modalities=("audio",),
    feature_views=("acoustic_summary", "speaker_split"),
    notes="No 'transcript' view: no licensed speech-recognition model is installed.",
)


class AudioClip(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(min_length=1)
    path: str
    label: str = Field(min_length=1)
    speaker: str = Field(min_length=1)


class AudioDataset(BaseModel):
    model_config = ConfigDict(extra="forbid")
    modality: Literal["audio"] = "audio"
    root: str
    sample_rate: int = Field(gt=0, description="every clip must use this rate")
    channels: int = Field(default=1, ge=1, le=2)
    clips: list[AudioClip] = Field(min_length=2)


def _parse(manifest: Mapping[str, Any]) -> AudioDataset:
    try:
        return AudioDataset.model_validate(manifest)
    except ValidationError as exc:
        raise PluginError(
            SCHEMA_INVALID,
            "audio dataset does not match the schema",
            {"errors": exc.errors(include_url=False)},
        ) from exc


def _resolve(ds: AudioDataset, clip: AudioClip) -> Path:
    root = Path(ds.root).resolve()
    path = (root / clip.path).resolve()
    if root not in path.parents:
        raise PluginError(SCHEMA_INVALID, f"clip {clip.id} points outside the dataset root")
    return path


def _read(path: Path) -> tuple[np.ndarray, int, int]:
    with wave.open(str(path), "rb") as w:
        rate, channels, width = w.getframerate(), w.getnchannels(), w.getsampwidth()
        if width != 2:
            raise PluginError(SCHEMA_INVALID, f"{path.name}: only 16-bit PCM is supported")
        data = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32)
    if channels > 1:
        data = data.reshape(-1, channels).mean(axis=1)
    return data / 32768.0, rate, channels


class AudioAdapter:
    def capabilities(self) -> Capability:
        return CAPABILITY

    def validate_dataset(self, dataset_manifest: Mapping[str, Any]) -> dict[str, Any]:
        ds = _parse(dataset_manifest)
        problems: list[str] = []
        durations = []
        for clip in ds.clips:
            path = _resolve(ds, clip)
            if not path.exists():
                problems.append(f"{clip.id}: file missing")
                continue
            try:
                samples, rate, channels = _read(path)
            except (wave.Error, EOFError) as exc:
                problems.append(f"{clip.id}: not a readable WAV file ({exc})")
                continue
            except PluginError as exc:
                problems.append(exc.message)
                continue
            if rate != ds.sample_rate:
                problems.append(f"{clip.id}: sample rate {rate} Hz, expected {ds.sample_rate} Hz")
            if channels != ds.channels:
                problems.append(f"{clip.id}: {channels} channels, expected {ds.channels}")
            durations.append(len(samples) / rate)
        speakers = sorted({c.speaker for c in ds.clips})
        return ValidationReport(
            ok=not problems,
            modality="audio",
            n_items=len(ds.clips),
            groups=speakers,
            problems=problems,
            stats={
                "labels": dict(Counter(c.label for c in ds.clips)),
                "total_seconds": round(float(sum(durations)), 3),
            },
        ).as_dict()

    def feature_view(self, name: str, dataset_manifest: Mapping[str, Any]) -> dict[str, Any]:
        require_view(CAPABILITY, name)
        ds = _parse(dataset_manifest)
        if name == "speaker_split":
            return {"view": name, **group_split([c.speaker for c in ds.clips])}
        rows = {}
        for clip in ds.clips:
            x, rate, _ = _read(_resolve(ds, clip))
            spectrum = np.abs(np.fft.rfft(x))
            freqs = np.fft.rfftfreq(len(x), 1 / rate)
            rows[clip.id] = {
                "seconds": round(len(x) / rate, 4),
                "rms": round(float(np.sqrt(np.mean(x**2))), 6),
                "zero_crossing_rate": round(float(np.mean(np.abs(np.diff(np.sign(x))) > 0)), 6),
                "spectral_centroid_hz": round(
                    float((freqs * spectrum).sum() / max(spectrum.sum(), 1e-12)), 2
                ),
            }
        return {"view": name, "rows": rows}


def write_fixture(root: Path, sample_rate: int = 8000) -> dict[str, Any]:
    """Synthetic tones per speaker (test use only). Returns the dataset manifest."""
    root.mkdir(parents=True, exist_ok=True)
    clips = []
    for i, (speaker, label, hz) in enumerate(
        [
            ("spk1", "alarm", 880),
            ("spk1", "normal", 220),
            ("spk2", "alarm", 990),
            ("spk2", "normal", 260),
            ("spk3", "alarm", 940),
            ("spk3", "normal", 240),
        ]
    ):
        t = np.arange(int(0.25 * sample_rate)) / sample_rate
        tone = (0.4 * np.sin(2 * np.pi * hz * t) * 32767).astype("<i2")
        name = f"clip{i}.wav"
        with wave.open(str(root / name), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(sample_rate)
            w.writeframes(tone.tobytes())
        clips.append({"id": f"c{i}", "path": name, "label": label, "speaker": speaker})
    return {"modality": "audio", "root": str(root), "sample_rate": sample_rate, "clips": clips}


__all__ = ["CAPABILITY", "AudioAdapter", "write_fixture"]
