"""Offline screening of saved speech WAVs; does not replace human listening.

Run from project root:
    /media/scai/scai_data/fhr/anaconda3/envs/cosyvoice/bin/python tests/acceptance_prosody_audit.py

The earlier loose autocorrelation detector includes weakly periodic consonant
frames and can report octave errors. This audit uses WORLD's DIO+StoneMask,
excludes low-energy/unvoiced frames, and checks sustained pitch transitions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import wave

import numpy as np
import pyworld
from scipy.signal import medfilt


ROOT = Path(__file__).resolve().parents[1]


def pcm_from_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as source:
        if source.getnchannels() != 1 or source.getsampwidth() != 2:
            raise ValueError(f"Expected mono PCM16 WAV: {path}")
        rate = source.getframerate()
        samples = np.frombuffer(source.readframes(source.getnframes()), dtype="<i2").astype(np.float64)
    return samples, rate


def stable_pitch_screen(samples: np.ndarray, rate: int) -> dict:
    audio = samples / 32768.0
    initial, times = pyworld.dio(audio, rate, f0_floor=80, f0_ceil=400, frame_period=10)
    track = pyworld.stonemask(audio, initial, times, rate)
    voiced = track.copy()
    gate = round(rate * 0.03)
    for index, second in enumerate(times):
        start = round(second * rate)
        frame = samples[start:start + gate]
        if len(frame) < gate or np.sqrt(np.mean(frame * frame)) < 300:
            voiced[index] = 0
    # DIO may mark the exact transition frame unvoiced even for a persistent
    # synthetic step. Bridge at most two frames, so that such a step remains
    # detectable without joining speech across a real pause.
    position = 0
    while position < len(voiced):
        if voiced[position] > 0:
            position += 1
            continue
        end = position
        while end < len(voiced) and voiced[end] <= 0:
            end += 1
        if 0 < position and end < len(voiced) and end - position <= 2:
            logarithms = np.linspace(np.log2(voiced[position - 1]), np.log2(voiced[end]), end - position + 2)
            voiced[position:end] = np.exp2(logarithms[1:-1])
        position = end
    jumps: list[tuple[float, float]] = []
    index = 0
    while index < len(voiced):
        if voiced[index] <= 0:
            index += 1
            continue
        end = index
        while end < len(voiced) and voiced[end] > 0:
            end += 1
        # The median window removes isolated pitch-picking glitches while
        # preserving a real step that lasts more than five 10-ms frames.
        if end - index >= 7:
            smooth = medfilt(voiced[index:end], kernel_size=5)
            jumps.extend(
                (float(times[index + pos]),
                 float(abs(12 * np.log2(smooth[pos + 1] / smooth[pos]))))
                for pos in range(2, len(smooth) - 3)
            )
        index = end
    values = [value for _, value in jumps]
    return {
        "energy_gated_voiced_frames": int(np.count_nonzero(voiced > 0)),
        "stable_pitch_transition_count": len(jumps),
        "stable_pitch_jump_p95_semitones": round(float(np.percentile(values, 95)), 2) if values else None,
        "stable_pitch_jump_max_semitones": round(max(values), 2) if values else None,
        "stable_pitch_jumps_over_4_semitones": sum(value > 4 for value in values),
        "stable_pitch_jump_events_over_4": [
            {"time_s": round(time, 2), "jump_semitones": round(value, 2)}
            for time, value in jumps if value > 4
        ],
    }


def quiet_intervals(samples: np.ndarray, rate: int, *, minimum_ms: int = 80) -> list[dict]:
    hop = max(1, round(rate * 0.01))
    rms = np.asarray([
        np.sqrt(np.mean(samples[start:start + hop] ** 2))
        for start in range(0, len(samples) - hop, hop)
    ])
    is_quiet = rms < 300
    boundaries = np.diff(np.r_[False, is_quiet, False].astype(int))
    starts, ends = np.flatnonzero(boundaries == 1), np.flatnonzero(boundaries == -1)
    return [
        {"start_s": round(start * hop / rate, 2), "duration_ms": round((end - start) * hop * 1000 / rate)}
        for start, end in zip(starts, ends)
        if (end - start) * hop * 1000 / rate >= minimum_ms
    ]


def synthetic_step_check(rate: int = 22050) -> float:
    time = np.arange(2 * rate) / rate
    frequency = np.where(time < 1, 180.0, 300.0)
    samples = 10000 * np.sin(2 * np.pi * np.cumsum(frequency) / rate)
    detected = stable_pitch_screen(samples, rate)["stable_pitch_jump_max_semitones"]
    if detected is None or detected < 4:
        raise AssertionError("Pitch screen failed to detect a persistent synthetic pitch step")
    return detected


def main(directory: Path, output: Path) -> None:
    directory = directory.expanduser().resolve()
    output = output.expanduser().resolve()
    paths = sorted(directory.glob("sample-*.wav"))
    if len(paths) != 20:
        raise ValueError(f"Expected 20 WAV samples, found {len(paths)} in {directory}")
    rows = []
    for path in paths:
        samples, rate = pcm_from_wav(path)
        rows.append({"file": str(path.relative_to(ROOT)),
                     **stable_pitch_screen(samples, rate),
                     "quiet_intervals_at_least_80ms": quiet_intervals(samples, rate)})
    output.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "method": "WORLD DIO+StoneMask, 10ms hop, 30ms RMS>=300 and voiced f0; 5-frame median within continuous voiced runs",
        "synthetic_180_to_300_hz_step_detected_semitones": synthetic_step_check(),
        "max_stable_pitch_jump_semitones": max(row["stable_pitch_jump_max_semitones"] or 0 for row in rows),
        "samples_with_stable_jump_over_4_semitones": sum(row["stable_pitch_jumps_over_4_semitones"] > 0 for row in rows),
        "scope_warning": "Acoustic screening cannot certify natural prosody or every punctuation pause; listen to the WAV files.",
        "samples": rows,
    }
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "samples"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio-dir", type=Path, default=ROOT / "exports/acceptance/final/audio")
    parser.add_argument("--output", type=Path, default=ROOT / "exports/acceptance/final/prosody-results.json")
    args = parser.parse_args()
    main(args.audio_dir, args.output)
