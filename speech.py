"""Offline SenseVoice ASR and local CosyVoice streaming TTS interfaces."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
import json
import os
from pathlib import Path
import re
import struct
import subprocess
from threading import Event, Lock
from time import perf_counter, sleep
from typing import Any, Iterator, Mapping
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen
import wave

import numpy as np
import sherpa_onnx

from project_config import PROJECT_ROOT, cosyvoice_python, project_path


DEFAULT_ASR_MODEL_PATH = project_path(
    "ASR_MODEL_PATH", "models/speech/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17"
)
DEFAULT_TTS_MODEL_PATH = project_path("TTS_MODEL_PATH", "models/speech/CosyVoice-300M-SFT")
DEFAULT_COSYVOICE_PYTHON = cosyvoice_python()
DEFAULT_COSYVOICE_URL = "http://127.0.0.1:8766"
DEFAULT_SAMPLE_RATE = 16_000
DEFAULT_TTS_SPEAKER = "中文女"

_SOURCE_FOOTERS = ("\n\n检索来源：", "\n\n检索判定：")
_FALLBACK_PREFIXES = (
    "根据当前保险资料无法确认，以下根据模型自身知识补充：",
    "根据当前讲义无法确认，以下根据模型自身知识补充：",
)
_DEFAULT_PRONUNCIATIONS = {
    "Qwen": "通义千问",
    "RAG": "检索增强生成",
    "ASR": "语音识别",
    "TTS": "语音合成",
}


@dataclass(frozen=True)
class ASRResult:
    text: str
    language: str
    emotion: str
    event: str
    audio_duration_ms: float
    decode_ms: float
    inference_ms: float
    total_ms: float
    timestamps_ms: tuple[float, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "language": self.language,
            "emotion": self.emotion,
            "event": self.event,
            "audio_duration_ms": self.audio_duration_ms,
            "decode_ms": self.decode_ms,
            "inference_ms": self.inference_ms,
            "total_ms": self.total_ms,
            "timestamps_ms": list(self.timestamps_ms),
        }


@dataclass(frozen=True)
class AudioChunk:
    index: int
    text: str
    pcm16: bytes
    sample_rate: int
    duration_ms: float
    synthesis_ms: float
    rtf: float

    def wav_bytes(self) -> bytes:
        output = BytesIO()
        with wave.open(output, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(self.sample_rate)
            wav.writeframes(self.pcm16)
        return output.getvalue()

    def save_wav(self, path: str | Path) -> Path:
        target = Path(path).expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(self.wav_bytes())
        return target


def _strip_tag(value: str) -> str:
    return value.removeprefix("<|").removesuffix("|>")


def _apply_corrections(text: str, corrections: Mapping[str, str] | None) -> str:
    if not corrections:
        return text
    for source, replacement in sorted(corrections.items(), key=lambda item: -len(item[0])):
        if not source:
            continue
        text = re.sub(re.escape(source), lambda _: replacement, text, flags=re.IGNORECASE)
    return text


_CHINESE_DIGITS = "零一二三四五六七八九"
_CHINESE_SMALL_UNITS = ("", "十", "百", "千")
_CHINESE_GROUP_UNITS = ("", "万", "亿", "兆")


def _integer_to_chinese(value: int) -> str:
    """Read an Arabic integer as a Chinese cardinal number."""
    if value == 0:
        return "零"
    if value < 0 or value >= 10**16:
        return "".join(_CHINESE_DIGITS[int(digit)] for digit in str(value))

    def read_group(group: int) -> str:
        output: list[str] = []
        zero_pending = False
        for position in range(3, -1, -1):
            unit_value = 10**position
            digit = group // unit_value % 10
            if digit == 0:
                if output and group % unit_value:
                    zero_pending = True
                continue
            if zero_pending:
                output.append("零")
                zero_pending = False
            if not (digit == 1 and position == 1 and not output):
                output.append(_CHINESE_DIGITS[digit])
            output.append(_CHINESE_SMALL_UNITS[position])
        return "".join(output)

    groups: list[int] = []
    while value:
        groups.append(value % 10_000)
        value //= 10_000

    output: list[str] = []
    zero_pending = False
    for position in range(len(groups) - 1, -1, -1):
        group = groups[position]
        if group == 0:
            if output:
                zero_pending = True
            continue
        if output and (zero_pending or group < 1_000):
            output.append("零")
        output.append(read_group(group) + _CHINESE_GROUP_UNITS[position])
        zero_pending = False
    return "".join(output)


def _number_to_chinese(value: str, *, digitwise: bool = False) -> str:
    value = value.replace(",", "")
    integer, dot, fraction = value.partition(".")
    if digitwise:
        whole = "".join(_CHINESE_DIGITS[int(digit)] for digit in integer)
    else:
        whole = _integer_to_chinese(int(integer))
    if not dot:
        return whole
    decimals = "".join(_CHINESE_DIGITS[int(digit)] for digit in fraction)
    return f"{whole}点{decimals}" if decimals else whole


def _normalize_spoken_numbers(text: str) -> str:
    """Convert Arabic numerals for speech only, leaving displayed answers intact."""
    number_pattern = re.compile(r"(?<![\d.])(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?![\d.])")

    def read_percent(match: re.Match[str]) -> str:
        return "百分之" + _number_to_chinese(match.group("number"))

    text = re.sub(
        r"(?P<number>(?<![\d.])(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)(?:\s*[%％])",
        read_percent,
        text,
    )

    def read_number(match: re.Match[str]) -> str:
        value = match.group()
        # Years and edition numbers are conventionally read digit by digit.
        suffix = text[match.end():].lstrip()
        is_year = len(value.replace(",", "")) == 4 and suffix.startswith(("年", "版", "款"))
        return _number_to_chinese(value, digitwise=is_year)

    return number_pattern.sub(read_number, text)


def _normalize_pcm16_loudness(
    pcm16: bytes,
    sample_rate: int,
    *,
    previous_gain: float = 1.0,
    target_rms: float = 3277.0,
    max_gain: float = 2.0,
) -> tuple[bytes, float]:
    """Level short speech windows so a quiet trailing syllable stays audible."""
    samples = np.frombuffer(pcm16, dtype="<i2").astype(np.float32)
    if not samples.size or sample_rate <= 0:
        return pcm16, previous_gain

    # Per-chunk RMS misses a quiet last syllable inside an otherwise loud
    # chunk. Estimate over 100 ms speech windows and interpolate the gain
    # smoothly; windows containing only silence/noise keep the previous gain.
    window_size = max(1, int(sample_rate * 0.1))
    centers: list[float] = [0.0]
    gains: list[float] = [float(previous_gain)]
    last_gain = float(previous_gain)
    for start in range(0, samples.size, window_size):
        end = min(samples.size, start + window_size)
        window = samples[start:end]
        active = np.abs(window) >= 328.0
        if int(np.count_nonzero(active)) >= max(1, int(window.size * 0.05)):
            measured_rms = float(np.sqrt(np.mean(np.square(window[active]))))
            last_gain = float(np.clip(target_rms / max(measured_rms, 1.0), 0.5, max_gain))
        centers.append((start + end) / 2)
        gains.append(last_gain)

    positions = np.arange(samples.size, dtype=np.float32)
    envelope = np.interp(positions, centers, gains).astype(np.float32)
    peak = float(np.max(np.abs(samples * envelope)))
    if peak:
        envelope *= min(1.0, 0.95 * 32767.0 / peak)
    gain = last_gain
    leveled = np.clip(np.rint(samples * envelope), -32768, 32767).astype("<i2")
    return leveled.tobytes(), gain


def sanitize_for_speech(
    text: str, *, pronunciations: Mapping[str, str] | None = None,
) -> str:
    """Convert answer Markdown to speakable body text and remove RAG metadata."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    for prefix in _FALLBACK_PREFIXES:
        text = text.removeprefix(prefix).lstrip()
    for marker in _SOURCE_FOOTERS:
        text = text.split(marker, 1)[0]
    text = re.sub(r"```[\s\S]*?```", "代码内容请查看文字界面。", text)
    text = re.sub(r"\[PDF第\d+页\]", "", text)
    text = re.sub(r"\[来源：[^\]]+\]", "", text)
    text = re.sub(r"\[资料[^\]]*\]", "", text)
    text = re.sub(r"!\[([^\]]*)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"^\s{0,3}#{1,6}\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*(?:[-*+] |\d+[.)]\s+)", "", text, flags=re.MULTILINE)
    text = text.replace("^{-1}", "的逆")
    replacements = {
        r"\times": "乘以", r"\cdot": "乘以", r"\div": "除以",
        "×": "乘以", "÷": "除以", "⁄": "除以", "+": "加", "=": "等于",
        "−": "减", r"\leq": "小于等于",
        r"\geq": "大于等于", r"\neq": "不等于", "**": "", "__": "",
        "`": "", "$": "", "{": "", "}": "",
    }
    for source, replacement in replacements.items():
        text = text.replace(source, replacement)
    # Keep parenthetical content, but do not ask TTS to speak the brackets.
    text = re.sub(r"[()（）]", "", text)
    mapping = dict(_DEFAULT_PRONUNCIATIONS)
    if pronunciations:
        mapping.update(pronunciations)
    text = _apply_corrections(text, mapping)
    text = _normalize_spoken_numbers(text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\s*\n+\s*", "。", text)
    text = re.sub(r"。{2,}", "。", text)
    return text.strip(" 。")


class SpeechTextSegmenter:
    """Incrementally turn streamed Markdown into natural TTS-sized clauses."""

    def __init__(
        self, *, first_max_chars: int = 16, max_chars: int = 32, min_chars: int = 4,
        pronunciations: Mapping[str, str] | None = None,
    ) -> None:
        if not 1 <= min_chars <= first_max_chars <= max_chars:
            raise ValueError("require 1 <= min_chars <= first_max_chars <= max_chars")
        self.first_max_chars = first_max_chars
        self.max_chars = max_chars
        self.min_chars = min_chars
        self.pronunciations = pronunciations
        self._buffer = ""
        self._emitted = 0
        self._discard_footer = False

    def _clean(self, text: str) -> str:
        return sanitize_for_speech(text, pronunciations=self.pronunciations)

    def _next_boundary(self) -> int | None:
        for match in re.finditer(r"[。！？!?；;\n]", self._buffer):
            if len(self._clean(self._buffer[:match.end()])) >= self.min_chars:
                return match.end()
        limit = self.first_max_chars if self._emitted == 0 else self.max_chars
        clean = self._clean(self._buffer)
        if len(clean) < limit:
            return None
        candidates = [
            index + 1 for index, char in enumerate(self._buffer)
            if char in "，,：:" and self.min_chars <= len(self._clean(self._buffer[:index + 1])) <= limit
        ]
        if candidates:
            return candidates[-1]
        for index in range(1, len(self._buffer) + 1):
            if len(self._clean(self._buffer[:index])) >= limit:
                return index
        return len(self._buffer)

    def feed(self, chunk: str) -> list[str]:
        if self._discard_footer or not chunk:
            return []
        self._buffer += chunk
        footer_at = min(
            (position for marker in _SOURCE_FOOTERS if (position := self._buffer.find(marker)) >= 0),
            default=-1,
        )
        if footer_at >= 0:
            self._buffer = self._buffer[:footer_at]
            self._discard_footer = True

        segments: list[str] = []
        while self._buffer:
            boundary = self._next_boundary()
            if boundary is None:
                break
            raw, self._buffer = self._buffer[:boundary], self._buffer[boundary:]
            clean = self._clean(raw)
            if clean:
                segments.append(clean)
                self._emitted += 1
        return segments

    def flush(self) -> list[str]:
        clean = self._clean(self._buffer)
        self._buffer = ""
        if clean:
            self._emitted += 1
            return [clean]
        return []


class ASRClient:
    """SenseVoice INT8 offline ASR for paths, encoded bytes, or wave arrays."""

    def __init__(
        self,
        model_path: str | Path = DEFAULT_ASR_MODEL_PATH,
        *,
        num_threads: int = 4,
        max_audio_seconds: float = 30.0,
        max_encoded_bytes: int = 50 * 1024 * 1024,
    ) -> None:
        self.model_path = Path(model_path).expanduser().resolve()
        required = ("model.int8.onnx", "tokens.txt")
        for name in required:
            if not (self.model_path / name).is_file():
                raise FileNotFoundError(f"ASR model file is missing: {self.model_path / name}")
        if num_threads <= 0 or max_audio_seconds <= 0 or max_encoded_bytes <= 0:
            raise ValueError("ASR limits and num_threads must be positive")
        self.max_audio_seconds = float(max_audio_seconds)
        self.max_encoded_bytes = int(max_encoded_bytes)
        self._lock = Lock()
        self.recognizer = sherpa_onnx.OfflineRecognizer.from_sense_voice(
            model=str(self.model_path / "model.int8.onnx"),
            tokens=str(self.model_path / "tokens.txt"),
            num_threads=num_threads,
            provider="cpu",
            language="auto",
            use_itn=True,
        )

    def _validate_samples(self, samples: np.ndarray) -> np.ndarray:
        samples = np.asarray(samples, dtype=np.float32).reshape(-1)
        if not len(samples):
            raise ValueError("audio is empty")
        if not np.isfinite(samples).all():
            raise ValueError("audio contains non-finite samples")
        if len(samples) > int(DEFAULT_SAMPLE_RATE * self.max_audio_seconds):
            raise ValueError(f"audio exceeds {self.max_audio_seconds:g} seconds")
        if float(np.max(np.abs(samples))) < 1e-5:
            raise ValueError("audio is silent")
        return np.clip(samples, -1.0, 1.0)

    @staticmethod
    def _resample(samples: np.ndarray, source_rate: int) -> np.ndarray:
        if source_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if source_rate == DEFAULT_SAMPLE_RATE:
            return samples.astype(np.float32, copy=False)
        output_length = round(len(samples) * DEFAULT_SAMPLE_RATE / source_rate)
        positions = np.arange(output_length, dtype=np.float64) * source_rate / DEFAULT_SAMPLE_RATE
        return np.interp(positions, np.arange(len(samples)), samples).astype(np.float32)

    def _array_audio(self, source: np.ndarray, sample_rate: int | None) -> np.ndarray:
        if sample_rate is None:
            raise ValueError("sample_rate is required for numpy audio")
        original = np.asarray(source)
        if original.ndim == 2:
            if original.shape[1] <= 8:
                original = original.mean(axis=1)
            elif original.shape[0] <= 8:
                original = original.mean(axis=0)
            else:
                raise ValueError("cannot infer the channel axis")
        if original.ndim != 1:
            raise ValueError("numpy audio must have one or two dimensions")
        if np.issubdtype(original.dtype, np.integer):
            scale = max(abs(np.iinfo(original.dtype).min), np.iinfo(original.dtype).max)
            original = original.astype(np.float32) / scale
        else:
            original = original.astype(np.float32)
        return self._resample(original, sample_rate)

    def _encoded_audio(self, source: str | Path | bytes | bytearray | memoryview) -> np.ndarray:
        input_bytes: bytes | None = None
        if isinstance(source, (bytes, bytearray, memoryview)):
            input_bytes = bytes(source)
            if len(input_bytes) > self.max_encoded_bytes:
                raise ValueError("encoded audio is too large")
            input_spec = "pipe:0"
        else:
            path = Path(source).expanduser().resolve()
            if not path.is_file():
                raise FileNotFoundError(f"audio file does not exist: {path}")
            if path.stat().st_size > self.max_encoded_bytes:
                raise ValueError("encoded audio is too large")
            input_spec = str(path)
        command = [
            "ffmpeg", "-nostdin", "-v", "error", "-i", input_spec,
            "-ac", "1", "-ar", str(DEFAULT_SAMPLE_RATE), "-t", str(self.max_audio_seconds + 0.02),
            "-f", "f32le", "pipe:1",
        ]
        try:
            completed = subprocess.run(
                command, input=input_bytes, capture_output=True, check=True, timeout=45,
            )
        except FileNotFoundError as exc:
            raise RuntimeError("ffmpeg is required for encoded audio input") from exc
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError("audio decoding timed out") from exc
        except subprocess.CalledProcessError as exc:
            detail = exc.stderr.decode("utf-8", "replace").strip()
            raise ValueError(f"cannot decode audio: {detail or 'unsupported or corrupt input'}") from exc
        return np.frombuffer(completed.stdout, dtype="<f4").copy()

    def transcribe(
        self,
        source: str | Path | bytes | bytearray | memoryview | np.ndarray,
        *,
        sample_rate: int | None = None,
        corrections: Mapping[str, str] | None = None,
    ) -> ASRResult:
        started = perf_counter()
        if isinstance(source, np.ndarray):
            samples = self._array_audio(source, sample_rate)
        else:
            samples = self._encoded_audio(source)
        samples = self._validate_samples(samples)
        decode_ms = (perf_counter() - started) * 1000
        stream = self.recognizer.create_stream()
        stream.accept_waveform(DEFAULT_SAMPLE_RATE, samples)
        inference_started = perf_counter()
        with self._lock:
            self.recognizer.decode_stream(stream)
        inference_ms = (perf_counter() - inference_started) * 1000
        result = stream.result
        text = _apply_corrections(result.text.strip(), corrections)
        if not text:
            raise ValueError("no speech was recognized")
        return ASRResult(
            text=text,
            language=_strip_tag(result.lang),
            emotion=_strip_tag(result.emotion),
            event=_strip_tag(result.event),
            audio_duration_ms=len(samples) / DEFAULT_SAMPLE_RATE * 1000,
            decode_ms=decode_ms,
            inference_ms=inference_ms,
            total_ms=(perf_counter() - started) * 1000,
            timestamps_ms=tuple(float(value) * 1000 for value in result.timestamps),
        )


class TTSClient:
    """Client for the isolated, local-only CosyVoice streaming worker."""

    def __init__(
        self,
        model_path: str | Path = DEFAULT_TTS_MODEL_PATH,
        *,
        speaker: str = DEFAULT_TTS_SPEAKER,
        speed: float = 1.0,
        server_url: str | None = None,
        device: str = "cuda:0",
        server_python: str | Path = DEFAULT_COSYVOICE_PYTHON,
        auto_start: bool = True,
        startup_timeout: float = 120.0,
        request_timeout: float = 120.0,
        seed: int = 1986,
        pronunciations: Mapping[str, str] | None = None,
    ) -> None:
        self.model_path = Path(model_path).expanduser().resolve()
        required = ("cosyvoice.yaml", "llm.pt", "flow.pt", "hift.pt", "spk2info.pt")
        for name in required:
            if not (self.model_path / name).is_file():
                raise FileNotFoundError(f"TTS model resource is missing: {self.model_path / name}")
        if not speaker or not 0.5 <= speed <= 2.0:
            raise ValueError("speaker is required and speed must be between 0.5 and 2.0")
        self.speaker = speaker
        self.speed = float(speed)
        self.device = device
        if not self.device.startswith("cuda:"):
            raise ValueError("CosyVoice service currently requires device='cuda:N'")
        device_index = int(self.device.split(":", 1)[1])
        self.server_url = (
            server_url.rstrip("/") if server_url is not None
            else f"http://127.0.0.1:{8766 + device_index}"
        )
        parsed_url = urlparse(self.server_url)
        if parsed_url.scheme != "http" or parsed_url.hostname not in {"127.0.0.1", "localhost"}:
            raise ValueError("CosyVoice server_url must be a local HTTP address")
        self._server_host = parsed_url.hostname
        self._server_port = parsed_url.port or 80
        self.server_python = Path(server_python).expanduser().resolve()
        self.auto_start = auto_start
        self.startup_timeout = float(startup_timeout)
        self.request_timeout = float(request_timeout)
        self.seed = int(seed)
        self.pronunciations = dict(pronunciations or {})
        self._lock = Lock()
        self._last_normalizer_gain = 1.0
        self._server_process: subprocess.Popen[bytes] | None = None
        self._ensure_server()

    def warmup(self) -> None:
        """Warm CosyVoice kernels without persisting the generated audio."""
        list(self.synthesize_stream("准备就绪。", sanitize=False))

    def _health(self, timeout: float = 1.0) -> dict[str, Any]:
        with urlopen(f"{self.server_url}/health", timeout=timeout) as response:
            return json.loads(response.read())

    def _ensure_server(self) -> None:
        try:
            health = self._health()
            if health.get("device") != self.device:
                raise ValueError(
                    f"CosyVoice service uses {health.get('device')}, expected {self.device}"
                )
            if self.speaker not in health.get("speakers", []):
                raise ValueError(f"CosyVoice speaker is unavailable: {self.speaker}")
            return
        except (OSError, ValueError, URLError, json.JSONDecodeError):
            if not self.auto_start:
                raise RuntimeError(f"CosyVoice service is not ready at {self.server_url}")
        if not self.server_python.is_file():
            raise FileNotFoundError(
                f"CosyVoice Python is missing: {self.server_python}. "
                "Set COSYVOICE_PYTHON to the cosyvoice Conda environment's Python, "
                "or start cosyvoice_server.py manually before the web service."
            )
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = self.device.split(":", 1)[1]
        env["PYTHONNOUSERSITE"] = "1"
        env["MODELSCOPE_OFFLINE"] = "1"
        log_path = PROJECT_ROOT / "data" / "cosyvoice_server.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log = log_path.open("ab")
        try:
            self._server_process = subprocess.Popen(
                [
                    str(self.server_python), str(PROJECT_ROOT / "cosyvoice_server.py"),
                    "--host", self._server_host, "--port", str(self._server_port),
                    "--model", str(self.model_path), "--physical-device", self.device,
                ],
                cwd=str(PROJECT_ROOT), env=env, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
            )
        finally:
            log.close()
        deadline = perf_counter() + self.startup_timeout
        last_error: BaseException | None = None
        while perf_counter() < deadline:
            if self._server_process.poll() is not None:
                detail = log_path.read_text("utf-8", errors="replace")[-4000:]
                raise RuntimeError(f"CosyVoice service failed to start:\n{detail}")
            try:
                health = self._health()
                if health.get("device") != self.device:
                    raise ValueError(
                        f"CosyVoice service uses {health.get('device')}, expected {self.device}"
                    )
                if self.speaker not in health.get("speakers", []):
                    raise ValueError(f"CosyVoice speaker is unavailable: {self.speaker}")
                return
            except (OSError, ValueError, URLError, json.JSONDecodeError) as exc:
                last_error = exc
                sleep(0.25)
        raise TimeoutError("CosyVoice service startup timed out") from last_error

    @staticmethod
    def _read_exact(response: Any, size: int) -> bytes:
        output = bytearray()
        while len(output) < size:
            block = response.read(size - len(output))
            if not block:
                raise EOFError("CosyVoice stream ended unexpectedly")
            output.extend(block)
        return bytes(output)

    def synthesize_stream(
        self,
        text: str,
        *,
        index: int = 0,
        speaker: str | None = None,
        speed: float | None = None,
        cancel_event: Event | None = None,
        sanitize: bool = True,
    ) -> Iterator[AudioChunk]:
        spoken = sanitize_for_speech(text, pronunciations=self.pronunciations) if sanitize else text.strip()
        if not spoken:
            raise ValueError("text contains no speakable body content")
        selected_speaker = self.speaker if speaker is None else speaker
        selected_speed = self.speed if speed is None else speed
        stop = cancel_event or Event()
        request = Request(
            f"{self.server_url}/synthesize",
            data=json.dumps({
                "text": spoken, "speaker": selected_speaker, "speed": selected_speed,
                "seed": self.seed,
            }, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"}, method="POST",
        )
        try:
            with self._lock, urlopen(request, timeout=self.request_timeout) as response:
                chunk_index = index
                while not stop.is_set():
                    meta_size_bytes = response.read(4)
                    if not meta_size_bytes:
                        raise EOFError("CosyVoice stream ended without a completion frame")
                    meta_size = struct.unpack("!I", meta_size_bytes)[0]
                    meta = json.loads(self._read_exact(response, meta_size))
                    pcm_size = struct.unpack("!I", self._read_exact(response, 4))[0]
                    pcm16 = self._read_exact(response, pcm_size) if pcm_size else b""
                    if meta.get("error"):
                        raise RuntimeError(meta["error"])
                    if meta.get("done"):
                        return
                    pcm16, self._last_normalizer_gain = _normalize_pcm16_loudness(
                        pcm16,
                        int(meta["sample_rate"]),
                        previous_gain=self._last_normalizer_gain,
                    )
                    duration_ms = float(meta["duration_ms"])
                    synthesis_ms = float(meta["synthesis_ms"])
                    yield AudioChunk(
                        index=chunk_index, text=spoken, pcm16=pcm16,
                        sample_rate=int(meta["sample_rate"]), duration_ms=duration_ms,
                        synthesis_ms=synthesis_ms,
                        rtf=synthesis_ms / duration_ms if duration_ms else 0.0,
                    )
                    chunk_index += 1
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")
            raise RuntimeError(f"CosyVoice request failed: {detail}") from exc
        except (OSError, URLError) as exc:
            raise RuntimeError("CosyVoice service connection failed") from exc
        if stop.is_set():
            raise InterruptedError("TTS synthesis was cancelled")

    def synthesize(
        self,
        text: str,
        *,
        index: int = 0,
        speaker: str | None = None,
        speed: float | None = None,
        cancel_event: Event | None = None,
        sanitize: bool = True,
        output_path: str | Path | None = None,
    ) -> AudioChunk:
        started = perf_counter()
        chunks = list(self.synthesize_stream(
            text, index=index, speaker=speaker, speed=speed,
            cancel_event=cancel_event, sanitize=sanitize,
        ))
        if not chunks:
            raise InterruptedError("TTS synthesis was cancelled")
        sample_rate = chunks[0].sample_rate
        pcm16 = b"".join(chunk.pcm16 for chunk in chunks)
        duration_ms = sum(chunk.duration_ms for chunk in chunks)
        spoken = chunks[0].text
        elapsed_ms = (perf_counter() - started) * 1000
        chunk = AudioChunk(
            index=index,
            text=spoken,
            pcm16=pcm16,
            sample_rate=sample_rate,
            duration_ms=duration_ms,
            synthesis_ms=elapsed_ms,
            rtf=elapsed_ms / duration_ms,
        )
        if output_path is not None:
            chunk.save_wav(output_path)
        return chunk


def save_audio_chunks(chunks: list[AudioChunk], path: str | Path) -> Path:
    """Save same-rate mono chunks as one WAV; called only on explicit request."""
    if not chunks:
        raise ValueError("chunks must not be empty")
    sample_rate = chunks[0].sample_rate
    if any(chunk.sample_rate != sample_rate for chunk in chunks):
        raise ValueError("all chunks must have the same sample rate")
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    pcm = b"".join(chunk.pcm16 for chunk in chunks)
    with wave.open(str(target), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm)
    return target


__all__ = [
    "ASRClient", "ASRResult", "AudioChunk", "DEFAULT_ASR_MODEL_PATH",
    "DEFAULT_COSYVOICE_PYTHON", "DEFAULT_COSYVOICE_URL", "DEFAULT_TTS_MODEL_PATH",
    "DEFAULT_TTS_SPEAKER", "SpeechTextSegmenter",
    "TTSClient", "sanitize_for_speech", "save_audio_chunks",
]
