"""ASR -> contextual Qwen/RAG -> buffered TTS pipeline for one local user."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from queue import Empty, Queue
import re
from threading import Event, Lock, RLock, Thread, current_thread
from time import perf_counter
from typing import Any

import numpy as np

from context_manager import ContextManager, ResponseStream
from speech import (
    ASRClient,
    ASRResult,
    AudioChunk,
    SpeechTextSegmenter,
    TTSClient,
    save_audio_chunks,
)


@dataclass(frozen=True)
class VoiceAnswer:
    transcript: str
    text: str
    audio_path: Path | None
    metadata: dict[str, Any]


class VoiceResponse(Iterator[AudioChunk]):
    """Audio chunks synthesized ahead of playback from one text response."""

    def __init__(
        self,
        response: ResponseStream,
        tts: TTSClient,
        asr_result: ASRResult,
        *,
        request_started: float,
        prebuffer_ms: float = 1_000,
        first_max_chars: int = 24,
        max_chars: int = 80,
        speed: float | None = None,
    ) -> None:
        if prebuffer_ms < 0:
            raise ValueError("prebuffer_ms cannot be negative")
        self.response = response
        self.tts = tts
        self.asr_result = asr_result
        self.prebuffer_ms = float(prebuffer_ms)
        self.speed = float(tts.speed if speed is None else speed)
        if not 0.5 <= self.speed <= 2.0:
            raise ValueError("speed must be between 0.5 and 2.0")
        self._request_started = request_started
        self._segmenter = SpeechTextSegmenter(
            first_max_chars=first_max_chars,
            max_chars=max_chars,
            pronunciations=tts.pronunciations,
        )
        self._stop = Event()
        self.text_done = Event()
        self._segments: Queue[str | None] = Queue()
        self._queue: Queue[AudioChunk | None] = Queue()
        self._lock = Lock()
        self._consumer_lock = Lock()
        self._error: BaseException | None = None
        self._status = "generating"
        self._first_text_ms: float | None = None
        self._first_audio_ms: float | None = None
        self._text_completed_ms: float | None = None
        self._audio_duration_ms = 0.0
        self._sample_rate: int | None = None
        self._synthesis_ms = 0.0
        self._segment_count = 0
        self._nonstream_segment_count = 0
        self._audio_chunk_count = 0
        self._released = False
        self._staged: list[AudioChunk] = []
        self._staged_ms = 0.0
        self._playback_until: float | None = None
        self._minimum_buffer_margin_ms: float | None = None
        self._predicted_underruns = 0
        self.response.update_metadata({"voice": self._voice_metadata()})
        self._worker = Thread(
            target=self._produce, name=f"voice-turn-{response.turn_id}", daemon=True,
        )
        self._text_worker = Thread(target=self._produce_text, name=f"voice-text-{response.turn_id}", daemon=True)
        self._text_worker.start()
        self._worker.start()

    @property
    def transcript(self) -> str:
        return self.asr_result.text

    @property
    def text(self) -> str:
        return self.response.text

    @property
    def status(self) -> str:
        with self._lock:
            return self._status

    @property
    def first_audio_ms(self) -> float | None:
        with self._lock:
            return self._first_audio_ms

    @property
    def metadata(self) -> dict[str, Any]:
        return self.response.metadata

    def _voice_metadata(self) -> dict[str, Any]:
        return {
            "transcript": self.asr_result.text,
            "asr_language": self.asr_result.language,
            "asr_emotion": self.asr_result.emotion,
            "input_audio_ms": self.asr_result.audio_duration_ms,
            "asr_decode_ms": self.asr_result.decode_ms,
            "asr_inference_ms": self.asr_result.inference_ms,
            "asr_total_ms": self.asr_result.total_ms,
            "llm_first_text_ms": self._first_text_ms,
            "first_audio_ms": self._first_audio_ms,
            "text_completed_ms": self._text_completed_ms,
            "tts_synthesis_ms": self._synthesis_ms,
            "output_audio_ms": self._audio_duration_ms,
            "tts_segment_count": self._segment_count,
            "tts_nonstream_segment_count": self._nonstream_segment_count,
            "tts_audio_chunk_count": self._audio_chunk_count,
            "recommended_prebuffer_ms": self.prebuffer_ms,
            "minimum_buffer_margin_ms": self._minimum_buffer_margin_ms,
            "predicted_buffer_underruns": self._predicted_underruns,
            "audio_format": "pcm_s16le_mono",
        }

    def _publish(self, chunk: AudioChunk) -> None:
        if self._stop.is_set():
            return
        self._staged.append(chunk)
        self._staged_ms += chunk.duration_ms
        if not self._released and self._staged_ms < self.prebuffer_ms:
            return
        if not self._released:
            self._released = True
            now = perf_counter()
            with self._lock:
                self._first_audio_ms = (now - self._request_started) * 1000
                self._playback_until = now + self._staged_ms / 1000
        else:
            now = perf_counter()
            with self._lock:
                assert self._playback_until is not None
                margin_ms = (self._playback_until - now) * 1000
                self._minimum_buffer_margin_ms = (
                    margin_ms if self._minimum_buffer_margin_ms is None
                    else min(self._minimum_buffer_margin_ms, margin_ms)
                )
                if margin_ms < 0:
                    self._predicted_underruns += 1
                self._playback_until = max(self._playback_until, now) + chunk.duration_ms / 1000
        for staged in self._staged:
            self._queue.put(staged)
        self._staged.clear()
        self._staged_ms = 0.0

    def _synthesize(self, text: str) -> None:
        if self._stop.is_set():
            return
        with self._lock:
            is_following_segment = self._segment_count > 0
            self._segment_count += 1
            sample_rate = self._sample_rate
        if is_following_segment and sample_rate:
            # Sentence segmentation strips terminal punctuation before TTS.
            # Add a short PCM gap between separately synthesized clauses so
            # concatenated playback still has a perceptible, stable pause.
            pause_ms = 180.0
            silence = AudioChunk(
                index=self._audio_chunk_count,
                text="",
                pcm16=bytes(round(sample_rate * pause_ms / 1000) * 2),
                sample_rate=sample_rate,
                duration_ms=pause_ms,
                synthesis_ms=0.0,
                rtf=0.0,
            )
            with self._lock:
                self._audio_duration_ms += pause_ms
            self._publish(silence)
        with self._lock:
            buffered_ms = max(0.0, (self._playback_until - perf_counter()) * 1000) if self._playback_until else 0.0
        # Full-utterance synthesis avoids audible articulation glitches in
        # some long technical clauses. Only select it once already-buffered
        # audio can cover a conservative estimate of the synthesis time.
        estimated_synthesis_ms = max(2500.0, len(text) * 100.0)
        stream = not (
            is_following_segment
            and self.speed == 1.0
            and buffered_ms >= estimated_synthesis_ms + 1000.0
        )
        if not stream:
            with self._lock:
                self._nonstream_segment_count += 1
        synthesis_options: dict[str, Any] = {"speed": self.speed, "stream": stream}
        for chunk in self.tts.synthesize_stream(
            text, index=self._audio_chunk_count, cancel_event=self._stop, sanitize=True,
            **synthesis_options,
        ):
            with self._lock:
                self._audio_chunk_count += 1
                self._audio_duration_ms += chunk.duration_ms
                self._synthesis_ms += chunk.synthesis_ms
                self._sample_rate = chunk.sample_rate
            self._publish(chunk)

    def _produce_text(self) -> None:
        pending_speech_segments: list[str] = []
        first_speech_segment = True

        def join_sentences(segments: list[str]) -> str:
            joined = segments[0]
            for segment in segments[1:]:
                separator = "" if re.search(r"[。！？!?；;]$", joined) else "。"
                joined += separator + segment
            return joined

        def queue_speech_segment(segment: str) -> None:
            nonlocal first_speech_segment
            if first_speech_segment:
                # Keep the first audio responsive; batch later sentences so TTS
                # does not repeatedly reset its prosody in the middle of a turn.
                self._segments.put(segment)
                first_speech_segment = False
                return
            pending_speech_segments.append(segment)
            if len(pending_speech_segments) >= 2:
                self._segments.put(join_sentences(pending_speech_segments))
                pending_speech_segments.clear()

        try:
            with self.response:
                for text_chunk in self.response:
                    if self._stop.is_set():
                        break
                    if self._first_text_ms is None and text_chunk:
                        with self._lock:
                            self._first_text_ms = (
                                perf_counter() - self._request_started
                            ) * 1000
                    for segment in self._segmenter.feed(text_chunk):
                        queue_speech_segment(segment)
            if not self._stop.is_set():
                for segment in self._segmenter.flush():
                    queue_speech_segment(segment)
                if pending_speech_segments:
                    self._segments.put(join_sentences(pending_speech_segments))
                    pending_speech_segments.clear()
                self._text_completed_ms = (perf_counter() - self._request_started) * 1000
        except BaseException as exc:
            with self._lock:
                if not self._stop.is_set():
                    self._error = exc
                    self._status = "failed"
            self._stop.set()
        finally:
            self._segments.put(None)
            try:
                self.response.update_metadata({"voice": self._voice_metadata()})
            except BaseException as exc:
                with self._lock:
                    self._error = self._error or exc
                    self._status = "failed"
                self._stop.set()
            self.text_done.set()

    def _produce(self) -> None:
        try:
            while not self._stop.is_set():
                segment = self._segments.get()
                if segment is None or self._stop.is_set():
                    break
                self._synthesize(segment)
            if not self._stop.is_set():
                if self._staged:
                    # A very short final answer may not reach the normal buffer target.
                    self._released = True
                    with self._lock:
                        if self._first_audio_ms is None:
                            now = perf_counter()
                            self._first_audio_ms = (now - self._request_started) * 1000
                            self._playback_until = now + self._staged_ms / 1000
                    for chunk in self._staged:
                        self._queue.put(chunk)
                    self._staged.clear()
                with self._lock:
                    self._status = self.response.status
        except InterruptedError:
            with self._lock:
                self._status = "interrupted"
        except BaseException as exc:
            with self._lock:
                self._error = exc
                self._status = "failed"
        finally:
            try:
                self.response.update_metadata({"voice": self._voice_metadata()})
            except BaseException as exc:
                with self._lock:
                    if self._error is None:
                        self._error = exc
                        self._status = "failed"
            self._queue.put(None)

    def __iter__(self) -> VoiceResponse:
        return self

    def __next__(self) -> AudioChunk:
        if not self._consumer_lock.acquire(blocking=False):
            raise RuntimeError("A VoiceResponse supports only one consumer")
        try:
            while True:
                if self._stop.is_set():
                    with self._lock:
                        if self._error is not None:
                            raise RuntimeError("Voice generation failed") from self._error
                    raise StopIteration
                try:
                    item = self._queue.get(timeout=0.1)
                except Empty:
                    continue
                if item is None:
                    with self._lock:
                        if self._error is not None:
                            raise RuntimeError("Voice generation failed") from self._error
                    raise StopIteration
                return item
        finally:
            self._consumer_lock.release()

    def request_cancel(self) -> bool:
        with self._lock:
            changed = (
                self._status == "generating"
                or bool(self._staged)
                or not self._queue.empty()
            )
            self._stop.set()
            if changed:
                self._status = "interrupted"
        request = getattr(self.response, "request_cancel", self.response.cancel)
        request()
        self._segments.put(None)
        self._queue.put(None)
        return changed

    def cancel(self) -> bool:
        changed = self.request_cancel()
        self.response.cancel()
        if self._text_worker is not current_thread():
            self._text_worker.join()
        if self._worker is not current_thread():
            self._worker.join()
        while True:
            try:
                self._queue.get_nowait()
            except Empty:
                break
        self._queue.put(None)
        with self._lock:
            if changed:
                self._status = "interrupted"
        return changed

    def close(self) -> None:
        self.cancel()

    def __enter__(self) -> VoiceResponse:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


class VoiceAssistant:
    """Single-user local voice wrapper around an existing ContextManager."""

    def __init__(
        self,
        context: ContextManager,
        asr: ASRClient,
        tts: TTSClient,
        *,
        prebuffer_ms: float = 1_000,
        first_max_chars: int = 24,
        max_chars: int = 80,
    ) -> None:
        self.context = context
        self.asr = asr
        self.tts = tts
        self.prebuffer_ms = prebuffer_ms
        self.first_max_chars = first_max_chars
        self.max_chars = max_chars
        self._gate = RLock()
        self._active: VoiceResponse | None = None

    def stream_chat(
        self,
        session_id: str,
        audio: str | Path | bytes | bytearray | memoryview | np.ndarray,
        *,
        sample_rate: int | None = None,
        corrections: Mapping[str, str] | None = None,
        tts_speed: float | None = None,
        cancel_event: Event | None = None,
        **generation_options: Any,
    ) -> VoiceResponse:
        with self._gate:
            if self._active is not None:
                self._active.cancel()
            started = perf_counter()
            if cancel_event is not None and cancel_event.is_set():
                raise InterruptedError("Voice request cancelled before ASR")
            asr_result = self.asr.transcribe(
                audio, sample_rate=sample_rate, corrections=corrections,
            )
            if cancel_event is not None and cancel_event.is_set():
                raise InterruptedError("Voice request cancelled during ASR")
            text_response = self.context.stream_chat(
                session_id, asr_result.text, **generation_options,
            )
            if cancel_event is not None and cancel_event.is_set():
                text_response.cancel()
                raise InterruptedError("Voice request cancelled before synthesis")
            response = VoiceResponse(
                text_response,
                self.tts,
                asr_result,
                request_started=started,
                prebuffer_ms=self.prebuffer_ms,
                first_max_chars=self.first_max_chars,
                max_chars=self.max_chars,
                speed=tts_speed,
            )
            self._active = response
            return response

    def stream_text_chat(
        self,
        session_id: str,
        text: str,
        *,
        tts_speed: float | None = None,
        cancel_event: Event | None = None,
        **generation_options: Any,
    ) -> VoiceResponse:
        """Stream one typed question through Qwen/RAG and the same TTS pipeline.

        The synthetic ASR result keeps the web protocol uniform while making it
        explicit in metadata that no speech recognition was performed.
        """
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be non-empty")
        with self._gate:
            if self._active is not None:
                self._active.cancel()
            started = perf_counter()
            if cancel_event is not None and cancel_event.is_set():
                raise InterruptedError("Text request cancelled")
            asr_result = ASRResult(
                text=text.strip(), language="text", emotion="", event="",
                audio_duration_ms=0.0, decode_ms=0.0, inference_ms=0.0,
                total_ms=0.0, timestamps_ms=(),
            )
            text_response = self.context.stream_chat(
                session_id, asr_result.text, **generation_options,
            )
            if cancel_event is not None and cancel_event.is_set():
                text_response.cancel()
                raise InterruptedError("Text request cancelled before synthesis")
            response = VoiceResponse(
                text_response,
                self.tts,
                asr_result,
                request_started=started,
                prebuffer_ms=self.prebuffer_ms,
                first_max_chars=self.first_max_chars,
                max_chars=self.max_chars,
                speed=tts_speed,
            )
            self._active = response
            return response

    def stream_transcribed_chat(
        self, session_id: str, asr_result: ASRResult, *,
        request_started: float, tts_speed: float | None = None,
        **generation_options: Any,
    ) -> VoiceResponse:
        """Reuse final ASR, preserving its metadata without recognizing twice."""
        with self._gate:
            if self._active is not None:
                self._active.cancel()
            text_response = self.context.stream_chat(
                session_id, asr_result.text, **generation_options,
            )
            response = VoiceResponse(
                text_response, self.tts, asr_result, request_started=request_started,
                prebuffer_ms=self.prebuffer_ms, first_max_chars=self.first_max_chars,
                max_chars=self.max_chars, speed=tts_speed,
            )
            self._active = response
            return response

    def interrupt(self) -> bool:
        """Stop text generation/TTS and discard all audio not yet delivered."""
        with self._gate:
            if self._active is None:
                return False
            return self._active.cancel()

    def request_interrupt(self) -> bool:
        # Do not wait on the construction gate: ASR may currently hold it.
        active = self._active
        return active.request_cancel() if active is not None else False

    def chat_to_wav(
        self,
        session_id: str,
        audio: str | Path | bytes | bytearray | memoryview | np.ndarray,
        *,
        output_path: str | Path | None = None,
        sample_rate: int | None = None,
        corrections: Mapping[str, str] | None = None,
        **generation_options: Any,
    ) -> VoiceAnswer:
        response = self.stream_chat(
            session_id,
            audio,
            sample_rate=sample_rate,
            corrections=corrections,
            **generation_options,
        )
        with closing(response):
            chunks = list(response)
        saved = save_audio_chunks(chunks, output_path) if output_path is not None else None
        return VoiceAnswer(
            transcript=response.transcript,
            text=response.text,
            audio_path=saved,
            metadata=response.metadata,
        )


__all__ = ["VoiceAnswer", "VoiceAssistant", "VoiceResponse"]
