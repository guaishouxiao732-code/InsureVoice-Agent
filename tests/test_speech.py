"""Speech formatting and voice pipeline tests without loading ONNX models."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock
from threading import Event
from time import perf_counter
import unittest

from speech import ASRResult, AudioChunk, SpeechTextSegmenter, sanitize_for_speech
from voice_assistant import VoiceResponse, VoiceAssistant
from context_manager import ContextManager
from rag import HANDOFF_MESSAGE, RAGClient, RetrievalResult
from cooperative import CooperativeClient


class FakeTextResponse:
    def __init__(self, chunks):
        self.turn_id = 7
        self._chunks = iter(chunks)
        self._text = ""
        self.status = "generating"
        self.metadata = {"knowledge_mode": "rag"}
        self.updates = []

    @property
    def text(self):
        return self._text

    def __iter__(self):
        return self

    def __next__(self):
        try:
            chunk = next(self._chunks)
        except StopIteration:
            self.status = "completed"
            raise
        self._text += chunk
        return chunk

    def update_metadata(self, metadata):
        self.updates.append(deepcopy(metadata))
        self.metadata.update(metadata)

    def cancel(self):
        changed = self.status == "generating"
        if changed:
            self.status = "interrupted"
        return changed

    def close(self):
        return None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class FakeTTS:
    pronunciations = {}

    def __init__(self):
        self.spoken = []

    def synthesize_stream(self, text, *, index, cancel_event, sanitize):
        self.spoken.append(text)
        yield AudioChunk(
            index=index,
            text=text,
            pcm16=b"\0\0" * 16_000,
            sample_rate=16_000,
            duration_ms=1_000,
            synthesis_ms=10,
            rtf=0.01,
        )


class SpeechTests(unittest.TestCase):
    def test_text_completion_does_not_wait_for_slow_tts(self):
        started, release = Event(), Event()
        class SlowTTS(FakeTTS):
            def synthesize_stream(self, *args, **kwargs):
                started.set()
                if not release.wait(3):
                    raise TimeoutError("test TTS timeout")
                yield from super().synthesize_stream(*args, **kwargs)
        text = FakeTextResponse(["第一句已经完成。", "第二句也已经完成。"])
        asr = ASRResult("测试", "zh", "", "", 0, 0, 0, 0, ())
        voice = VoiceResponse(text, SlowTTS(), asr, request_started=perf_counter(), prebuffer_ms=1000)
        try:
            self.assertTrue(started.wait(1))
            self.assertTrue(voice.text_done.wait(1))
            self.assertEqual(voice.text, "第一句已经完成。第二句也已经完成。")
            self.assertLess(voice.metadata["voice"]["text_completed_ms"], 1000)
            self.assertTrue(voice._worker.is_alive())
            release.set()
            self.assertEqual(len(list(voice)), 2)
        finally:
            release.set()
            voice.close()

    def test_request_cancel_returns_before_blocked_tts_and_discards_late_audio(self):
        started, release = Event(), Event()
        class SlowTTS(FakeTTS):
            def synthesize_stream(self, *args, **kwargs):
                started.set()
                release.wait(3)
                yield from super().synthesize_stream(*args, **kwargs)
        asr = ASRResult("测试", "zh", "", "", 0, 0, 0, 0, ())
        voice = VoiceResponse(FakeTextResponse(["这是一段测试回答。"]), SlowTTS(), asr,
                              request_started=perf_counter(), prebuffer_ms=0)
        try:
            self.assertTrue(started.wait(1))
            before = perf_counter()
            voice.request_cancel()
            self.assertLess(perf_counter()-before, .2)
            self.assertEqual(list(voice), [])
            release.set()
            voice.close()
            self.assertEqual(list(voice), [])
        finally:
            release.set()
            voice.close()

    def test_voice_recognition_uses_cooperative_clarification_and_speaks_it(self):
        qwen, retriever, asr = Mock(), Mock(), Mock()
        asr.transcribe.return_value = ASRResult(
            text="有高血压，住院能赔吗", language="zh", emotion="", event="Speech",
            audio_duration_ms=1000, decode_ms=1, inference_ms=20, total_ms=21, timestamps_ms=(),
        )
        client = CooperativeClient(RAGClient(qwen, retriever))
        tts = FakeTTS()
        with TemporaryDirectory() as directory, ContextManager(client, db_path=Path(directory) / "coop-voice.sqlite3") as context:
            session = context.create_session()
            voice = VoiceAssistant(context, asr, tts, prebuffer_ms=0)
            with voice.stream_chat(session, b"fake ASR input", thinking_mode="auto") as response:
                self.assertTrue(list(response))
            metadata = context.get_history(session)[0]["metadata"]
        self.assertIn("能否投保", "".join(tts.spoken))
        self.assertIn("住院理赔", "".join(tts.spoken))
        self.assertEqual(metadata["cooperation"]["decision"], "clarify")
        self.assertEqual(metadata["voice"]["asr_inference_ms"], 20)
        qwen.stream_chat.assert_not_called()
        retriever.search.assert_not_called()

    def test_voice_pipeline_speaks_handoff_and_keeps_metadata_without_qwen(self):
        qwen = Mock()
        qwen.stream_chat.side_effect = AssertionError("禁止调用 Qwen")
        retriever = Mock()
        retriever.search.return_value = RetrievalResult(
            query="资料外问题", sufficient=False, threshold=0.47,
            lexical_threshold=0.08, elapsed_ms=1, results=(),
        )
        client = RAGClient(qwen, retriever)
        tts = FakeTTS()
        with TemporaryDirectory() as directory:
            with ContextManager(client, db_path=Path(directory) / "voice.sqlite3") as context:
                session = context.create_session()
                assistant = VoiceAssistant(context, Mock(), tts, prebuffer_ms=0)
                with assistant.stream_text_chat(session, "资料外问题") as voice:
                    chunks = list(voice)
                self.assertTrue(chunks)
                turn = context.get_history(session)[0]
        self.assertIn("需转人工处理", "".join(tts.spoken))
        self.assertIn("尚未接入人工客服", "".join(tts.spoken))
        self.assertNotIn("检索判定", "".join(tts.spoken))
        self.assertEqual(turn["metadata"]["handoff"]["status"], "pending")
        self.assertIn("voice", turn["metadata"])
        qwen.stream_chat.assert_not_called()

    def test_handoff_body_is_spoken_without_retrieval_diagnostics(self):
        spoken = sanitize_for_speech(HANDOFF_MESSAGE + "\n\n检索判定：保险资料证据不足，待人工处理。")
        self.assertIn("需转人工处理", spoken)
        self.assertIn("尚未接入人工客服", spoken)
        self.assertNotIn("检索判定", spoken)
        segmenter = SpeechTextSegmenter(first_max_chars=24, max_chars=80)
        segments = segmenter.feed(HANDOFF_MESSAGE)
        segments.extend(segmenter.flush())
        self.assertIn("需转人工处理", "".join(segments))
        self.assertIn("待人工处理状态", "".join(segments))

    def test_sanitizer_removes_sources_markdown_and_fallback_disclosure(self):
        text = (
            "根据当前保险资料无法确认，以下根据模型自身知识补充：\n\n"
            "**Qwen** 可以回答问题[来源：保险资料.pdf，第2页]。"
            "\n\n检索来源：来源：保险资料.pdf，第2页"
        )
        spoken = sanitize_for_speech(text)
        self.assertEqual(spoken, "通义千问 可以回答问题")
        self.assertNotIn("PDF", spoken)

    def test_sanitizer_reads_amounts_percentages_and_edition_years_in_chinese(self):
        spoken = sanitize_for_speech(
            "尊享E生2026版：我30岁，年缴318元，赔付比例100%，免赔额1万元；保费3,318.50元。"
        )
        self.assertEqual(
            spoken,
            "尊享E生二零二六版：我三十岁，年缴三百一十八元，赔付比例百分之一百，"
            "免赔额一万元；保费三千三百一十八点五零元",
        )
        self.assertNotRegex(spoken, r"\d")

    def test_segmenter_prefers_punctuation_and_discards_source_footer(self):
        segmenter = SpeechTextSegmenter(first_max_chars=10, max_chars=16)
        output = []
        for chunk in ("可逆矩阵是方阵。", "它存在逆矩阵。", "\n\n检索来源：PDF第79页"):
            output.extend(segmenter.feed(chunk))
        output.extend(segmenter.flush())
        self.assertEqual(output, ["可逆矩阵是方阵", "它存在逆矩阵"])

    def test_voice_response_streams_audio_and_records_only_metadata(self):
        response = FakeTextResponse([
            "可逆矩阵是一种方阵。",
            "它具有唯一的逆矩阵。",
            "\n\n检索来源：PDF第79页",
        ])
        tts = FakeTTS()
        asr = ASRResult(
            text="什么是可逆矩阵？",
            language="zh",
            emotion="NEUTRAL",
            event="Speech",
            audio_duration_ms=1_200,
            decode_ms=3,
            inference_ms=80,
            total_ms=83,
            timestamps_ms=(),
        )
        voice = VoiceResponse(
            response, tts, asr, request_started=0, prebuffer_ms=1_000,
        )
        chunks = list(voice)
        self.assertEqual(len(chunks), 2)
        self.assertEqual(tts.spoken, ["可逆矩阵是一种方阵", "它具有唯一的逆矩阵"])
        metadata = response.metadata["voice"]
        self.assertEqual(metadata["transcript"], "什么是可逆矩阵？")
        self.assertEqual(metadata["tts_segment_count"], 2)
        self.assertEqual(metadata["tts_audio_chunk_count"], 2)
        self.assertEqual(metadata["predicted_buffer_underruns"], 0)
        self.assertNotIn("pcm16", str(metadata))

    def test_cancel_discards_audio_already_buffered_for_playback(self):
        response = FakeTextResponse(["这是一段已经合成但尚未播放的回答。"])
        asr = ASRResult(
            text="测试", language="zh", emotion="NEUTRAL", event="Speech",
            audio_duration_ms=500, decode_ms=1, inference_ms=2, total_ms=3,
            timestamps_ms=(),
        )
        voice = VoiceResponse(
            response, FakeTTS(), asr, request_started=0, prebuffer_ms=0,
        )
        voice._worker.join(timeout=2)
        voice.cancel()
        self.assertEqual(list(voice), [])


if __name__ == "__main__":
    unittest.main()
