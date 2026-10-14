"""Speech formatting and voice pipeline tests without loading ONNX models."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock
import re
from threading import Event
from time import perf_counter
import unittest

import numpy as np

from speech import (
    ASRResult,
    AudioChunk,
    SpeechTextSegmenter,
    _normalize_pcm16_loudness,
    classify_english_token,
    sanitize_for_speech,
)
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
    speed = 1.0

    def __init__(self):
        self.spoken = []
        self.speeds = []
        self.stream_modes = []

    def synthesize_stream(self, text, *, index, cancel_event, sanitize, speed, stream):
        self.spoken.append(text)
        self.speeds.append(speed)
        self.stream_modes.append(stream)
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
    def test_ai_is_read_as_two_letters_without_changing_other_words(self):
        self.assertEqual(sanitize_for_speech("我是 AI 保险业务助手。"), "我是 A I 保险业务助手")
        self.assertEqual(sanitize_for_speech("AI和ai都用于说明；AIGC与PAID不变。"),
                         "A I和A I都用于说明；A I G C与paid不变")

    def test_english_pronunciation_distinguishes_words_from_initialisms(self):
        for token in ("AI", "ICU", "BNCT", "VIP", "PDF", "AIGC", "pdf", "XYZ"):
            with self.subTest(token=token):
                self.assertEqual(classify_english_token(token), "letters")
        for token in ("WHO", "IT", "LED", "US"):
            with self.subTest(token=token):
                self.assertEqual(classify_english_token(token), "letters")
        for token in ("Hello", "HELLO", "NASA", "NATO", "COVID", "PLUS", "PAID", "policy"):
            with self.subTest(token=token):
                self.assertEqual(classify_english_token(token), "word")
        for token in ("who", "Who", "it", "led", "us"):
            with self.subTest(token=token):
                self.assertEqual(classify_english_token(token), "word")
        self.assertEqual(
            sanitize_for_speech("ICU使用BNCT。PDF可读，Hello和NASA不逐字读。"),
            "ICU使用B N C T。P D F可读，Hello和nasa不逐字读",
        )
        self.assertEqual(sanitize_for_speech("ChatGPT与OpenAI"), "Chat G P T与Open A I")
        self.assertEqual(sanitize_for_speech("WHO与IT确认LED，who led us"),
                         "W H O与I T确认L E D，who led us")
        self.assertEqual(sanitize_for_speech("PLUS（2025）版"), "普拉斯二零二五版")
        self.assertEqual(sanitize_for_speech("storage和RAG"), "storage和检索增强生成")
        self.assertEqual(sanitize_for_speech("BNCT治疗", pronunciations={"BNCT": "硼中子俘获"}),
                         "硼中子俘获治疗")
        self.assertEqual(classify_english_token("DON'T"), "word")

    def test_streamed_english_tokens_are_not_cut_in_the_middle(self):
        for source, expected in (
            ("本次保险责任包BNCT治疗费用。", "B N C T"),
            ("他说OpenAI模型可以回答。", "Open A I"),
        ):
            with self.subTest(source=source):
                segmenter = SpeechTextSegmenter(first_max_chars=8, max_chars=16, min_chars=4)
                parts = []
                for character in source:
                    parts.extend(segmenter.feed(character))
                parts.extend(segmenter.flush())
                self.assertIn(expected, "".join(parts))
                self.assertTrue(any(expected in part for part in parts), parts)

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
            output_chunks = list(voice)
            self.assertEqual(len(output_chunks), 3)
            self.assertEqual(output_chunks[1].duration_ms, 180.0)
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
            "尊享伊生二零二六版：我三十岁，年缴三百一十八元，赔付比例百分之一百，"
            "免赔额一万元；保费三千三百一十八点五零元",
        )
        self.assertNotRegex(spoken, r"\d")

    def test_product_name_e_is_spoken_as_english_letter(self):
        for written in ("尊享e生", "尊享E生", "尊享 e 生", "尊享 E 生"):
            with self.subTest(written=written):
                self.assertEqual(sanitize_for_speech(written), "尊享伊生")
        self.assertEqual(sanitize_for_speech("尊享一生和电子邮件"), "尊享一生和电子邮件")

    def test_streaming_segments_do_not_split_product_name(self):
        for prefix_length in range(10, 19):
            for product_name in ("尊享e生", "尊享 E 生"):
                with self.subTest(prefix_length=prefix_length, product_name=product_name):
                    segmenter = SpeechTextSegmenter(first_max_chars=16, max_chars=32)
                    parts = []
                    for character in "请" * prefix_length + product_name + "可以咨询。":
                        parts.extend(segmenter.feed(character))
                    parts.extend(segmenter.flush())
                    self.assertEqual("".join(parts), "请" * prefix_length + "尊享伊生可以咨询")
                    self.assertEqual(sum("尊享伊生" in part for part in parts), 1)

    def test_sanitizer_reads_division_and_omits_parentheses(self):
        spoken = sanitize_for_speech("计算12÷3，括号内（12+6）乘以2。")
        self.assertEqual(spoken, "计算十二除以三，括号内十二加六乘以二")
        self.assertNotRegex(spoken, r"[()（）÷+×]")

    def test_version_list_reads_years_and_plus_without_speaking_slashes(self):
        source = "2025/2024/2023/PLUS（2025）版/PLUS"
        expected = "二零二五、二零二四、二零二三、普拉斯二零二五版，以及普拉斯"
        self.assertEqual(sanitize_for_speech(source), expected)
        self.assertEqual(sanitize_for_speech("PLUS（2025）版"), "普拉斯二零二五版")
        self.assertEqual(sanitize_for_speech("13元/月"), "每月十三元")
        self.assertEqual(sanitize_for_speech("12/3"), "十二/三")

        segmenter = SpeechTextSegmenter(first_max_chars=24, max_chars=80)
        segments = []
        for character in "可选版本：" + source + "。":
            segments.extend(segmenter.feed(character))
        segments.extend(segmenter.flush())
        self.assertEqual("".join(segments), "可选版本：" + expected)

    def test_sanitizer_reads_age_ranges_and_annual_rates_naturally(self):
        original = (
            "根据当前费率表，2岁属于0–4岁档，有社保、年缴基础费率为806元/年。"
            "实际保费以投保页面、核保结果及保险合同为准。"
        )
        self.assertEqual(
            sanitize_for_speech(original),
            "根据当前费率表，二岁属于零到四岁档，有社保、年缴基础费率为每年八百零六元。"
            "实际保费以投保页面、核保结果及保险合同为准",
        )
        for range_text in ("0—4岁", "0-4岁", "0～4岁"):
            with self.subTest(range_text=range_text):
                self.assertEqual(sanitize_for_speech(range_text), "零到四岁")
        self.assertEqual(sanitize_for_speech("806元／年"), "每年八百零六元")
        self.assertEqual(sanitize_for_speech("13元/月"), "每月十三元")
        self.assertEqual(sanitize_for_speech("13元／月"), "每月十三元")

        # Autoplay receives model output incrementally; normalization must
        # still work when the symbols arrive in separate text deltas.
        segmenter = SpeechTextSegmenter(first_max_chars=24, max_chars=80)
        segments = []
        for character in original:
            segments.extend(segmenter.feed(character))
        segments.extend(segmenter.flush())
        self.assertIn("零到四岁档", "".join(segments))
        self.assertIn("每年八百零六元", "".join(segments))
        self.assertNotRegex("".join(segments), r"[–—/／]")

        monthly = SpeechTextSegmenter(first_max_chars=24, max_chars=80)
        monthly_segments = []
        for character in "基础费率13元/月，实际保费以合同为准。":
            monthly_segments.extend(monthly.feed(character))
        monthly_segments.extend(monthly.flush())
        self.assertIn("每月十三元", "".join(monthly_segments))
        self.assertNotIn("/", "".join(monthly_segments))

    def test_pcm_loudness_is_leveled_without_clipping(self):
        timeline = np.arange(16_000, dtype=np.float32) / 16_000
        previous_gain = 1.0
        measured_levels = []
        for amplitude in (8_000, 5_000):
            pcm = np.rint(amplitude * np.sin(2 * np.pi * 220 * timeline)).astype("<i2").tobytes()
            normalized, previous_gain = _normalize_pcm16_loudness(
                pcm, 16_000, previous_gain=previous_gain,
            )
            samples = np.frombuffer(normalized, dtype="<i2").astype(np.float32)
            active = samples[np.abs(samples) >= 328]
            measured_levels.append(float(np.sqrt(np.mean(active**2))))
            self.assertLessEqual(float(np.max(np.abs(samples))), 32767)
        self.assertLess(abs(measured_levels[0] - measured_levels[1]), 400)

    def test_pcm_loudness_levels_quiet_speech_at_end_of_same_chunk(self):
        sample_rate = 16_000
        timeline = np.arange(sample_rate, dtype=np.float32) / sample_rate
        amplitudes = np.where(timeline < 0.6, 8_000.0, 3_000.0)
        pcm = np.rint(amplitudes * np.sin(2 * np.pi * 220 * timeline)).astype("<i2").tobytes()
        normalized, _ = _normalize_pcm16_loudness(pcm, sample_rate)
        samples = np.frombuffer(normalized, dtype="<i2").astype(np.float32)
        first_rms = float(np.sqrt(np.mean(samples[int(0.2 * sample_rate):int(0.55 * sample_rate)] ** 2)))
        final_rms = float(np.sqrt(np.mean(samples[int(0.7 * sample_rate):int(0.95 * sample_rate)] ** 2)))
        self.assertLess(abs(first_rms - final_rms), 250)

    def test_all_tts_segments_keep_the_same_speed(self):
        tts = FakeTTS()
        asr = ASRResult("测试", "zh", "", "", 0, 0, 0, 0, ())
        response = FakeTextResponse(["第一句话。", "第二句话。", "第三句话。"])
        voice = VoiceResponse(
            response, tts, asr, request_started=perf_counter(), prebuffer_ms=0, speed=0.9,
        )
        list(voice)
        self.assertEqual(tts.speeds, [0.9, 0.9])
        self.assertEqual(tts.stream_modes, [True, True])
        self.assertEqual(tts.spoken, ["第一句话", "第二句话。第三句话"])

    def test_following_segment_can_use_full_synthesis_only_with_enough_buffer(self):
        class LongFirstTTS(FakeTTS):
            def synthesize_stream(self, text, **options):
                chunk = next(super().synthesize_stream(text, **options))
                if len(self.spoken) == 1:
                    yield AudioChunk(
                        index=chunk.index, text=chunk.text,
                        pcm16=b"\0\0" * (16_000 * 20), sample_rate=16_000,
                        duration_ms=20_000, synthesis_ms=10, rtf=0.0005,
                    )
                else:
                    yield chunk

        tts = LongFirstTTS()
        asr = ASRResult("测试", "zh", "", "", 0, 0, 0, 0, ())
        response = FakeTextResponse(["第一句话。", "第二句话。", "第三句话。"])
        voice = VoiceResponse(response, tts, asr, request_started=perf_counter(), prebuffer_ms=0)
        list(voice)
        self.assertEqual(tts.stream_modes, [True, False])
        self.assertEqual(response.metadata["voice"]["tts_nonstream_segment_count"], 1)

    def test_cancel_during_full_synthesis_returns_without_waiting(self):
        started, release = Event(), Event()

        class BlockedFullTTS(FakeTTS):
            def synthesize_stream(self, text, **options):
                chunk = next(super().synthesize_stream(text, **options))
                if options["stream"]:
                    yield AudioChunk(
                        index=chunk.index, text=chunk.text,
                        pcm16=b"\0\0" * (16_000 * 20), sample_rate=16_000,
                        duration_ms=20_000, synthesis_ms=10, rtf=0.0005,
                    )
                else:
                    started.set()
                    release.wait(3)
                    yield chunk

        tts = BlockedFullTTS()
        asr = ASRResult("测试", "zh", "", "", 0, 0, 0, 0, ())
        voice = VoiceResponse(
            FakeTextResponse(["第一句话。", "第二句话。", "第三句话。"]),
            tts, asr, request_started=perf_counter(), prebuffer_ms=0,
        )
        try:
            self.assertTrue(started.wait(1))
            before = perf_counter()
            self.assertTrue(voice.request_cancel())
            self.assertLess(perf_counter() - before, 0.2)
            self.assertEqual(tts.stream_modes, [True, False])
            self.assertEqual(list(voice), [])
        finally:
            release.set()
            voice.close()

    def test_batched_sentences_do_not_add_duplicate_sentence_punctuation(self):
        tts = FakeTTS()
        asr = ASRResult("测试", "zh", "", "", 0, 0, 0, 0, ())
        response = FakeTextResponse(["第一句话！", "第二句话。", "第三句话。"])
        voice = VoiceResponse(
            response, tts, asr, request_started=perf_counter(), prebuffer_ms=0,
        )
        audio_chunks = list(voice)
        self.assertEqual(tts.spoken, ["第一句话！", "第二句话。第三句话"])
        self.assertEqual(len(audio_chunks), 3)
        self.assertEqual(audio_chunks[1].text, "")
        self.assertEqual(audio_chunks[1].duration_ms, 180.0)

    def test_segmenter_prefers_punctuation_and_discards_source_footer(self):
        segmenter = SpeechTextSegmenter(first_max_chars=10, max_chars=16)
        output = []
        for chunk in ("可逆矩阵是方阵。", "它存在逆矩阵。", "\n\n检索来源：PDF第79页"):
            output.extend(segmenter.feed(chunk))
        output.extend(segmenter.flush())
        self.assertEqual(output, ["可逆矩阵是方阵", "它存在逆矩阵"])

    def test_long_complete_first_sentence_releases_earliest_natural_clause(self):
        source = (
            "根据当前费率表，35岁属于31–35岁档，"
            "有社保、年缴基础费率为423元/年。"
            "实际保费以投保页面、核保结果及保险合同为准。"
        )
        segmenter = SpeechTextSegmenter(first_max_chars=24, max_chars=80)
        output = segmenter.feed(source)
        output.extend(segmenter.flush())
        self.assertEqual(output[0], "根据当前费率表，")
        self.assertEqual(
            "".join(output),
            re.sub(r"[。！？!?；;]", "", sanitize_for_speech(source)),
        )

        # The low-latency boundary remains outside protected product names and
        # English initialisms.
        protected = SpeechTextSegmenter(first_max_chars=24, max_chars=80)
        protected_source = "根据尊享e生和BNCT责任，这是一条较长的完整回答。"
        protected_output = protected.feed(protected_source)
        protected_output.extend(protected.flush())
        self.assertEqual(protected_output[0], "根据尊享伊生和B N C T责任，")
        self.assertEqual(
            "".join(protected_output),
            re.sub(r"[。！？!?；;]", "", sanitize_for_speech(protected_source)),
        )

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
        self.assertEqual(len(chunks), 3)
        self.assertEqual(chunks[1].duration_ms, 180.0)
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
