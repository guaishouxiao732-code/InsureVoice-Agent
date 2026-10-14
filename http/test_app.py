import base64
import os
import sys
import unittest
import wave
from types import SimpleNamespace
from threading import Event
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch, Mock, ANY
from tempfile import TemporaryDirectory
from io import BytesIO
from pathlib import Path

os.environ["WEB_SKIP_MODEL_LOAD"] = "1"
HTTP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(HTTP_DIR))

from fastapi.testclient import TestClient
from app import _decode_browser_audio, _mode_options, app
import app as gateway


class FakeContext:
    def __init__(self):
        self.history = []

    def list_sessions(self):
        return []

    def create_session(self):
        return "test-session"

    def get_history(self, _):
        return list(self.history)


class FakeResponse:
    def __init__(self, text):
        self.text, self.status, self.metadata = text, "complete", {}
        self.response = SimpleNamespace(update_metadata=self.metadata.update)
        self.iterator = iter([SimpleNamespace(pcm16=bytes(3200), sample_rate=16000, duration_ms=100)])

    def __next__(self):
        return next(self.iterator)

    def close(self):
        pass


class VoiceProtocolTests(unittest.TestCase):
    def test_interrupt_ack_does_not_wait_for_blocked_asr_and_does_not_start_model(self):
        from context_manager import ContextManager
        from voice_assistant import VoiceAssistant
        from speech import ASRResult
        started, release = Event(), Event()
        model, asr, tts = Mock(), Mock(), Mock()
        def recognize(*args, **kwargs):
            started.set()
            if not release.wait(3):
                raise TimeoutError("test ASR timeout")
            return ASRResult("年缴多少钱", "zh", "", "", 1000, 0, 0, 0, ())
        asr.transcribe.side_effect = recognize
        with TemporaryDirectory() as directory:
            context = ContextManager(model, db_path=Path(directory)/"cancel-asr.sqlite3")
            runtime = gateway.AssistantRuntime()
            runtime.state, runtime.context = "ready", context
            runtime.voice = VoiceAssistant(context, asr, tts)
            with patch.object(gateway, "runtime", runtime), patch.object(gateway, "_decode_browser_audio", return_value=b"wav"), TestClient(app) as client:
                with client.websocket_connect("/ws") as ws:
                    self.receive_until(ws, "history")
                    ws.send_json({"type":"voice", "audio":"encoded"})
                    self.receive_until(ws, "stage")
                    self.assertTrue(started.wait(1))
                    try:
                        ws.send_json({"type":"interrupt"})
                        with ThreadPoolExecutor() as pool:
                            ack = pool.submit(self.receive_until, ws, "interrupted").result(timeout=1)
                        self.assertEqual(ack["type"], "interrupted")
                    finally:
                        release.set()
            model.stream_chat.assert_not_called()
            tts.synthesize_stream.assert_not_called()

    def receive_until(self, ws, kind):
        for _ in range(30):
            message = ws.receive_json()
            if message["type"] == kind:
                return message
        self.fail(f"没有收到 {kind}")

    def test_removed_continuous_commands_are_rejected_without_asr(self):
        asr, voice = Mock(), Mock()
        fake = SimpleNamespace(state="ready", context=FakeContext(), asr=asr,
                               voice=voice, close=lambda: None)
        with patch.object(gateway, "runtime", fake), TestClient(app) as client:
            with client.websocket_connect("/ws") as ws:
                self.receive_until(ws, "history")
                for command in ("listen_start", "listen_stop"):
                    ws.send_json({"type": command, "sample_rate": 16000})
                    self.assertIn("连续语音已移除", self.receive_until(ws, "error")["message"])
        asr.transcribe.assert_not_called()
        voice.stream_chat.assert_not_called()

    def test_binary_continuous_audio_is_rejected(self):
        fake = SimpleNamespace(state="ready", context=FakeContext(), close=lambda: None)
        with patch.object(gateway, "runtime", fake), TestClient(app) as client:
            with client.websocket_connect("/ws") as ws:
                self.receive_until(ws, "history")
                ws.send_bytes(b"old-continuous-audio")
                self.assertIn("连续语音已移除", self.receive_until(ws, "error")["message"])

    def test_recorded_audio_still_recognizes_answers_and_sends_audio(self):
        response = FakeResponse("测试回答")
        response.transcript = "年缴多少钱"
        voice = SimpleNamespace(stream_chat=Mock(return_value=response))
        fake = SimpleNamespace(
            state="ready", context=FakeContext(), voice=voice,
            interrupt=lambda _session_id: False, close=lambda: None,
        )
        with patch.object(gateway, "runtime", fake), patch.object(gateway, "_decode_browser_audio", return_value=b"wav"), TestClient(app) as client:
            with client.websocket_connect("/ws") as ws:
                self.receive_until(ws, "history")
                ws.send_json({"type": "voice", "audio": "encoded"})
                started = self.receive_until(ws, "turn_start")
                self.assertEqual(started["question"], "年缴多少钱")
                self.assertTrue(started["from_voice"])
                audio = self.receive_until(ws, "audio_chunk")
                self.assertEqual(audio["sample_rate"], 16000)
                self.assertEqual(self.receive_until(ws, "turn_done")["text"], "测试回答")
        voice.stream_chat.assert_called_once_with("test-session", b"wav", tts_speed=1.0, cancel_event=ANY, thinking_mode="auto")

    def test_replay_uses_full_utterance_synthesis(self):
        chunk = SimpleNamespace(pcm16=bytes(3200), sample_rate=16000, duration_ms=100)
        def audio_chunks(*_args, **_kwargs):
            yield chunk
        tts = SimpleNamespace(synthesize_stream=Mock(side_effect=audio_chunks))
        fake = SimpleNamespace(state="ready", context=FakeContext(), tts=tts, close=lambda: None)
        with patch.object(gateway, "runtime", fake), TestClient(app) as client:
            with client.websocket_connect("/ws") as ws:
                self.receive_until(ws, "history")
                ws.send_json({"type": "speak", "text": "2岁属于0–4岁档，806元/年。", "speed": 1.0})
                self.receive_until(ws, "audio_chunk")
                self.receive_until(ws, "audio_end")
        tts.synthesize_stream.assert_called_once_with(
            "2岁属于0–4岁档，806元/年。",
            speed=1.0, cancel_event=ANY, stream=False,
        )

    def test_text_handoff_metadata_reaches_websocket_and_restored_history(self):
        from context_manager import ContextManager
        from rag import RAGClient, RetrievalResult, HANDOFF_MESSAGE
        from cooperative import CooperativeClient
        qwen, retriever = Mock(), Mock()
        qwen.stream_chat.side_effect = AssertionError("证据不足不得调用 Qwen")
        retriever.search.return_value = RetrievalResult(
            query="资料外问题", sufficient=False, threshold=0.47,
            lexical_threshold=0.08, elapsed_ms=1, results=(),
        )
        with TemporaryDirectory() as directory:
            context = ContextManager(CooperativeClient(RAGClient(qwen, retriever)), db_path=Path(directory) / "web.sqlite3")
            fake = SimpleNamespace(state="ready", context=context, close=context.close,
                                   interrupt=lambda session: context.interrupt(session))
            with patch.object(gateway, "runtime", fake), TestClient(app) as client:
                with client.websocket_connect("/ws") as ws:
                    self.receive_until(ws, "history")
                    ws.send_json({"type": "chat", "text": "资料外问题", "autoplay": False})
                    done = self.receive_until(ws, "turn_done")
                    self.assertIn(HANDOFF_MESSAGE, done["text"])
                    self.assertEqual(done["metadata"]["handoff"]["status"], "pending")
                    sessions = self.receive_until(ws, "sessions")
                    session = sessions["items"][0]["session_id"]
                    ws.send_json({"type": "load_session", "session_id": session})
                    history = self.receive_until(ws, "history")
                    self.assertEqual(history["turns"][0]["metadata"]["handoff"]["status"], "pending")
        qwen.stream_chat.assert_not_called()

    def test_auto_default_routes_clarification_then_evidence_answer(self):
        from context_manager import ContextManager
        from cooperative import CooperativeClient
        from rag import RAGClient, RetrievalResult, RetrievedChunk
        qwen, retriever = Mock(), Mock()
        qwen.stream_chat.side_effect = lambda *a, **k: iter(["测试回答"])
        retriever.search.side_effect = lambda query: RetrievalResult(
            query=query, sufficient=True, threshold=.47, lexical_threshold=.08, elapsed_ms=1,
            results=(RetrievedChunk(1, 1, 1, "测试资料", .8, .5),),
        )
        with TemporaryDirectory() as directory:
            context = ContextManager(CooperativeClient(RAGClient(qwen, retriever)), db_path=Path(directory) / "auto.sqlite3")
            fake = SimpleNamespace(state="ready", context=context, close=context.close,
                                   interrupt=lambda session: context.interrupt(session))
            with patch.object(gateway, "runtime", fake), TestClient(app) as client:
                with client.websocket_connect("/ws") as ws:
                    self.receive_until(ws, "history")
                    for text, pending in (("年缴多少钱", "age"), ("30", "social_insurance"), ("有社保", None)):
                        ws.send_json({"type": "chat", "text": text, "autoplay": False})
                        self.assertEqual(self.receive_until(ws, "turn_start")["mode"], "auto")
                        plan = self.receive_until(ws, "cooperation")["plan"]
                        self.assertEqual(plan["pending_slot"], pending)
                        done = self.receive_until(ws, "turn_done")
                        self.assertEqual(done["metadata"]["cooperation"]["requested_mode"], "auto")
                        self.receive_until(ws, "sessions")
        self.assertEqual(qwen.stream_chat.call_count, 1)
        self.assertEqual(qwen.stream_chat.call_args.kwargs["max_new_tokens"], 192)


class WebAppTests(unittest.TestCase):
    def test_health_and_frontend(self):
        with TestClient(app) as client:
            self.assertEqual(client.get("/healthz").status_code, 200)
            page = client.get("/")
            self.assertEqual(page.status_code, 200)
            self.assertIn("智能业务助手", page.text)
            self.assertIn("vendor/katex/katex.min.css", page.text)
            self.assertIn("vendor/marked/marked.umd.js", page.text)
            self.assertNotIn('value="continuous"', page.text)
            self.assertNotIn('id="voice-input-mode"', page.text)
            self.assertEqual(client.get("/static/pcm-worklet.js").status_code, 404)
            self.assertEqual(
                client.get("/static/vendor/katex/katex.min.js").status_code, 200,
            )

    def test_local_mode_does_not_require_password(self):
        with TestClient(app) as client:
            self.assertEqual(
                client.get("/api/auth").json(),
                {"required": False, "authenticated": True},
            )

    def test_thinking_modes(self):
        self.assertLess(
            _mode_options("fast")["max_new_tokens"],
            _mode_options("slow")["max_new_tokens"],
        )

    def test_browser_audio_is_converted_to_16k_mono_wav(self):
        source = BytesIO()
        with wave.open(source, "wb") as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(8000)
            output.writeframes(b"\x00\x00" * 800)
        converted = _decode_browser_audio(base64.b64encode(source.getvalue()).decode())
        with wave.open(BytesIO(converted), "rb") as result:
            self.assertEqual(result.getframerate(), 16000)
            self.assertEqual(result.getnchannels(), 1)


if __name__ == "__main__":
    unittest.main()
