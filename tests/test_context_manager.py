"""Deterministic conversation lifecycle tests; no model/GPU required."""

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
from threading import Event, Lock
import unittest

from context_manager import ContextManager


class FakeClient:
    def __init__(self):
        self.calls = []
        self.lock = Lock()
        self.buffered = Event()
        self.stopped = Event()

    def stream_chat(self, messages, *, cancel_event, **options):
        with self.lock:
            self.calls.append((deepcopy(messages), options))
        prompt = messages[-1]["content"]
        if prompt == "long":
            try:
                yield "已展示。"
                yield "尚未展示，不能加入历史。"
                self.buffered.set()
                if not cancel_event.wait(5):
                    raise TimeoutError("test cancellation did not arrive")
            finally:
                self.stopped.set()
        elif prompt == "fail":
            raise RuntimeError("simulated model failure")
        elif prompt == "empty":
            return
        else:
            yield "答："
            yield prompt


class ContextTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.path = Path(self.tmp.name) / "history.sqlite3"
        self.client = FakeClient()
        self.manager = ContextManager(self.client, db_path=self.path)
        self.sid = self.manager.create_session(system_prompt="测试助手", session_id="alice")

    def tearDown(self):
        self.manager.close()
        self.tmp.cleanup()

    def test_history_is_automatic_and_survives_restart(self):
        self.assertEqual(self.manager.chat(self.sid, "我叫小明"), "答：我叫小明")
        self.manager.close()
        self.manager = ContextManager(self.client, db_path=self.path)
        self.manager.chat(self.sid, "我叫什么？")
        messages, _ = self.client.calls[-1]
        self.assertEqual(messages, [
            {"role": "system", "content": "测试助手"},
            {"role": "user", "content": "我叫小明"},
            {"role": "assistant", "content": "答：我叫小明"},
            {"role": "user", "content": "我叫什么？"},
        ])
        history = self.manager.get_history(self.sid)
        self.assertEqual([t["status"] for t in history], ["completed", "completed"])
        self.assertGreaterEqual(history[0]["first_chunk_ms"], 0)

    def test_new_request_cancels_paused_consumer_and_discards_buffer(self):
        old = self.manager.stream_chat(self.sid, "long")
        self.assertEqual(next(old), "已展示。")
        self.assertTrue(self.client.buffered.wait(2))
        new = self.manager.stream_chat(self.sid, "先说价格")
        self.assertEqual("".join(new), "答：先说价格")
        self.assertTrue(self.client.stopped.is_set())
        self.assertEqual(list(old), [])
        self.assertEqual(old.status, "interrupted")
        history = self.manager.get_history(self.sid)
        self.assertEqual(history[0]["assistant_text"], "已展示。")
        self.assertEqual(history[0]["status"], "interrupted")
        prompt = str(self.client.calls[-1][0])
        self.assertIn("已展示。", prompt)
        self.assertIn("被打断", prompt)
        self.assertNotIn("尚未展示", prompt)

    def test_interrupt_before_reading_saves_no_assistant_text(self):
        response = self.manager.stream_chat(self.sid, "long")
        self.assertTrue(self.client.buffered.wait(2))
        self.assertTrue(self.manager.interrupt(self.sid))
        self.assertFalse(self.manager.interrupt(self.sid))
        self.assertEqual(response.text, "")
        self.assertEqual(self.manager.get_history(self.sid)[0]["assistant_text"], "")

    def test_context_exit_cancels_early_break(self):
        with self.manager.stream_chat(self.sid, "long") as response:
            self.assertEqual(next(response), "已展示。")
        self.assertEqual(response.status, "interrupted")
        self.assertTrue(self.client.stopped.is_set())

    def test_sessions_remain_isolated_with_concurrent_calls(self):
        bob = self.manager.create_session(session_id="bob")
        with ThreadPoolExecutor(max_workers=2) as pool:
            alice_result = pool.submit(self.manager.chat, self.sid, "Alice-secret")
            bob_result = pool.submit(self.manager.chat, bob, "Bob-secret")
            self.assertIn("Alice-secret", alice_result.result(timeout=2))
            self.assertIn("Bob-secret", bob_result.result(timeout=2))
        self.manager.chat(bob, "追问")
        messages = str(self.client.calls[-1][0])
        self.assertIn("Bob-secret", messages)
        self.assertNotIn("Alice-secret", messages)

    def test_failure_is_persisted_and_does_not_block_next_turn(self):
        with self.assertRaisesRegex(RuntimeError, "generation failed"):
            self.manager.chat(self.sid, "fail")
        failed = self.manager.get_history(self.sid)[0]
        self.assertEqual(failed["status"], "failed")
        self.assertIn("simulated model failure", failed["error"])
        self.assertEqual(self.manager.chat(self.sid, "继续"), "答：继续")

    def test_tools_and_token_budget_pass_through(self):
        tools = [{"type": "function", "function": {"name": "ping"}}]
        self.manager.chat(self.sid, "test", tools=tools, max_input_tokens=1024)
        self.assertEqual(self.client.calls[-1][1], {"tools": tools, "max_input_tokens": 1024})

    def test_clear_only_affects_selected_session(self):
        bob = self.manager.create_session(session_id="bob")
        self.manager.chat(bob, "keep")
        self.manager.chat(self.sid, "drop")
        self.manager.clear_history(self.sid)
        self.assertEqual(self.manager.get_history(self.sid), [])
        self.assertEqual(len(self.manager.get_history(bob)), 1)
        self.manager.chat(self.sid, "fresh")
        self.assertEqual(len(self.client.calls[-1][0]), 2)
        self.assertEqual(self.client.calls[-1][0][0]["content"], "测试助手")

    def test_delete_session_removes_session_and_turns(self):
        bob = self.manager.create_session(session_id="bob")
        self.manager.chat(self.sid, "delete me")
        self.manager.chat(bob, "keep me")
        self.manager.delete_session(self.sid)
        self.assertEqual(
            [item["session_id"] for item in self.manager.list_sessions()], ["bob"],
        )
        self.assertEqual(len(self.manager.get_history(bob)), 1)
        with self.assertRaises(KeyError):
            self.manager.get_history(self.sid)

    def test_duplicate_owner_is_rejected_without_damaging_history(self):
        with self.assertRaisesRegex(RuntimeError, "active ContextManager"):
            ContextManager(self.client, db_path=self.path)
        self.assertEqual(self.manager.chat(self.sid, "ok"), "答：ok")

    def test_crash_recovery_marks_unfinished_turn_interrupted(self):
        self.manager.chat(self.sid, "persisted")
        self.manager.close()
        # Simulate the committed state left behind by a terminated process.
        with sqlite3.connect(self.path) as db:
            db.execute("UPDATE context_turns SET status='generating', finished_at=NULL")
        self.manager = ContextManager(self.client, db_path=self.path)
        turn = self.manager.get_history(self.sid)[0]
        self.assertEqual(turn["status"], "interrupted")
        self.assertEqual(turn["assistant_text"], "答：persisted")
        self.assertIsNotNone(turn["finished_at"])

    def test_validation_does_not_create_invalid_turns(self):
        with self.assertRaises(ValueError):
            self.manager.create_session(session_id=self.sid)
        with self.assertRaises(KeyError):
            self.manager.chat("missing", "hello")
        with self.assertRaises(ValueError):
            self.manager.chat(self.sid, "  ")
        with self.assertRaises(TypeError):
            self.manager.chat(self.sid, "hello", cancel_event=Event())
        self.assertEqual(self.manager.get_history(self.sid), [])

    def test_empty_response_and_closed_manager(self):
        self.assertEqual(self.manager.chat(self.sid, "empty"), "")
        self.assertEqual(self.manager.get_history(self.sid)[0]["status"], "completed")
        self.manager.close()
        self.manager.close()
        with self.assertRaisesRegex(RuntimeError, "closed"):
            self.manager.chat(self.sid, "test")


if __name__ == "__main__":
    unittest.main()
