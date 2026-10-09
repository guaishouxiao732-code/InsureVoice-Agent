"""Fast/slow orchestration regressions without GPU/model resources."""
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event
import unittest
from unittest.mock import Mock

from context_manager import ContextManager
from cooperative import CooperativeClient, plan_conversation
from rag import RAGClient, RetrievalResult, RetrievedChunk


def messages(*texts):
    return [{"role": "user", "content": text} for text in texts]


class Retriever:
    def __init__(self, sufficient=True):
        self.sufficient, self.queries = sufficient, []

    def search(self, query):
        self.queries.append(query)
        return RetrievalResult(query, self.sufficient, .47, .08, 1,
            (RetrievedChunk(1, 1, 1, "测试保险资料，不构成实际业务规则。", .8, .5),) if self.sufficient else ())


class Model:
    def __init__(self):
        self.calls = []

    def stream_chat(self, messages, **options):
        self.calls.append((deepcopy(messages), options))
        return iter(["依据资料的测试回答"])


class CooperativeTests(unittest.TestCase):
    def test_ambiguous_health_question_asks_one_question_without_model_or_retrieval(self):
        model, retriever = Model(), Retriever()
        client = CooperativeClient(RAGClient(model, retriever))
        response = client.stream_chat(messages("有高血压，住院能赔吗"))
        self.assertEqual(response.metadata["decision"], "clarify")
        text = "".join(response)
        self.assertEqual(text.count("？"), 1)
        self.assertIn("能否投保", text)
        self.assertIn("住院理赔", text)
        self.assertEqual(model.calls, [])
        self.assertEqual(retriever.queries, [])

    def test_clarify_then_resume_slow_analysis_in_same_session(self):
        model = Model()
        client = CooperativeClient(RAGClient(model, Retriever()))
        with TemporaryDirectory() as directory:
            path = Path(directory) / "history.sqlite3"
            with ContextManager(client, db_path=path) as context:
                session = context.create_session()
                self.assertIn("投保", context.chat(session, "我65岁，有高血压，住院能赔吗", thinking_mode="auto"))
                self.assertIn("原因", context.chat(session, "理赔", thinking_mode="auto"))
                self.assertIn("测试回答", context.chat(session, "因为摔伤住院", thinking_mode="auto"))
                history = context.get_history(session)
            with ContextManager(client, db_path=path) as context:
                self.assertEqual(context.get_history(session)[-1]["metadata"]["cooperation"]["slots"]["age"]["value"], 65)
        self.assertEqual(len(model.calls), 1)
        self.assertEqual(history[-1]["metadata"]["cooperation"]["resolved_mode"], "slow")
        self.assertEqual(history[-1]["metadata"]["cooperation"]["context_version"], 3)
        self.assertEqual(model.calls[0][1]["max_new_tokens"], 512)

    def test_quote_conditions_are_requested_one_at_a_time(self):
        sequence = ["年缴多少钱", "30", "有社保"]
        self.assertEqual(plan_conversation(messages(sequence[0])).pending_slot, "age")
        self.assertEqual(plan_conversation(messages(*sequence[:2])).pending_slot, "social_insurance")
        plan = plan_conversation(messages(*sequence))
        self.assertEqual(plan.decision, "answer")
        self.assertEqual(plan.resolved_mode, "fast")
        self.assertIn("30岁", plan.query)
        self.assertIn("有社保", plan.query)

    def test_person_change_does_not_transfer_age_social_insurance_or_health(self):
        plan = plan_conversation(messages("我30岁有社保有高血压，年缴多少钱", "不是我，是我父亲，65岁"))
        self.assertEqual(plan.intent, "quote")
        self.assertEqual(plan.slots["age"]["value"], 65)
        self.assertEqual(plan.slots["insured_person"]["value"], "父亲")
        self.assertNotIn("social_insurance", plan.slots)
        self.assertNotIn("health", plan.slots)
        self.assertEqual(plan.pending_slot, "social_insurance")
        self.assertNotIn("30岁", plan.query)

    def test_latest_correction_and_payment_followup_drive_query(self):
        plan = plan_conversation(messages("我30岁有社保，年缴多少钱", "不是30岁，是35岁", "那月缴呢"))
        self.assertEqual(plan.slots["age"]["value"], 35)
        self.assertEqual(plan.slots["payment"]["value"], "月缴")
        self.assertIn("35岁", plan.query)
        self.assertNotIn("30岁", plan.query)

    def test_unknown_condition_hands_off_instead_of_repeating_question(self):
        model = Model()
        client = CooperativeClient(RAGClient(model, Retriever()))
        response = client.stream_chat(messages("年缴多少钱", "不知道年龄"))
        self.assertEqual(response.metadata["decision"], "handoff")
        self.assertEqual(response.metadata["handoff"]["reason"], "missing_user_conditions")
        self.assertIn("待人工处理", "".join(response))
        self.assertEqual(model.calls, [])

    def test_claim_progress_never_fabricates_business_records(self):
        qwen, retriever = Mock(), Mock()
        response = CooperativeClient(RAGClient(qwen, retriever)).stream_chat(messages("我的理赔进度到哪一步了"))
        self.assertEqual(response.metadata["handoff"]["reason"], "business_record_unavailable")
        self.assertIn("未接入授权理赔记录", "".join(response))
        qwen.stream_chat.assert_not_called()
        retriever.search.assert_not_called()

    def test_simple_and_complex_tasks_share_one_model_with_different_budgets(self):
        model = Model()
        client = CooperativeClient(RAGClient(model, Retriever()))
        simple = client.stream_chat(messages("等待期是什么"))
        complex_stream = client.stream_chat(messages("比较等待期和免赔额的区别"))
        self.assertEqual(simple.metadata["cooperation"]["resolved_mode"], "fast")
        self.assertEqual(complex_stream.metadata["cooperation"]["resolved_mode"], "slow")
        self.assertEqual([c[1]["max_new_tokens"] for c in model.calls], [192, 512])
        self.assertTrue(all("thinking_mode" not in c[1] for c in model.calls))
        simple.close()
        complex_stream.close()

    def test_manual_modes_do_not_bypass_missing_conditions_or_evidence(self):
        model = Model()
        client = CooperativeClient(RAGClient(model, Retriever(False)))
        for mode in ("fast", "slow"):
            response = client.stream_chat(messages("年缴多少钱"), thinking_mode=mode)
            self.assertEqual(response.metadata["decision"], "clarify")
            response = client.stream_chat(messages("我30岁有社保，年缴多少钱"), thinking_mode=mode)
            self.assertEqual(response.metadata["handoff"]["status"], "pending")
        self.assertEqual(model.calls, [])

    def test_cancelled_static_response_and_no_session_state_leak(self):
        cancel = Event()
        cancel.set()
        client = CooperativeClient(RAGClient(Model(), Retriever()))
        self.assertEqual(list(client.stream_chat(messages("你好"), cancel_event=cancel)), [])
        with TemporaryDirectory() as directory, ContextManager(client, db_path=Path(directory) / "isolated.sqlite3") as context:
            alice, bob = context.create_session(), context.create_session()
            context.chat(alice, "我30岁有社保，年缴多少钱")
            self.assertIn("多少岁", context.chat(bob, "年缴多少钱"))
            self.assertNotIn("age", context.get_history(bob)[0]["metadata"]["cooperation"]["slots"])

    def test_conflicting_ages_are_not_silently_selected(self):
        plan = plan_conversation(messages("我30岁，父亲65岁，年缴多少钱"))
        self.assertEqual(plan.slots["age"]["status"], "conflict")
        self.assertEqual(plan.pending_slot, "age")

    def test_negated_health_condition_is_not_carried_forward(self):
        plan = plan_conversation(messages("我65岁有高血压能买保险吗", "我没有高血压"))
        self.assertIsNone(plan.slots["health"]["value"])
        self.assertEqual(plan.slots["health_denials"]["value"], "高血压")

    def test_general_followup_and_comparison_are_not_lost(self):
        plan = plan_conversation(messages("等待期是什么", "它怎么算"))
        self.assertIn("等待期是什么", plan.query)
        plan = plan_conversation(messages("30岁和65岁年缴保费各多少钱"))
        self.assertIsNone(plan.pending_slot)
        self.assertEqual(plan.resolved_mode, "slow")

    def test_preview_does_not_commit_or_ask_clarification(self):
        retriever = Retriever()
        client = CooperativeClient(RAGClient(Model(), retriever))
        client.retrieve(messages("我30岁有社保年缴多少钱"))
        plan = plan_conversation(messages("年缴多少钱"))
        self.assertEqual(plan.pending_slot, "age")

    def test_chinese_age_and_service_identity(self):
        plan = plan_conversation(messages("我三十五岁有社保年缴多少钱"))
        self.assertEqual(plan.slots["age"]["value"], 35)
        response = CooperativeClient(RAGClient(Mock(), Mock())).stream_chat(messages("你是人工客服吗"))
        self.assertIn("AI", "".join(response))
        self.assertEqual(response.metadata["knowledge_mode"], "service")

    def test_traditional_corrections_short_replies_and_age_negation(self):
        plan = plan_conversation(messages("我30歲有社保年繳多少錢", "不是我，是我父親65歲", "沒有"))
        self.assertEqual(plan.slots["age"]["value"], 65)
        self.assertEqual(plan.slots["social_insurance"]["value"], "无社保")
        self.assertEqual(plan.slots["insured_person"]["value"], "父亲")
        plan = plan_conversation(messages("我30岁有社保年缴多少钱", "我不是30岁"))
        self.assertIsNone(plan.slots["age"]["value"])
        self.assertEqual(plan.pending_slot, "age")

    def test_hypothetical_conditions_are_not_used_as_actual_user_facts(self):
        plan = plan_conversation(messages("假设我30岁有社保年缴多少钱", "我35岁，年缴多少钱"))
        self.assertNotIn("social_insurance", plan.slots)
        self.assertEqual(plan.pending_slot, "social_insurance")

    def test_interruption_during_retrieval_does_not_launch_model(self):
        started, release = Event(), Event()
        class BlockingRetriever(Retriever):
            def search(self, query):
                started.set()
                if not release.wait(3):
                    raise TimeoutError("test retrieval timeout")
                return super().search(query)
        model = Model()
        client = CooperativeClient(RAGClient(model, BlockingRetriever()))
        with TemporaryDirectory() as directory, ContextManager(client, db_path=Path(directory) / "cancel.sqlite3") as context:
            session = context.create_session()
            old = context.stream_chat(session, "我30岁有社保年缴多少钱")
            self.assertTrue(started.wait(2))
            with ThreadPoolExecutor(max_workers=1) as pool:
                cancelling = pool.submit(context.interrupt, session)
                self.assertTrue(old._stop.wait(2))
                release.set()
                self.assertTrue(cancelling.result(timeout=2))
            self.assertEqual(old.status, "interrupted")
            self.assertEqual(list(old), [])
            self.assertEqual(model.calls, [])
            self.assertIn("您好", context.chat(session, "你好"))


if __name__ == "__main__":
    unittest.main()
