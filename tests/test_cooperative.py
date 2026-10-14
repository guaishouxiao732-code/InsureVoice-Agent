"""Fast/slow orchestration regressions without GPU/model resources."""
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event
import unittest
from unittest.mock import Mock

from context_manager import ContextManager
from cooperative import CooperativeClient, _lookup_fee_rate, _previous_source_answer, plan_conversation
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
    def test_rate_table_lookup_uses_exact_age_band_and_payment_period(self):
        self.assertEqual(_lookup_fee_rate(7, "有社保", "年缴"), (279, 5, 10))
        self.assertEqual(_lookup_fee_rate(33, "无社保", "月缴"), (96, 31, 35))
        self.assertEqual(_lookup_fee_rate(71, "有社保", "年缴"), (3928, 71, 75))

    def test_rate_answer_is_deterministic_when_rate_table_is_retrieved(self):
        class RateRetriever(Retriever):
            def search(self, query):
                self.queries.append(query)
                item = RetrievedChunk(
                    9, 1, 1, "## 年缴费率 | 5-10岁 | 有社保 279元", .8, .5,
                    "尊享E生2026缴费率.pdf", "pdf", "费率表 第1页",
                )
                return RetrievalResult(query, True, .47, .08, 1, (item,))

        model, retriever = Model(), RateRetriever()
        client = CooperativeClient(RAGClient(model, retriever))
        response = client.stream_chat(messages("7岁有社保，尊享E生2026年缴多少钱？"))
        answer = "".join(response)
        self.assertIn("279元/年", answer)
        self.assertIn("5–10岁档", answer)
        self.assertEqual(model.calls, [])
        self.assertTrue(response.metadata["evidence_sources"])

    def test_prior_quote_condition_recap_keeps_premium_and_correct_rate_evidence(self):
        class MixedRetriever(Retriever):
            def search(self, query):
                self.queries.append(query)
                hits = (
                    RetrievedChunk(1, 1, 1,
                        "年缴费率（单位：元/年）\n| 年龄 | 有社保 | 无社保 |\n| 31-35岁 | 423 | 971 |",
                        .9, .8, "尊享E生百万医疗2026年缴费率.pdf", "pdf_transcription", "第1页费率表"),
                    RetrievedChunk(2, 1, 1,
                        "月缴费率（单位：元/月）\n| 31-35岁 | 42 | 96 |",
                        .8, .7, "尊享E生百万医疗2026月缴费率.pdf", "pdf_transcription", "第1页费率表"),
                    RetrievedChunk(3, 1, 1,
                        "以有基本医保身份参保，但未按医保身份结算时赔付比例为60%",
                        .7, .6, "投保须知.pdf", "pdf", "第1页赔付比例"),
                )
                return RetrievalResult(query, True, .47, .08, 1, hits)

        model, retriever = Model(), MixedRetriever()
        client = CooperativeClient(RAGClient(model, retriever))
        response = client.stream_chat(messages(
            "我30岁有基本医保，年缴保费是多少？",
            "不是30岁，而是35岁，其他条件不变。",
            "如果没有基本医保，保费是多少？",
            "再改回有基本医保。你先告诉我现在记录的年龄、医保情况和缴费方式分别是什么。",
        ), turn_history=[{"assistant_text": "年缴基础费率为971元/年。",
                          "metadata": {"cooperation": {"intent": "quote"}}}])
        answer = "".join(response)
        self.assertIn("年龄35岁", answer)
        self.assertIn("有基本医保", answer)
        self.assertIn("年缴", answer)
        self.assertIn("423元/年", answer)
        self.assertEqual(model.calls, [])
        self.assertIn("医疗保险保费报价", retriever.queries[0])
        self.assertEqual(len(response.metadata["evidence_sources"]), 1)
        evidence = response.metadata["evidence_sources"][0]
        self.assertIn("年缴费率.pdf", evidence["source_name"])
        self.assertIn("31-35岁 | 423 | 971", evidence["text"])
        self.assertNotIn("赔付比例", evidence["text"])
        self.assertEqual(response.metadata["cooperation"]["reason"], "quote_condition_recap")

    def test_condition_recap_without_prior_quote_does_not_add_rate(self):
        model, retriever = Model(), Retriever()
        client = CooperativeClient(RAGClient(model, retriever))
        response = client.stream_chat(messages(
            "我35岁，有基本医保，按年缴费。",
            "现在记录的年龄、医保情况和缴费方式分别是什么？",
        ))
        self.assertNotIn("423元", "".join(response))
        self.assertNotIn("exact_rate_lookup", response.metadata["cooperation"]["stages"])

    def test_prior_quote_recap_without_rate_evidence_hands_off(self):
        model, retriever = Model(), Retriever(False)
        client = CooperativeClient(RAGClient(model, retriever))
        response = client.stream_chat(messages(
            "我35岁有基本医保，年缴保费是多少？",
            "现在记录的年龄、医保情况和缴费方式分别是什么？",
        ), turn_history=[{"assistant_text": "年缴基础费率为423元/年。",
                          "metadata": {"cooperation": {"intent": "quote"}}}])
        self.assertEqual(response.metadata["decision"], "handoff")
        self.assertEqual(response.metadata.get("evidence_sources", []), [])
        self.assertIn("转人工", "".join(response))
        self.assertEqual(model.calls, [])

    def test_quote_recap_survives_intervening_service_and_handoff_turns(self):
        class RateRetriever(Retriever):
            def search(self, query):
                self.queries.append(query)
                item = RetrievedChunk(
                    7, 1, 1, "年缴费率（单位：元/年）\n| 31-35岁 | 423 | 971 |",
                    .8, .6, "尊享E生百万医疗2026年缴费率.pdf",
                    "pdf_transcription", "第1页费率表",
                )
                return RetrievalResult(query, True, .47, .08, 1, (item,))

        model, retriever = Model(), RateRetriever()
        client = CooperativeClient(RAGClient(model, retriever))
        history = [
            {"assistant_text": "971元/年", "metadata": {"cooperation": {"intent": "quote"}}},
            {"assistant_text": "我是 AI 保险业务助手", "metadata": {"cooperation": {"intent": "service"}}},
            {"assistant_text": "需转人工", "metadata": {"decision": "handoff",
                                          "cooperation": {"intent": "general"}}},
        ]
        response = client.stream_chat(messages(
            "我35岁，没有基本医保，年缴多少钱？",
            "你好，你是谁？",
            "再改回有基本医保。你先告诉我现在记录的年龄、医保情况和缴费方式分别是什么。",
        ), turn_history=history)
        self.assertIn("423元/年", "".join(response))
        self.assertEqual(response.metadata["cooperation"]["reason"], "quote_condition_recap")
        self.assertEqual(len(response.metadata["evidence_sources"]), 1)
        self.assertEqual(model.calls, [])

    def test_fee_source_followup_uses_persisted_previous_answer_citation(self):
        class RateRetriever(Retriever):
            def search(self, query):
                self.queries.append(query)
                hits = tuple(RetrievedChunk(
                    number, 1, 1, "费率表", .8, .5,
                    f"尊享E生百万医疗2026{period}费率.pdf", "pdf_transcription", "第1页费率表",
                ) for number, period in ((1, "年缴"), (2, "月缴")))
                return RetrievalResult(query, True, .47, .08, 1, hits)

        model, retriever = Model(), RateRetriever()
        client = CooperativeClient(RAGClient(model, retriever))
        with TemporaryDirectory() as directory:
            path = Path(directory) / "history.sqlite3"
            with ContextManager(client, db_path=path) as context:
                session = context.create_session()
                quote = context.chat(session, "我30岁有社保，年缴基础保费是多少？", thinking_mode="auto")
                self.assertIn("318元/年", quote)
                first = context.get_history(session)[0]
                self.assertEqual(len(first["metadata"]["cited_sources"]), 1)
                self.assertIn("年缴费率.pdf", first["metadata"]["cited_sources"][0])
            with ContextManager(client, db_path=path) as context:
                answer = context.chat(session, "你刚才给出的保费依据哪份资料？具体出处是什么？", thinking_mode="auto")
                self.assertIn("年缴费率.pdf", answer)
                self.assertNotIn("月缴费率.pdf", answer)
                self.assertNotIn("待人工", answer)
                self.assertEqual(context.get_history(session)[-1]["metadata"]["decision"], "answer")
        self.assertEqual(model.calls, [])
        self.assertEqual(len(retriever.queries), 1)

    def test_source_followup_without_prior_citation_does_not_invent_one(self):
        model, retriever = Model(), Retriever()
        client = CooperativeClient(RAGClient(model, retriever))
        response = client.stream_chat(messages("你刚才的报价依据是什么？"), turn_history=[{
            "assistant_text": "基础费率为318元/年。", "metadata": {"decision": "answer"},
        }])
        self.assertEqual(response.metadata["decision"], "handoff")
        self.assertEqual(response.metadata["cited_sources"], [])
        self.assertIn("不能推测", "".join(response))
        self.assertEqual(model.calls, [])
        self.assertEqual(retriever.queries, [])

    def test_old_premium_answer_cannot_cite_unrelated_policy_clause(self):
        history = [{"assistant_text": "对应保费为423元/年。", "metadata": {
            "decision": "answer",
            "cited_sources": ["来源：投保须知.pdf，第1页医保赔付比例"],
            "evidence_sources": [{"source_name": "投保须知.pdf", "text": "医保赔付比例为60%"}],
        }}]
        self.assertIsNone(_previous_source_answer(history))

    def test_previous_source_collapses_two_locations_on_same_pdf_page(self):
        history = [{"assistant_text": "年缴基础费率为318元/年。", "metadata": {
            "decision": "answer", "cited_sources": [
                "来源：尊享E生百万医疗2026年缴费率.pdf，第1页费率表（人工核对转录）",
                "来源：尊享E生百万医疗2026年缴费率.pdf，第1页",
                "来源：尊享E生百万医疗2026月缴费率.pdf，第1页",
            ],
        }}]
        answer, metadata = _previous_source_answer(history)
        self.assertEqual(answer.count("年缴费率.pdf"), 1)
        self.assertNotIn("月缴费率.pdf", answer)
        self.assertEqual(len(metadata["cited_sources"]), 1)

    def test_retry_after_old_source_handoff_uses_original_quote(self):
        history = [
            {"user_text": "我30岁有社保，年缴多少钱？", "assistant_text": "基础费率为318元/年。",
             "metadata": {"decision": "answer", "cited_sources": [
                 "来源：尊享E生百万医疗2026年缴费率.pdf，第1页",
             ]}},
            {"user_text": "刚才的保费依据是什么？", "assistant_text": "需转人工处理。",
             "metadata": {"decision": "handoff"}},
        ]
        result = _previous_source_answer(history)
        self.assertIsNotNone(result)
        self.assertIn("年缴费率.pdf", result[0])

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

    def test_heldout_paraphrases_do_not_prompt_for_known_conditions(self):
        quote = plan_conversation(messages("我今年35岁，没有基本医保，月付这份保险要多少钱？"))
        self.assertEqual(quote.intent, "quote")
        self.assertIsNone(quote.pending_slot)
        self.assertEqual(quote.slots["social_insurance"]["value"], "无社保")
        self.assertEqual(quote.slots["payment"]["value"], "月缴")
        for question in (
            "如果我在加拿大医院接受质子治疗，能直接按先进疗法责任申请赔付吗？",
            "特药处方一次开40天，超出30天的那10天药量一定能报吗？",
            "牙齿因意外受伤要治疗，能直接说所有牙科治疗都免责吗？",
        ):
            with self.subTest(question=question):
                plan = plan_conversation(messages(question))
                self.assertEqual(plan.intent, "general")
                self.assertIsNone(plan.pending_slot)

    def test_explicit_underwriting_guarantee_hands_off_without_age_prompt(self):
        plan = plan_conversation(messages("我有高血压但其他信息未提供，能不能保证这次投保必然通过？"))
        self.assertEqual(plan.decision, "handoff")
        self.assertEqual(plan.reason, "individual_underwriting_unavailable")

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
        contextual = plan_conversation(messages(
            "我35岁有基本医保，尊享E生一般医疗责任的免赔额是多少？",
            "根据前文，我现在的条件下这个免赔额怎么算？",
        ))
        self.assertIn("一般医疗责任", contextual.query)
        self.assertIn("35岁", contextual.query)
        self.assertIn("有社保", contextual.query)
        plan = plan_conversation(messages("30岁和65岁年缴保费各多少钱"))
        self.assertIsNone(plan.pending_slot)
        self.assertEqual(plan.resolved_mode, "slow")

    def test_retrieval_query_expands_hospital_scope_and_bnct_terms(self):
        hospital = plan_conversation(messages("一般医疗赔付需要满足怎样的医院等级和科室范围？"))
        self.assertIn("二级及二级以上公立医院", hospital.query)
        bnct = plan_conversation(messages("BNCT责任的指定医院有哪些？"))
        self.assertIn("硼中子俘获治疗", bnct.query)

    def test_specific_hospital_membership_question_hands_off_without_guessing(self):
        model, retriever = Model(), Retriever()
        response = CooperativeClient(RAGClient(model, retriever)).stream_chat(
            messages("资料没有列出某个具体医院是否属于指定民营医疗机构，能否确认？")
        )
        self.assertEqual(response.metadata["handoff"]["reason"], "specific_hospital_list_unavailable")
        self.assertIn("转人工", "".join(response))
        self.assertEqual(model.calls, [])
        self.assertEqual(retriever.queries, [])

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
