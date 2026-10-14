"""Real-browser checks for the read-only answer-evidence highlighter."""

from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
import unittest

from playwright.sync_api import sync_playwright


WEB_DIR = Path(__file__).resolve().parent


class QuietStaticHandler(SimpleHTTPRequestHandler):
    def log_message(self, *_args):
        pass


class EvidenceHighlightTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(
            ("127.0.0.1", 0), partial(QuietStaticHandler, directory=str(WEB_DIR))
        )
        cls.thread = Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.playwright = sync_playwright().start()
        cls.browser = cls.playwright.chromium.launch(args=["--no-sandbox"])

    @classmethod
    def tearDownClass(cls):
        cls.browser.close()
        cls.playwright.stop()
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=2)

    def setUp(self):
        self.page = self.browser.new_page(viewport={"width": 390, "height": 844})
        self.errors = []
        self.page.on("pageerror", lambda error: self.errors.append(str(error)))
        self.page.goto(f"http://127.0.0.1:{self.server.server_port}/", wait_until="domcontentloaded")
        self.page.wait_for_function("typeof renderEvidence === 'function'")
        self.page.add_style_tag(content="*, *::before, *::after { transition: none !important; animation: none !important; }")

    def tearDown(self):
        self.assertFalse(self.errors, self.errors)
        self.page.close()

    def test_question_terms_manual_search_navigation_and_literal_text(self):
        tail = "年缴费率为318元，保费以合同为准。<img src=x onerror=alert(1)>"
        source_text = "前文" * 190 + tail
        self.page.evaluate("""source => {
          createTurn('年缴费率是多少？', 'auto', {complete:true, text:'请查看依据',
            metadata:{knowledge_mode:'rag', evidence_sources:[{
              source_name:'2026年缴费率表', location:'第2页', score:0.91, text:source
            }]}});
        }""", source_text)
        self.page.locator('.turn [data-action="sources"]').click()
        panel = self.page.locator("#evidence-content")
        self.assertEqual(panel.locator(".source-card p").inner_text(), source_text)
        self.assertGreater(panel.locator("mark.evidence-highlight").count(), 0)
        self.assertEqual(panel.locator(".source-card img").count(), 0)
        self.assertTrue(panel.locator(".evidence-search input").is_visible())
        self.page.wait_for_function("""() => {
          const panel = document.querySelector('#evidence-panel').getBoundingClientRect();
          const input = document.querySelector('#evidence-content .evidence-search input').getBoundingClientRect();
          return input.x >= panel.x - 1;
        }""")
        panel_box = self.page.locator("#evidence-panel").bounding_box()
        input_box = panel.locator(".evidence-search input").bounding_box()
        button_box = panel.locator(".evidence-search button").bounding_box()
        self.assertGreaterEqual(input_box["x"], panel_box["x"])
        self.assertLessEqual(button_box["x"] + button_box["width"], panel_box["x"] + panel_box["width"])

        search = panel.locator(".evidence-search input")
        search.fill("318")
        self.assertEqual(panel.locator("mark.evidence-highlight").all_inner_texts(), ["318"])
        self.assertIn("找到 1 处", panel.locator(".evidence-search small").inner_text())
        panel.locator(".evidence-search button").click()
        self.assertEqual(panel.locator("mark.evidence-highlight.active").inner_text(), "318")

        search.fill("不存在")
        self.assertEqual(panel.locator("mark.evidence-highlight").count(), 0)
        self.assertIn("未找到", panel.locator(".evidence-search small").inner_text())
        search.fill("")
        self.assertGreater(panel.locator("mark.evidence-highlight").count(), 0)

    def test_no_sources_has_no_search_control(self):
        self.page.evaluate("renderEvidence({knowledge_mode:'handoff', sources:[]}, '保费是多少？')")
        panel = self.page.locator("#evidence-content")
        self.assertEqual(panel.locator(".evidence-search").count(), 0)
        self.assertIn("没有可展示", panel.inner_text())

    def test_followup_highlight_uses_effective_retrieval_query(self):
        self.page.evaluate("""renderEvidence({knowledge_mode:'rag',
          query:'一般医疗责任免赔额 35岁 有社保',
          evidence_sources:[{source_name:'保障责任.xlsx',location:'第1行',score:0.81,
            text:'一般医疗责任免赔额为一万元。'}]
        }, '那这个呢？', '一般医疗责任免赔额为一万元。')""")
        panel = self.page.locator("#evidence-content")
        self.assertIn("一般医疗", "".join(panel.locator("mark.evidence-highlight").all_inner_texts()))
        self.assertIn("有效上下文", panel.locator(".evidence-search small").inner_text())

    def test_claim_source_pair_and_unverified_claim_are_visible(self):
        self.page.evaluate("""renderEvidence({knowledge_mode:'rag',
          evidence_audit:{matched_claims:1,unverified_claims:1},
          evidence_claims:[
            {claim:'一般医疗扣除一万元免赔额。',status:'source_match',source_ids:[31]},
            {claim:'该责任必然赔付五百万元。',status:'unverified',source_ids:[]}
          ], evidence_sources:[{chunk_id:31,source_name:'保障责任.xlsx',location:'第8行',score:0.8,
            text:'一般医疗扣除一万元免赔额后给付。'}]
        }, '一般医疗如何赔付？', '一般医疗扣除一万元免赔额。该责任必然赔付五百万元。')""")
        panel = self.page.locator('#evidence-content')
        self.assertEqual(panel.locator('.evidence-claim').count(), 2)
        self.assertIn('待人工核实', panel.locator('.evidence-claim.unverified').inner_text())
        self.assertEqual(panel.locator('.source-card').count(), 1)
        panel.locator('.evidence-claim.matched button').click()

    def test_unverified_claim_count_is_visible_on_answer_card(self):
        self.page.evaluate("""createTurn('问题', 'auto', {complete:true, text:'待核实回答',
          metadata:{evidence_audit:{unverified_claims:2}}})""")
        self.assertIn('2条待核实', self.page.locator('.turn [data-action="sources"]').last.inner_text())

    def test_joint_rule_inference_is_labeled_and_links_both_rows(self):
        self.page.evaluate("""renderEvidence({knowledge_mode:'rag',
          evidence_audit:{matched_claims:1,unverified_claims:0},
          evidence_claims:[{claim:'其他情况按100%给付',status:'source_match',
            support_kind:'base_exception_inference',source_ids:[305,306]}],
          evidence_sources:[
            {chunk_id:305,source_name:'保障责任.xlsx',location:'第8行',text:'一般规则按100%给付'},
            {chunk_id:306,source_name:'保障责任.xlsx',location:'第9行',text:'特定条件按60%给付'}]
        }, '赔付比例是多少？', '其他情况按100%给付')""")
        claim = self.page.locator('.evidence-claim.matched')
        self.assertIn('一般规则与例外规则推导', claim.inner_text())
        self.assertEqual(claim.locator('button').count(), 2)

    def test_exact_rate_answer_shows_only_matching_rate_row(self):
        self.page.evaluate("""renderEvidence({
          knowledge_mode:'rag', retrieval_ms:46, top_score:0.71,
          cooperation:{stages:['fast:intent_context','fast:exact_rate_lookup'],
            slots:{age:{value:35,status:'explicit'},
              social_insurance:{value:'有社保',status:'explicit'},
              payment:{value:'年缴',status:'explicit'}}, context_version:4},
          sources:[{source_name:'投保须知.pdf',text:'医保身份影响赔付比例'}],
          evidence_sources:[{source_name:'2026年缴费率.pdf',location:'第1页费率表',score:0.71,
            text:'年缴费率（单位：元/年）\\n| 年龄 | 有社保 | 无社保 |\\n| 31-35岁 | 423 | 971 |'}]
        }, '现在记录的年龄、医保情况和缴费方式分别是什么？',
           '当前记录年龄35岁，有基本医保，年缴基础费率为423元/年。')""")
        panel = self.page.locator("#evidence-content")
        self.assertIn("按当前记录的用户条件", panel.locator(".evidence-summary").first.inner_text())
        self.assertEqual(panel.locator(".source-card").count(), 1)
        self.assertIn("2026年缴费率.pdf", panel.locator(".source-card header").inner_text())
        self.assertIn("31-35岁 | 423 | 971", panel.locator(".source-card p").inner_text())
        self.assertNotIn("赔付比例", panel.inner_text())

    def test_old_premium_without_matching_table_is_not_misrepresented(self):
        self.page.evaluate("""renderEvidence({knowledge_mode:'rag',
          evidence_sources:[{source_name:'投保须知.pdf',location:'第1页',score:0.65,
            text:'有基本医保但未按医保结算时，赔付比例调整为60%'}]
        }, '当前记录的条件是什么？', '年龄35岁，年缴基础保费为423元/年。')""")
        panel = self.page.locator("#evidence-content")
        self.assertEqual(panel.locator(".source-card").count(), 0)
        self.assertIn("不能证明该金额", panel.inner_text())
        self.assertIn("未找到与该保费匹配", panel.inner_text())

    def test_desktop_multiple_keywords_and_next_match(self):
        self.page.set_viewport_size({"width": 1280, "height": 800})
        self.page.evaluate("""renderEvidence({knowledge_mode:'rag', evidence_sources:[
          {source_name:'年缴费率表', location:'第1页', text:'保费标准和费率见本页。'},
          {source_name:'产品条款', location:'第2页', text:'保费由费率表和核保结果决定。'}
        ]}, '保费怎么计算？')""")
        panel = self.page.locator("#evidence-content")
        panel.locator(".evidence-search input").fill("保费，费率")
        marks = panel.locator("mark.evidence-highlight")
        self.assertGreaterEqual(marks.count(), 4)
        self.assertIn("保费标准和费率见本页", panel.locator(".source-card p").first.inner_text())
        panel.locator(".evidence-search button").click()
        self.assertEqual(panel.locator("mark.evidence-highlight.active").count(), 1)
        panel.locator(".evidence-search button").click()
        self.assertEqual(panel.locator("mark.evidence-highlight.active").count(), 1)
        self.assertTrue(panel.locator(".evidence-search input").is_visible())
        panel_box = self.page.locator("#evidence-panel").bounding_box()
        button_box = panel.locator(".evidence-search button").bounding_box()
        self.assertLessEqual(button_box["x"] + button_box["width"], panel_box["x"] + panel_box["width"])


if __name__ == "__main__":
    unittest.main()
