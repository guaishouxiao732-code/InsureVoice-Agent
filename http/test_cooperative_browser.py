"""Optional real-browser automatic cooperation smoke test against a warmed service."""
import os
from playwright.sync_api import sync_playwright


def main():
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        page = browser.new_page()
        errors, sessions = [], []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.add_init_script("""
          window.received=[];
          const Original=window.WebSocket;
          window.WebSocket=class extends Original {
            constructor(...args) {super(...args);this.addEventListener('message',e=>window.received.push(JSON.parse(e.data)));}
          };
        """)

        def new_session():
            previous = page.evaluate("state.sessionId")
            page.click("#new-chat")
            page.wait_for_function("old => state.sessionId && state.sessionId !== old", arg=previous)
            sessions.append(page.evaluate("state.sessionId"))

        def ask(text):
            count = page.evaluate("window.received.filter(m=>m.type==='turn_done').length")
            page.fill("#prompt", text)
            page.click("#send")
            page.wait_for_function("n => window.received.filter(m=>m.type==='turn_done').length>n", arg=count, timeout=60000)
            return page.evaluate("window.received.filter(m=>m.type==='turn_done').at(-1)")

        try:
            page.goto(os.environ.get("TEST_WEB_URL", "http://127.0.0.1:8011"))
            page.wait_for_function("window.received.some(m=>m.type==='history')", timeout=120000)
            assert page.evaluate("state.mode") == "auto"
            page.click("#settings-open")
            page.uncheck("#autoplay")
            page.click("#settings-close")
            new_session()
            first = ask("年缴多少钱")
            assert first["metadata"]["cooperation"]["pending_slot"] == "age"
            second = ask("30")
            assert second["metadata"]["cooperation"]["pending_slot"] == "social_insurance"
            quote = ask("有社保")
            assert quote["metadata"]["knowledge_mode"] == "rag"
            assert quote["metadata"]["cooperation"]["resolved_mode"] == "fast"
            assert "318" in quote["text"], quote["text"]
            corrected = ask("不是我，是我父亲，65岁")
            assert corrected["metadata"]["cooperation"]["pending_slot"] == "social_insurance"
            assert corrected["metadata"]["cooperation"]["slots"]["age"]["value"] == 65
            assert "social_insurance" not in corrected["metadata"]["cooperation"]["slots"]
            new_session()
            complex_answer = ask("如果我30岁有社保，年缴多少钱")
            assert complex_answer["metadata"]["knowledge_mode"] == "rag"
            assert complex_answer["metadata"]["cooperation"]["resolved_mode"] == "slow"
            assert "318" in complex_answer["text"], complex_answer["text"]
            handoff = ask("Python列表和元组有什么区别")
            assert handoff["metadata"]["handoff"]["status"] == "pending"
            assert "尚未接入人工客服" in handoff["text"]
            new_session()
            clinical = ask("我65岁，有高血压，住院能赔吗")
            assert clinical["metadata"]["decision"] == "clarify"
            assert "已有保单" in clinical["text"]
            reply = ask("理赔")
            assert reply["metadata"]["cooperation"]["pending_slot"] == "event_reason"
            result = ask("因为摔伤住院")
            assert result["metadata"]["cooperation"]["resolved_mode"] == "slow"
            assert result["metadata"]["cooperation"]["slots"]["age"]["value"] == 65
            page.reload()
            page.wait_for_function("document.querySelectorAll('.turn').length>=3")
            assert "快" in page.locator(".answer-head").last.inner_text() or "待人工" in page.locator(".answer-head").last.inner_text()
            page.locator('[data-action="sources"]').last.click()
            assert "用户表述" in page.locator("#evidence-content").inner_text()
            assert not errors, errors
            print("PASS: auto default; staged clarification; real fast and slow RAG/Qwen quotes; person correction; handoff; restored route and evidence UI")
        finally:
            for session in sessions:
                before = page.evaluate("window.received.filter(m=>m.type==='sessions').length")
                page.evaluate("id => send({type:'delete_session',session_id:id})", session)
                page.wait_for_function("n => window.received.filter(m=>m.type==='sessions').length>n", arg=before)
            browser.close()


if __name__ == "__main__":
    main()
