"""Optional real-model regression; use an isolated test service/database."""
import os
from playwright.sync_api import sync_playwright


def main():
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        errors, sessions = [], []
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.add_init_script("""
          window.received=[];
          const Original=window.WebSocket;
          window.WebSocket=class extends Original {
            constructor(...args) {super(...args);this.addEventListener('message',e=>window.received.push({...JSON.parse(e.data),arrived:performance.now()}));}
          };
        """)
        try:
            page.goto(os.environ.get("TEST_WEB_URL", "http://127.0.0.1:8011"))
            page.wait_for_function("window.received.some(m=>m.type==='history')", timeout=120000)
            previous = page.evaluate("state.sessionId")
            page.click("#new-chat")
            page.wait_for_function("old=>state.sessionId&&state.sessionId!==old", arg=previous)
            sessions.append(page.evaluate("state.sessionId"))
            page.fill("#prompt", "我30岁有社保，尊享E生百万医疗2026年缴多少钱？")
            page.click("#send")
            page.wait_for_function("window.received.some(m=>m.type==='text_done')", timeout=60000)
            text_done = page.evaluate("window.received.find(m=>m.type==='text_done')")
            assert "318" in text_done["text"], text_done
            assert "检索来源" not in text_done["text"] and "[来源" not in text_done["text"]
            assert text_done["metadata"]["evidence_sources"]
            page.wait_for_function("window.received.some(m=>m.type==='turn_done')", timeout=60000)
            done = page.evaluate("window.received.find(m=>m.type==='turn_done')")
            assert text_done["arrived"] < done["arrived"]
            page.wait_for_function("state.audioSources.size>0")
            assert page.locator("#stop").is_visible(), "stop must remain available after generation finishes"
            page.click("#stop")
            assert page.evaluate("state.audioSources.size") == 0
            assert not page.evaluate("state.generating")
            page.wait_for_function("!state.stopping")
            page.locator('[data-action="sources"]').last.click()
            assert "年缴费率.pdf" in page.locator("#evidence-content").inner_text()
            page.click("#evidence-panel .close-panel")
            audio_count = page.evaluate("window.received.filter(m=>m.type==='audio_chunk').length")
            page.locator('[data-action="speak"]').last.click()
            page.wait_for_function("n=>window.received.filter(m=>m.type==='audio_chunk').length>n", arg=audio_count, timeout=60000)
            assert page.locator("#stop").is_visible()
            page.click("#stop")
            page.wait_for_function("!state.stopping")
            count = page.evaluate("window.received.filter(m=>m.type==='turn_start').length")
            page.fill("#prompt", "详细解释尊享E生百万医疗2026的一般医疗保险责任与重大疾病医疗保险责任的区别，逐项说明条件和限制。")
            page.click("#send")
            page.wait_for_function("n=>window.received.filter(m=>m.type==='turn_start').length>n", arg=count)
            page.click("#stop")
            page.wait_for_function("!state.stopping")
            text = page.locator(".answer-body").last.inner_text()
            page.wait_for_timeout(1000)
            assert page.locator(".answer-body").last.inner_text() == text
            assert page.evaluate("state.audioSources.size") == 0
            assert not page.evaluate("state.generating")
            count = page.evaluate("window.received.filter(m=>m.type==='turn_done').length")
            page.fill("#prompt", "你好")
            page.click("#send")
            page.wait_for_function("n=>window.received.filter(m=>m.type==='turn_done').length>n", arg=count, timeout=60000)
            assert not errors, errors
            print(f"PASS: text done {text_done['metadata']['voice']['text_completed_ms']:.0f} ms; all audio sent {done['arrived']-text_done['arrived']:.0f} ms later; stop after turn_done, during replay and during generation; next question works; sources only in evidence")
        finally:
            for session in sessions:
                count = page.evaluate("window.received.filter(m=>m.type==='sessions').length")
                page.evaluate("id=>{stopAudio();cancelRecording();send({type:'delete_session',session_id:id});}", session)
                page.wait_for_function("n=>window.received.filter(m=>m.type==='sessions').length>n", arg=count, timeout=30000)
            browser.close()


if __name__ == "__main__":
    main()
