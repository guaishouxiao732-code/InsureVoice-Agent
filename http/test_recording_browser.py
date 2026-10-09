"""Optional real ASR/TTS browser regression, using only test-created sessions."""
import os
from pathlib import Path
from playwright.sync_api import sync_playwright


def main():
    url = os.environ.get("TEST_WEB_URL", "http://127.0.0.1:8011")
    microphone = Path(os.environ["TEST_MIC_WAV"]).resolve()
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(args=[
            "--use-fake-ui-for-media-stream", "--use-fake-device-for-media-stream",
            f"--use-file-for-fake-audio-capture={microphone}",
        ])
        context = browser.new_context(permissions=["microphone"])
        page = context.new_page()
        errors, sessions = [], []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.add_init_script("""
          localStorage.setItem('assistant-voice-mode-v2','continuous');
          window.received=[]; window.sent=[];
          const Original=window.WebSocket;
          window.WebSocket=class extends Original {
            constructor(...args) {super(...args);this.addEventListener('message',e=>window.received.push(JSON.parse(e.data)));}
            send(data) {window.sent.push(JSON.parse(data));super.send(data);}
          };
        """)
        try:
            page.goto(url)
            page.wait_for_function("window.received.some(m=>m.type==='history')", timeout=120000)
            assert page.locator("#voice-input-mode").count() == 0
            previous = page.evaluate("state.sessionId")
            page.click("#new-chat")
            page.wait_for_function("old=>state.sessionId&&state.sessionId!==old", arg=previous)
            sessions.append(page.evaluate("state.sessionId"))
            for button in ("#hero-mic", "#mic"):
                done = page.evaluate("window.received.filter(m=>m.type==='turn_done').length")
                sent = page.evaluate("window.sent.filter(m=>m.type==='voice').length")
                page.click(button)
                page.wait_for_function("state.recorder?.state==='recording'")
                assert page.locator("#mic").get_attribute("aria-label") == "结束录音并发送"
                page.wait_for_timeout(1900)
                assert page.evaluate("window.sent.filter(m=>m.type==='voice').length") == sent
                page.click("#mic")
                page.wait_for_function("n=>window.received.filter(m=>m.type==='turn_done').length>n", arg=done, timeout=60000)
                assert page.evaluate("window.sent.filter(m=>m.type==='voice').length") == sent + 1
                assert page.evaluate("state.recorder.stream.getTracks().every(t=>t.readyState==='ended')")
                assert page.locator("#mic").get_attribute("aria-label") == "开始录音"
            messages = page.evaluate("window.received")
            assert any(m["type"] == "turn_start" and m.get("from_voice") and m.get("question") for m in messages)
            assert any(m["type"] == "audio_chunk" for m in messages)
            assert not page.evaluate("window.sent.some(m=>m.type==='listen_start')")
            assert not errors, errors
            print("PASS: real MediaRecorder → ASR → answer → TTS; no upload before second click; continuous entry removed")
        finally:
            for session in sessions:
                count = page.evaluate("window.received.filter(m=>m.type==='sessions').length")
                page.evaluate("id=>{cancelRecording();send({type:'delete_session',session_id:id});}", session)
                page.wait_for_function("n=>window.received.filter(m=>m.type==='sessions').length>n", arg=count, timeout=30000)
            browser.close()


if __name__ == "__main__":
    main()
