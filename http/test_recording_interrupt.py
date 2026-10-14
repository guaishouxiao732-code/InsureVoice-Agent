"""Browser regression for stopping speech when click-to-record begins."""

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


class RecordingInterruptTests(unittest.TestCase):
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
        self.page = self.browser.new_page()
        self.errors = []
        self.page.on("pageerror", lambda error: self.errors.append(str(error)))
        self.page.add_init_script("""(() => {
          window.sentPackets = [];
          window.recorders = [];
          window.stoppedSources = 0;
          window.stoppedTracks = 0;
          window.WebSocket = class {
            static OPEN = 1;
            constructor() { this.readyState = 1; }
            send(raw) { window.sentPackets.push(JSON.parse(raw)); }
          };
          window.MediaRecorder = class {
            static isTypeSupported() { return true; }
            constructor(_stream, options) {
              this.mimeType = options.mimeType;
              this.state = 'inactive';
              window.recorders.push(this);
            }
            start() { this.state = 'recording'; }
            stop() {
              this.state = 'inactive';
              this.ondataavailable({data: new Blob(['recording-test'])});
              this.onstop();
            }
          };
          Object.defineProperty(navigator, 'mediaDevices', {configurable: true,
            value: {getUserMedia: () => new Promise(resolve => {
              window.resolveMic = () => resolve({getTracks: () => [{stop() {
                window.stoppedTracks++;
              }}]});
            })}});
        })()""")
        self.page.goto(f"http://127.0.0.1:{self.server.server_port}/", wait_until="domcontentloaded")
        self.page.wait_for_function("state.ws?.readyState === WebSocket.OPEN")

    def tearDown(self):
        self.assertFalse(self.errors, self.errors)
        self.page.close()

    def test_start_stops_old_speech_and_second_click_sends(self):
        self.page.evaluate("""() => {
          state.audioContext = {state: 'running'};
          state.responseId = 6;
          state.generating = true;
          state.audioSources.add({stop() { window.stoppedSources++; }});
          state.pendingAudio = 1;
        }""")
        self.page.locator("#mic").click()
        immediate = self.page.evaluate("""() => ({
          stoppedSources, pendingAudio: state.pendingAudio,
          packets: sentPackets.map(packet => packet.type),
          recorderCount: recorders.length, micStarting: state.micStarting
        })""")
        self.assertEqual(immediate["stoppedSources"], 1)
        self.assertEqual(immediate["pendingAudio"], 0)
        self.assertEqual(immediate["packets"], ["interrupt"])
        self.assertEqual(immediate["recorderCount"], 0)
        self.assertTrue(immediate["micStarting"])

        self.page.evaluate("window.resolveMic()")
        self.page.wait_for_function("recorders[0]?.state === 'recording'")
        self.assertEqual(self.page.locator("#mic").get_attribute("aria-label"), "结束录音并发送")
        self.page.evaluate("""() => {
          handleMessage({type:'response_epoch', response_id:7});
          handleMessage({type:'interrupted'});
          handleMessage({type:'audio_chunk', response_id:6, pcm:'AAA=', sample_rate:16000});
        }""")
        self.assertTrue(self.page.evaluate("recorders[0].state === 'recording'"))
        self.assertEqual(self.page.evaluate("state.audioSources.size"), 0)
        self.assertEqual(self.page.evaluate("sentPackets.filter(item => item.type === 'voice').length"), 0)

        self.page.locator("#mic").click()
        self.page.wait_for_function("sentPackets.some(item => item.type === 'voice')")
        self.assertEqual(self.page.evaluate("sentPackets.map(item => item.type)"), ["interrupt", "voice"])
        self.assertEqual(self.page.evaluate("window.stoppedTracks"), 1)
        self.assertEqual(self.page.locator("#mic").get_attribute("aria-label"), "开始录音")

    def test_idle_recording_does_not_send_interrupt(self):
        self.page.evaluate("state.audioContext = {state:'running'}")
        self.page.locator("#mic").click()
        self.assertEqual(self.page.evaluate("sentPackets.length"), 0)
        self.page.evaluate("window.resolveMic()")
        self.page.wait_for_function("recorders[0]?.state === 'recording'")
        self.page.locator("#mic").click()
        self.page.wait_for_function("sentPackets.length === 1")
        self.assertEqual(self.page.evaluate("sentPackets[0].type"), "voice")

    def test_playback_after_answer_completed_stops_immediately(self):
        self.page.evaluate("""() => {
          state.audioContext = {state:'running'};
          state.generating = false;
          state.currentTurn = null;
          state.audioSources.add({stop() { window.stoppedSources++; }});
        }""")
        self.page.locator("#mic").click()
        self.assertEqual(self.page.evaluate("window.stoppedSources"), 1)
        self.assertEqual(self.page.evaluate("sentPackets.map(item => item.type)"), ["interrupt"])
        self.page.evaluate("window.resolveMic()")
        self.page.wait_for_function("recorders[0]?.state === 'recording'")


if __name__ == "__main__":
    unittest.main()
