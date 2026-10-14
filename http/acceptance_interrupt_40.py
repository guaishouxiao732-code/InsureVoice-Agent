"""Run 40 distinct question-specific stop trials against an isolated web app.

TEST_WEB_URL=http://127.0.0.1:8011 python http/acceptance_interrupt_40.py
The browser presses the real Stop button; this does not substitute a fake TTS.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
from time import perf_counter

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests.acceptance_benchmark import interrupt_cases
from acceptance_browser import AUDIT_JS


OUTPUT = Path(os.environ.get(
    "ACCEPTANCE_INTERRUPT_OUTPUT", ROOT / "exports/acceptance-interrupt-40-20261013.json",
))


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    errors: list[str] = []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            args=["--no-sandbox", "--autoplay-policy=no-user-gesture-required"],
        )
        page = browser.new_page(viewport={"width": 1280, "height": 900})
        page.add_init_script(AUDIT_JS)
        page.on("pageerror", lambda error: errors.append(str(error)))
        try:
            page.goto(os.environ.get("TEST_WEB_URL", "http://127.0.0.1:8011"))
            page.wait_for_function("acceptancePackets.some(m=>m.type==='history')", timeout=180000)
            for ordinal, (mode, source_id, case) in enumerate(interrupt_cases(), 1):
                phase = "generation" if 11 <= ordinal <= 20 else "playback"
                record = {"id": ordinal, "mode": mode, "source_id": source_id,
                          "question": case.question, "phase": phase, "pass": False}
                try:
                    history_count = page.evaluate("acceptancePackets.filter(m=>m.type==='history').length")
                    page.click("#new-chat")
                    page.wait_for_function(
                        "n=>acceptancePackets.filter(m=>m.type==='history').length>n",
                        arg=history_count,
                    )
                    if ordinal % 5 == 0:
                        seed_count = page.evaluate("acceptancePackets.filter(m=>m.type==='turn_done').length")
                        page.evaluate("send({type:'chat',text:'我30岁有社保，年缴基础费率是多少？',mode:'fast',autoplay:false})")
                        page.wait_for_function(
                            "n=>acceptancePackets.filter(m=>m.type==='turn_done').length>n",
                            arg=seed_count, timeout=90000,
                        )
                        seed = page.evaluate("acceptancePackets.filter(m=>m.type==='turn_done').at(-1)")
                        record["seed_pass"] = seed.get("status") == "completed" and "318" in seed.get("text", "")
                    turn_count = page.evaluate("acceptancePackets.filter(m=>m.type==='turn_start').length")
                    page.evaluate(
                        "x=>send({type:'chat',text:x.question,mode:x.mode,autoplay:true,speed:1})",
                        {"question": case.question, "mode": mode},
                    )
                    page.wait_for_function(
                        "n=>acceptancePackets.filter(m=>m.type==='turn_start').length>n",
                        arg=turn_count, timeout=90000,
                    )
                    if phase == "playback":
                        page.wait_for_function(
                            "state.audioSources.size>0&&audioRms()>0.001", timeout=90000,
                        )
                    else:
                        page.wait_for_timeout(100)
                    record["pre_stop_rms"] = page.evaluate("audioRms()")
                    ack_count = page.evaluate("acceptancePackets.filter(m=>m.type==='interrupted').length")
                    old_stops = page.evaluate("audioAudit.stops")
                    started = perf_counter()
                    page.click("#stop")
                    page.wait_for_function(
                        "n=>acceptancePackets.filter(m=>m.type==='interrupted').length>n",
                        arg=ack_count, timeout=10000,
                    )
                    record["ack_ms"] = round((perf_counter() - started) * 1000, 2)
                    body = page.locator(".answer-body")
                    prefix = body.last.inner_text() if body.count() else ""
                    page.wait_for_timeout(300)
                    record.update(page.evaluate("""()=>({
                        post_stop_rms:audioRms(),active_sources:state.audioSources.size,
                        pending_audio:state.pendingAudio,generating:state.generating,
                        stop_calls:audioAudit.stops,
                        local_stop_ms:audioAudit.lastStop-audioAudit.lastClick
                    })"""))
                    stable_text = not body.count() or body.last.inner_text() == prefix
                    record["stable_text"] = stable_text
                    record["pass"] = (
                        record["active_sources"] == 0 and record["pending_audio"] == 0
                        and not record["generating"] and stable_text
                        and (phase == "generation" or (
                            record["stop_calls"] > old_stops
                            and record["post_stop_rms"] < 0.0001
                        ))
                    )
                    if ordinal % 5 == 0:
                        done_count = page.evaluate("acceptancePackets.filter(m=>m.type==='turn_done').length")
                        page.evaluate("send({type:'chat',text:'改成35岁，年缴基础费率是多少？',mode:'fast',autoplay:false})")
                        page.wait_for_function(
                            "n=>acceptancePackets.filter(m=>m.type==='turn_done').length>n",
                            arg=done_count, timeout=90000,
                        )
                        followup = page.evaluate("acceptancePackets.filter(m=>m.type==='turn_done').at(-1)")
                        slots = followup.get("metadata", {}).get("cooperation", {}).get("slots", {})
                        record["followup_pass"] = (
                            followup.get("status") == "completed"
                            and "423" in followup.get("text", "")
                            and slots.get("age", {}).get("value") == 35
                            and slots.get("social_insurance", {}).get("value") == "有社保"
                        )
                except Exception as error:
                    record["error"] = str(error)
                    page.screenshot(path=str(OUTPUT.parent / f"interrupt-40-failure-{ordinal}.png"))
                    page.evaluate("stopReply()")
                    page.wait_for_timeout(500)
                records.append(record)
                OUTPUT.write_text(json.dumps({"records": records, "page_errors": errors},
                                             ensure_ascii=False, indent=2), encoding="utf-8")
                if ordinal % 5 == 0:
                    print(f"interrupt {ordinal}/40: stop_pass={sum(r['pass'] for r in records)}", flush=True)
        finally:
            browser.close()
    print(f"RESULT stop={sum(r['pass'] for r in records)}/40, "
          f"followup={sum(r.get('followup_pass', False) for r in records)}/8, "
          f"page_errors={len(errors)}", flush=True)


if __name__ == "__main__":
    main()
