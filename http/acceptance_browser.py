"""100 real-service stop-button trials, including Web Audio signal checks.

Use an isolated localhost service/database and run with qwen3-vl Python:
    TEST_WEB_URL=http://127.0.0.1:8011 python http/acceptance_browser.py
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from time import perf_counter

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = Path(os.environ.get("ACCEPTANCE_BROWSER_OUTPUT", ROOT / "exports/acceptance-browser.json"))
QUESTION = "请详细比较一般医疗与重大疾病医疗的免赔额、赔付比例、医保结算要求和责任限额。"
REPLAY_TEXT = (
    "一般医疗责任与重大疾病医疗责任存在不同的免赔额要求。"
    "一般医疗扣除一万元年免赔额后，通常按百分之一百赔付。"
    "重大疾病医疗责任没有免赔额，但仍须满足合同约定的疾病和就医条件。"
    "如果有基本医保却没有按医保身份结算，应核对相应的比例调整。"
    "这是一段用于检验停止播放功能的语音，点击停止后应立即停止声音。"
)

AUDIT_JS = """
window.acceptancePackets=[];
window.audioAudit={stops:0,ended:0,lastClick:0,lastStop:0,analyser:null};
const Socket=window.WebSocket;
window.WebSocket=class extends Socket {
 constructor(...args) {super(...args);this.addEventListener('message',e=>{
   const m=JSON.parse(e.data);delete m.pcm;window.acceptancePackets.push(m);
 });}
};
const AC=window.AudioContext||window.webkitAudioContext;
const create=AC.prototype.createBufferSource;
AC.prototype.createBufferSource=function(...args){
 const source=create.apply(this,args);
 if(!audioAudit.analyser){
   audioAudit.analyser=this.createAnalyser();audioAudit.analyser.fftSize=2048;
   const mute=this.createGain();mute.gain.value=0;
   audioAudit.analyser.connect(mute);mute.connect(this.destination);
 }
 source.connect(audioAudit.analyser);
 const stop=source.stop.bind(source);
 source.stop=(...a)=>{audioAudit.stops++;audioAudit.lastStop=performance.now();return stop(...a);};
 source.addEventListener('ended',()=>audioAudit.ended++);
 return source;
};
window.audioRms=()=>{
 if(!audioAudit.analyser)return 0;
 const a=new Float32Array(2048);audioAudit.analyser.getFloatTimeDomainData(a);
 return Math.sqrt(a.reduce((s,v)=>s+v*v,0)/a.length);
};
document.addEventListener('click',e=>{
 if(e.target.closest('#stop'))audioAudit.lastClick=performance.now();
},true);
"""


def main():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    records = []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(args=["--no-sandbox", "--autoplay-policy=no-user-gesture-required"])
        page = browser.new_page(viewport={"width": 1280, "height": 900})
        page.add_init_script(AUDIT_JS)
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        try:
            page.goto(os.environ.get("TEST_WEB_URL", "http://127.0.0.1:8011"))
            page.wait_for_function("acceptancePackets.some(m=>m.type==='history')", timeout=180000)
            for i in range(100):
                phase = "generation" if i < 20 else "playback" if i < 60 else "replay"
                record = {"id": i + 1, "phase": phase, "pass": False}
                try:
                    before = page.evaluate("acceptancePackets.filter(m=>m.type==='history').length")
                    page.click("#new-chat")
                    page.wait_for_function("n=>acceptancePackets.filter(m=>m.type==='history').length>n", arg=before)
                    count = page.evaluate("acceptancePackets.filter(m=>m.type==='turn_start').length")
                    if phase == "replay":
                        page.evaluate("text=>send({type:'speak',text,speed:1})", REPLAY_TEXT)
                    else:
                        page.fill("#prompt", QUESTION)
                        page.click("#send")
                        page.wait_for_function("n=>acceptancePackets.filter(m=>m.type==='turn_start').length>n", arg=count, timeout=60000)
                    if phase != "generation":
                        page.wait_for_function("state.audioSources.size>0&&audioRms()>0.001", timeout=90000)
                    else:
                        if i >= 10:
                            page.wait_for_function("state.currentTurn&&state.currentTurn._rawText.length>=12&&state.generating", timeout=90000)
                        page.wait_for_timeout(100 + (i % 4) * 50)
                    record["pre_stop_rms"] = page.evaluate("audioRms()")
                    ack_count = page.evaluate("acceptancePackets.filter(m=>m.type==='interrupted').length")
                    stops = page.evaluate("audioAudit.stops")
                    started = perf_counter()
                    page.click("#stop")
                    page.wait_for_function("n=>acceptancePackets.filter(m=>m.type==='interrupted').length>n", arg=ack_count, timeout=10000)
                    record["ack_ms"] = round((perf_counter() - started) * 1000, 2)
                    prefix = page.locator(".answer-body").last.inner_text() if page.locator(".answer-body").count() else ""
                    page.wait_for_timeout(300)
                    record.update(page.evaluate("""()=>({
                      post_stop_rms:audioRms(),active_sources:state.audioSources.size,
                      pending_audio:state.pendingAudio,generating:state.generating,
                      stop_calls:audioAudit.stops,
                      local_stop_ms:audioAudit.lastStop-audioAudit.lastClick
                    })"""))
                    stable_text = not page.locator(".answer-body").count() or page.locator(".answer-body").last.inner_text() == prefix
                    record["stable_text"] = stable_text
                    record["pass"] = (
                        record["active_sources"] == 0 and record["pending_audio"] == 0
                        and not record["generating"] and stable_text
                        and (phase == "generation" or (
                            record["stop_calls"] > stops and record["post_stop_rms"] < 0.0001
                        ))
                    )
                except Exception as error:
                    record["error"] = str(error)
                    page.screenshot(path=str(OUTPUT.parent / f"browser-failure-{i+1}.png"))
                    page.evaluate("stopReply()")
                    page.wait_for_timeout(500)
                records.append(record)
                OUTPUT.write_text(json.dumps({"records": records, "page_errors": errors}, ensure_ascii=False, indent=2), encoding="utf-8")
                if (i + 1) % 10 == 0:
                    print(f"browser interrupt {i+1}/100: pass={sum(r['pass'] for r in records)}", flush=True)
            page.screenshot(path=str(OUTPUT.parent / "acceptance-browser.png"))
            before = page.evaluate("acceptancePackets.filter(m=>m.type==='turn_done').length")
            page.evaluate("send({type:'chat',text:'我30岁有社保，年缴多少钱？',mode:'auto',autoplay:true})")
            page.wait_for_function("n=>acceptancePackets.filter(m=>m.type==='turn_done').length>n", arg=before, timeout=60000)
            page.wait_for_function("state.audioSources.size>0", timeout=60000)
            page.click("#stop")
            page.wait_for_function("!state.stopping")
            before = page.evaluate("acceptancePackets.filter(m=>m.type==='turn_done').length")
            page.evaluate("send({type:'chat',text:'改成35岁，年缴保费是多少？',mode:'auto',autoplay:false})")
            page.wait_for_function("n=>acceptancePackets.filter(m=>m.type==='turn_done').length>n", arg=before, timeout=60000)
            recovery = page.evaluate("acceptancePackets.filter(m=>m.type==='turn_done').at(-1)")
            slots = recovery.get("metadata", {}).get("cooperation", {}).get("slots", {})
            recovery_ok = "423" in recovery.get("text", "") and slots.get("age", {}).get("value") == 35 and slots.get("social_insurance", {}).get("value") == "有社保"
            OUTPUT.write_text(json.dumps({"records": records, "page_errors": errors, "post_interrupt_recovery": {"pass": recovery_ok, "result": recovery}}, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"RESULT browser_interrupt={sum(r['pass'] for r in records)}/100; context_recovery={recovery_ok}; errors={errors}", flush=True)
        finally:
            browser.close()


if __name__ == "__main__":
    main()
