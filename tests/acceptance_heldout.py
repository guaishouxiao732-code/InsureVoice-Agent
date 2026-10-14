"""Run a small, unseen paraphrase set through the real web interface.

These cases are deliberately kept separate from acceptance_benchmark.py. Their
full answers still require human review; expected facts are review prompts,
not a substring-based accuracy score.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

import websockets


CASES = [
    ("fast", "我今年35岁，没有基本医保，月付这份保险要多少钱？", "35岁位于31-35岁档；无社保月缴96元。"),
    ("fast", "我满71周岁了还能直接按费率表买这款产品吗？", "超过被保险人最高投保年龄70周岁，不应编造71岁的费率。"),
    ("fast", "有医保但这次没有用医保结算，一般医疗还按百分百赔吗？", "一般医疗在满足相关参保前提下调整为60%，并扣1万元免赔额。"),
    ("fast", "这份保险的疾病等待期和药省保加油包等待期一样吗？", "分别为30天和7天。"),
    ("fast", "一般医疗去指定民营医院的特需部是否属于普通部保障？", "不能把特需部当普通部。"),
    ("fast", "重大疾病医疗有没有一万元免赔额？", "无免赔额。"),
    ("fast", "定点康复机构普通部与指定民营医院普通部的比例一样吗？", "分别为80%和60%，不能混同。"),
    ("fast", "互联网药品加油包一整年最高能赔多少钱？", "年累计1000元。"),
    ("fast", "能给我指定民营医院的完整名称清单吗？", "资料未给完整名单，应说明无法核实并引导人工。"),
    ("fast", "我这笔已经提交的理赔现在审核到哪一步了？", "无授权理赔系统，不能编造进度，应转人工。"),
    ("slow", "如果康复治疗在确诊后第181天才住院，能否仅凭这项康复责任报销？", "180日内含第180日，第181天超出该时间条件；不得直接判断全部理赔。"),
    ("slow", "有医保却没用医保结算，住指定民营医院普通部做康复，比例是80%、60%还是36%？", "指定民营医院通常60%，满足未按医保结算前提时为36%，非80%。"),
    ("slow", "恶性肿瘤的BNCT和光免疫疗法是同一种治疗吗？资料分别列了什么机构？", "两种不同疗法；分别列出对应日本机构，不能混同。"),
    ("slow", "如果我在加拿大医院接受质子治疗，能直接按先进疗法责任申请赔付吗？", "质子治疗医疗机构限中国大陆境内经审核认定的机构；不能据此承诺境外赔付。"),
    ("slow", "特药处方一次开40天，超出30天的那10天药量一定能报吗？", "处方剂量超出30天的部分不予赔付；不应把整张处方都判定不赔。"),
    ("slow", "一般医疗和一般门急诊各有300万，所以两项合计能到600万吗？", "两项共享300万元保额，不能简单相加。"),
    ("slow", "重大疾病医疗300万和先进疗法600万能否合并成同一个900万额度？", "来源未明确合并关系；不能断言900万，应说明不确定并转人工。"),
    ("slow", "我有高血压但其他信息未提供，能不能保证这次投保必然通过？", "缺个人健康告知及核保结果，不能保证，应转人工。"),
    ("slow", "牙齿因意外受伤要治疗，能直接说所有牙科治疗都免责吗？", "牙科免责存在意外所致例外，不能一概而论；具体理赔须核实。"),
    ("slow", "只要是研究性治疗就必然不赔吗，条款具体排除的条件是什么？", "仅明确排除未经科学或医学认可的试验性、研究性治疗及后果，不可扩大为所有研究性治疗。"),
]


async def run(server_url: str, output: Path) -> None:
    url = server_url.replace("http://", "ws://").replace("https://", "wss://").rstrip("/") + "/ws"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as file:
        async with asyncio.timeout(3600), websockets.connect(
            url, max_size=16 * 1024 * 1024, ping_timeout=120,
        ) as socket:
            async def receive(kind: str) -> dict:
                while True:
                    packet = json.loads(await asyncio.wait_for(socket.recv(), timeout=180))
                    if packet.get("type") == "error":
                        raise RuntimeError(packet.get("message"))
                    if packet.get("type") == kind:
                        return packet

            await receive("history")
            for index, (mode, question, expected) in enumerate(CASES, 1):
                await socket.send(json.dumps({"type": "new_session"}))
                await receive("history")
                await socket.send(json.dumps({"type": "chat", "text": question, "mode": mode, "autoplay": False}, ensure_ascii=False))
                packet = await receive("turn_done")
                row = {"id": index, "mode": mode, "question": question,
                       "review_expected": expected, "answer": packet.get("text", ""),
                       "status": packet.get("status"), "metadata": packet.get("metadata", {})}
                file.write(json.dumps(row, ensure_ascii=False) + "\n")
                file.flush()
                print(f"held-out {index}/{len(CASES)} {mode}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-url", default="http://127.0.0.1:8011")
    parser.add_argument("--output", type=Path, default=Path("exports/acceptance-heldout-20.jsonl"))
    args = parser.parse_args()
    asyncio.run(run(args.server_url, args.output))
