"""Live 200-question acceptance benchmark for the local insurance assistant.

Run from the project root in qwen3-vl after loading .env:
    python tests/acceptance_benchmark.py --list
    python tests/acceptance_benchmark.py --run

Accuracy is scored against fact markers transcribed from the checked-in
knowledge base, with an evidence-source requirement for generated answers.
This is a practical regression score, not a substitute for human claim review.
"""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass
import json
from pathlib import Path
import re
import sys
from time import perf_counter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@dataclass(frozen=True)
class Case:
    question: str
    expected: tuple[str, ...]
    category: str
    handoff: bool = False


def _case(question: str, expected: str | tuple[str, ...], category: str) -> Case:
    return Case(question, (expected,) if isinstance(expected, str) else expected, category)


RATE_BANDS = (
    # age, annual insured/uninsured, monthly insured/uninsured
    (2, 806, 1679, 80, 166),
    (7, 279, 575, 27, 56),
    (13, 142, 302, 13, 30),
    (16, 205, 374, 20, 37),
    (19, 205, 374, 20, 37),
    (23, 255, 520, 25, 51),
    (28, 318, 679, 31, 67),
    (33, 423, 971, 42, 96),
    (38, 517, 1330, 51, 131),
    (43, 636, 2130, 63, 212),
    (48, 984, 2974, 97, 294),
    (53, 1226, 4042, 121, 399),
    (58, 1645, 5436, 162, 539),
    (63, 2563, 8898, 253, 881),
    (68, 3316, 11926, 328, 1181),
)


def fast_cases() -> list[Case]:
    # Sample different fee bands and payment modes without expanding one
    # question template into sixty near-duplicates.
    rates = (
        ("孩子2岁、有基本医保，2026版按年缴的基础费率是多少？", "806"),
        ("给4岁没有社保的孩子选月缴，费率表每月列多少钱？", "166"),
        ("7岁孩子没有医保，若一年交一次应查到哪个金额？", "575"),
        ("10岁参保儿童打算月付，表内对应的月缴数字是多少？", "27"),
        ("13岁有社保，投保页面核保前的年缴基础费率是多少？", "142"),
        ("19岁未参加基本医保，按月缴费的表列金额是多少？", "37"),
        ("23岁有医保，先只看年缴费率表应是多少？", "255"),
        ("28岁没社保，2026版年缴这一栏列了多少元？", "679"),
        ("33岁无社保，如果改成月缴，基础费率是多少？", "96"),
        ("38岁有社保，年缴报价对应哪个年龄档金额？", "517"),
        ("43岁未参保，按年交的基础费率请报数值。", "2130"),
        ("48岁有社保，想分月支付，费率表每月多少钱？", "97"),
        ("53岁没有医保，年缴基础保费查到多少？", "4042"),
        ("58岁参保，选择年缴时表列多少元？", "1645"),
        ("63岁无社保，每月缴费的基础费率是多少？", "881"),
        ("68岁有社保，月缴金额在费率表中是多少？", "328"),
        ("70岁没有社保，按年缴费的表列金额是多少？", "11926"),
        ("刚满30岁并有医保，按月交费该看哪一档、多少钱？", "31"),
        ("35岁有社保，年缴时表列的基础费率是多少？", "423"),
        ("60岁未参保，如果选择月缴，表列多少元？", "539"),
    )
    result = [_case(question, amount, "费率边界与缴费方式") for question, amount in rates]
    facts = (
        ("尊享E生2026必选计划的保险期间是多久？", "1年", "基本产品信息"),
        ("这个产品必选计划的疾病等待期有几天？", "30天", "等待期"),
        ("药省保加油包等待期多少天？", "7天", "加油包等待期"),
        ("被保险人最低多少岁可以投保？", "30天", "投保年龄"),
        ("被保险人最高投保年龄是多少？", "70周岁", "投保年龄"),
        ("一般医疗及外购药械责任的年免赔额是多少？", "1万元", "一般医疗免赔额"),
        ("一般医疗责任通常按多少比例赔付？", "100%", "一般医疗赔付比例"),
        ("未按基本医保身份结算时一般医疗比例是多少？", "60%", "未医保结算比例"),
        ("一般医疗年度累计保额上限多少？", "300万元", "一般医疗保额"),
        ("一般医疗与一般门急诊共用免赔额吗？", "共享", "共享免赔额"),
        ("重大疾病医疗责任有没有免赔额？", "无免赔额", "重大疾病免赔额"),
        ("重大疾病医疗责任包含多少种重大疾病？", "100种", "重疾种类"),
        ("国内医疗总保额是多少？", "600万元", "总保额"),
        ("普通医疗责任的医院等级要求是什么？", "二级及二级以上", "医院范围"),
        ("普通医疗责任能否在医院特需部直接按普通部责任理赔？", "普通部", "医院部别"),
        ("普通医疗包含住院前后多长时间的门急诊费用？", "30天", "住院前后门急诊"),
        ("特定疾病康复住院责任覆盖多少种特定疾病？", "16种", "康复疾病种类"),
        ("康复住院治疗要在确诊后多少天内？", "180日", "康复治疗时间"),
        ("指定康复机构普通部的康复责任比例是多少？", "80%", "康复比例"),
        ("公立医院普通部康复责任比例是多少？", "60%", "康复比例"),
        ("康复责任年度保额多少？", "100万元", "康复保额"),
        ("康复住院转运救护车单次限额是多少？", "1000元", "康复转运限额"),
        ("恶性肿瘤先进疗法年度限额是多少？", "600万元", "先进疗法限额"),
        ("先进疗法床位费每天限额是多少？", "1500元", "先进疗法床位费"),
        ("先进疗法责任的免赔额是多少？", "无免赔额", "先进疗法免赔额"),
        ("重大疾病异地转诊交通和住宿费用年度限额是多少？", "1万元", "异地转诊限额"),
        ("异地转诊住宿费每天上限是多少？", "500元/天", "异地转诊住宿"),
        ("重疾住院护工每天给付多少？", "500元/天", "护工费用"),
        ("重疾住院护工单次住院最多给付多少天？", "30日", "护工天数"),
        ("住院补偿加油包的年累计限额是多少？", "1万元", "住院补偿加油包"),
        ("一般住院津贴的日额是多少？", "100元/日", "住院津贴"),
        ("一般住院津贴每次住院免赔几天？", "3天", "住院津贴免赔天数"),
        ("ICU住院津贴每天是多少？", "500元/日", "ICU津贴"),
        ("互联网药品责任的月赔付限额是多少？", "100元", "互联网药品限额"),
        ("互联网药品责任的年累计限额是多少？", "1000元", "互联网药品年限额"),
        ("门急诊加油包单次限额是多少？", "300元", "门急诊限额"),
        ("门急诊加油包年度限额是多少？", "1万元", "门急诊年限额"),
        ("门急诊加油包通常赔付比例是多少？", "50%", "门急诊比例"),
        ("未按医保身份就诊结算的门急诊比例是多少？", "30%", "门急诊未医保比例"),
        ("先进疗法协助服务电话是多少？", "952299", "服务电话"),
    )
    result.extend(_case(*fact) for fact in facts)
    additional_facts = (
        ("投保人至少要满多少周岁？", "18周岁", "投保人资格"),
        ("投保人与被保险人之间需要具备什么保险上的关系？", "可保利益", "投保人与被保险人关系"),
        ("这款产品的互联网销售区域是否包含港澳台？", "中国大陆", "销售地域"),
        ("家庭共享免赔额加油包是否把一般门急诊也纳入共享？", "不在", "共享免赔额范围"),
        ("恶性肿瘤特定药品应通过哪类药店购买？", "指定药店", "特药购药"),
        ("买特定药品前是否要经过处方审核？", "处方审核", "特药审核"),
        ("一次特药处方超过多少天的药量，超出部分不予赔付？", "30天", "特药剂量上限"),
        ("医生确认对某特药已经耐药后，继续购买费用是否承担？", "不承担", "特药耐药限制"),
        ("普通门急诊费用中，诊疗费是否属于列明的费用类型？", "诊疗费", "门急诊费用"),
        ("一般门急诊责任能直接套用特殊门诊吗？", "特殊门诊", "门急诊责任边界"),
        ("先进疗法中，BNCT的中文名称是什么？", "硼中子俘获", "先进疗法名称"),
        ("光免疫疗法是否被列在恶性肿瘤先进疗法之中？", "光免疫疗法", "先进疗法类别"),
        ("质子重离子治疗的医疗机构地域限于哪里？", "中国大陆", "质子治疗地域"),
        ("先进疗法床位费是否覆盖单人病房？", "不包括单人病房", "先进疗法房型"),
        ("先进疗法责任是否承担治疗过程中使用的化疗药费？", "不包括", "先进疗法药费"),
        ("重大疾病异地转诊时，飞机票可按什么舱位报销？", "经济舱", "转诊交通标准"),
        ("异地转诊住宿可选套房还是标准间？", "标准间", "转诊住宿房型"),
        ("重疾异地转诊最多包括几位陪同亲属？", "一名亲属", "转诊陪同人数"),
        ("申请重疾异地转诊是否需要转出医院的证明？", "转院证明", "转诊手续"),
        ("重疾住院护工能否自行聘用后再直接报销？", "不接受", "护工服务流程"),
        ("特定疾病康复住院的180日期限包括第180日吗？", "含第180日", "康复期限边界"),
        ("指定康复机构有医保却未按医保身份结算时，康复赔付比例是多少？", "48%", "康复结算比例"),
        ("指定民营医院普通部康复，未按医保身份结算时比例是多少？", "36%", "康复民营医院比例"),
        ("康复住院责任是否设置年度免赔额？", "无免赔额", "康复免赔额"),
        ("重疾关爱加油包约定的额外保险金是多少？", "5万元", "重疾关爱加油包"),
        ("扩展特需医疗加油包的年度限额是多少？", "300万元", "特需加油包限额"),
        ("一般住院津贴单次住院最多累计给付多少天？", "30天", "住院津贴天数"),
        ("ICU住院津贴每次住院免赔几天？", "0天", "ICU津贴免赔"),
        ("ICU津贴能否与一般住院津贴叠加？", "叠加", "津贴叠加"),
        ("互联网药品加油包对6周岁及以下儿童能开处方药吗？", "无法开具", "儿童互联网处方"),
        ("同一原因再次住院，间隔不超过多少日按同一次住院看？", "30日", "连续住院认定"),
        ("责任免除对美容整形费用如何处理？", "美容", "美容整形免责"),
        ("常规健康体检能直接作为本产品医疗赔付费用吗？", "健康体检", "体检免责"),
        ("牙科治疗因意外伤害发生时，免责条款有没有例外？", "意外", "牙科意外例外"),
        ("跳伞和攀岩属于哪类风险活动？", "高风险运动", "高风险运动"),
        ("未经科学或医学认可的试验性治疗是否被列入责任免除？", "未经科学或医学认可", "试验性治疗"),
        ("只给年龄就要求保证承保，是否需要看个人健康告知和核保？", "核保", "个案承保边界"),
        ("我这笔理赔现在审核到哪一步？系统没有授权记录时怎么办？", "无法确认", "理赔进度安全", True),
        ("请给出指定民营医院完整名单；资料未列全时应如何处理？", "无法确认", "医院名单证据不足", True),
        ("高血压患者能否保证买到该产品？没有核保结论时怎么办？", "无法确认", "个案核保安全", True),
    )
    for question, expected, category, *flags in additional_facts:
        result.append(Case(question, (expected,), category, handoff=bool(flags and flags[0])))
    assert len(result) == 100, len(result)
    return result


def slow_cases() -> list[Case]:
    """Compound questions with short, auditable gold-fact markers."""
    raw: list[tuple[str, str | tuple[str, ...], str]] = [
        # General medical + outpatient: deductible sharing and settlement rules.
        ("请综合说明2026版一般医疗及外购药械责任的疾病等待期、年免赔额、正常赔付比例和年限额。", ("30天", "1万元", "100%", "300万元"), "一般医疗组合"),
        ("有基本医保但住院没有按医保身份结算时，一般医疗要扣多少免赔额、按多少比例赔？", ("1万元", "60%"), "一般医疗未医保结算"),
        ("一般医疗和一般门急诊责任是各有一份免赔额和保额，还是共用？请列出数值。", ("共享", "1万元", "300万元"), "一般与门急诊共用"),
        ("一般门急诊医疗责任覆盖哪些常见费用？请列出至少四类。", ("诊疗费", "急诊室费", "检查检验费", "药品费"), "门急诊费用构成"),
        ("一般门急诊责任明确不包括一般医疗责任中的哪些门诊项目？", ("特殊门诊", "门诊手术", "住院前后"), "门急诊范围边界"),
        ("一般医疗责任的住院前后门急诊费用分别覆盖住院前后多少天？", ("住院前", "出院后", "30天"), "住院前后门急诊"),
        ("一般医疗赔付需要满足怎样的医院等级和科室范围？", ("二级及二级以上", "普通部"), "一般医疗医院要求"),
        ("若有医保身份但未使用医保结算，一般医疗赔付比例是否仍为100%？正确比例是多少？", "60%", "医保结算条件"),
        ("说明一般医疗与一般门急诊共享限额后，两个责任的合计年度保险金额是多少？", ("共享", "300万元"), "共享年度保额"),
        ("一般门急诊责任等待期、免赔额、正常赔付比例和年度限额分别是什么？", ("30天", "1万元", "100%", "300万元"), "门急诊责任组合"),
        # Major diseases: no deductible, listed disease condition, limits.
        ("重大疾病医疗责任适用于多少种疾病？疾病等待期和一般医疗等待期相同吗？", ("100种", "30天"), "重疾种类与等待期"),
        ("重大疾病医疗责任是否扣除1万元免赔额？正常结算时赔付比例和年限额是什么？", ("无免赔额", "100%", "300万元"), "重疾免赔与比例"),
        ("重大疾病医疗未按基本医保身份就诊结算时，赔付比例是多少？", "60%", "重疾未医保"),
        ("重大疾病医疗责任可包括住院、特殊门诊、门诊手术和住院前后门急诊吗？住院前后各多久？", ("住院", "特殊门诊", "门诊手术", "30天"), "重疾医疗费用范围"),
        ("重大疾病责任的普通医院要求是什么？是否能据此确认任意医院都可赔？", ("二级及二级以上", "普通部"), "重疾医院范围"),
        ("重大疾病医疗年累计300万元是否和先进疗法600万元合并为900万元？请按资料区分责任限额。", ("无法确认", "人工"), "重疾与先进疗法限额", True),
        ("资料中重大疾病医疗责任写了100%赔付，有医保但未按医保结算时是否仍按100%？", "60%", "重疾结算条件"),
        ("重大疾病医疗是否有年免赔额？回答时请区别一般医疗的免赔额。", ("无免赔额", "1万元"), "重疾与一般医疗免赔"),
        ("重大疾病医疗是否承担必需且合理的外购药械费用？具体给付还要满足什么特别提示？", ("外购药品", "特定条件"), "重疾外购药械"),
        ("请说明一般医疗和重大疾病医疗在免赔额、正常赔付比例、年限额上的区别。", ("1万元", "无免赔额", "100%", "300万元"), "一般与重疾比较"),
        # Rehabilitation: disease count, 180-day time window, differing venue ratios.
        ("特定疾病康复住院责任涵盖多少种疾病，确诊后几天内住院治疗才在条款期限内？", ("16种", "180日"), "康复疾病与期限"),
        ("康复住院在指定康复医疗机构普通部和二级以上公立医院普通部的赔付比例分别是多少？", ("80%", "60%"), "康复机构比例比较"),
        ("康复责任在有医保但未按医保身份结算时，两类医院的赔付比例分别是多少？", ("48%", "36%"), "康复未医保比例"),
        ("康复住院责任的年度限额、免赔额和转运救护车单次上限是什么？", ("100万元", "无免赔额", "1000元"), "康复限额组合"),
        ("康复住院是否只要在确诊后180日内就可以报销任何康复费用？说明治疗目的和住院要求。", ("180日", "住院", "治疗该特定疾病"), "康复条件解释"),
        ("康复治疗可包括哪些费用类型？请至少提及物理治疗、床位或护理中的两类。", ("物理治疗", "护理费"), "康复费用项目"),
        ("康复住院转运救护车是否没有单次上限？正确限额是多少？", "1000元", "康复救护车限额"),
        ("在指定康复机构治疗和在指定民营医院普通部治疗，康复比例是否一样？", ("80%", "60%"), "康复医院比较"),
        ("康复责任要求确诊后180日内住院，这180日是否包括第180日？", "含第180日", "康复期限边界"),
        ("请同时列出康复责任的疾病数量、赔付比例范围和年度保额，不要把比例说成统一一个数。", ("16种", "80%", "60%", "100万元"), "康复责任综合"),
        # Advanced cancer therapies: locations, cap, exclusions and no deductible.
        ("恶性肿瘤先进疗法责任有哪些治疗类别？请提及质子重离子和BNCT。", ("质子", "硼中子俘获", "光免疫疗法"), "先进疗法类别"),
        ("恶性肿瘤先进疗法责任的等待期、年度给付限额、免赔额分别是什么？", ("30天", "600万元", "0"), "先进疗法限额"),
        ("先进疗法责任床位费每天限额多少？单人病房是否包含在这一限额保障内？", ("1500元", "不包括单人病房"), "先进疗法床位与房型"),
        ("日本BNCT治疗和中国大陆质子重离子治疗是否都在资料列明的先进疗法范围？", ("日本", "质子重离子"), "先进疗法地域"),
        ("先进疗法责任是否包括治疗过程中的化疗药、免疫治疗药、内分泌药或靶向药费用？", "不包括", "先进疗法药品排除"),
        ("先进疗法责任列明哪些日本机构可进行光免疫疗法？请说明资料列举的机构范围。", ("日本国立癌症研究中心东医院", "东京医科大学病院"), "光免疫机构"),
        ("质子重离子责任的医疗机构地域范围是否包括香港、澳门、台湾或境外？", ("中国大陆", "不包括", "香港"), "质子重离子地域"),
        ("先进疗法就医协助服务的适用对象是谁、何时可使用、电话是什么？", ("被保险人本人", "等待期", "952299"), "先进疗法协助服务"),
        ("BNCT责任能否在任意日本医院治疗后都主张给付？资料明确列出的机构有什么限制？", ("日本大阪关西BNCT中心", "日本国立癌症研究中心"), "BNCT指定机构"),
        ("先进疗法600万元限额是否有免赔额？请同时说明床位费限制。", ("无免赔额", "600万元", "1500元"), "先进疗法责任综合"),
        # Special drugs and gene testing.
        ("恶性肿瘤重度特定药品责任通常要求等待期后初次确诊吗，请一并说明药品应在哪里购买？", ("30天", "指定药店"), "特药等待期与购买"),
        ("购买特定药品前是否可以直接购买后再申请，而不经过审核？", ("处方审核", "指定流程"), "特药处方审核"),
        ("恶性肿瘤重度特定药品基因检测需满足什么治疗关联要求，并说明哪些检测不符合条件？", ("初次确诊", "用药方案"), "特药基因检测"),
        ("被保险人经专科医生审核确定对特定药品已经耐药后仍继续购买，该药品费用是否承担？", ("耐药", "不承担"), "耐药限制"),
        ("外购药械费用是不是只要有发票就必然赔，请说明资料对合同释义和给付边界的提醒？", ("特定条件", "合同"), "外购药械限制"),
        ("特药责任中的购买药店、处方审核和药品适应症分别有什么要求？", ("指定药店", "处方审核", "适应症"), "特药流程"),
        ("因恶性肿瘤重度特定药品而发生的基因检测，未经医生建议是否符合资料要求？", ("未经医生建议", "不承担"), "基因检测限制"),
        ("特药处方剂量超过30天的部分，能否依据资料确认给付？", ("30天", "不予赔付"), "处方剂量限制"),
        ("特药适应症与药品说明书不符时，资料如何说明保险责任？", ("适应症", "不承担"), "药品适应症限制"),
        ("重度恶性肿瘤特药服务的服务电话是什么，是否可跳过处方审核？", ("952299", "处方审核"), "特药服务电话"),
        # Care and referral benefits.
        ("重大疾病住院护工费用的日标准、单次住院天数上限和年度累计天数分别是多少？", ("500元/天", "30日", "30天"), "护工限额组合"),
        ("重疾护工费用能否由被保险人自行雇佣护工后直接申请报销？", ("指定或认可", "直接结算", "不接受"), "护工申请流程"),
        ("重大疾病异地转诊交通住宿责任的年度限额、住宿日限额及陪同人数是什么？", ("1万元", "500元/天", "一名亲属"), "异地转诊条件"),
        ("异地转诊的飞机舱位和火车席别分别有什么上限？", ("经济舱", "软卧", "一等座"), "异地转诊交通标准"),
        ("异地转诊住宿是否可以报销套房？资料规定的住宿房型和日限额是什么？", ("标准间", "500元/天"), "异地住宿限制"),
        ("重大疾病异地转诊是否需要转出医院开具转院证明？", "转院证明", "异地转诊手续"),
        ("重疾护工责任要求在哪种医院住院、由谁提供服务？", ("二级及二级以上", "指定或认可"), "护工医院与服务"),
        ("重疾护工责任需要先提交材料并审核吗？费用是直接结算还是个人自行申请？", ("审核", "直接结算", "不接受"), "护工结算方式"),
        ("重疾护工年度累计和单次住院分别最多给付多少天，请分别列出？", ("30天", "30日"), "护工天数"),
        ("异地转诊责任是否覆盖所有出行人员的全部费用？说明陪同人员和年度限额。", ("一名亲属", "1万元"), "异地转诊范围"),
        # Optional packages.
        ("家庭共享免赔额加油包的共享范围是什么？门急诊是否参与共享？", ("1万元", "不在", "一般门急诊"), "家庭共享免赔额"),
        ("购买重疾关爱加油包后，确诊约定重大疾病额外给付多少保险金？", "5万元", "重疾关爱加油包"),
        ("重大疾病扩展特需医疗加油包的年限额、赔付比例和床位费上限是什么？", ("300万元", "100%", "1500元"), "重疾特需加油包"),
        ("重疾扩展特需医疗加油包是否对任何疾病都开放特需部？", ("重大疾病", "特需医疗部"), "特需责任条件"),
        ("住院医疗费用补偿加油包的年限额和医保身份结算规则是什么？", ("1万元", "100%", "60%"), "住院补偿条件"),
        ("住院补偿加油包治疗结节、息肉、囊肿等，资料所列赔付比例是多少？", "20%", "住院补偿特定疾病"),
        ("住院补偿加油包对颈椎病或腰椎间盘突出治疗费用如何约定，请说明赔付和免责边界？", ("脊椎疾病", "不承担"), "脊椎疾病除外"),
        ("一般住院津贴每天多少、每次住院免赔几天、单次累计最多几天？", ("100元/日", "3天", "30天"), "一般住院津贴组合"),
        ("ICU住院津贴每天多少、次免赔天数是多少？能否与一般住院津贴叠加？", ("500元/日", "0天", "叠加"), "ICU住院津贴"),
        ("互联网药品加油包赔付比例、月度限额、年度限额分别是什么？", ("50%", "100元", "1000元"), "互联网药品责任"),
        # Eligibility and policy context.
        ("投保人最低年龄、民事行为能力和与被保险人关系要求是什么？", ("18周岁", "完全民事行为能力", "可保利益"), "投保人资格"),
        ("被保险人的年龄范围与投保人年龄限制是否相同？请分别说明。", ("30天", "70周岁", "18周岁"), "投保年龄区分"),
        ("产品保险期间是一年，资料是否承诺来年一定按原价格续保？", ("无法确认", "人工"), "续保承诺边界", True),
        ("保费是否会随着年龄增长而变化？资料有没有承诺每年保费固定不变？", ("年龄", "变化", "未承诺"), "费率变化"),
        ("投保地区和被保险人与投保人的关系有哪些基本要求？", ("中国大陆", "本人", "配偶", "子女", "父母"), "投保资格"),
        ("当前知识库是否包含个体高血压的核保结论？不能确认时应如何回答？", ("无法确认", "人工"), "个案核保安全", True),
        ("某人已患高血压，能否仅凭投保须知断言一定承保？", ("不能", "核保"), "个案核保安全", True),
        ("产品资料中等待期后的疾病保障是否代表既往症必然承保？", ("不代表", "既往症"), "既往症边界"),
        ("用户没有提供保单版本和生效日期时，能否根据2026版资料确定其已有保单理赔？", ("不能", "无法确认"), "保单版本边界", True),
        ("用户只给年龄，询问个案能否投保；要给出承保结论还是说明需要健康告知与核保？", ("健康告知", "核保"), "核保前置条件", True),
        # Exclusions and evidence limits.
        ("责任免除资料对美容整形、健康体检和牙科治疗分别如何约定？", ("美容", "健康体检", "牙科"), "责任免除"),
        ("资料是否将所有既往症引起的住院医疗费用都承诺赔付？", ("既往症", "不承担"), "既往症除外"),
        ("未经医生建议自行购买药品，或药品使用超出说明书适应症，责任免除如何约定？", ("未经医生建议", "适应症", "免责情形"), "药品使用限制"),
        ("高风险运动导致受伤是否明确属于责任免除事项？请举出资料列举的运动。", ("高风险运动", "跳伞", "攀岩"), "高风险活动除外"),
        ("在香港、澳门、台湾或境外接受治疗，相关外购药械和特定费用是否在资料明示责任范围？", ("不承担", "香港", "境外"), "境外治疗除外"),
        ("试验性或研究性治疗相关费用是否属于资料明确承担的费用？", ("未经科学或医学认可", "不能据此认定所有"), "试验治疗除外"),
        ("出险后仅凭当前资料能否确认保险公司最终会赔付？回答应保留哪些审核条件？", ("无法确认", "人工"), "理赔结论边界", True),
        ("资料中没有列出某个具体医院是否属于指定民营医疗机构，能否凭空确认？", ("无法确认", "人工"), "医院名单证据不足", True),
        ("我的理赔案件现在处理到哪一步？如果没有授权理赔记录系统，应如何答复？", ("未接入", "无法确认", "人工"), "理赔进度安全", True),
        ("请列出合同中外购药械定义的全部申请材料，但当前资料只提示查阅合同释义时应怎么办？", ("无法确认", "合同"), "材料清单证据不足"),
        ("一般门急诊责任里的外购药械费用是否与一般医疗的免赔额和保额共享？", ("共享", "1万元", "300万元"), "门急诊外购药械共享"),
        ("重大疾病住院前后门急诊是否要求与该次住院原因相同？", ("无法确认", "人工"), "重疾住院前后条件", True),
        ("特定疾病康复责任是否可以在普通门诊而非住院时申请康复住院费用？", ("住院", "康复"), "康复住院形式"),
        ("恶性肿瘤先进疗法责任中的质子重离子治疗是否包含化疗药品费用？", "不包括", "先进疗法用药排除"),
        ("恶性肿瘤特定药品需在哪类药店购买，且购买前应完成什么审核？", ("指定药店", "处方审核"), "特药购买与审核"),
        ("互联网药品责任中，6周岁及以下儿童能否开具处方药品？", ("6周岁", "无法开具"), "儿童处方限制"),
        ("一般住院津贴对同一原因再次住院且间隔不超过多少日的情形如何认定？", ("30日", "同一次住院"), "连续住院认定"),
        ("ICU津贴和一般住院津贴能否同时给付？分别是多少日额？", ("500元/日", "100元/日", "叠加"), "津贴叠加"),
        ("当前资料没有给出具体个人的核保结论，回答时能否使用大模型常识替代核保？", ("不能", "核保"), "核保信息不足"),
        ("尊享家庭共享免赔额加油包是否把一般门急诊医疗费用也纳入共享免赔额累计？", ("不在", "一般门急诊"), "共享免赔额排除范围"),
    ]
    result: list[Case] = []
    for row in raw:
        question, expected, category, *flags = row
        markers = expected if isinstance(expected, tuple) else (expected,)
        result.append(Case(question, markers, category, handoff=bool(flags and flags[0])))
    assert len(result) == 100, len(result)
    return result


def normalized(text: str) -> str:
    return re.sub(r"[\s,，。:：;；（）()【】\[\]￥¥]", "", text).lower()


MARKER_ALIASES = {
    "二级及二级以上": ("二级及二级以上", "二级及以上"),
    "指定药店": ("指定药店", "指定的药店", "指定合规药店"),
    "一名亲属": ("一名亲属", "1名亲属"),
    "特需医疗部": ("特需医疗部", "特需部"),
    "治疗该特定疾病": ("治疗该特定疾病", "治疗该疾病"),
    "500元/天": ("500元/天", "500元/日", "500元每天", "每日500元", "500元"),
    "30日": ("30日", "30天"),
    "30天": ("30天", "30日"),
    "适应症": ("适应症", "适应证"),
    "不包括单人病房": ("不包括单人病房", "不包含单人病房"),
    "不包括": ("不包括", "不包含"),
    "不承担": ("不承担", "不予赔付", "不予给付", "属于责任免除", "属于免责", "不纳入保险承担", "不属于资料明确承担", "不属于责任范围", "不在责任范围", "保险责任不覆盖"),
    "不接受": ("不接受", "不支持"),
    "不在": ("不在", "不参与共享", "不纳入共享", "不共享"),
    "可保利益": ("可保利益", "保险利益"),
    "不代表": ("不代表", "不等于"),
    "无法确认": ("无法确认", "无法确定", "不能确认", "未明确", "未列明", "资料未提供", "未提及"),
    "不能": ("不能", "无法确认", "不能确认", "不可以"),
    "未承诺": ("未承诺", "没有承诺", "未作承诺", "未明确承诺", "没有保证", "不能保证", "未提供保证"),
}


def marker_matches(marker: str, answer: str) -> bool:
    answer_norm = normalized(answer)
    return any(normalized(candidate) in answer_norm
               for candidate in MARKER_ALIASES.get(marker, (marker,)))


def score_case(case: Case, answer: str, meta: dict) -> bool:
    sources = meta.get("evidence_sources", [])
    if case.handoff:
        return bool(meta.get("handoff", {}).get("required")) and not sources and "人工" in answer and bool(re.search(r"无法确认|无法可靠确认|缺少|未接入", answer))
    valid = bool(sources) and all(marker_matches(marker, answer) for marker in case.expected)
    if re.search(r"BNCT[（(]光免疫|投保人年龄(?:仅影响|无下限)|不明确属于责任免除|不构成责任免除情形|否则仅承担60%|限被保险人直系亲属", answer):
        valid = False
    return valid


async def run_server(url: str, mode: str, output: Path, resume: bool = False) -> None:
    """Exercise the same WebSocket, mode settings and model used by the UI."""
    import websockets
    output.parent.mkdir(parents=True, exist_ok=True)
    previous = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()] if resume and output.exists() else []
    completed = {(record["mode"], record["id"]): record for record in previous}
    if not resume:
        output.write_text("", encoding="utf-8")
    url = url.replace("http://", "ws://").replace("https://", "wss://").rstrip("/") + "/ws"
    async with websockets.connect(url, max_size=16 * 1024 * 1024, ping_timeout=120) as socket:
        async def receive(kind):
            while True:
                packet = json.loads(await asyncio.wait_for(socket.recv(), timeout=180))
                if packet.get("type") == "error":
                    raise RuntimeError(packet.get("message"))
                if packet.get("type") == kind:
                    return packet
        await receive("history")
        for group, cases in (("fast", fast_cases()), ("slow", slow_cases())):
            if mode != "all" and mode != group:
                continue
            passed = sum(int(record["pass"]) for key, record in completed.items() if key[0] == group)
            for index, case in enumerate(cases, 1):
                if (group, index) in completed:
                    continue
                await socket.send(json.dumps({"type": "new_session"}))
                await receive("history")
                started = perf_counter()
                await socket.send(json.dumps({"type": "chat", "text": case.question, "mode": group, "autoplay": False}, ensure_ascii=False))
                packet = await receive("turn_done")
                answer, meta = packet["text"], packet.get("metadata", {})
                ok = packet.get("status") == "completed" and score_case(case, answer, meta)
                passed += int(ok)
                record = {
                    "mode": group, "id": index, "question": case.question,
                    "expected": list(case.expected), "category": case.category,
                    "expected_handoff": case.handoff, "answer": answer,
                    "metadata": meta, "pass": ok, "transport": "real_websocket",
                    "status": packet.get("status"), "first_chunk_ms": packet.get("first_chunk_ms"),
                    "elapsed_ms": round((perf_counter() - started) * 1000, 2),
                }
                with output.open("a", encoding="utf-8") as result_file:
                    result_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                if index % 10 == 0:
                    print(f"server {group} {index}/100: pass={passed}", flush=True)
            print(f"RESULT server {group}={passed}/100", flush=True)


def list_cases() -> None:
    for mode, cases in (("快思考", fast_cases()), ("慢思考", slow_cases())):
        print(f"## {mode}（{len(cases)}题）")
        for index, case in enumerate(cases, 1):
            print(f"{index}. {case.question}")


def interrupt_cases() -> list[tuple[str, int, Case]]:
    """A fixed, non-template subset of the 200 questions for browser stop trials."""
    selected = {
        "fast": (21, 25, 31, 40, 50, 60, 66, 75, 88, 99),
        "slow": (1, 4, 7, 10, 12, 15, 20, 22, 25, 29, 31, 35, 39, 42, 45,
                 50, 52, 55, 59, 62, 65, 69, 72, 75, 79, 82, 85, 89, 95, 100),
    }
    groups = {"fast": fast_cases(), "slow": slow_cases()}
    subset = [(mode, index, groups[mode][index - 1])
              for mode, indices in selected.items() for index in indices]
    assert len(subset) == 40 and len({case.question for _, _, case in subset}) == 40
    return subset


def write_markdown(path: Path) -> None:
    lines = [
        "# 快慢思考验收题库", "",
        "> 快思考与慢思考各 100 题。费率表只占快思考 20 题，其余覆盖条款事实、条件边界、责任免除与证据不足转人工。",
        "> 答案核验、首响与完整回答时延须对实际运行结果统计；停播成功率由独立浏览器操作测试统计。", "",
    ]
    for mode, cases in (("快思考", fast_cases()), ("慢思考", slow_cases())):
        lines.extend((f"## {mode}（{len(cases)}题）", ""))
        lines.extend(f"{index}. {case.question}" for index, case in enumerate(cases, 1))
        lines.append("")
    lines.extend(("## 打断响应专项（从以上 200 题选取 40 题）", "",
                  "前 10 题覆盖快思考，后 30 题覆盖慢思考；测试时在生成中或播放中点击停止，并检查停止后的残留输出及后续会话。", ""))
    for index, (mode, original_id, case) in enumerate(interrupt_cases(), 1):
        lines.append(f"{index}. [{mode} #{original_id}] {case.question}")
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def run_live(mode: str = "all", output: Path = PROJECT_ROOT / "exports/acceptance-results.jsonl") -> None:
    """Load local RAG/Qwen once and score all question cases against evidence."""
    from api import Qwen3Client
    from cooperative import CooperativeClient
    from rag import RAGClient, RAGRetriever

    all_cases = (("fast", fast_cases()), ("slow", slow_cases()))
    cases_by_mode = all_cases if mode == "all" else tuple(item for item in all_cases if item[0] == mode)
    print("Loading local embedding model and FAISS index…", flush=True)
    retriever = RAGRetriever(embedding_device="cpu")
    print("Loading Qwen3 on cuda:1…", flush=True)
    qwen = Qwen3Client(device="cuda:1")
    qwen.warmup()
    client = CooperativeClient(RAGClient(qwen, retriever))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("", encoding="utf-8")
    total_pass = total = 0
    started = perf_counter()
    for mode, cases in cases_by_mode:
        passed = 0
        failures: list[tuple[int, str, str, str]] = []
        print(f"\nRunning {mode}: {len(cases)} questions", flush=True)
        for index, case in enumerate(cases, 1):
            case_started = perf_counter()
            response = client.stream_chat(
                [{"role": "user", "content": case.question}],
                thinking_mode=mode,
            )
            answer = "".join(response)
            meta = response.metadata
            sources = meta.get("evidence_sources", [])
            markers_ok = all(marker_matches(marker, answer) for marker in case.expected)
            handoff_meta = meta.get("handoff", {})
            if case.handoff:
                evidence_ok = bool(handoff_meta.get("required")) and not sources
                markers_ok = "人工" in answer and bool(re.search(r"无法确认|无法可靠确认|缺少|未接入", answer))
            else:
                evidence_ok = bool(sources)
            ok = markers_ok and evidence_ok
            if re.search(r"BNCT[（(]光免疫|投保人年龄(?:仅影响|无下限)|不明确属于责任免除|不构成责任免除情形", answer):
                ok = False
            record = {
                "mode": mode, "id": index, "question": case.question,
                "expected": list(case.expected), "category": case.category,
                "expected_handoff": case.handoff, "answer": answer,
                "metadata": meta, "pass": ok,
                "elapsed_ms": round((perf_counter() - case_started) * 1000, 2),
            }
            with output.open("a", encoding="utf-8") as result_file:
                result_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            if ok:
                passed += 1
            else:
                failures.append((index, case.question, answer, ",".join(case.expected)))
            response.close()
            if index % 10 == 0 or index == len(cases):
                print(f"{mode} {index}/{len(cases)}; pass={passed}; latest={'PASS' if ok else 'FAIL'}", flush=True)
        total_pass += passed
        total += len(cases)
        print(f"RESULT {mode}: {passed}/{len(cases)} ({passed/len(cases)*100 if cases else 0:.1f}%)", flush=True)
        for index, question, answer, expected in failures:
            print(f"FAIL {mode} #{index}: {question}\n  expected={expected}\n  answer={answer[:240]}", flush=True)
    print(f"TOTAL: {total_pass}/{total} ({total_pass/total*100 if total else 0:.1f}%) in {(perf_counter()-started)/60:.1f} min", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--list", action="store_true", help="print all generated questions")
    parser.add_argument("--run", action="store_true", help="run live Qwen/RAG scoring")
    parser.add_argument("--write-markdown", action="store_true", help="regenerate the 200-question review document")
    parser.add_argument("--markdown-output", type=Path, default=PROJECT_ROOT / "docs/快慢思考验收题库.md")
    parser.add_argument("--mode", choices=("all", "fast", "slow"), default="all")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "exports/acceptance-results.jsonl")
    parser.add_argument("--server-url", help="test the real isolated web service instead of loading another model")
    parser.add_argument("--resume", action="store_true", help="retain completed WebSocket cases and finish remaining cases")
    args = parser.parse_args()
    if args.write_markdown:
        write_markdown(args.markdown_output)
    elif args.run:
        if args.server_url:
            asyncio.run(run_server(args.server_url, args.mode, args.output, args.resume))
        else:
            run_live(args.mode, args.output)
    else:
        list_cases()
