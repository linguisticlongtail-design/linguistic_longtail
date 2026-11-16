import json
from pathlib import Path

source = Path("data/density_eval_v1.jsonl")
output = Path("data/density_eval_zh_5.jsonl")

translations = {
    0: [
        "丹尼尔星期三去了波士顿。",
        "丹尼尔乘火车出行。",
        "丹尼尔参加了一场会议。",
        "会议的主题是语言学。",
        "丹尼尔在波士顿住了两晚。"
    ],
    1: [
        "丹尼尔去了亚特兰大。",
        "索菲娅也去了亚特兰大。",
        "丹尼尔参加了一场医学研讨会。",
        "索菲娅参加了一场机器人技术会议。",
        "丹尼尔比索菲娅先到。",
        "丹尼尔和索菲娅同一天回家。"
    ],
    2: [
        "玛雅星期一抵达波士顿。",
        "玛雅星期二参加了一场研讨会。",
        "研讨会的主题是机器人技术。",
        "玛雅星期三参观了博物馆。",
        "玛雅星期四做了一场报告。",
        "玛雅星期五回家了。"
    ],
    3: [
        "卢卡斯乘飞机去了奥斯汀。",
        "飞机延误了30分钟。",
        "这次延误导致卢卡斯迟到。",
        "卢卡斯错过了开幕报告。",
        "卢卡斯参加了剩余的会议环节。",
        "卢卡斯第二天回家了。"
    ],
    4: [
        "伊森审阅了5篇论文。",
        "丹尼尔审阅了8篇论文。",
        "丹尼尔审阅的论文比伊森多。",
        "伊森的分会场有40名参加者。",
        "丹尼尔的分会场有60名参加者。",
        "两个分会场都聚焦气候科学。"
    ]
}

instruction_map = {
    "Write a paragraph based on the information below.":
        "根据以下信息写一段话。",
    "Turn the following facts into a paragraph.":
        "将以下事实整理成一段话。"
}

rows = []

with source.open(encoding="utf-8") as f:
    for line in f:
        example = json.loads(line)

        if example["id"] not in translations:
            continue

        example["facts"] = translations[example["id"]]
        example["instruction"] = instruction_map[example["instruction"]]

        assert len(example["facts"]) == example["num_facts"]
        rows.append(example)

assert len(rows) == 5
assert [x["id"] for x in rows] == list(range(5))

with output.open("w", encoding="utf-8") as f:
    for example in rows:
        f.write(json.dumps(example, ensure_ascii=False) + "\n")

print(f"Saved {len(rows)} Chinese examples to {output}")
