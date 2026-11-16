
import argparse
import json
import random
from pathlib import Path


NAMES = [
    "玛雅", "丹尼尔", "普丽娅", "卢卡斯",
    "阿米娜", "索菲娅", "伊森", "诺拉",
]

CITIES = [
    "波士顿", "芝加哥", "西雅图",
    "丹佛", "奥斯汀", "亚特兰大",
]

TOPICS = [
    "机器人学",
    "语言学",
    "经济学",
    "气候科学",
    "医学",
    "天文学",
]

DAYS = [
    "星期一",
    "星期二",
    "星期三",
    "星期四",
    "星期五",
]

TRANSPORT = [
    "火车",
    "公共汽车",
    "飞机",
]

INSTRUCTIONS = [
    "请将以下信息自然地描述成一段话。",
    "请根据以下信息写一段话。",
    "请自然地表达以下信息。",
    "请将以下事实整理成一段话。",
]


def sample_distinct(rng, items, n):
    return rng.sample(items, n)


def basic_event(rng):
    name = rng.choice(NAMES)
    city = rng.choice(CITIES)
    day = rng.choice(DAYS)
    mode = rng.choice(TRANSPORT)
    topic = rng.choice(TOPICS)

    facts = [
        f"{name}于{day}前往{city}。",
        f"{name}乘坐{mode}出行。",
        f"{name}参加了一场会议。",
        f"会议的主题是{topic}。",
        f"{name}在{city}住了两晚。",
    ]

    return facts


def multi_entity(rng):
    a, b = sample_distinct(rng, NAMES, 2)
    city = rng.choice(CITIES)
    topic_a, topic_b = sample_distinct(rng, TOPICS, 2)

    facts = [
        f"{a}前往了{city}。",
        f"{b}也前往了{city}。",
        f"{a}参加了一场关于{topic_a}的研讨会。",
        f"{b}参加了一场关于{topic_b}的会议。",
        f"{a}比{b}先到达。",
        f"{a}和{b}在同一天返回家中。",
    ]

    return facts


def temporal(rng):
    name = rng.choice(NAMES)
    city = rng.choice(CITIES)
    topic = rng.choice(TOPICS)

    facts = [
        f"{name}于星期一抵达{city}。",
        f"{name}于星期二参加了一场研讨会。",
        f"研讨会的主题是{topic}。",
        f"{name}于星期三参观了一家博物馆。",
        f"{name}于星期四做了一场报告。",
        f"{name}于星期五返回家中。",
    ]

    return facts


def causal(rng):
    name = rng.choice(NAMES)
    city = rng.choice(CITIES)
    mode = rng.choice(TRANSPORT)
    delay = rng.choice([30, 45, 60, 90])

    facts = [
        f"{name}乘坐{mode}前往{city}。",
        f"{mode}延误了{delay}分钟。",
        f"这次延误导致{name}迟到。",
        f"{name}错过了开场报告。",
        f"{name}参加了其余的会议环节。",
        f"{name}于第二天返回家中。",
    ]

    return facts


def numerical(rng):
    a, b = sample_distinct(rng, NAMES, 2)
    topic = rng.choice(TOPICS)

    papers_a = rng.choice([3, 4, 5])
    papers_b = papers_a + rng.choice([2, 3, 4])

    attendees_a = rng.choice([40, 50, 60])
    attendees_b = attendees_a + rng.choice([20, 30, 40])

    facts = [
        f"{a}审阅了{papers_a}篇论文。",
        f"{b}审阅了{papers_b}篇论文。",
        f"{b}审阅的论文比{a}多。",
        f"{a}的会议环节有{attendees_a}名参与者。",
        f"{b}的会议环节有{attendees_b}名参与者。",
        f"两个会议环节的主题都是{topic}。",
    ]

    return facts


def compressible(rng):
    name = rng.choice(NAMES)
    city = rng.choice(CITIES)
    day = rng.choice(DAYS)
    mode = rng.choice(TRANSPORT)
    topic = rng.choice(TOPICS)

    facts = [
        f"{name}前往了{city}。",
        f"这次旅行发生在{day}。",
        f"{name}乘坐{mode}出行。",
        f"{name}参加了一场会议。",
        f"会议的主题是{topic}。",
        f"会议在{city}举行。",
        f"{name}住了两晚。",
    ]

    return facts


def low_compressibility(rng):
    a, b, c = sample_distinct(rng, NAMES, 3)
    city_a, city_b = sample_distinct(rng, CITIES, 2)
    topic = rng.choice(TOPICS)

    facts = [
        f"{a}居住在{city_a}。",
        f"{b}居住在{city_b}。",
        f"{c}研究{topic}。",
        f"{a}拥有一辆自行车。",
        f"{b}养了两只狗。",
        f"{c}每周六工作。",
    ]

    return facts


GENERATORS = {
    "basic": basic_event,
    "multi_entity": multi_entity,
    "temporal": temporal,
    "causal": causal,
    "numerical": numerical,
    "compressible": compressible,
    "low_compressibility": low_compressibility,
}


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output",
        default="data/density_eval_v1_zh.jsonl",
    )

    parser.add_argument(
        "--n",
        type=int,
        default=500,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    rng = random.Random(args.seed)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    categories = list(GENERATORS.keys())

    counts = {category: 0 for category in categories}

    with output.open("w", encoding="utf-8") as f:

        for i in range(args.n):

            # Cycle categories to preserve the original distribution.
            category = categories[i % len(categories)]

            facts = GENERATORS[category](rng)

            instruction = rng.choice(INSTRUCTIONS)

            example = {
                "id": i,
                "category": category,
                "facts": facts,
                "instruction": instruction,
                "num_facts": len(facts),
            }

            f.write(
                json.dumps(
                    example,
                    ensure_ascii=False,
                ) + "\n"
            )

            counts[category] += 1

    print(f"\nSaved {args.n} examples to {output}\n")

    print("Category counts:")
    for category, count in counts.items():
        print(f"  {category}: {count}")


if __name__ == "__main__":
    main()
