import argparse
import json
import random
from pathlib import Path


NAMES = [
    "Maya", "Daniel", "Priya", "Lucas",
    "Amina", "Sofia", "Ethan", "Nora",
]

CITIES = [
    "Boston", "Chicago", "Seattle",
    "Denver", "Austin", "Atlanta",
]

TOPICS = [
    "robotics",
    "linguistics",
    "economics",
    "climate science",
    "medicine",
    "astronomy",
]

DAYS = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
]

TRANSPORT = [
    "train",
    "bus",
    "plane",
]

INSTRUCTIONS = [
    "Describe the following information naturally in a paragraph.",
    "Write a paragraph based on the information below.",
    "Communicate the following information naturally.",
    "Turn the following facts into a paragraph.",
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
        f"{name} traveled to {city} on {day}.",
        f"{name} traveled by {mode}.",
        f"{name} attended a conference.",
        f"The conference focused on {topic}.",
        f"{name} stayed in {city} for two nights.",
    ]

    return facts


def multi_entity(rng):
    a, b = sample_distinct(rng, NAMES, 2)
    city = rng.choice(CITIES)
    topic_a, topic_b = sample_distinct(rng, TOPICS, 2)

    facts = [
        f"{a} traveled to {city}.",
        f"{b} also traveled to {city}.",
        f"{a} attended a workshop on {topic_a}.",
        f"{b} attended a conference on {topic_b}.",
        f"{a} arrived before {b}.",
        f"{a} and {b} returned home on the same day.",
    ]

    return facts


def temporal(rng):
    name = rng.choice(NAMES)
    city = rng.choice(CITIES)
    topic = rng.choice(TOPICS)

    facts = [
        f"{name} arrived in {city} on Monday.",
        f"{name} attended a workshop on Tuesday.",
        f"The workshop focused on {topic}.",
        f"{name} visited a museum on Wednesday.",
        f"{name} gave a presentation on Thursday.",
        f"{name} returned home on Friday.",
    ]

    return facts


def causal(rng):
    name = rng.choice(NAMES)
    city = rng.choice(CITIES)
    mode = rng.choice(TRANSPORT)
    delay = rng.choice([30, 45, 60, 90])

    facts = [
        f"{name} traveled to {city} by {mode}.",
        f"The {mode} was delayed by {delay} minutes.",
        f"The delay caused {name} to arrive late.",
        f"{name} missed the opening presentation.",
        f"{name} attended the remaining sessions.",
        f"{name} returned home the following day.",
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
        f"{a} reviewed {papers_a} papers.",
        f"{b} reviewed {papers_b} papers.",
        f"{b} reviewed more papers than {a}.",
        f"{a}'s session had {attendees_a} attendees.",
        f"{b}'s session had {attendees_b} attendees.",
        f"Both sessions focused on {topic}.",
    ]

    return facts


def compressible(rng):
    name = rng.choice(NAMES)
    city = rng.choice(CITIES)
    day = rng.choice(DAYS)
    mode = rng.choice(TRANSPORT)
    topic = rng.choice(TOPICS)

    facts = [
        f"{name} traveled to {city}.",
        f"The trip happened on {day}.",
        f"{name} traveled by {mode}.",
        f"{name} attended a conference.",
        f"The conference focused on {topic}.",
        f"The conference took place in {city}.",
        f"{name} stayed for two nights.",
    ]

    return facts


def low_compressibility(rng):
    a, b, c = sample_distinct(rng, NAMES, 3)
    city_a, city_b = sample_distinct(rng, CITIES, 2)
    topic = rng.choice(TOPICS)

    facts = [
        f"{a} lives in {city_a}.",
        f"{b} lives in {city_b}.",
        f"{c} studies {topic}.",
        f"{a} owns a bicycle.",
        f"{b} has two dogs.",
        f"{c} works on Saturdays.",
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
        default="data/density_eval_v1.jsonl",
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

            # Cycle categories so they're approximately balanced.
            category = categories[i % len(categories)]

            facts = GENERATORS[category](rng)

            instruction = rng.choice(INSTRUCTIONS)

            example = {
                "id": i,
                "category": category,

                # Each item is deliberately an atomic
                # ground-truth proposition.
                "facts": facts,

                "instruction": instruction,

                "num_facts": len(facts),
            }

            f.write(
                json.dumps(
                    example,
                    ensure_ascii=False,
                )
                + "\n"
            )

            counts[category] += 1

    print(f"\nSaved {args.n} examples to {output}\n")

    print("Category counts:")

    for category, count in counts.items():
        print(f"  {category:20s} {count}")


if __name__ == "__main__":
    main()