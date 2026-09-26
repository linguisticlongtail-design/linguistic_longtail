import json
import random
from pathlib import Path

random.seed(42)

OUT = Path("data/density_eval.jsonl")

names = ["Maya", "Daniel", "Priya", "Lucas", "Amina", "Sofia", "Ethan", "Nora"]
cities = ["Boston", "Chicago", "Seattle", "Denver", "Austin", "Atlanta"]
transport = ["train", "bus", "plane"]
topics = ["robotics", "climate science", "linguistics", "medicine", "economics"]
days = ["Monday", "Tuesday", "Wednesday"]
delays = [20, 30, 40, 50, 60]

examples = []

for i in range(200):
    name = random.choice(names)
    city = random.choice(cities)
    mode = random.choice(transport)
    topic = random.choice(topics)
    day = random.choice(days)
    delay = random.choice(delays)

    facts = [
        f"{name} traveled to {city} on {day}.",
        f"{name} traveled by {mode}.",
        f"The {mode} arrived {delay} minutes late.",
        f"{name} attended a conference.",
        f"The conference focused on {topic}.",
        f"{name} stayed in {city} for two nights.",
    ]

    examples.append({
        "id": i,
        "facts": facts,
        "instruction": "Describe the following information naturally in a paragraph."
    })

OUT.parent.mkdir(parents=True, exist_ok=True)

with OUT.open("w", encoding="utf-8") as f:
    for example in examples:
        f.write(json.dumps(example, ensure_ascii=False) + "\n")

print(f"Saved {len(examples)} examples to {OUT}")