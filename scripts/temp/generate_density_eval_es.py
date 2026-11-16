
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
    "robótica", "lingüística", "economía",
    "ciencias del clima", "medicina", "astronomía",
]

DAYS = [
    "lunes", "martes", "miércoles",
    "jueves", "viernes",
]

TRANSPORT = [
    "tren", "autobús", "avión",
]

INSTRUCTIONS = [
    "Describe la siguiente información de forma natural en un párrafo.",
    "Escribe un párrafo basado en la siguiente información.",
    "Comunica la siguiente información de forma natural.",
    "Convierte los siguientes hechos en un párrafo.",
]


def sample_distinct(rng, items, n):
    return rng.sample(items, n)


def basic_event(rng):
    name = rng.choice(NAMES)
    city = rng.choice(CITIES)
    day = rng.choice(DAYS)
    mode = rng.choice(TRANSPORT)
    topic = rng.choice(TOPICS)

    return [
        f"{name} viajó a {city} el {day}.",
        f"{name} viajó en {mode}.",
        f"{name} asistió a una conferencia.",
        f"La conferencia trató sobre {topic}.",
        f"{name} se quedó en {city} durante dos noches.",
    ]


def multi_entity(rng):
    a, b = sample_distinct(rng, NAMES, 2)
    city = rng.choice(CITIES)
    topic_a, topic_b = sample_distinct(rng, TOPICS, 2)

    return [
        f"{a} viajó a {city}.",
        f"{b} también viajó a {city}.",
        f"{a} asistió a un taller sobre {topic_a}.",
        f"{b} asistió a una conferencia sobre {topic_b}.",
        f"{a} llegó antes que {b}.",
        f"{a} y {b} regresaron a casa el mismo día.",
    ]


def temporal(rng):
    name = rng.choice(NAMES)
    city = rng.choice(CITIES)
    topic = rng.choice(TOPICS)

    return [
        f"{name} llegó a {city} el lunes.",
        f"{name} asistió a un taller el martes.",
        f"El taller trató sobre {topic}.",
        f"{name} visitó un museo el miércoles.",
        f"{name} dio una presentación el jueves.",
        f"{name} regresó a casa el viernes.",
    ]


def causal(rng):
    name = rng.choice(NAMES)
    city = rng.choice(CITIES)
    mode = rng.choice(TRANSPORT)
    delay = rng.choice([30, 45, 60, 90])

    return [
        f"{name} viajó a {city} en {mode}.",
        f"El {mode} sufrió un retraso de {delay} minutos.",
        f"El retraso hizo que {name} llegara tarde.",
        f"{name} se perdió la presentación inaugural.",
        f"{name} asistió a las sesiones restantes.",
        f"{name} regresó a casa al día siguiente.",
    ]


def numerical(rng):
    a, b = sample_distinct(rng, NAMES, 2)
    topic = rng.choice(TOPICS)

    papers_a = rng.choice([3, 4, 5])
    papers_b = papers_a + rng.choice([2, 3, 4])

    attendees_a = rng.choice([40, 50, 60])
    attendees_b = attendees_a + rng.choice([20, 30, 40])

    return [
        f"{a} revisó {papers_a} artículos académicos.",
        f"{b} revisó {papers_b} artículos académicos.",
        f"{b} revisó más artículos académicos que {a}.",
        f"La sesión de {a} tuvo {attendees_a} asistentes.",
        f"La sesión de {b} tuvo {attendees_b} asistentes.",
        f"Ambas sesiones trataron sobre {topic}.",
    ]


def compressible(rng):
    name = rng.choice(NAMES)
    city = rng.choice(CITIES)
    day = rng.choice(DAYS)
    mode = rng.choice(TRANSPORT)
    topic = rng.choice(TOPICS)

    return [
        f"{name} viajó a {city}.",
        f"El viaje ocurrió el {day}.",
        f"{name} viajó en {mode}.",
        f"{name} asistió a una conferencia.",
        f"La conferencia trató sobre {topic}.",
        f"La conferencia tuvo lugar en {city}.",
        f"{name} se quedó durante dos noches.",
    ]


def low_compressibility(rng):
    a, b, c = sample_distinct(rng, NAMES, 3)
    city_a, city_b = sample_distinct(rng, CITIES, 2)
    topic = rng.choice(TOPICS)

    return [
        f"{a} vive en {city_a}.",
        f"{b} vive en {city_b}.",
        f"{c} estudia {topic}.",
        f"{a} tiene una bicicleta.",
        f"{b} tiene dos perros.",
        f"{c} trabaja los sábados.",
    ]


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
        default="data/density_eval_v1_es.jsonl",
    )
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    categories = list(GENERATORS.keys())
    counts = {category: 0 for category in categories}

    with output.open("w", encoding="utf-8") as f:
        for i in range(args.n):
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
                json.dumps(example, ensure_ascii=False) + "\n"
            )
            counts[category] += 1

    print(f"\nSaved {args.n} examples to {output}")
    print("Category counts:")
    for category, count in counts.items():
        print(f"  {category}: {count}")


if __name__ == "__main__":
    main()
