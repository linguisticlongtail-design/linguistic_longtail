import argparse
import json
import re
from statistics import mean


STOPWORDS = {
    "a", "an", "the", "to", "in", "on", "at", "by",
    "of", "for", "and", "or", "was", "were", "is", "are",
    "has", "had", "have", "with", "from", "that", "this",
    "his", "her", "their", "also"
}


def normalize(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def content_units(fact):
    """Extract meaningful lexical/numerical units from a gold fact."""
    words = normalize(fact).split()

    return [
        w for w in words
        if w not in STOPWORDS and len(w) > 1
    ]


def score_fact(fact, response):
    """Fraction of meaningful units from this fact appearing in response."""
    units = content_units(fact)
    response_words = set(normalize(response).split())

    if not units:
        return 1.0

    hits = sum(unit in response_words for unit in units)
    return hits / len(units)


def main():
    paths = {
        "Base": "outputs/density_v1_base.jsonl",
        "Step 50": "outputs/density_v1_poly_step50.jsonl",
        "Step 100": "outputs/density_v1_poly_step100.jsonl",
        "Step 150": "outputs/density_v1_poly_step150.jsonl",
        "Step 200": "outputs/density_v1_poly_step200.jsonl",
        "Step 243": "outputs/density_v1_poly243.jsonl",
    }

    print("\nDOSE RESPONSE")
    print("-" * 76)
    print(
        f"{'Checkpoint':<12}"
        f"{'Words':>10}"
        f"{'Tokens':>10}"
        f"{'Recall':>10}"
        f"{'Density/W':>14}"
        f"{'Density/T':>14}"
    )

    for model, path in paths.items():
        with open(path, encoding="utf-8") as f:
            xs = [json.loads(line) for line in f if line.strip()]

        scored = []
        for x in xs:
            fact_scores = [score_fact(fact, x["response"]) for fact in x["facts"]]
            recall = mean(fact_scores)
            words = x["response_word_count"]
            tokens = x["response_token_count"]

            scored.append({
                "recall": recall,
                "words": words,
                "tokens": tokens,
                "density_words": recall * len(x["facts"]) / words,
                "density_tokens": recall * len(x["facts"]) / tokens,
            })

        print(
            f"{model:<12}"
            f"{mean(x['words'] for x in scored):>10.1f}"
            f"{mean(x['tokens'] for x in scored):>10.1f}"
            f"{mean(x['recall'] for x in scored):>10.3f}"
            f"{mean(x['density_words'] for x in scored):>14.4f}"
            f"{mean(x['density_tokens'] for x in scored):>14.4f}"
        )


if __name__ == "__main__":
    main()