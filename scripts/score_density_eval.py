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
    parser = argparse.ArgumentParser()

    parser.add_argument("--base", required=True)
    parser.add_argument("--poly", required=True)
    parser.add_argument("--analytic", required=True)
    parser.add_argument("--english", required=True)

    args = parser.parse_args()

    paths = {
        "Base": args.base,
        "Poly": args.poly,
        "Analytic": args.analytic,
        "English": args.english,
    }

    all_results = {}

    for model, path in paths.items():

        with open(path, encoding="utf-8") as f:
            xs = [
                json.loads(line)
                for line in f
                if line.strip()
            ]

        scored = []

        for x in xs:

            fact_scores = [
                score_fact(fact, x["response"])
                for fact in x["facts"]
            ]

            recall = mean(fact_scores)

            words = x["response_word_count"]
            tokens = x["response_token_count"]

            scored.append({
                "id": x["id"],
                "category": x["category"],
                "recall": recall,
                "words": words,
                "tokens": tokens,
                "density_words":
                    recall * len(x["facts"]) / words,
                "density_tokens":
                    recall * len(x["facts"]) / tokens,
            })

        all_results[model] = scored

    # --------------------------------------------------
    # Overall
    # --------------------------------------------------

    print("\nOVERALL")
    print("-" * 72)

    print(
        f"{'Model':<12}"
        f"{'Words':>10}"
        f"{'Tokens':>10}"
        f"{'Recall':>10}"
        f"{'Density/W':>14}"
        f"{'Density/T':>14}"
    )

    for model, xs in all_results.items():

        print(
            f"{model:<12}"
            f"{mean(x['words'] for x in xs):>10.1f}"
            f"{mean(x['tokens'] for x in xs):>10.1f}"
            f"{mean(x['recall'] for x in xs):>10.3f}"
            f"{mean(x['density_words'] for x in xs):>14.4f}"
            f"{mean(x['density_tokens'] for x in xs):>14.4f}"
        )

    # --------------------------------------------------
    # Categories
    # --------------------------------------------------

    categories = sorted({
        x["category"]
        for xs in all_results.values()
        for x in xs
    })

    # --------------------------------------------------
    # Recall by category
    # --------------------------------------------------

    print("\nFACT RECALL BY CATEGORY")
    print("-" * 70)

    print(
        f"{'Category':<22}"
        f"{'Base':>10}"
        f"{'Analytic':>10}"
        f"{'Poly':>10}"
        f"{'English':>10}"
    )

    for cat in categories:

        vals = {}

        for model, xs in all_results.items():
            subset = [
                x["recall"]
                for x in xs
                if x["category"] == cat
            ]

            vals[model] = mean(subset)

        print(
            f"{cat:<22}"
            f"{vals['Base']:>10.3f}"
            f"{vals['Analytic']:>10.3f}"
            f"{vals['Poly']:>10.3f}"
            f"{vals['English']:>10.3f}"
        )

    # --------------------------------------------------
    # Density by category
    # --------------------------------------------------

    print("\nINFORMATION DENSITY (FACTS / WORD)")
    print("-" * 70)

    print(
        f"{'Category':<22}"
        f"{'Base':>10}"
        f"{'Analytic':>10}"
        f"{'Poly':>10}"
        f"{'English':>10}"
    )

    for cat in categories:

        vals = {}

        for model, xs in all_results.items():
            subset = [
                x["density_words"]
                for x in xs
                if x["category"] == cat
            ]

            vals[model] = mean(subset)

        print(
            f"{cat:<22}"
            f"{vals['Base']:>10.4f}"
            f"{vals['Analytic']:>10.4f}"
            f"{vals['Poly']:>10.4f}"
            f"{vals['English']:>10.4f}"
        )


if __name__ == "__main__":
    main()