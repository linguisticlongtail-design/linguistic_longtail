import argparse
import json
import re
import numpy as np
from scipy.stats import ttest_rel
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
    return [
        w for w in normalize(fact).split()
        if w not in STOPWORDS and len(w) > 1
    ]


def score_fact(fact, response):
    units = content_units(fact)
    response_words = set(normalize(response).split())

    if not units:
        return 1.0

    return sum(u in response_words for u in units) / len(units)


def load(path):
    with open(path, encoding="utf-8") as f:
        xs = [json.loads(line) for line in f if line.strip()]

    results = {}

    for x in xs:
        recall = mean([
            score_fact(fact, x["response"])
            for fact in x["facts"]
        ])

        words = x["response_word_count"]
        tokens = x["response_token_count"]

        results[x["id"]] = {
            "words": words,
            "tokens": tokens,
            "recall": recall,
            "density": recall * len(x["facts"]) / words,
        }

    return results


def compare(poly, control, control_name):
    ids = sorted(set(poly) & set(control))

    print(f"\nPOLY vs {control_name}")
    print("-" * 65)

    for metric in ["words", "tokens", "recall", "density"]:

        p = np.array([poly[i][metric] for i in ids])
        c = np.array([control[i][metric] for i in ids])

        diff = p - c

        # Paired t-test
        _, pvalue = ttest_rel(p, c)

        # 95% CI for paired mean difference
        mean_diff = np.mean(diff)
        se = np.std(diff, ddof=1) / np.sqrt(len(diff))
        ci_low = mean_diff - 1.96 * se
        ci_high = mean_diff + 1.96 * se

        # Paired Cohen's dz
        sd = np.std(diff, ddof=1)
        effect = mean_diff / sd if sd > 0 else 0

        print(
            f"{metric:<10} "
            f"Poly={np.mean(p):>8.3f}  "
            f"{control_name}={np.mean(c):>8.3f}  "
            f"Δ={mean_diff:>8.3f}  "
            f"95%CI=[{ci_low:.3f}, {ci_high:.3f}]  "
            f"p={pvalue:.3e}  "
            f"dz={effect:.3f}"
        )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--base", required=True)
    parser.add_argument("--poly", required=True)
    parser.add_argument("--analytic", required=True)
    parser.add_argument("--english", required=True)

    args = parser.parse_args()

    base = load(args.base)
    poly = load(args.poly)
    analytic = load(args.analytic)
    english = load(args.english)

    compare(poly, base, "Base")
    compare(poly, analytic, "Analytic")
    compare(poly, english, "English")


if __name__ == "__main__":
    main()