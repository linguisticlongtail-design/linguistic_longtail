import json
import re
import numpy as np
from statistics import mean
from scipy.stats import wilcoxon, page_trend_test

SEED = 42
N_BOOT = 10000

FILES = {
    0: "outputs/density_v1_base.jsonl",
    50: "outputs/density_v1_poly_step50.jsonl",
    100: "outputs/density_v1_poly_step100.jsonl",
    150: "outputs/density_v1_poly_step150.jsonl",
    200: "outputs/density_v1_poly_step200.jsonl",
    243: "outputs/density_v1_poly243.jsonl",
}

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

    out = {}

    for x in xs:
        recall = mean(
            score_fact(fact, x["response"])
            for fact in x["facts"]
        )

        words = x["response_word_count"]
        tokens = x["response_token_count"]
        n_facts = len(x["facts"])

        out[x["id"]] = {
            "words": words,
            "tokens": tokens,
            "recall": recall,
            "density_words": recall * n_facts / words,
            "density_tokens": recall * n_facts / tokens,
        }

    return out

data = {step: load(path) for step, path in FILES.items()}

ids = sorted(data[0])
for step in data:
    assert set(data[step]) == set(ids)

steps = list(FILES.keys())

def vals(step, metric):
    return np.array(
        [data[step][i][metric] for i in ids],
        dtype=float
    )

rng = np.random.default_rng(SEED)

def bootstrap_mean_ci(x):
    n = len(x)
    boot = np.empty(N_BOOT)

    for b in range(N_BOOT):
        idx = rng.integers(0, n, n)
        boot[b] = x[idx].mean()

    return np.percentile(boot, [2.5, 97.5])

def bootstrap_paired_diff_ci(a, b):
    """CI for mean(b - a), preserving pairing."""
    diff = b - a
    n = len(diff)
    boot = np.empty(N_BOOT)

    for k in range(N_BOOT):
        idx = rng.integers(0, n, n)
        boot[k] = diff[idx].mean()

    return np.percentile(boot, [2.5, 97.5])

# --------------------------------------------------
# Means + bootstrap confidence intervals
# --------------------------------------------------

for metric in ["words", "recall", "density_words"]:
    print(f"\n{metric.upper()}: MEAN AND 95% BOOTSTRAP CI")
    print("-" * 68)

    for step in steps:
        x = vals(step, metric)
        lo, hi = bootstrap_mean_ci(x)

        print(
            f"{step:>3}: "
            f"mean = {x.mean():.4f} | "
            f"95% CI [{lo:.4f}, {hi:.4f}]"
        )

# --------------------------------------------------
# Paired Base -> checkpoint differences
# --------------------------------------------------

print("\nWORDS: PAIRED CHANGE FROM BASE")
print("-" * 78)

base = vals(0, "words")

for step in steps[1:]:
    x = vals(step, "words")
    diff = x - base

    lo, hi = bootstrap_paired_diff_ci(base, x)
    _, p = wilcoxon(diff)

    print(
        f"0 -> {step:<3} | "
        f"Δ = {diff.mean():7.3f} | "
        f"95% CI [{lo:7.3f}, {hi:7.3f}] | "
        f"p = {p:.3e}"
    )

# --------------------------------------------------
# Adjacent checkpoint differences
# --------------------------------------------------

print("\nWORDS: ADJACENT PAIRED CHANGES")
print("-" * 78)

for a, b in zip(steps[:-1], steps[1:]):
    xa = vals(a, "words")
    xb = vals(b, "words")
    diff = xb - xa

    lo, hi = bootstrap_paired_diff_ci(xa, xb)
    _, p = wilcoxon(diff)

    print(
        f"{a:>3} -> {b:<3} | "
        f"Δ = {diff.mean():7.3f} | "
        f"95% CI [{lo:7.3f}, {hi:7.3f}] | "
        f"p = {p:.3e}"
    )

# --------------------------------------------------
# Ordered repeated-measures trend
# Page test: negate words so expected shortening
# becomes an increasing ordered trend.
# --------------------------------------------------

word_matrix = np.column_stack([
    vals(step, "words") for step in steps
])

page_words = page_trend_test(-word_matrix)

print("\nORDERED TREND TEST")
print("-" * 68)
print(
    "Words decrease with increasing CPT exposure:\n"
    f"Page L = {page_words.statistic:.3f}, "
    f"p = {page_words.pvalue:.3e}"
)

# Recall expected to increase
recall_matrix = np.column_stack([
    vals(step, "recall") for step in steps
])

page_recall = page_trend_test(recall_matrix)

print(
    "\nRecall increases with increasing CPT exposure:\n"
    f"Page L = {page_recall.statistic:.3f}, "
    f"p = {page_recall.pvalue:.3e}"
)

# Density expected to increase
density_matrix = np.column_stack([
    vals(step, "density_words") for step in steps
])

page_density = page_trend_test(density_matrix)

print(
    "\nDensity increases with increasing CPT exposure:\n"
    f"Page L = {page_density.statistic:.3f}, "
    f"p = {page_density.pvalue:.3e}"
)