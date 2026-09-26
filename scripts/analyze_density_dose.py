import json
import numpy as np
from scipy.stats import wilcoxon

FILES = {
    0: "outputs/density_v1_base.jsonl",
    50: "outputs/density_v1_poly_step50.jsonl",
    100: "outputs/density_v1_poly_step100.jsonl",
    150: "outputs/density_v1_poly_step150.jsonl",
    200: "outputs/density_v1_poly_step200.jsonl",
    243: "outputs/density_v1_poly243.jsonl",
}

def load(path):
    with open(path) as f:
        xs = [json.loads(line) for line in f if line.strip()]
    return {x["id"]: x for x in xs}

data = {step: load(path) for step, path in FILES.items()}

# Make sure we're comparing exactly the same examples
ids = set(data[0])
for step in data:
    assert set(data[step]) == ids

ids = sorted(ids)

def values(step, key):
    return np.array([data[step][i][key] for i in ids], dtype=float)

print("\nWORDS: EACH CHECKPOINT VS BASE")
print("-" * 70)

base = values(0, "response_word_count")

for step in [50, 100, 150, 200, 243]:
    x = values(step, "response_word_count")
    diff = x - base
    stat, p = wilcoxon(diff)

    print(
        f"0 -> {step:<3} | "
        f"mean Δ = {diff.mean():7.3f} | "
        f"median Δ = {np.median(diff):6.1f} | "
        f"p = {p:.3e}"
    )

print("\nWORDS: ADJACENT CHECKPOINTS")
print("-" * 70)

pairs = [(0,50), (50,100), (100,150), (150,200), (200,243)]

for a, b in pairs:
    xa = values(a, "response_word_count")
    xb = values(b, "response_word_count")
    diff = xb - xa
    stat, p = wilcoxon(diff)

    print(
        f"{a:>3} -> {b:<3} | "
        f"mean Δ = {diff.mean():7.3f} | "
        f"median Δ = {np.median(diff):6.1f} | "
        f"p = {p:.3e}"
    )