import json
import numpy as np
from scipy.stats import wilcoxon, binomtest

FILES = {
    "Base": {
        "standard": "outputs/cot_base_full.jsonl",
        "more": "outputs/cot_base_more_full.jsonl",
    },
    "Poly": {
        "standard": "outputs/cot_poly243_full.jsonl",
        "more": "outputs/cot_poly243_more_full.jsonl",
    },
    "Analytic": {
        "standard": "outputs/cot_analytic243_full.jsonl",
        "more": "outputs/cot_analytic243_more_full.jsonl",
    },
    "English": {
        "standard": "outputs/cot_english243_full.jsonl",
        "more": "outputs/cot_english243_more_full.jsonl",
    },
}


def load(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def get_correct(x):
    return bool(x["correct"])


def get_tokens(x):
    return int(x["response_token_count"])


data = {
    model: {prompt: load(path) for prompt, path in paths.items()}
    for model, paths in FILES.items()
}


# ============================================================
# 1. Overall results
# ============================================================

print("\nOVERALL RESULTS")
print("=" * 78)
print(
    f"{'Model':<12} {'Prompt':<10} {'N':>6} "
    f"{'Acc':>9} {'Mean tok':>11} {'Median':>10}"
)
print("-" * 78)

for model in FILES:
    for prompt in ["standard", "more"]:
        rows = data[model][prompt]

        correct = np.array([get_correct(x) for x in rows])
        lengths = np.array([get_tokens(x) for x in rows])

        print(
            f"{model:<12} {prompt:<10} {len(rows):>6} "
            f"{correct.mean()*100:>8.2f}% "
            f"{lengths.mean():>11.2f} "
            f"{np.median(lengths):>10.1f}"
        )


# ============================================================
# 2. Effect of "think more steps"
# ============================================================

print("\nSTANDARD -> MORE STEPS")
print("=" * 70)
print(
    f"{'Model':<12} {'Standard':>12} {'More':>12} "
    f"{'Delta':>12} {'p':>14}"
)
print("-" * 70)

for model in FILES:
    standard = np.array(
        [get_tokens(x) for x in data[model]["standard"]]
    )
    more = np.array(
        [get_tokens(x) for x in data[model]["more"]]
    )

    _, p = wilcoxon(more, standard)

    print(
        f"{model:<12} "
        f"{standard.mean():>12.2f} "
        f"{more.mean():>12.2f} "
        f"{(more - standard).mean():>+12.2f} "
        f"{p:>14.3g}"
    )


# ============================================================
# 3. Shared-correct subsets
# ============================================================

shared_correct = {}

for prompt in ["standard", "more"]:

    masks = [
        np.array([get_correct(x) for x in data[model][prompt]])
        for model in FILES
    ]

    shared = np.logical_and.reduce(masks)
    idx = np.where(shared)[0]

    shared_correct[prompt] = idx

    print(f"\nSHARED-CORRECT: {prompt.upper()}")
    print(f"N = {len(idx)} / 1319")
    print("=" * 60)

    for model in FILES:
        lengths = np.array(
            [get_tokens(x) for x in data[model][prompt]]
        )[idx]

        print(
            f"{model:<12} "
            f"mean={lengths.mean():8.2f} "
            f"median={np.median(lengths):7.1f}"
        )


# ============================================================
# 4. Poly vs controls under MORE STEPS
#    Only questions all four models answered correctly
# ============================================================

idx = shared_correct["more"]

poly = np.array(
    [get_tokens(x) for x in data["Poly"]["more"]]
)[idx]

print("\nPOLY VS CONTROLS: MORE STEPS, SHARED-CORRECT")
print("=" * 75)

for control in ["Base", "Analytic", "English"]:

    other = np.array(
        [get_tokens(x) for x in data[control]["more"]]
    )[idx]

    diff = poly - other

    _, p = wilcoxon(poly, other)

    print(
        f"Poly vs {control:<8} "
        f"mean difference = {diff.mean():>8.2f} tokens | "
        f"p = {p:.3g}"
    )


# ============================================================
# 5. Accuracy change: Standard -> More Steps
#    Exact McNemar test
# ============================================================

print("\nACCURACY CHANGE: STANDARD -> MORE STEPS")
print("=" * 75)

for model in FILES:

    standard = np.array(
        [get_correct(x) for x in data[model]["standard"]]
    )

    more = np.array(
        [get_correct(x) for x in data[model]["more"]]
    )

    # Correct under standard, wrong under more
    lost = np.sum(standard & ~more)

    # Wrong under standard, correct under more
    gained = np.sum(~standard & more)

    # Exact McNemar test
    if lost + gained:
        p = binomtest(
            min(lost, gained),
            lost + gained,
            0.5
        ).pvalue
    else:
        p = 1.0

    print(
        f"{model:<12} "
        f"{standard.mean()*100:6.2f}% -> "
        f"{more.mean()*100:6.2f}% | "
        f"lost={lost:3d} gained={gained:3d} | "
        f"p={p:.3g}"
    )

# ============================================================
# 6. Difference-in-differences
#    Does Poly respond less to verbosity pressure?
# ============================================================

print("\nDIFFERENCE-IN-DIFFERENCES: RESPONSE TO VERBOSITY PRESSURE")
print("=" * 82)

def lengths(model, prompt):
    return np.array([
        get_tokens(x) for x in data[model][prompt]
    ])

# Per-question change induced by "think more steps"
poly_delta = lengths("Poly", "more") - lengths("Poly", "standard")

print(
    f"{'Comparison':<22} "
    f"{'Poly Δ':>10} "
    f"{'Control Δ':>12} "
    f"{'DiD':>10} "
    f"{'p':>14}"
)
print("-" * 82)

for control in ["Base", "Analytic", "English"]:

    control_delta = (
        lengths(control, "more")
        - lengths(control, "standard")
    )

    # Difference-in-differences:
    # negative = Poly expands less than control
    did = poly_delta - control_delta

    _, p = wilcoxon(poly_delta, control_delta)

    print(
        f"Poly vs {control:<14} "
        f"{poly_delta.mean():>10.2f} "
        f"{control_delta.mean():>12.2f} "
        f"{did.mean():>+10.2f} "
        f"{p:>14.3g}"
    )