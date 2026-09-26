import json
import numpy as np

FILES = {
    "Base": "outputs/eos_base.jsonl",
    "Poly": "outputs/eos_poly243.jsonl",
    "Analytic": "outputs/eos_analytic243.jsonl",
    "English": "outputs/eos_english243.jsonl",
}

# Relative positions through each response.
POSITIONS = [0.10, 0.25, 0.50, 0.75, 0.90, 0.95]


def load(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def prob_at_relative_position(probs, frac):
    """
    Get EOS probability at a given fraction of the response,
    excluding the final terminating EOS step.
    """
    # Remove actual final EOS decision
    pre = probs[:-1]

    if not pre:
        return np.nan

    idx = int(round(frac * (len(pre) - 1)))
    idx = max(0, min(idx, len(pre) - 1))

    return pre[idx]


print("\nEOS PROBABILITY BY RELATIVE RESPONSE POSITION")
print("=" * 88)

header = f"{'Model':<12}"
for p in POSITIONS:
    header += f"{int(p*100):>11}%"
print(header)

print("-" * 88)

for name, path in FILES.items():

    rows = load(path)

    values = []

    for frac in POSITIONS:

        probs = [
            prob_at_relative_position(
                r["eos_probs"],
                frac
            )
            for r in rows
        ]

        probs = [
            x for x in probs
            if not np.isnan(x)
        ]

        values.append(np.mean(probs))

    line = f"{name:<12}"

    for x in values:
        line += f"{x:>11.6f}"

    print(line)


print("\nEOS PROBABILITY IN FINAL RESPONSE QUARTER")
print("=" * 50)

for name, path in FILES.items():

    rows = load(path)

    vals = []

    for r in rows:

        pre = r["eos_probs"][:-1]

        if not pre:
            continue

        start = int(len(pre) * 0.75)

        vals.append(
            np.mean(pre[start:])
        )

    print(
        f"{name:<12} {np.mean(vals):.6f}"
    )