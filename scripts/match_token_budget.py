import argparse
import random
from pathlib import Path

from transformers import AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_file", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--target_tokens", type=int, required=True)

    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-7B",
    )

    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def main():
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    with open(args.input_file, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    rng = random.Random(args.seed)
    rng.shuffle(lines)

    selected = []
    total_tokens = 0

    for line in lines:
        n_tokens = len(
            tokenizer.encode(
                line,
                add_special_tokens=False,
            )
        ) + 1  # EOS between sentences

        if total_tokens + n_tokens > args.target_tokens:
            continue

        selected.append(line)
        total_tokens += n_tokens

        if total_tokens >= args.target_tokens:
            break

    output = Path(args.output_file)
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("w", encoding="utf-8") as f:
        for line in selected:
            f.write(line + "\n")

    print(f"Selected sentences: {len(selected):,}")
    print(f"Token budget:       {total_tokens:,}")
    print(f"Target:             {args.target_tokens:,}")
    print(f"Difference:         {args.target_tokens - total_tokens:,}")


if __name__ == "__main__":
    main()