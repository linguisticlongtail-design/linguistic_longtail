import argparse
import json
import re
from pathlib import Path
from statistics import mean

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_file", required=True)
    parser.add_argument("--output_file", required=True)

    parser.add_argument(
        "--nli_model",
        default="facebook/bart-large-mnli",
    )

    parser.add_argument(
        "--entailment_threshold",
        type=float,
        default=0.70,
    )

    return parser.parse_args()


def split_claims(text):
    """
    Conservative first-pass claim decomposition:
    split generated text into sentence-like units.
    """
    claims = re.split(r"(?<=[.!?])\s+", text.strip())
    return [c.strip() for c in claims if c.strip()]


def main():
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.nli_model)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.nli_model,
        dtype=torch.float16,
    ).cuda()

    model.eval()

    # BART-MNLI labels:
    # 0 = contradiction
    # 1 = neutral
    # 2 = entailment
    entailment_id = 2

    with open(args.input_file, encoding="utf-8") as f:
        examples = [json.loads(line) for line in f if line.strip()]

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_supported = 0
    total_unsupported = 0

    with output_path.open("w", encoding="utf-8") as out:

        for i, x in enumerate(examples):

            premise = " ".join(x["facts"])
            claims = split_claims(x["response"])

            claim_results = []

            for claim in claims:

                inputs = tokenizer(
                    premise,
                    claim,
                    return_tensors="pt",
                    truncation=True,
                    max_length=1024,
                ).to("cuda")

                with torch.no_grad():
                    logits = model(**inputs).logits

                probs = torch.softmax(
                    logits,
                    dim=-1,
                )[0]

                entailment_prob = probs[entailment_id].item()

                supported = (
                    entailment_prob >= args.entailment_threshold
                )

                claim_results.append({
                    "claim": claim,
                    "entailment_prob": entailment_prob,
                    "supported": supported,
                })

            supported_count = sum(
                c["supported"] for c in claim_results
            )

            unsupported_count = (
                len(claim_results) - supported_count
            )

            unsupported_rate = (
                unsupported_count / len(claim_results)
                if claim_results
                else 0.0
            )

            total_supported += supported_count
            total_unsupported += unsupported_count

            result = {
                "id": x["id"],
                "category": x["category"],
                "response": x["response"],
                "claims": claim_results,
                "supported_claims": supported_count,
                "unsupported_claims": unsupported_count,
                "unsupported_rate": unsupported_rate,
            }

            out.write(
                json.dumps(
                    result,
                    ensure_ascii=False,
                ) + "\n"
            )

            if (i + 1) % 25 == 0:
                print(f"{i + 1}/{len(examples)}")

    total_claims = total_supported + total_unsupported

    print("\nDone")
    print(f"Supported claims:   {total_supported}")
    print(f"Unsupported claims: {total_unsupported}")

    if total_claims:
        print(
            f"Unsupported rate:   "
            f"{total_unsupported / total_claims:.3f}"
        )

    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()