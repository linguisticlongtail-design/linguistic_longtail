import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--eval_file",
        required=True,
        help="Path to JSONL evaluation dataset.",
    )

    parser.add_argument(
        "--output_file",
        required=True,
        help="Where to save generations.",
    )

    parser.add_argument(
        "--base_model",
        default="Qwen/Qwen2.5-7B",
    )

    parser.add_argument(
        "--adapter",
        default=None,
        help="Path to LoRA adapter. Omit for the base model.",
    )

    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=150,
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only evaluate the first N examples.",
    )

    return parser.parse_args()


def load_examples(path, limit=None):
    examples = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))

    if limit is not None:
        examples = examples[:limit]

    return examples


def build_prompt(example):
    facts = "\n".join(
        f"- {fact}"
        for fact in example["facts"]
    )

    return (
        f"{example['instruction']}\n\n"
        f"{facts}\n\n"
        f"Paragraph:"
    )


def main():
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        dtype=torch.float16,
        device_map="auto",
    )

    if args.adapter:
        print(f"Loading LoRA adapter: {args.adapter}")

        model = PeftModel.from_pretrained(
            model,
            args.adapter,
        )
    else:
        print("Running BASE model")

    model.eval()

    examples = load_examples(
        args.eval_file,
        args.limit,
    )

    print(f"Evaluating {len(examples)} examples")

    output_path = Path(args.output_file)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open("w", encoding="utf-8") as out:

        for i, example in enumerate(examples):

            prompt = build_prompt(example)

            inputs = tokenizer(
                prompt,
                return_tensors="pt",
            ).to(model.device)

            with torch.no_grad():
                generated = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,

                    # Deterministic decoding
                    do_sample=False,

                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

            prompt_length = inputs["input_ids"].shape[1]

            new_tokens = generated[
                0,
                prompt_length:
            ]

            response = tokenizer.decode(
                new_tokens,
                skip_special_tokens=True,
            ).strip()

            result = {
                **example,
                "prompt": prompt,
                "response": response,
                "response_word_count": len(response.split()),
                "response_token_count": len(new_tokens),
            }

            out.write(
                json.dumps(
                    result,
                    ensure_ascii=False,
                ) + "\n"
            )

            print(
                f"[{i + 1}/{len(examples)}] "
                f"{len(response.split())} words | "
                f"{len(new_tokens)} tokens"
            )

    print(f"\nSaved results to: {output_path}")


if __name__ == "__main__":
    main()