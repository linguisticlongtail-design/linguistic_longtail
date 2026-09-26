import argparse
import json
import re
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--eval_file", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--base_model", default="Qwen/Qwen2.5-7B")
    parser.add_argument("--adapter", required=True)

    parser.add_argument(
        "--scale_layers",
        required=True,
        help="Layer range to scale, e.g. 7-13",
    )

    parser.add_argument(
        "--scale",
        type=float,
        required=True,
        help="Multiplier applied to LoRA contribution in selected layers.",
    )

    parser.add_argument("--max_new_tokens", type=int, default=150)
    parser.add_argument("--limit", type=int, default=None)

    return parser.parse_args()


def parse_layers(spec):
    layers = set()

    for part in spec.split(","):
        part = part.strip()

        if "-" in part:
            start, end = map(int, part.split("-"))
            layers.update(range(start, end + 1))
        else:
            layers.add(int(part))

    return layers


def scale_lora_layers(model, layers, scale):
    """
    Multiply the LoRA contribution in selected transformer layers.

    scale = 0.0 -> remove LoRA contribution in these layers
    scale = 0.5 -> half-strength LoRA
    scale = 1.0 -> original Poly model
    scale = 1.5 -> amplified LoRA
    """

    changed = []

    for name, module in model.named_modules():

        match = re.search(r"\.layers\.(\d+)\.", name)

        if not match:
            continue

        layer = int(match.group(1))

        if layer not in layers:
            continue

        if hasattr(module, "scaling") and isinstance(module.scaling, dict):

            for adapter_name in list(module.scaling.keys()):
                module.scaling[adapter_name] *= scale

            changed.append(name)

    print(f"Scaled layers: {sorted(layers)}")
    print(f"LoRA multiplier: {scale}")
    print(f"Modified {len(changed)} LoRA modules")

    if len(changed) == 0:
        raise RuntimeError(
            "No LoRA modules were modified. Check layer naming."
        )


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

    print(f"Loading LoRA adapter: {args.adapter}")

    model = PeftModel.from_pretrained(
        model,
        args.adapter,
    )

    layers = parse_layers(args.scale_layers)

    scale_lora_layers(
        model,
        layers,
        args.scale,
    )

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

                    # Same deterministic decoding as main evaluation
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

                "response_word_count":
                    len(response.split()),

                "response_token_count":
                    len(new_tokens),

                "scale_layers":
                    args.scale_layers,

                "scale":
                    args.scale,
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

    print(
        f"\nSaved results to: {output_path}"
    )


if __name__ == "__main__":
    main()
