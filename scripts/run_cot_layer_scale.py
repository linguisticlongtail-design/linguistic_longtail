import argparse
import json
import re
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--eval_file", required=True)
    p.add_argument("--output_file", required=True)
    p.add_argument("--base_model", default="Qwen/Qwen2.5-7B")
    p.add_argument("--adapter", required=True)
    p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument("--prompt_type", choices=["standard", "more"], default="more")
    p.add_argument("--limit", type=int, default=None)

    p.add_argument(
        "--scale_layers",
        required=True,
        help="Layer range to scale, e.g. 7-13",
    )
    p.add_argument(
        "--scale",
        type=float,
        required=True,
        help="Multiplier for LoRA contribution in selected layers.",
    )

    return p.parse_args()


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

    if not changed:
        raise RuntimeError("No LoRA modules were modified.")


def load_examples(path, limit=None):
    with open(path, encoding="utf-8") as f:
        examples = [json.loads(x) for x in f if x.strip()]
    return examples[:limit] if limit else examples


def build_prompt(example, prompt_type):
    instruction = (
        "Let's think step by step."
        if prompt_type == "standard"
        else "Let's think step by step, you must think more steps."
    )
    return f"Q: {example['question']}\nA: {instruction}"


def extract_gold(gold):
    m = re.search(
        r"####\s*([-+]?\d[\d,]*(?:\.\d+)?)",
        gold,
    )
    return m.group(1).replace(",", "") if m else None


def truncate_first_answer(text):
    answer_pattern = re.compile(
        r"(.*?\bThe answer(?:\s+is)?\s*:?\s*"
        r"[-+]?\$?\d[\d,]*(?:\.\d+)?"
        r"(?:\s*[A-Za-z]+)?[.!]?)",
        re.IGNORECASE | re.DOTALL,
    )

    m = answer_pattern.match(text)

    if m:
        return m.group(1).strip()

    split = re.split(
        r"\n\s*(?:Q:|\[Question\]|Question:)",
        text,
        maxsplit=1,
        flags=re.IGNORECASE,
    )

    return split[0].strip()


def extract_answer(text):
    m = re.search(
        r"\bThe answer(?:\s+is)?\s*:?\s*\$?"
        r"([-+]?\d[\d,]*(?:\.\d+)?)",
        text,
        re.IGNORECASE,
    )

    if m:
        return m.group(1).replace(",", "")

    nums = re.findall(
        r"[-+]?\d[\d,]*(?:\.\d+)?",
        text,
    )

    return nums[-1].replace(",", "") if nums else None


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
    print(f"Prompt type: {args.prompt_type}")

    output_path = Path(args.output_file)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open("w", encoding="utf-8") as out:

        for i, example in enumerate(examples):

            prompt = build_prompt(
                example,
                args.prompt_type,
            )

            inputs = tokenizer(
                prompt,
                return_tensors="pt",
            ).to(model.device)

            with torch.no_grad():
                generated = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

            prompt_length = inputs["input_ids"].shape[1]
            raw_tokens = generated[0, prompt_length:]

            raw_response = tokenizer.decode(
                raw_tokens,
                skip_special_tokens=True,
            ).strip()

            response = truncate_first_answer(
                raw_response
            )

            response_tokens = tokenizer(
                response,
                add_special_tokens=False,
            )["input_ids"]

            pred = extract_answer(response)
            gold = extract_gold(example["gold"])

            result = {
                **example,
                "prompt": prompt,
                "response": response,
                "raw_response": raw_response,
                "response_word_count": len(response.split()),
                "response_token_count": len(response_tokens),
                "raw_token_count": len(raw_tokens),
                "pred_answer": pred,
                "gold_answer": gold,
                "correct": pred == gold,
                "scale_layers": args.scale_layers,
                "scale": args.scale,
            }

            out.write(
                json.dumps(
                    result,
                    ensure_ascii=False,
                ) + "\n"
            )

            print(
                f"[{i+1}/{len(examples)}] "
                f"{len(response_tokens)} tokens | "
                f"pred={pred} gold={gold} | "
                f"{'✓' if pred == gold else '✗'}"
            )

    print(f"\nSaved results to: {output_path}")


if __name__ == "__main__":
    main()
