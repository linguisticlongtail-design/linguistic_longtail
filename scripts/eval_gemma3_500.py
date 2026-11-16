import argparse
import json
import os

import torch
from transformers import AutoProcessor, Gemma3ForConditionalGeneration
from peft import PeftModel

BASE = "google/gemma-3-4b-pt"
ADAPTER = "outputs/gemma3_4b_poly_eos_seed42/final"
DATASET = "data/density_eval_v1.jsonl"

def load_prompts(path):
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]

    assert len(rows) == 500, f"Expected 500 prompts; found {len(rows)}"

    for i, row in enumerate(rows):
        if not isinstance(row.get("instruction"), str) or not isinstance(row.get("facts"), list):
            raise ValueError(
                f"Row {i} has no string 'prompt' field. "
                f"Available keys: {list(row.keys())}. "
                "Check the dataset schema before running."
            )

    return rows

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", choices=["base", "poly"], required=True)
    parser.add_argument("--max_new_tokens", type=int, default=150)
    args = parser.parse_args()

    torch.manual_seed(42)

    rows = load_prompts(DATASET)

    output = f"outputs/gemma3_4b_{args.condition}_density500.jsonl"
    os.makedirs("outputs", exist_ok=True)

    processor = AutoProcessor.from_pretrained(BASE)
    tokenizer = processor.tokenizer

    model = Gemma3ForConditionalGeneration.from_pretrained(
        BASE,
        dtype=torch.float32,
        low_cpu_mem_usage=True,
    ).to("cuda:0")

    if args.condition == "poly":
        model = PeftModel.from_pretrained(model, ADAPTER)

    model.eval()

    # Resume safely if a generation run was interrupted.
    completed = set()

    if os.path.exists(output):
        with open(output, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    completed.add(json.loads(line)["id"])

    print(f"Condition: {args.condition}")
    print(f"Total examples: {len(rows)}")
    print(f"Already completed: {len(completed)}")
    print(f"Output: {output}", flush=True)

    with open(output, "a", encoding="utf-8") as f:
        for index, row in enumerate(rows, 1):
            example_id = row.get("id", index)

            if example_id in completed:
                continue

            facts = "\n".join(
                f"- {fact}" for fact in row["facts"]
            )
            prompt = (
                f"{row['instruction']}\n\n"
                f"{facts}\n\n"
                f"Paragraph:"
            )

            inputs = tokenizer(
                prompt,
                return_tensors="pt",
            ).to("cuda:0")

            prompt_length = inputs["input_ids"].shape[-1]

            with torch.inference_mode():
                generated = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                )

            ids = generated[0, prompt_length:].tolist()

            response = tokenizer.decode(
                ids,
                skip_special_tokens=True,
            )

            result = {
                "id": example_id,
                "category": row.get("category"),
                "prompt": prompt,
                "condition": args.condition,
                "response": response,
                "response_token_count": len(ids),
                "eos_reached": tokenizer.eos_token_id in ids,
                "truncated": (
                    tokenizer.eos_token_id not in ids
                    and len(ids) >= args.max_new_tokens
                ),
                "pad_count": ids.count(tokenizer.pad_token_id)
                    if tokenizer.pad_token_id is not None else 0,
            }

            f.write(json.dumps(result, ensure_ascii=False) + "\n")
            f.flush()

            if index % 25 == 0 or index == len(rows):
                print(
                    f"[{args.condition}] {index}/500 | "
                    f"tokens={len(ids)} | "
                    f"EOS={result['eos_reached']}",
                    flush=True,
                )

    print(f"DONE: {output}", flush=True)

if __name__ == "__main__":
    main()
