import argparse
import json
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from tqdm import tqdm


def load_model(base_model, adapter=None):
    tokenizer = AutoTokenizer.from_pretrained(base_model)

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        dtype=torch.float16,
        device_map="auto",
    )

    if adapter:
        model = PeftModel.from_pretrained(model, adapter)

    model.eval()

    return model, tokenizer


def build_prompt(x):
    """
    Construct prompt from instruction + facts.
    """
    return (
        x["instruction"]
        + "\n\n"
        + "\n".join(f"- {fact}" for fact in x["facts"])
        + "\n\nParagraph:"
    )


@torch.no_grad()
def analyze_prompt(model, tokenizer, prompt, max_new_tokens=150):

    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    ).to(model.device)

    current_ids = inputs["input_ids"]

    eos_id = tokenizer.eos_token_id

    eos_probs = []
    generated_tokens = []

    for step in range(max_new_tokens):

        outputs = model(current_ids)

        logits = outputs.logits[:, -1, :]

        probs = torch.softmax(
            logits.float(),
            dim=-1
        )

        eos_prob = probs[0, eos_id].item()
        eos_probs.append(eos_prob)

        # Greedy generation
        next_token = torch.argmax(
            logits,
            dim=-1
        )

        token_id = next_token.item()
        generated_tokens.append(token_id)

        if token_id == eos_id:
            break

        current_ids = torch.cat(
            [
                current_ids,
                next_token.unsqueeze(0)
            ],
            dim=1
        )

    # EOS probability before the actual terminating step
    if len(eos_probs) > 1:
        pre_eos_probs = eos_probs[:-1]
    else:
        pre_eos_probs = eos_probs

    return {
        "length": len(generated_tokens),

        "mean_eos_prob": float(
            np.mean(eos_probs)
        ),

        "mean_pre_eos_prob": float(
            np.mean(pre_eos_probs)
        ),

        "max_eos_prob": float(
            np.max(eos_probs)
        ),

        "final_eos_prob": float(
            eos_probs[-1]
        ),

        # Save full trajectory for later
        "eos_probs": eos_probs,
    }


def run_model(
    name,
    base_model,
    adapter,
    eval_file,
    output_file,
    limit=None
):

    print(f"\nLoading {name}...")

    model, tokenizer = load_model(
        base_model,
        adapter
    )

    with open(eval_file, encoding="utf-8") as f:
        examples = [
            json.loads(line)
            for line in f
            if line.strip()
        ]

    if limit:
        examples = examples[:limit]

    results = []

    for x in tqdm(
        examples,
        desc=name
    ):

        prompt = build_prompt(x)

        result = analyze_prompt(
            model,
            tokenizer,
            prompt
        )

        result["id"] = x["id"]
        result["category"] = x["category"]

        results.append(result)

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as f:

        for x in results:
            f.write(
                json.dumps(x)
                + "\n"
            )

    print(
        f"Saved: {output_file}"
    )

    # Free GPU before loading next condition
    del model

    torch.cuda.empty_cache()


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--eval_file",
        default="data/density_eval_v1.jsonl"
    )

    parser.add_argument(
        "--base_model",
        default="Qwen/Qwen2.5-7B"
    )

    parser.add_argument(
        "--poly",
        required=True
    )

    parser.add_argument(
        "--analytic",
        required=True
    )

    parser.add_argument(
        "--english",
        required=True
    )

    parser.add_argument(
        "--limit",
        type=int
    )

    args = parser.parse_args()

    conditions = [
        (
            "base",
            None,
            "outputs/eos_base.jsonl"
        ),
        (
            "poly",
            args.poly,
            "outputs/eos_poly243.jsonl"
        ),
        (
            "analytic",
            args.analytic,
            "outputs/eos_analytic243.jsonl"
        ),
        (
            "english",
            args.english,
            "outputs/eos_english243.jsonl"
        ),
    ]

    for name, adapter, output_file in conditions:

        run_model(
            name=name,
            base_model=args.base_model,
            adapter=adapter,
            eval_file=args.eval_file,
            output_file=output_file,
            limit=args.limit,
        )


if __name__ == "__main__":
    main()