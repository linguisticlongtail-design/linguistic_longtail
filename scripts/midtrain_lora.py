import argparse
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-7B",
    )
    parser.add_argument(
        "--train_file",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
    )

    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--max_steps", type=int, default=-1)

    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--warmup_steps", type=int, default=22)
    parser.add_argument(
        "--lr_scheduler_type",
        type=str,
        default="cosine",
    )

    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=16,
    )

    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def load_text_dataset(path):
    path = Path(path)

    with path.open("r", encoding="utf-8") as f:
        texts = [line.strip() for line in f if line.strip()]

    print(f"Loaded {len(texts):,} sentences from {path}")

    return Dataset.from_dict({"text": texts})


def main():
    args = parse_args()

    print(f"\nModel:       {args.model}")
    print(f"Train file:  {args.train_file}")
    print(f"Output:      {args.output_dir}")
    print(f"Epochs:      {args.epochs}")
    print(f"Max steps:   {args.max_steps}")
    print(f"LR:          {args.learning_rate}")
    print(f"Scheduler:   {args.lr_scheduler_type}")
    print(f"Warmup:      {args.warmup_steps} steps\n")

    # ---------------------------------------------------------
    # Tokenizer
    # ---------------------------------------------------------

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---------------------------------------------------------
    # Dataset
    # ---------------------------------------------------------

    dataset = load_text_dataset(args.train_file)

    # Reproducible shuffle
    dataset = dataset.shuffle(seed=args.seed)

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            add_special_tokens=False,
        )

    tokenized = dataset.map(
        tokenize,
        batched=True,
        remove_columns=["text"],
        desc="Tokenizing",
    )

    # ---------------------------------------------------------
    # Pack sentences into contiguous token sequences.
    #
    # Add EOS between sentences, concatenate, then split into
    # fixed max_length blocks for efficient causal-LM training.
    # ---------------------------------------------------------

    eos_id = tokenizer.eos_token_id

    def group_texts(examples):
        concatenated = []

        for ids in examples["input_ids"]:
            concatenated.extend(ids)
            concatenated.append(eos_id)

        usable_length = (
            len(concatenated) // args.max_length
        ) * args.max_length

        blocks = [
            concatenated[i:i + args.max_length]
            for i in range(
                0,
                usable_length,
                args.max_length,
            )
        ]

        return {
            "input_ids": blocks,
            "attention_mask": [
                [1] * len(block)
                for block in blocks
            ],
        }

    packed = tokenized.map(
        group_texts,
        batched=True,
        batch_size=1000,
        remove_columns=tokenized.column_names,
        desc="Packing",
    )

    print(f"Packed dataset: {len(packed):,} sequences")
    print(
        f"Approx tokens used: "
        f"{len(packed) * args.max_length:,}"
    )

    # ---------------------------------------------------------
    # Base model
    # ---------------------------------------------------------

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.float16,
    )

    model.config.use_cache = False

    # ---------------------------------------------------------
    # LoRA
    # ---------------------------------------------------------

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,

        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,

        bias="none",

        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
        ],
    )

    model = get_peft_model(
        model,
        lora_config,
    )

    model.print_trainable_parameters()

    # ---------------------------------------------------------
    # Causal next-token prediction
    # ---------------------------------------------------------

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    # ---------------------------------------------------------
    # Training configuration
    # ---------------------------------------------------------

    training_args = TrainingArguments(
        output_dir=args.output_dir,

        num_train_epochs=args.epochs,
        max_steps=args.max_steps,

        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        lr_scheduler_type=args.lr_scheduler_type,

        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=(
            args.gradient_accumulation_steps
        ),

        fp16=True,
        gradient_checkpointing=True,

        logging_strategy="steps",
        logging_steps=10,

        save_strategy="steps",
        save_steps=50,
        save_total_limit=10,

        report_to="none",

        remove_unused_columns=False,
        seed=args.seed,

        dataloader_num_workers=4,

        # Avoid unnecessary DDP autograd graph traversal.
        ddp_find_unused_parameters=False,
    )

    # ---------------------------------------------------------
    # Trainer
    # ---------------------------------------------------------

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=packed,
        data_collator=data_collator,
        processing_class=tokenizer,
    )

    trainer.train()

    # ---------------------------------------------------------
    # Save final LoRA adapter.
    # Only rank 0 writes during distributed training.
    # ---------------------------------------------------------

    if trainer.is_world_process_zero():
        final_dir = Path(args.output_dir) / "final"
        final_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        model.save_pretrained(final_dir)
        tokenizer.save_pretrained(final_dir)

        print(
            f"\nSaved final LoRA adapter to: "
            f"{final_dir}"
        )


if __name__ == "__main__":
    main()