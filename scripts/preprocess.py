from pathlib import Path
import argparse
from transformers import AutoTokenizer

RAW_DIR = Path("data/raw")
OUT_DIR = Path("data/processed")
MODEL = "Qwen/Qwen2.5-7B"

OUT_DIR.mkdir(parents=True, exist_ok=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL)


def clean_line(line):
    line = line.strip()

    if not line:
        return None

    # Remove Leipzig sentence ID
    parts = line.split("\t", 1)
    if len(parts) == 2 and parts[0].strip().isdigit():
        line = parts[1].strip()

    return line or None


def process_file(filename):
    path = RAW_DIR / filename
    seen = set()
    lines = []

    with path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            text = clean_line(raw_line)

            if text and text not in seen:
                seen.add(text)
                lines.append(text)

    output_path = OUT_DIR / filename
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    n_tokens = sum(
        len(tokenizer.encode(text, add_special_tokens=False))
        for text in lines
    )

    print(
        f"{filename}: "
        f"{len(lines):,} sentences | "
        f"{n_tokens:,} Qwen tokens | "
        f"{n_tokens / len(lines):.1f} tokens/sentence"
    )


parser = argparse.ArgumentParser()
parser.add_argument("files", nargs="+")
args = parser.parse_args()

for filename in args.files:
    process_file(filename)