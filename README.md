# Reproducibility Repository

Code and selected artifacts for an anonymous submission studying how continued pretraining (CPT) on low-resource polysynthetic languages affects model behavior.

## Data

### Continued Pretraining

CPT data was obtained from the [Leipzig Corpora Collection](https://wortschatz.uni-leipzig.de/en/download).

The polysynthetic condition uses:
- Navajo: 10k sentences
- Kalaallisut: 30k web + 30k news sentences

Controls use Indonesian/Vietnamese (analytic-language control) and English.

The Leipzig data is not redistributed. Download the relevant corpora into `data/raw/`, then preprocess with:

```bash
python scripts/preprocess.py FILE1 FILE2 ...
```

`match_token_budget.py` is used to construct matched-token training corpora where required.

### Evaluation

`data/density_eval_v1.jsonl` contains the exact 500-example controlled English evaluation set used in the paper.

GSM8K is used for the reasoning experiments. It is not redistributed here and can be obtained from its original release or Hugging Face Datasets.

## Continued Pretraining

The primary experiments use `Qwen/Qwen2.5-7B` with LoRA CPT:

```text
Sequence length: 1024
LoRA r / alpha: 16 / 32
Learning rate: 2e-4
Warmup: 22 steps
Scheduler: cosine
Seed: 42
```

Example:

```bash
python scripts/midtrain_lora.py \
  --model Qwen/Qwen2.5-7B \
  --train_file data/processed/polysynthetic_train.txt \
  --output_dir checkpoints/poly \
  --gradient_accumulation_steps 4 \
  --seed 42
```

## Evaluation

Generate controlled English responses:

```bash
python scripts/run_density_eval.py \
  --eval_file data/density_eval_v1.jsonl \
  --output_file outputs/example.jsonl \
  --base_model Qwen/Qwen2.5-7B \
  --adapter PATH_TO_ADAPTER
```

Omit `--adapter` for the base model.

Score the included Qwen2.5-7B outputs:

```bash
python scripts/score_density_eval.py \
  --base outputs/density_v1_base.jsonl \
  --poly outputs/density_v1_poly243.jsonl \
  --analytic outputs/density_v1_analytic243.jsonl \
  --english outputs/density_v1_english243.jsonl
```

Run paired statistical tests with the same four files using `scripts/statistical_tests.py`.

## Scripts

| Script | Purpose |
|---|---|
| `preprocess.py` | Preprocess Leipzig corpora |
| `match_token_budget.py` | Match CPT token budgets |
| `midtrain_lora.py` | LoRA continued pretraining |
| `generate_density_eval_v1.py` | Generate controlled evaluation set |
| `run_density_eval.py` | Main English evaluation |
| `score_density_eval.py` | Length, recall, and density metrics |
| `statistical_tests.py` | Paired statistical tests |
| `analyze_density_dose*.py` | Dose-response analysis |
| `eos_analysis.py` | EOS analysis |
| `run_cot_eval.py` | GSM8K reasoning evaluation |
| `analyze_cot_eval.py` | Reasoning analysis |
| `run_density_layer_ablation.py` | Layer ablations |
| `run_density_layer_scale.py` | Layer scaling |
| `run_cot_layer_scale.py` | Layer scaling on reasoning |

## Included Outputs

Raw generations are provided for the principal Qwen2.5-7B conditions:

- Base
- Polysynthetic CPT
- Analytic CPT
- English CPT

and the Gemma-2-9B Base and Polysynthetic-CPT replication.

Everything can be regenerated using the provided scripts.
