# Dataset: `eriksalt/reddit-rpg-rules-question-classification`

## What it is
A small, curated **binary text-classification** dataset intended to train a model to decide whether a Reddit post from tabletop-RPG communities is a **rules question** or **not a rules question**.

Hugging Face Hub page: https://huggingface.co/datasets/eriksalt/reddit-rpg-rules-question-classification

## Labels
The dataset uses an integer `ClassLabel` with two named classes:

- `0` → **Rules Question**
- `1` → **Other**

## Row schema
Each example is a single Reddit post (as plain text) with three fields:

- `id` *(string)*: A stable identifier that also encodes the source file and line number (e.g. `blades_posts.txt:755`).
- `content` *(string)*: The post text used for classification (typically includes the post title plus body/description where present).
- `label` *(ClassLabel int)*: `0` for rules questions, `1` for everything else.

## Splits and size
The dataset is published in Parquet format with one config (`default`) and four splits:

- `train`: **1,559** rows
- `validation`: **195** rows
- `test`: **195** rows
- `large`: **25** rows

Total: **1,974** rows.

## Notable characteristics
- **Source hinting via `id`:** IDs commonly look like `blades_posts.txt:<n>` or `mothership_posts.txt:<n>`, which makes it easy to trace examples back to the original extraction batch.
- **Wide length range:** `content` ranges from very short titles to multi-paragraph posts (the dataset viewer shows examples up to ~16k characters).
- **"large" split:** A small extra set of data that overlaps with train, validation, and test. It is the longest rows in the data, and can be used for stress-testing longer or more complex posts, or as a manual spot-check set. 

## Intended use
- Fine-tuning / instruction-tuning a classifier (e.g., Qwen2.5-14B-Instruct) to output one of two labels.
- Training/evaluating a cheaper routing model (e.g., fast filter → expensive model only when likely rules-related).
- Building a rules-QA pipeline where only "Rules Question" posts get routed into downstream answer extraction.

## Loading example
```python
from datasets import load_dataset

ds = load_dataset("eriksalt/reddit-rpg-rules-question-classification")
print(ds)
print(ds["train"].features)
```
