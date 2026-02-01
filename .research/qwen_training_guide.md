# Qwen (Unsloth) Training Guide — `qwen_training_guide.md`

This guide summarizes the official Unsloth “Qwen 2.5 fine-tuning” notebook workflow (the **Qwen2.5-(7B)-Alpaca** notebook), explains each **key function call**, and adds practical recommendations—especially for **text classification** (your use case).

> **Core idea:** the notebook demonstrates *instruction-style supervised fine-tuning* (SFT) using **LoRA adapters** on top of a pretrained Qwen model, trained on an Alpaca-formatted dataset, then shows inference and export paths.

---

## 1) High-level flow

1. **Load base model + tokenizer** (optionally 4-bit)
2. **Attach LoRA adapters** (PEFT)
3. **Prepare dataset** by creating a `"text"` column with a prompt template + EOS token
4. **Configure SFTTrainer** (batching, LR, optimizer, steps/epochs, etc.)
5. **Train** (`trainer.train()`)
6. **Run inference** (tokenize prompt → `model.generate`)
7. **Save/export** (adapters-only, merged model, GGUF)

---

## 2) Key function calls (what they do)

### Model load
```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen2.5-7B",
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,
)
```
**What it does**
- Downloads/loads the base model
- Applies Unsloth speed/memory patches
- Configures max sequence length (supports RoPE scaling internally)
- Returns `(model, tokenizer)`

---

### Attach LoRA adapters (PEFT)
```python
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
    use_rslora=False,
    loftq_config=None,
)
```
**What it does**
- Adds LoRA adapter weights to specific layers in the transformer
- Keeps the base model mostly frozen; trains the adapters

---

### Dataset transform (create a `text` column)
```python
dataset = dataset.map(formatting_prompts_func, batched=True)
```
**What it does**
- Applies your function across the dataset
- **Returns a new dataset**
- Adds/overwrites columns based on the dict you return
- With `return {"text": texts}`, it adds a new `"text"` column
- Does **not** remove existing columns unless `remove_columns=[...]` is used

---

### SFTTrainer setup
```python
from trl import SFTConfig, SFTTrainer

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    packing=False,
    args=SFTConfig(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        max_steps=60,
        learning_rate=2e-4,
        optim="adamw_8bit",
        lr_scheduler_type="linear",
        weight_decay=0.001,
        logging_steps=1,
        seed=3407,
        output_dir="outputs",
        report_to="none",
    ),
)
```
**What it does**
- Builds a training loop around your model + dataset + hyperparameters.

---

### Train
```python
trainer_stats = trainer.train()
```
**What it does**
- Runs optimizer updates for `max_steps` (or for epochs if you set `num_train_epochs`)
- Updates LoRA weights (not the entire base weights)

---

### Inference optimization
```python
FastLanguageModel.for_inference(model)
```
**What it does**
- Switches to inference-optimized mode (speed/memory tweaks)
- Conceptually similar to “eval mode + no grad” plus Unsloth specifics

---

### Generate
```python
inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=64, use_cache=True)
decoded = tokenizer.batch_decode(outputs)
```

---

### Save adapters + tokenizer
```python
model.save_pretrained("lora_model")
tokenizer.save_pretrained("lora_model")
```
**What it saves**
- Adapter weights + adapter config + tokenizer files
- Usually **not** a full standalone base model

---

### Export merged model (standalone)
```python
model.save_pretrained_merged("model-merged", tokenizer, save_method="merged_16bit")
model.save_pretrained_merged("model-merged-4bit", tokenizer, save_method="merged_4bit")
```
**What it saves**
- A model with LoRA “baked into” base weights

---

### Export GGUF (llama.cpp / Ollama)
```python
model.save_pretrained_gguf("model-q4_k_m-gguf", tokenizer, quantization_method="q4_k_m")
```

---

## 3) Data preparation in the notebook (Alpaca format)

### Template used
```python
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""
```

### Formatting function
```python
EOS_TOKEN = tokenizer.eos_token

def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs = examples["input"]
    outputs = examples["output"]
    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
        text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
        texts.append(text)
    return {"text": texts}
```

### Resulting dataset columns
If the dataset initially has `"instruction"`, `"input"`, `"output"`, then after mapping it has **four columns**:
- `instruction`
- `input`
- `output`
- `text` (added)

> If you want “text-only,” pass `remove_columns=[...]` to `map`.

---

## 4) Parameter definitions and how to think about them

### Base model load params
- **`max_seq_length`**
  - Max tokens per example (prompt + response)
  - Higher = more VRAM + slower
- **`dtype`**
  - `None` lets Unsloth auto-pick (often FP16 on older GPUs, BF16 on newer)
- **`load_in_4bit`**
  - `True` loads base model weights quantized → big VRAM savings

### LoRA/PEFT params
- **`r` (LoRA rank)**
  - *“How many parallel ‘edit channels’ the adapter has”*
  - Higher rank = more capacity, more VRAM/compute
- **`target_modules`**
  - Which linear layers get LoRA adapters
  - Full set (attention + MLP) improves quality; attention-only reduces VRAM
- **`lora_alpha`**
  - Scaling for the adapter contribution
  - Too high can destabilize; too low can underfit
- **`lora_dropout`**
  - Regularization on LoRA path
  - `0` is fastest; small values (0.05) can reduce overfit on small datasets
- **`bias`**
  - `"none"` trains fewer params and is often fastest
- **`use_gradient_checkpointing`**
  - Trades compute for memory
  - `"unsloth"` is Unsloth’s memory-optimized variant

### rsLoRA params
- **`use_rslora`**
  - Rank-stabilized LoRA changes scaling behavior so higher ranks stay “audible”
  - Particularly relevant at `r >= 32`

### LoftQ params
- **`loftq_config`**
  - A LoRA initialization approach to reduce quantization error (especially in low-bit setups)
  - Can create a **one-time VRAM spike** during initialization

### Trainer params
- **`per_device_train_batch_size`**
  - Micro-batch size (rows per forward/backward pass)
- **`gradient_accumulation_steps`**
  - Number of micro-batches accumulated before an optimizer step
- **Optimizer step meaning**
  - **One “step” = one optimizer update**
  - Rows per step = `batch_size * grad_accum * num_gpus`
- **`warmup_steps`**
  - Ramps learning rate up gradually at the start
  - Prevents unstable early updates
- **`max_steps` vs `num_train_epochs`**
  - `max_steps` = fixed number of optimizer updates
  - `num_train_epochs` = full passes over dataset
- **`packing`**
  - Packs multiple short samples into one sequence (can speed up short-text training)
  - Can change behavior; simplest is `False`

---

## 5) Recommendations under different conditions

### A) Memory tight training (general)
**First levers (biggest impact):**
1. Lower `per_device_train_batch_size`
2. Lower `max_seq_length`
3. Use `use_gradient_checkpointing="unsloth"`
4. Keep `load_in_4bit=True`
5. Use `optim="adamw_8bit"`

**If still tight:**
- Reduce `target_modules`:
  - **Attention-only LoRA**:
    ```python
    target_modules=["q_proj","k_proj","v_proj","o_proj"]
    ```
  - Add MLP modules later if you can afford them.

**Packing**
- If your inputs are short, `packing=True` may greatly increase throughput.
- Keep it off at first to reduce surprises; enable once behavior is understood.

---

### B) rsLoRA: when to use it (and when not)
**Use rsLoRA if:**
- You want **higher rank** (e.g., `r=32/64/128`)
- Increasing rank in vanilla LoRA **does not improve validation**
- Training at higher rank feels “wasteful” or plateaus unexpectedly

**Don’t bother if:**
- You are at `r <= 16` and results are good and stable

**Downsides**
- Effective adapter contribution can be “louder” than your tuned vanilla settings
- You may need to lower `lora_alpha` and/or learning rate

**Good starting point (r=32, memory tight)**
- `use_rslora=True`
- `r=32`
- Start conservative on scaling:
  - `lora_alpha=16` (then try 24 → 32 if too weak)
- LR: `1e-4` to `2e-4` (start lower if you saw instability)

---

### C) LoftQ: when it’s worth it
**Try LoftQ if:**
- You are training with aggressive quantization (4-bit base)
- You suspect “quantization error” is hurting quality
- You have some VRAM headroom for a possible initialization spike

**Avoid LoftQ if:**
- You are *right at the edge* of OOM already
- You cannot tolerate a one-time VRAM spike during init

---

### D) Warmup: why it matters
Warmup is “gentle throttle at the start.” It reduces the chance that the first few updates overshoot.

**If you see early loss spikes**
- Small spikes that settle quickly may not matter
- Big spikes can:
  - waste training budget
  - reduce run stability
  - increase odds of a worse final solution

**Rule of thumb**
- 1–5% of total optimizer steps, or “a handful of steps” for short demo runs

**Recommendation**
Do one run with it off, and then one run with it set to 1% of training steps, and see if metrics improve.
---

## 6) Classification-specific guidance (your use case)

### A) Instruct vs base model (Qwen2.5-14B)
With **~2k rows** and a strict requirement “output only the label,” an **Instruct** checkpoint is usually best because it:
- follows instructions better
- stays within output constraints more reliably

**Choose base model instead if:**
- your data is “messy” and triggers refusals/moralizing, and you need classification no matter what
- you’re doing logit-based scoring (not free-form generation)

### B) Labels and tokenization bias
Longer labels often tokenize into more pieces, which can introduce subtle biases or formatting errors.

To measure token counts:
```python
len(tokenizer.encode("Rules Question", add_special_tokens=False))
len(tokenizer.encode("Other", add_special_tokens=False))
```

**Recommendation**
- Train model to output short labels, e.g.:
  - `RQ` and `O`
- Map them back in your app:
  - `RQ → Rules Question`
  - `O → Other`

### C) Imbalance (your 90/10 split)
If you train directly on a 90/10 dataset, the model can learn a “safe default” of predicting the majority class.

**Best practice**
- Keep your *evaluation* set as **natural 90/10** (matches production)
- Create an additional *balanced or semi-balanced* validation set to measure minority learning
- For training:
  - oversample minority to **70/30** or **50/50**
  - but evaluate on both:
    - **Val A:** natural 90/10
    - **Val B:** balanced-ish

**What to watch**
- Accuracy on 90/10 can look great even if minority recall is terrible
- Track per-class precision/recall/F1

### D) Prompt formatting for instruct models
This notebook uses an Alpaca-style template. For **classification**, especially on an **Instruct** checkpoint, you usually get best results by formatting examples as chat:

- **system**: your rubric (the instruction)
- **user**: the post text
- **assistant**: the label only (`RQ`/`O`)

This matches the “chat template” style you’ve seen in other training, where text contains special role markers (e.g., `"<|im_*|>"`) via the tokenizer’s chat template utilities.

---

## 7) Memory stats: what “reserved memory” means

The notebook prints GPU memory using:
```python
torch.cuda.max_memory_reserved()
```

**Reserved vs allocated**
- *Reserved* is what PyTorch’s caching allocator has grabbed from the GPU
- It may be larger than currently used tensors (it’s cached for speed)

**OOM expectation**
- You usually won’t see `reserved > total VRAM` printed.
- Instead, you’ll OOM when PyTorch can’t satisfy a new allocation.
- Fragmentation can cause OOM even before you reach 100% reserved.

**Rule of thumb**
- >90% peak reserved → close
- >95% peak reserved → likely to OOM with small changes

---

## 8) Saving and deployment choices

### Adapters-only
- Small artifact
- Requires the base model to be available at load time

### Merged model
- Larger artifact
- Self-contained
- Often easier to deploy (no adapter attach step)

### GGUF export
- Useful for llama.cpp / Ollama style runtimes
- Choose quantization by your quality/speed constraints (q4 variants are popular)

---

## 9) Practical “starter configs” (quick recipes)

### Memory tight + moderate quality
- `load_in_4bit=True`
- `use_gradient_checkpointing="unsloth"`
- `r=16` (or `r=32` if you can afford it)
- `target_modules`: attention + MLP if possible; otherwise attention-only
- `per_device_train_batch_size=1–2`
- `gradient_accumulation_steps`: increase to reach desired effective batch

### Memory tight + r=32
- Consider `use_rslora=True`
- Start `lora_alpha=16`, LR ~ `1e-4`–`1.5e-4`

### Small dataset (2k rows)
- Prefer **Instruct** checkpoint
- Use short labels (`RQ`/`O`)
- Consider `lora_dropout=0.05` if you see overfitting

### 90/10 class imbalance (matches production)
- Train on 70/30 or 50/50 (oversample minority)
- Evaluate on:
  - natural 90/10 (production-like)
  - balanced-ish (boundary learning)

---

## 10) General advice
- Always build a small **sanity-check inference** prompt early (before full training)
- Track **minority recall** explicitly on a production-like validation set
- Keep decoding deterministic for classification:
  - `do_sample=False`, `temperature=0`, small `max_new_tokens`
- Save run metadata (LoRA config, LR, rank, dataset sampling recipe) for reproducibility
- When changing multiple knobs, change one at a time and keep a stable eval set

---

**End.**
