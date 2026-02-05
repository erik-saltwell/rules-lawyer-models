## FastLanguageModel.from_pretrained

### dtype
#### Definition
The datatype used for compute after dequantization.  Does not need to match what the base model was trained with, though it needs to match your trainer flags. Must be compatible with your hardware. In examples this is set to float16 for T4/v100 class GPUs and to bfloat16 for Ampere class GPUs.

#### Recommendation
A good approach is to leave this as None and let unsloth autoselect.  If you want to be explicit, here is code to check which is supported.

``` python
dtype = torch.bfloat16 if is_bfloat16_supported() else torch.float16
```

### load_in_4bit
#### Definition
Whether the base model you are loading is 4 bit.

#### Recommendation
The tradeoff here is that bigger models with more parameters require more RAM.  You want as much model as you can get without causing an out-of-memory error.  Because you can often load a bigger model if you load the 4-bit version, and this should be your starting approach.  If you have crazy gpus or your next-bigger-model would blow your budget (but loading not in 4-bit keeps you in budget) you will see a small improvement in accuracy.


## FastLanguageModel.get_peft_model

### r (lora rank)
#### Definition
A LoRA adapter is like adding an EQ plugin on top of that master. You’re not re-recording every instrument (full fine-tune); you’re shaping how the sound comes through.

rank_lora (a.k.a. LoRA rank r) = how many “EQ bands / knobs” you have.

- Low rank (e.g., 8): a few big, broad EQ moves. Like a simple 3–6 band EQ: bass / mids / treble adjustments. Harder to overdo, but limited nuance.
- Higher rank (e.g., 32–64): more bands, more surgical control. Like a parametric EQ with many bands: you can sculpt very specific resonances… but you can also “over-EQ” and bake in weird artifacts (overfitting).

#### Impact
If your rank is too low,  the adapter can run out of capacity and learning “falls off” compared to higher-rank adapters. If your rank is too high, it can overfit your data.
- Underfitting (rank too low):
 - Training loss won’t go down much, or it plateaus early.
 - Outputs don’t adopt the new style/behavior reliably.
 - Fix: increase rank (16 → 32 → 64) or apply LoRA to more modules (often better than cranking rank alone). Unsloth notes targeting both attention + MLP tends to perform best.
- Overfitting (rank too high for your data):
 - Training loss keeps dropping, validation stops improving or worsens.
 - Model becomes overly “on-rails” or brittle.
 - Fix: drop rank (32 → 16), add dropout, or reduce LoRA strength (Unsloth even suggests scaling down alpha impact when overfitting).

#### Simple Default

These are widely-used “first try” ranks, and they match what Hugging Face docs describe as typical LoRA ranks: 8, 16, 32, 64.
- Most tasks (instruction / formatting / light domain adaptation): r = 16 (or 32 if your task is more complex)
- Small dataset / risk of overfitting: r = 8 or 16. The higher the rank, the more data you need to compute each 'channels' setting.
- Harder domain shift / you need more “new behavior”: r = 32 (then 64 if clearly underfitting)

#### Detailed Configuration
Treat rank like “adapter capacity budget”:
- Higher rank = more capacity. Capacity is useful when:
- The transformation you want is genuinely complex (new domain reasoning, structured outputs across many cases).
- You have enough varied data to justify that capacity.
- Otherwise, higher rank mostly increases:
 - trainable parameters
 - training time / memory
 - risk of memorization

 Dataset size → ranks to consider
- < 500 examples (10²-ish)
 - Consider: r = 4–8
 - Why: you’re capacity-limited by data; higher rank tends to memorize phrasing.
 - If you need more power, first apply LoRA to more modules (attention + MLP) before raising rank.
- 500–2,000 examples (10³-ish, low)
 - Consider: r = 8–16
 - Typical sweet spot for “small-but-real” tasks.
 - If overfitting: stay at 8–16 and increase dropout / regularization.
- 2,000–10,000 examples (10³–10⁴)
 - Consider: r = 16–32
 - This is where nuance starts benefiting from extra capacity.
 - For complex text classification, 16 is the safe start; 32 if you see underfitting.
- 10,000–50,000 examples (10⁴-ish)
 - Consider: r = 32–64
 - You usually have enough variety to justify higher rank, especially with bigger label sets or harder boundaries.
 - 50,000–200,000 examples (10⁵-ish)
 - Consider: r = 64 (and only go higher if you can prove it helps)
 - Beyond 64, returns diminish fast; it’s often better to:
  - tune learning rate / schedule,
  - widen target modules,
  - or use a better base model.
- 200,000+ examples (10⁵–10⁶)
 - Consider: r = 64–128 only if you have evidence
 - At this scale, full fine-tuning or more sophisticated adapter setups may be competitive.
 - If you do go high-rank, consider rank-stabilized LoRA (rsLoRA) style options (if available in your stack).

Two “gotchas” that matter more than raw row count
- Effective data size (diversity)
 - 2,000 highly repetitive examples behave like 500.
 - 2,000 diverse examples can behave like 5,000.
- Module coverage beats rank bumps
 - If you’re only adapting attention, moving to attention+MLP often beats doubling r.

#### Tuning Process
A simple decision procedure that works
 - For any bucket, don’t guess—sweep 3 ranks and pick the smallest that wins on validation:
 - low / mid / high (e.g., 8, 16, 32 for your scale)
 - keep alpha/r constant (e.g., alpha = r or 2r)
 - same split, same seed, same steps

Pick the smallest rank that reaches the best validation metric / qualitative eval.

### lora_alpha
#### Definition
If LoRA rank is “how many EQ bands you have,” then lora_alpha is the output gain (or wet/dry intensity) of the EQ curve you dialed in.

You can have a perfectly tuned EQ (rank provides the knobs), but alpha decides how hard that EQ gets applied to the mastered track (the base model).

In most LoRA implementations, the adapter’s contribution is multiplied by a fixed scale that depends on lora_alpha and rank (commonly proportional to alpha / r; PEFT also supports rsLoRA scaling).

#### Impact
Higher alpha (relative to rank)
 - Adapter has more influence per step
 - Often learns faster early
 - Higher risk of overshooting / overfitting / “overriding” base behavior
- Lower alpha (relative to rank)
 - Adapter influence is softer
 - Learning can be slower / may underfit
 - Often more stable and conservative

Symptoms of lora_alpha too high - Think “EQ gain is cranked and you’re clipping.”
You’ll often see:
- Training loss drops fast, but validation stalls or worsens early (adapter is overpowering / overfitting).
- Instability: loss becomes noisy/oscillatory; sometimes spikes.
- Model behavior feels “over-tuned”: it latches onto superficial cues, becomes brittle on paraphrases, or gets weirdly confident.
- In classification specifically: you may see class collapse (predicting one label too often) or very high confidence on wrong classes.
- Unsloth even calls out “LoRA alpha scaling” as an overfitting mitigation: multiply alpha by 0.5 after training to make the finetune less pronounced.

Symptoms of lora_alpha too low - Think “EQ is on, but the gain is so low you can barely hear the difference.”
You’ll often see:
 - Loss decreases slowly or plateaus early.
 - Your eval metric improves only slightly vs base model.
 - Qualitatively, the model “mostly behaves like the base model,” ignoring the new decision boundary nuance.

#### Simple Default
The usual range is r (lora rank) or 2r (twice the lora rank).  Start with 2r

#### Tuning Process
Best practices for computing lora_alpha before a sweep
The goal is to pick an alpha that’s predictably sane so your sweep is about rank/task fit—not about accidentally changing the adapter “volume.”

- Choose a strength ratio first (k = 1 or 2)
 - Conservative / small data / worried about overfitting → k = 1 (alpha = r)
 - Want faster learning / complex boundary → k = 2 (alpha = 2r)
 - - Unsloth’s own recommendation is exactly these two ratios.
- Lock alpha to rank for the sweep
 - When you sweep rank (8/16/32), keep k constant:
 - r=8 → alpha=16
 - r=16 → alpha=32
 - r=32 → alpha=64
 - This keeps the adapter “gain staging” consistent so you’re comparing ranks fairly. (Unsloth frames this as keeping the alpha/rank relationship at 1 or 2.)
- If you’re using Hugging Face PEFT with rsLoRA, remember the scaling convention changes
 - PEFT notes that standard LoRA uses alpha/r, while rsLoRA uses alpha/sqrt(r).
 - Practical upshot: rsLoRA can behave “less touchy” at higher ranks. You can still start with alpha = 2r, but be aware the normalization differs.
- Have a fast “sanity check” before the full sweep
 - Run ~100–300 steps and check:
 - does training loss move at all? (if not, alpha may be too low or LR too low)
 - does it instantly overfit / destabilize? (alpha too high or LR too high)
 - Then proceed with the full sweep.

### lora_dropout
#### Definition
If LoRA is your EQ plugin:
- rank = how many EQ bands you can tweak
- alpha = how loud the EQ curve is applied
- lora_dropout = intermittently bypassing some of the EQ’s adjustments while you’re learning the mix

It’s like telling your engineer brain: “While we’re dialing this EQ in, randomly turn off a few bands each moment so we don’t obsessively carve a perfect curve for this exact song.” That forces the EQ settings to generalize instead of memorizing.

Concretely: lora_dropout randomly drops parts of the LoRA update during training, which acts as regularization and helps reduce overfitting. (This is the standard dropout idea applied to LoRA adapters.)

#### Impact
- Higher dropout
 - Reduces effective adapter capacity each step
 - Slows learning, but improves generalization
 - Useful when you have small data or you see overfitting
- Lower dropout (including 0)
 - Makes the adapter fully “active” every step
 - Learns faster, but more prone to overfitting
 - Often fine when you have lots of data or strong regularization elsewhere

Unsloth’s guidance frames it exactly as an overfitting control: keep it low (often 0) when you have plenty of data, raise it a bit for smaller datasets / overfitting.

Symptoms of lora_dropout too high - Think: you keep bypassing your EQ so often you never really learn what it should sound like.
You’ll see:
- Underfitting: training loss decreases slowly or plateaus early
- Validation doesn’t improve much either (because the adapter isn’t learning strong, consistent signals)
- Model outputs look “base-model-ish,” not adopting the new boundary/style reliably
- For classification: persistent confusion between hard classes; low confidence; gains are small even as you train longer
Symptoms of lora_dropout too low - Think: the EQ is always on, and you end up “mixing for the room,” overfitting to your training tracks.
You’ll see:
- Training metric improves quickly
- Validation improvement stalls early or degrades (classic overfit gap)
- Brittle behavior: does great on familiar phrasing, worse on paraphrases / slight distribution shift
- For classification: confidence becomes over-peaked; errors concentrate in edge cases

#### Simple Default
For smaller datasets (5-10k)
- lora_dropout = 0.05
Then:
 - If you see overfitting, go to 0.1
 - If you are clearly underfitting, try 0.0–0.03

This aligns with common PEFT practice where dropout is often 0.0–0.1 depending on overfitting risk, and Unsloth similarly treats dropout as a small regularization knob.

#### Detailed Configuration
The big idea: keep dropout fixed while you sweep rank, unless overfitting/underfitting forces you to change it. Otherwise you’re comparing apples to oranges.

- Decide “overfit risk level” from your setup
 - Small dataset (≤10k), complex text, noisy labels → start 0.05
 - Extremely small dataset (<1k) or very flexible adapter (high rank, many modules) → start 0.1
 - Large dataset (≥50k) and stable task → start 0.0–0.03
- Lock dropout for your rank sweep
 - Example: sweep ranks {8, 16, 32} with dropout fixed at 0.05
 - That way any differences are mostly due to capacity (rank), not changing regularization.
- Use the train–val gap as your “meter”
 - If train improves but val doesn’t → increase dropout one notch (0.05 → 0.1)
 - If neither train nor val improves (and LR is reasonable) → decrease dropout (0.05 → 0.0–0.03) or increase rank/target modules
- Change dropout in “notches,” not tiny increments
 - In practice, these levels cover almost all cases:
  - 0.0, 0.03, 0.05, 0.1
  - More granularity usually isn’t worth it early.
- If you raise rank or widen target modules, re-check dropout
 - More capacity often needs a touch more regularization:
 - r=16 @ 0.05 might be fine
 - r=32 with attention+MLP might want 0.05–0.1

#### Tuning Process
- Fix: dropout = 0.05
- Sweep: r ∈ {8, 16, 32}
- Keep: alpha = 2r (or r) constant ratio across runs
- If the winner overfits: rerun winner rank with dropout = 0.1
- If all underfit: rerun best rank with dropout = 0.0–0.03

For text classifiction with  minority class and you are optimizing F1:
- For each run: compute minority precision/recall/F1 and choose best threshold for minority F1 on val
- If best run is recall-limited: rerun that rank with dropout 0.03
- If best run is precision-limited: rerun that rank with dropout 0.1

### bias
#### Definition
**The “graphic equalizer” analogy (LoRA = EQ bands)**

Think of LoRA as a **graphic equalizer** you bolt onto a song:

- The **LoRA matrices** are the *EQ sliders* that reshape the sound *selectively* — boosting/cutting “frequencies” (i.e., directions in weight space) without rewriting the entire track.
- The **bias terms** are more like a **DC offset / baseline knob** on the mixing board: they don’t reshape frequencies; they shift the *zero point* (the “resting level”) of certain channels.

Mathematically, a typical linear layer is:

\[
y = xW^\top + b
\]

LoRA adapts the weight matrix by adding a low-rank update:

\[
W' = W + \Delta W,\quad \Delta W = \frac{\alpha}{r}\,BA
\]

The **`bias`** setting decides what happens to the layer bias **`b`** during training:

- **`"none"`**: keep all `b` frozen (default in PEFT).
- **`"lora_only"`**: only biases *attached to modules where LoRA is injected* become trainable.
- **`"all"`**: every bias parameter in the model becomes trainable.

These options match Hugging Face PEFT’s `LoraConfig.bias` semantics. PEFT warns that if you train biases, then even if you “disable” adapters, outputs won’t exactly match the original base model anymore because the base biases changed.

---
#### Impact and Symptoms
**What `bias` changes in training behavior**

1. **Capacity vs. “surgical-ness”**
   - Training biases adds a small amount of extra flexibility.
   - But it’s **less surgical** than pure LoRA because it changes baseline activations directly.

2. **Reversibility / “adapter as a plugin”**
   - If biases are trainable, then *turning off* adapters does **not** return you to the pristine base model output.
   - This matters when you want LoRA to behave like an on/off effect.

3. **Compute + memory**
   - More trainable parameters → slightly more optimizer state and gradients.
   - In Unsloth’s own hyperparameter guide, they recommend leaving this as `"none"` for faster training and reduced memory, and because it “adds trainable parameters for little to no practical gain.”

4. **Overfitting risk**
   - Bias vectors are “easy knobs” to turn; on small datasets they can move quickly and contribute to **fast training loss drops** with **worse validation**.

**Observable symptoms when `bias` is non-optimal**

If `bias="all"` (too permissive) on many SFT setups
- **Validation metrics stall or worsen** while training loss keeps improving (classic overfit pattern).
- **Style or refusal behavior drifts** more than expected.
- **Base-model drift when adapters are disabled** (you expected base outputs but they changed).
- Slightly higher VRAM use / slower steps (usually modest, but real).

If `bias="none"` (too restrictive) *for your specific task*
This is less common, but can happen if your adapter budget is extremely small (low rank, narrow target modules) and the task needs a strong “baseline shift.”
- Training **plateaus early** (loss stops improving) even after you’ve tried reasonable `r`, `lora_alpha`, LR, and target modules.
- Model learns some patterns but fails on systematic calibration (e.g., consistent label priors in classification, or systematic domain shift).

If you change `bias` and **nothing changes**
Unsloth users have reported versions where `bias="all"` / `"lora_only"` did not actually mark bias params trainable (trainable parameter counts stayed constant). If you see this, treat the sweep result as invalid until verified.

---
#### Simple Default
Set bias = None

**Why:**
- Unsloth’s current LoRA hyperparameter guidance explicitly says to leave bias as `"none"` for speed + memory and because it rarely helps in practice.
- Hugging Face PEFT docs recommend trying **no bias first**, then `lora_only`, then `all` only if needed.

**When to deviate immediately:**
- You *do not* need “adapter on/off returns base model,” and
- you have evidence of underfitting after you’ve already tuned rank/alpha/LR reasonably.

In that case, start with `bias="lora_only"` (not `"all"`).

#### Detailed Configuration

Step 0 — Decide what you need from deployment

**If you need LoRA to behave like an EQ plugin** (turn it off → you get the original song back), choose:
- ✅ `bias="none"`

**If you will always run the adapted model** (and don’t care about exact base outputs when adapters are disabled), you *may* consider:
- `bias="lora_only"` (first)
- `bias="all"` (last resort)

Step 1 — Verify the setting actually takes effect (critical)

Before you tune, do a quick “sanity check”:

- Run `model.print_trainable_parameters()` after `get_peft_model(...)`.
- Additionally, count how many parameters with `"bias"` in their name have `requires_grad=True`.

Expected behavior:
- With `"none"`: bias trainables ≈ 0 (aside from any separately-trainable heads you configured).
- With `"lora_only"` or `"all"`: trainable parameter count should increase.

If **the counts don’t change**, your experiment can’t inform tuning — update packages / check Unsloth version behavior.

Step 2 — Only consider `bias` after the “big levers” are sane

Before sweeping `bias`, ensure these are not obviously wrong:
- Target modules include the major linear layers you intend to adapt.
- Rank `r` is in a reasonable range (e.g., 8–64 for many SFT tasks; higher only if you know you need it).
- `lora_alpha` matches your scaling strategy (common: `r` or `2r`).
- Learning rate and effective batch size are stable.

Bias tuning is typically a **second-order** lever.

Step 3 — Guardrails if you enable bias training

If you try `"lora_only"` or `"all"`:
- **Expect stronger overfit pressure** → tighten regularization:
  - Fewer epochs, earlier stopping, or stronger eval-based checkpointing.
  - Consider modest `weight_decay` (if your trainer applies it to those params).
- Track “base drift”:
  - Save a small set of prompts and compare outputs with adapters disabled vs. the original base model.

---
#### Tuning Process
A systematic sweep that’s cheap and informative

**Goal:** decide whether bias training provides measurable gains *for your task*.

1. **Pick one solid baseline config**
   - Fix dataset, prompt format, trainer, LR schedule, rank, alpha, target modules.
   - Use a short pilot run length (e.g., a fraction of an epoch or a fixed number of steps).

2. **Run the 3-point sweep**
   - `bias="none"`
   - `bias="lora_only"`
   - `bias="all"`

3. **For each run, record**
   - Train loss curve + validation metric(s)
   - Time per step and peak VRAM
   - **Trainable parameter count** (must differ if the setting is active)
   - “Base drift check” (optional but valuable):
     - Compare outputs of:
       1) base model
       2) adapted model with adapters enabled
       3) adapted model with adapters disabled

4. **Decision rule (practical)**
   - If `"lora_only"` is **clearly better** on validation (not just training loss), keep it.
   - If `"all"` is only marginally better (or worse), prefer `"lora_only"` or `"none"` — it’s easier to manage and less likely to overfit.
   - If none of them differ meaningfully, default to `"none"`.

Follow-up refinement (only if `bias` mattered)

If `"lora_only"` wins:
- Re-run with 2 seeds to confirm it’s not noise.
- Then tune *rank/alpha/LR* again; enabling bias can change the best region slightly.

If `"all"` wins:
- Treat it as a red flag that your adapter capacity elsewhere might be too constrained.
- First try increasing rank or broadening target modules; often you can recover the gain without training all biases.

---
#### Notes specific to Unsloth users

- Unsloth’s own docs currently recommend keeping `bias="none"` for LoRA fine-tuning in Unsloth.
- Some users have reported that certain Unsloth versions did not actually enable bias training even when `bias="all"`/`"lora_only"` was set. Always verify trainable parameter counts before trusting results.

### use_rslora

#### Definition
Think of a LoRA adapter like a **graphic equalizer (EQ)** you bolt onto an existing song:

- **Rank `r`** is the *number of EQ sliders* you add (more sliders = more freedom to shape the sound).
- **`lora_alpha`** is like a *master gain knob* that controls how loud the EQ changes are.
- **`use_rslora`** decides **how the master gain compensates for adding more sliders**.

Concretely (as implemented in PEFT and many downstream trainers), LoRA adds an update:
- **Standard LoRA scaling:** `gain = lora_alpha / r`
- **Rank-Stabilized LoRA (rsLoRA) scaling:** `gain = lora_alpha / sqrt(r)`

So `use_rslora=True` switches the “gain compensation rule” from **divide-by-r** to **divide-by-sqrt(r)**.

Why this matters in the EQ analogy:
- Standard LoRA is like *turning the master gain down too aggressively* as you add sliders, so every individual slider becomes too quiet to matter at higher `r`.
- rsLoRA is like a *smarter loudness compensation*: you still turn gain down as you add sliders (to keep things stable), but not so much that the EQ becomes inaudible.

#### Impact and Symptoms
**What it changes in training behavior**
- With **`use_rslora=False`**, increasing rank often gives diminishing returns because the update is scaled down by `1/r`. For larger `r`, the adapter’s effective step size can become so small that it learns slowly or not at all.
- With **`use_rslora=True`**, the scale shrinks more gently (`1/sqrt(r)`), which tends to:
  - keep adapter activations / gradients in a healthier range across ranks,
  - make **higher ranks actually usable** (i.e., you can trade more trainable parameters/compute for better fit without changing inference cost).

**Observable “it’s set wrong” symptoms**
- **If you leave `use_rslora=False` and try high rank (e.g., 64–256):**
  - Training loss decreases *very slowly* compared to low-rank runs.
  - Validation improvements plateau early; high rank looks “no better than r=8/16”.
  - LoRA weight norms and LoRA gradient norms stay unusually small (underpowered adapter).
- **If you set `use_rslora=True` but keep a LoRA-alpha tuned for standard LoRA (common mistake):**
  - Because the denominator got smaller, your effective gain can jump.
  - Symptoms: loss spikes early, instability/NaNs with mixed precision, sharp overfitting, or outputs become “over-steered” / degraded.

A useful sanity metric is the **effective adapter gain**:
- `g_standard = lora_alpha / r`
- `g_rs = lora_alpha / sqrt(r)`
If `g_rs` is much larger than your usual stable `g_standard`, you likely need to reduce `lora_alpha` (or LR).

#### Simple Default
**Recommended initial value:** `use_rslora = True`

Rationale:
- It is a low-friction switch that often makes rank scaling behave more predictably, and is explicitly supported in major tooling (PEFT / TRL and popular fine-tuning stacks).

**One-line default recipe (safe starting point)**
- If you used standard LoRA with a stable effective gain `g` (often around 1–4), keep that same gain when turning rsLoRA on:
  - Choose `lora_alpha ≈ g * sqrt(r)` (rounded to an integer).

Example:
- If you liked `r=16, lora_alpha=32` under standard LoRA → `g_standard=32/16=2`.
- Switching to rsLoRA at `r=64`: set `lora_alpha ≈ 2 * sqrt(64) = 16` → `g_rs = 16/8 = 2` (similar “loudness”).

This “gain-matching” default avoids surprises and gives rsLoRA a fair comparison.

#### Detailed Configuration
A best-practice way to configure `use_rslora` before any big sweep:

1. **Pick a baseline rank first (don’t sweep everything at once).**
   - Start with `r ∈ {8, 16, 32}` depending on budget.
   - Decide whether you’re rank-constrained (tiny GPU) or quality-constrained (want higher r).

2. **Compute and lock a target effective gain `g`.**
   - If you have a known-good LoRA config, reuse its gain:
     - `g_target = lora_alpha / r` (from your standard LoRA run).
   - If you don’t, start with `g_target ≈ 2` as a conservative “audible but not blaring” EQ gain.

3. **Turn on rsLoRA and set `lora_alpha` by gain-matching.**
   - `use_rslora=True`
   - `lora_alpha = round(g_target * sqrt(r))`

4. **Do a short “smoke test” run (hundreds to a few thousand steps).**
   - Watch for early instability (loss spikes, overflow, NaNs).
   - If unstable: reduce `lora_alpha` first (e.g., ÷2), then consider reducing LR.
   - If clearly underfitting: increase `lora_alpha` (×1.5–2) *or* increase rank.

5. **Only after the above, consider increasing rank.**
   - rsLoRA is most valuable when you want to explore larger ranks; it reduces the “high-rank learns nothing” failure mode common with standard scaling.

Practical notes:
- **Framework support:** In Hugging Face PEFT, `use_rslora` is part of `LoraConfig`; TRL exposes `--use_rslora` in its CLI integration. Some older PEFT versions may not accept the argument—upgrade if you see an “unexpected keyword” error.
- **Interacts with LR:** Switching `use_rslora` changes the magnitude of the injected update, so treat it similarly to changing the learning-rate scale: re-check stability.

#### Tuning Process
Because `use_rslora` is a boolean, the tuning process is a clean A/B test—**but you should compare fairly** by controlling for effective gain.

**Step 0 — Define what “better” means**
- Primary metric: validation loss / task metric (accuracy, Rouge, pass@k, etc.).
- Secondary: training stability (no spikes/NaNs), convergence speed, generalization gap.

**Step 1 — Fair A/B comparison (same rank, matched gain)**
For a chosen rank `r` and target gain `g_target`:
- Run A: `use_rslora=False`, set `lora_alpha = round(g_target * r)`
- Run B: `use_rslora=True`, set `lora_alpha = round(g_target * sqrt(r))`

Keep everything else identical (data order, LR, warmup, batch/accum, dropout).

**What to look for**
- Does run B reach a better validation metric at the same step budget?
- Does run B remain stable without extra tricks (lower LR, gradient clipping)?
- Do higher ranks improve meaningfully under B?

**Step 2 — If rsLoRA wins or ties, lock it on and sweep rank**
Now fix `use_rslora=True` and sweep `r`:
- Suggested sweep: `r ∈ {8, 16, 32, 64}` (add 128/256 only if budget allows)
- For each `r`, keep `g_target` constant initially: `lora_alpha = round(g_target * sqrt(r))`

This isolates “does more rank help?” without also changing the effective adapter loudness.

**Step 3 — Micro-sweep `lora_alpha` (gain) at the best rank**
Once you find a promising `r`, sweep gain around `g_target`:
- `g ∈ {0.5*g_target, 0.75*g_target, 1.0*g_target, 1.5*g_target, 2.0*g_target}`
- Convert each to `lora_alpha = round(g * sqrt(r))`

**Step 4 — Re-check LR coupling (small sweep)**
If your best rsLoRA config is either:
- slightly unstable → sweep LR down (e.g., ÷2, ÷4),
- clearly underfitting → sweep LR up modestly (×1.5).

**Step 5 — Confirm on a longer run**
Finally, run the top 1–2 configs to full budget and compare:
- final metric,
- robustness to different seeds,
- qualitative output.

### loftq_config
#### Definition
`loftq_config` is the **sub-configuration that turns on and controls LoftQ initialization** for a LoRA adapter in **Hugging Face PEFT**.

**Equalizer analogy (LoRA = graphic EQ on a song):**
- Think of your *base model weights* as the original studio master, and *quantization* (e.g., 4-bit) as exporting that song to a highly compressed format.
- **LoRA** is your graphic equalizer: a small set of sliders that can “restore” what compression damaged—without rewriting the whole track.
- **LoftQ** is an *automatic pre-EQ calibration step* done **before training**: it chooses initial LoRA sliders so that **(quantized weights + LoRA correction)** matches the original full‑precision weights as closely as possible.

So `loftq_config` is the “**calibration preset**” that tells LoftQ:
- **how aggressively you compress** during calibration (`loftq_bits`), and
- **how many calibration passes** to run (`loftq_iter`).

In PEFT, `loftq_config` is passed inside `LoraConfig(...)` and is only used when:
- `init_lora_weights="loftq"`.

Typical fields (as exposed by `LoftQConfig`) are:
- `loftq_bits` (int, commonly 4): quantization bit-width used during LoftQ calibration.
- `loftq_iter` (int, commonly 1–5): number of alternating optimization iterations (quantize step ↔ low‑rank approximation step).

#### Impact and Symptoms
`loftq_config` mostly affects **the starting point** of training (initial adapter weights), not the steady-state training dynamics like learning rate or batch size.

**What it changes in practice**
- Lower **initial loss / perplexity** for quantized fine-tuning runs compared to vanilla initialization (especially at low precision).
- Faster early improvement (you “start closer” to the unquantized model’s behavior).
- Potentially better final quality under the same compute budget, because fewer steps are spent recovering from quantization error.

**When it’s set non-optimally (what you’ll notice)**

1) **`loftq_bits` too low (over-compression during calibration)**
- **Symptoms**
  - Initial eval quality is noticeably worse than expected for your base model.
  - Training loss starts higher and needs more steps to reach the “usual” QLoRA baseline.
  - Model becomes more brittle: smaller prompt changes cause larger output degradation.
- **Why (EQ analogy)**
  - You crushed the audio too hard; the pre-EQ sliders can’t fully reconstruct what’s gone.

2) **`loftq_bits` too high (under-compression during calibration)**
- **Symptoms**
  - Little/no improvement vs standard LoRA init on a quantized run.
  - You pay LoftQ overhead but don’t meaningfully reduce quantization damage.
- **Why**
  - Your “export” is so mild that there’s not much quantization error for LoftQ to cancel.

3) **`loftq_iter` too low (not enough calibration passes)**
- **Symptoms**
  - LoftQ behaves like “almost normal init”: early-loss advantage is small.
  - Quantization error metrics (see below) improve only marginally.
- **Why (EQ analogy)**
  - You ran auto-EQ for 10 seconds and stopped before it converged.

4) **`loftq_iter` too high (diminishing returns / overhead)**
- **Symptoms**
  - Initialization step becomes slow enough to dominate your workflow.
  - Any further gains are tiny relative to the additional wall time.
- **Why**
  - LoftQ is a greedy alternating procedure; beyond a few iterations, gains often saturate.

5) **Misuse / integration issues (common gotchas)**
- **Symptoms**
  - No benefit at all, or errors during initialization.
  - Confusion about VRAM usage (“why didn’t LoftQ reduce memory like load_in_4bit?”).
- **Common causes**
  - Not setting `init_lora_weights="loftq"` (then `loftq_config` is ignored).
  - Attempting LoftQ initialization on CPU (PEFT notes it must be on GPU for the init step).
  - Missing SciPy (PEFT checks for SciPy when using LoftQ init).
  - Targeting too few layers: LoftQ can only help layers that have LoRA adapters.

**Observable metrics to confirm LoftQ is working**
- **Before training**, compare logits of:
  - (A) original FP model vs
  - (B) quantized + LoftQ-initialized adapter model
- A simple **MSE / cosine similarity** of logits on a small calibration set is a good sanity check.
- If you use PEFT’s `replace_lora_weights_loftq(...)`, you can use its callback mechanism to accept/reject replacements based on measured improvement.

#### Simple Default
A solid default for most QLoRA-style setups:

- `loftq_bits = 4`
- `loftq_iter = 4`

**Why this default is a good starting point**
- 4-bit is the common “sweet spot” for memory efficiency and quality in practice.
- The LoftQ paper reports using **~5 iterations** in some benchmark setups, so 4 is a practical compromise between benefit and overhead.

Minimal example:

```python
from peft import LoftQConfig, LoraConfig

loftq_config = LoftQConfig(loftq_bits=4, loftq_iter=4)

lora_config = LoraConfig(
    # your usual LoRA settings…
    init_lora_weights="loftq",
    loftq_config=loftq_config,
)
```

#### Detailed Configuration
Use this checklist to configure `loftq_config` *correctly before* doing any sweeps.

1) **Decide whether you should use LoftQ at all**
Use LoftQ when:
- you are doing **quantized** fine-tuning (e.g., QLoRA-style 4-bit),
- you care about squeezing maximum quality out of low precision.

Skip LoftQ when:
- you are fine-tuning in full precision (bf16/fp16) and not quantizing weights,
- your training budget is tiny and the init overhead is not worth it.

2) **Make sure LoftQ is actually enabled**
- Set `init_lora_weights="loftq"` inside `LoraConfig`.
- Pass `loftq_config=LoftQConfig(...)` (or a dict with the same keys).

3) **Target enough layers**
For LoftQ to have maximum effect, you generally want LoRA on as many linear layers as feasible (e.g., `target_modules="all-linear"`), because layers without LoRA cannot be “corrected” by LoftQ.

4) **Align with your quantization strategy**
If you’ll train a 4-bit model, prefer NF4 quantization in your bitsandbytes config (`bnb_4bit_quant_type="nf4"`) when applicable.

5) **Plan for initialization constraints**
- LoftQ initialization is typically **GPU-only**.
- Ensure **SciPy** is installed (PEFT checks this for LoftQ init).
- Expect a non-trivial initialization step (it’s doing per-layer alternating optimization).

6) **Optional: decide between two LoftQ workflows**
- **Workflow A: LoftQ init at adapter creation time**
  - Load base model in full precision.
  - Create PEFT model with `init_lora_weights="loftq"` + `loftq_config=...`.
- **Workflow B: “weight replacement” convenience path**
  - Start from a quantized PEFT model with LoRA layers.
  - Run `replace_lora_weights_loftq(...)` to swap in LoftQ-initialized adapter weights using a safetensors copy of the original weights.
  - Use the callback to verify each replacement improves your chosen metric.

#### Tuning Process
Treat `loftq_config` tuning as an **initialization-quality vs overhead** trade-off.

**Step 0 — Fix everything else first**
Before sweeping LoftQ:
- fix your LoRA topology (target modules, rank `r`, alpha, dropout),
- fix quantization approach (4-bit vs 8-bit),
- set a stable training recipe (LR schedule, batch/grad-accum, eval cadence).

**Step 1 — Establish baselines (tiny cost, high value)**
Run 3 short experiments:
1. **QLoRA baseline** (quantized model, standard LoRA init)
2. **LoftQ init** with `(bits=4, iter=1)`
3. **LoftQ init** with `(bits=4, iter=4)`

Compare:
- pre-training eval loss/logit similarity vs FP model,
- early-step learning curves (first 200–500 steps),
- final validation metric at a fixed small step budget.

**Step 2 — Sweep `loftq_iter` (usually most sensitive)**
Recommended sweep:
- `loftq_iter ∈ {1, 2, 4, 5, 8}`

Rules of thumb:
- If iter=1 already gives most gains, stop—don’t over-optimize.
- If iter=4–5 materially improves early eval *and* final quality, keep it.
- If iter>5 gives tiny gains, prefer fewer iters.

**Step 3 — Sweep `loftq_bits` only if you have a reason**
Most people will stay at `4`. Consider a sweep only if:
- you are exploring 8-bit quantization for compatibility/performance, or
- you’re intentionally investigating extremely low precision.

Suggested sweep:
- `loftq_bits ∈ {4, 8}` (practical)
- optionally `{2, 4}` (researchy / may degrade; validate carefully)

**Step 4 — Pick using a simple decision metric**
Select the configuration that maximizes:

> (final validation score at fixed compute) – λ × (init_time)

Where λ reflects how much you care about workflow speed.

**Step 5 — Lock it in, then tune other hyperparameters**
Once LoftQ is set, go back to the normal knobs (rank, target modules, LR, batch size). LoftQ is usually a second-order improvement compared to those.

---

### References
- Hugging Face PEFT docs: LoRA & quantization guides (LoftQ initialization notes)
- PEFT/LoRA config source showing `loftq_config` usage and LoftQConfig defaults
- LoftQ paper (ICLR 2024): “LoftQ: LoRA-Fine-Tuning-Aware Quantization for Large Language Models”
- Microsoft Research blog: overview of LoftQ’s alternating quantize ↔ low-rank steps


## SFTTrainer

### packing
#### Definition
Think of your LoRA adapters like a **graphic equalizer (EQ)** you’re training to “shape” the model’s behavior.
The `packing` parameter is **not an EQ band**—it’s the way you **queue up the audio** so the EQ always hears signal instead of silence.

Concretely, in TRL’s `SFTTrainer`, `packing=True` means:

- Instead of padding each training example up to `max_length`, the trainer **concatenates multiple shorter examples** (with separator tokens, typically EOS) and then **cuts the stream into fixed-length blocks** of exactly `max_length`.
- This reduces wasted compute on padding and increases GPU utilization.

Older TRL versions accepted `packing` directly in `SFTTrainer(...)`; newer versions set it in `SFTConfig(...)` (passed as `args` to `SFTTrainer`).

#### Impact and Symptoms
**What it changes in training behavior**
- **Higher token-throughput / less padding waste.** When your dataset has many short examples, packing can significantly increase “useful tokens per forward pass.”
- **Different notion of “epoch.”** Because the training stream is repacked into fixed blocks, “one pass through the dataset” becomes less intuitive. If you set `max_steps`, you may inadvertently train for *more* (or less) “epochs” than you think.
- **Example boundaries are no longer aligned to batch items.** Your model sees one example immediately followed by the next inside the same context window (usually separated by EOS / chat delimiters). This is usually fine for causal LM SFT, but it does change the context distribution.

**Observable symptoms when `packing` is non-optimal**
- **Packing should be ON, but it’s OFF**
  - Low GPU utilization.
  - Tokens/sec much lower than expected.
  - Batch “effective tokens” are far below `batch_size × max_length` because most of the batch is padding.
- **Packing should be OFF, but it’s ON**
  - Out-of-memory errors (because you now *consistently* hit `max_length`, whereas padding previously kept many batches shorter).
  - Validation/generations show **boundary artifacts**: the model “bleeds” from one conversation/example into the next (e.g., continues a prior answer, mishandles separators).
  - Metrics become harder to interpret (per-example loss/perplexity isn’t directly comparable if eval is also packed).

**Compatibility & correctness gotchas**
- In TRL, packing is most commonly paired with attention implementations that correctly respect example boundaries in packed / padding-free regimes. TRL’s docs note packing is intended for SFT and setups using FlashAttention (or variants), and TRL exposes a `packing_strategy` that interacts with padding-free behavior.
- For multimodal / vision-language SFT, packing may be unsupported.

#### Simple Default
Use this as a reliable starting point:

- **Default = `packing=True`** if your dataset is mostly **short** examples (e.g., many < 25–50% of `max_length`) and you care about throughput.
- **Default = `packing=False`** if:
  - you are debugging data/formatting,
  - you need clean per-example accounting (loss, filtering, logging),
  - your examples are already near `max_length` (packing won’t help much),
  - you see boundary artifacts in generations.

#### Detailed Configuration
A best-practice way to configure packing **before** hyperparameter sweeps:

1. **Choose `max_length` first**
   - Plot/inspect your token length distribution.
   - Pick `max_length` so you aren’t discarding lots of tail tokens, but don’t blow VRAM.
   - Remember: with packing ON, you will hit `max_length` *much more consistently* → plan batch size accordingly.

2. **Verify boundary tokens are correct**
   - For instruction/chat data, ensure each example ends with a clear delimiter (EOS, `<|eot_id|>`, etc.).
   - If you train “assistant-only” or “completion-only,” confirm your masking logic still works when multiple examples share one packed block.

3. **Decide whether eval should be packed**
   - Prefer `eval_packing=False` so validation reflects real example boundaries and your metrics are easier to interpret.
   - Only pack eval if you explicitly want throughput comparisons and you understand the metric shift.

4. **Pick a packing strategy (if available in your TRL version)**
   - Many TRL versions default to a near-optimal bin-packing style strategy (often documented as best-fit decreasing / similar).
   - Older “wrapped” strategies can reduce padding aggressively but may break sequence continuity more often.

5. **Sanity-check with a tiny run**
   - Train for ~50–200 steps and do quick generations from a few prompts.
   - Look specifically for boundary bleed: repeated role tokens, merged conversations, abrupt truncations, or “assistant continues into next sample.”

6. **Make throughput measurable**
   - Log tokens/sec and VRAM.
   - When comparing packing settings, compare by **tokens processed** (or wall-clock to reach a target eval loss), not just “steps.”

#### Tuning Process
`packing` is mostly a **binary** decision, but you can tune it systematically:

1. **Baseline A/B**
   - Run two short experiments with identical:
     - model, data, `max_length`,
     - optimizer/lr schedule,
     - **total tokens** (adjust `max_steps` if needed).
   - Compare:
     - tokens/sec,
     - peak VRAM,
     - eval loss/quality,
     - generation boundary cleanliness.

2. **If `packing=True` wins on speed but hurts quality**
   - Keep packing ON but tighten formatting:
     - ensure EOS between samples,
     - use a consistent chat template,
     - double-check completion/assistant-only masking.
   - Ensure your attention implementation supports packed regimes correctly (especially if you also use padding-free / flattening).

3. **If `packing=True` causes OOM**
   - Reduce `per_device_train_batch_size` or `max_length`.
   - Prefer increasing `gradient_accumulation_steps` to preserve global batch tokens.

4. **Joint sweeps (recommended)**
   Packing interacts most with:
   - `max_length` (sequence length),
   - batch size / grad accumulation (tokens per optimizer step),
   - `packing_strategy` and any padding-free/flash-attn settings.

   A practical sweep grid:
   - `packing ∈ {False, True}`
   - `max_length ∈ {512, 1024, 2048}` (depending on model/context and VRAM)
   - keep **global tokens per step** roughly constant across conditions

5. **Decision rule**
   Choose `packing=True` when it:
   - materially improves tokens/sec (often the main point),
   - does not introduce boundary artifacts in generations,
   - keeps eval quality (loss or task metric) within tolerance.

---

**Quick reference snippet**

```python
from trl import SFTConfig, SFTTrainer

args = SFTConfig(
    max_length=1024,
    packing=True,
    eval_packing=False,   # keep eval interpretable
    # packing_strategy="bfd",  # if supported in your TRL version
)

trainer = SFTTrainer(
    model=model,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    args=args,
    tokenizer=tokenizer,
)
trainer.train()
```


### gradient_accumulation_steps

#### Definition
`gradient_accumulation_steps` tells the trainer **how many micro-batches to “stack up” (accumulate gradients for) before doing one optimizer update**.

**Graphic-equalizer (LoRA) analogy:**
Imagine your LoRA adapters are a graphic equalizer you’re tuning while listening to a song. Each micro-batch is like listening to a **tiny snippet** (a fraction of a beat) and deciding how to nudge the EQ sliders. If you set `gradient_accumulation_steps=1`, you move the sliders after every snippet—fast feedback, but noisy. If you set it to 8, you listen to **8 snippets**, average what you heard, and then move the sliders once—slower feedback, but smoother and more stable adjustments.

**Math you’ll use constantly:**
- **Effective (global) batch size (in “examples”)**
  `effective_batch = per_device_train_batch_size × gradient_accumulation_steps × world_size`
- In token-packed SFT, it’s often more useful to think in **tokens per optimizer update**, but the same idea applies: you’re increasing *effective* batch without increasing *micro-batch* memory.

#### Impact and Symptoms
What this knob changes in training behavior:

1. **Stability vs responsiveness**
   - Higher accumulation → **less gradient noise**, often smoother loss curves and more stable updates.
   - Lower accumulation → **more frequent updates** (more responsive), but noisier gradients.

2. **Optimizer/scheduler “time”**
   - Most trainers/schedulers advance on **optimizer steps** (updates), not micro-batches.
   - Increasing `gradient_accumulation_steps` reduces the number of optimizer steps per epoch, which affects:
     - warmup duration (in steps)
     - decay schedule timing
     - checkpoint/logging/eval cadence (often measured in steps)

3. **Throughput tradeoff**
   - Accumulation is primarily a **memory workaround** to simulate larger batches. It generally **does not make training faster**, and can be slower than using a larger true batch if that fits in memory.

Observable symptoms when it’s set non-optimally:

- **Too low (effective batch too small)**
  - Loss curve looks “spiky” or erratic.
  - Grad norm fluctuates wildly; occasional divergence/NaNs (especially with higher LR).
  - Validation metrics vary a lot run-to-run; training feels “temperamental.”

- **Too high (updates too infrequent / effective batch too large for your LR/schedule)**
  - Training “feels sluggish”: loss improves slowly per wall-clock hour.
  - You may see under-training symptoms when `max_steps` is fixed (because you’re doing fewer optimizer updates than you think).
  - If you *don’t* adjust LR/schedule, you can get either:
    - overly conservative learning (underfitting), or
    - instability if you “scale LR up” too aggressively.

- **Configuration mismatch symptoms**
  - Warmup ends too early/late relative to the *token* budget you intended.
  - Logging/eval/checkpoint happens “too rarely” after you increased accumulation.

#### Simple Default
A practical default that works for many SFT/LoRA runs:

- **Start with `gradient_accumulation_steps = 1`** if you can fit a reasonable per-device batch (e.g., 4–8 sequences) at your chosen sequence length.
- Otherwise, **target an effective batch of ~16 sequences** and set:
  - `gradient_accumulation_steps = ceil(16 / (per_device_train_batch_size × world_size))`

**Concrete initial recommendation (most common single-GPU QLoRA case):**
- If you can only fit `per_device_train_batch_size=1`, start with **`gradient_accumulation_steps=16`**.
- If you can fit `per_device_train_batch_size=2`, start with **`gradient_accumulation_steps=8`**.

This gives you a stabilizing effective batch without making updates extremely infrequent.

#### Detailed Configuration
A best-practice way to set this knob *before* sweeping anything:

1. **Decide the memory-limited micro-batch first**
   - Pick `max_seq_length` (or packing strategy) and the model/precision/PEFT method.
   - Increase `per_device_train_batch_size` until you’re just below OOM.
   - This is your “micro-batch” (what fits on one device).

2. **Choose an effective batch target (examples or tokens)**
   - For instruction SFT, a common target is **16–64 sequences per optimizer update**.
   - If using packing, also sanity-check tokens per update:
     `tokens_per_update ≈ (avg_tokens_per_example) × effective_batch`

3. **Compute `gradient_accumulation_steps` from the target**
   - `grad_accum = target_effective_batch / (per_device_batch × world_size)` rounded up.
   - Avoid extremely large values unless you have a reason—very large accumulation makes updates rare and can interact poorly with fixed `max_steps`.

4. **Align learning-rate schedule with optimizer updates**
   - Warmup/decay are usually expressed in optimizer steps.
   - If you change `gradient_accumulation_steps`, revisit:
     - `warmup_steps` (or warmup ratio)
     - `max_steps` / `num_train_epochs`
     - `logging_steps`, `eval_steps`, `save_steps`
   - Rule of thumb: if you multiply `gradient_accumulation_steps` by K and want the *same* logging/eval frequency in examples/tokens, divide these step-based intervals by K.

5. **Distributed training detail (why this matters)**
   - In DDP, naïve accumulation can cause extra gradient synchronization overhead. Libraries like Accelerate/Trainer typically handle this by syncing only on the final accumulation step.
   - If you write custom loops, ensure you’re not synchronizing every micro-step.

6. **Sanity checks**
   - Verify `global_step` increments at the rate you expect (it usually increments on optimizer update).
   - Track:
     - loss smoothness
     - grad_norm
     - tokens/sec
     - validation metric trend

#### Tuning Process
A structured sweep process that avoids wasted runs:

**Phase 0 — Lock the true constraints**
1. Fix:
   - model + tokenizer + dataset + formatting
   - sequence length / packing
   - precision + PEFT settings
2. Find max stable `per_device_train_batch_size` (no OOM, stable step time).

**Phase 1 — Coarse sweep on effective batch (via accumulation)**
- Sweep `gradient_accumulation_steps` by powers of two around your initial guess, e.g.:
  - `{1, 2, 4, 8, 16, 32}`
- Keep *either*:
  - learning rate fixed (to isolate stability/throughput effects), **or**
  - scale LR cautiously with effective batch (only if you know you want that regime).
- For each run, compare:
  - validation metric at a fixed token budget
  - loss stability (spikes/NaNs)
  - wall-clock to reach the same metric
  - throughput (tokens/sec)

**Phase 2 — Narrow in**
- Pick the best 2–3 candidates and run longer, ideally with 2 random seeds.
- Watch for:
  - run-to-run variance (small effective batch tends to be higher variance)
  - overfitting behavior (very stable training can still overfit; check val)

**Phase 3 — Couple with schedule**
- Once you’ve chosen `gradient_accumulation_steps`, *then* refine:
  - learning rate
  - warmup ratio/steps
  - max_steps/epochs
- Because changing accumulation changes how many optimizer updates you get for the same token budget, you want the schedule tuned *after* accumulation is settled.

**Practical “stop” rule**
- Prefer the smallest `gradient_accumulation_steps` that gives:
  - stable training (no spikes/NaNs)
  - acceptable variance across seeds
  - good throughput
  - and the validation metric you care about

That’s usually the best balance of stability and training efficiency.


### warmup_steps
#### Definition
`warmup_steps` controls **how long the learning rate “ramps up” at the start of training** before it reaches your configured `learning_rate`.

**Equalizer (LoRA) analogy:**
Imagine your LoRA adapters are a **graphic equalizer** you’re dialing in on a song. The learning rate is like the **master gain** knob feeding the equalizer. If you crank the gain instantly at the first beat, you can get a nasty pop/clipping (unstable updates). `warmup_steps` is the **fade‑in duration**: for the first *N* optimizer updates, you smoothly raise the gain from ~0 to the target level, then continue with the chosen schedule (linear/cosine/etc.).

**Precise semantics (Transformers / TRL SFT):**
- It is the **number of optimizer update steps** used for **linear warmup from 0 → `learning_rate`**.
- In recent `transformers`, `warmup_steps` can be:
  - an **integer** (exact number of warmup steps), or
  - a **float in [0, 1)**, interpreted as a **ratio of total training steps** (e.g., `0.05` means 5% of total steps).
- If you also set `warmup_ratio`, `warmup_steps` takes precedence when it’s non‑zero (because it specifies warmup explicitly).

#### Impact and Symptoms
**What it changes in training behavior**
- **Stabilizes the first phase of SFT.** Warmup reduces the chance that early large updates (especially with mixed precision, high LR, or noisy batches) push the model into a bad region.
- **Shifts where “useful learning” happens.** A longer warmup means fewer steps at full LR (and fewer at decay, depending on scheduler), which can slightly change convergence speed and final quality.

**Symptoms `warmup_steps` is too low**
- Loss spikes or oscillates sharply in the first few dozen updates.
- Gradient norms jump dramatically; occasional NaNs/infs (often seen with fp16).
- Early evaluation looks erratic; training may “recover” but ends worse than expected.

**Symptoms `warmup_steps` is too high**
- Training loss decreases very slowly at the start (“stuck in first gear”).
- Underfitting on short runs (you spend most of the run ramping up).
- You need more total steps/epochs to reach the same quality.

#### Simple Default
A practical starting point for **LoRA/QLoRA SFT**:

- **`warmup_steps = 0.03`** (3% warmup) *if your Transformers version supports ratio floats*,
  **or** `warmup_steps = round(0.03 * total_optimizer_steps)` as an integer.

Guardrails:
- For **very short runs** (< ~500 optimizer steps), use an integer floor like **10–25 steps** (otherwise 3% may be only a couple steps).
- For **large global batch sizes / higher learning rates / very long runs**, starting at **0.05–0.10** warmup is common.

Why this is a good “first try”:
- It’s usually enough to prevent the “initial pop” (instability) without spending a big chunk of training ramping up.

#### Detailed Configuration
Use this sequence before doing any hyperparameter sweeps.

1) **Pick your LR schedule first**
- Decide `lr_scheduler_type` (e.g., `cosine` or `linear`) and whether you will use decay to 0.
- Warmup interacts with the schedule: warmup happens *before* the main schedule.

2) **Compute total optimizer steps (the unit warmup uses)**
Warmup is counted in **optimizer updates**, not forward passes.
Compute:

- **effective_batch = per_device_train_batch_size × world_size × gradient_accumulation_steps**
- **steps_per_epoch ≈ ceil(num_training_examples / effective_batch)**
- **total_optimizer_steps = steps_per_epoch × num_train_epochs**
  (or `max_steps` if you set it)

If your dataset length is *unknown* (iterable/generator or heavy packing), the scheduler can be misconfigured unless you set an explicit `max_steps`.

3) **Choose how to express warmup**
- If supported: prefer **ratio form** for portability across dataset sizes:
  - `warmup_steps = 0.03` means 3% warmup even if you later change epochs, batch size, or filtering.
- If using integer warmup:
  - Set `warmup_steps = int(total_optimizer_steps × warmup_fraction)`.

4) **Sanity-check the warmup in logs**
- Log learning rate every few steps (`logging_steps` small early on).
- Verify LR starts near 0 and reaches the configured `learning_rate` exactly at step `warmup_steps`.

5) **Match warmup to your risk profile**
Increase warmup if you have any of the following:
- Higher LR (common for LoRA/QLoRA), small datasets, or very long sequences.
- fp16 instability or frequent gradient clipping.
- Large global batch sizes.

Decrease warmup if:
- Total training is extremely short, or you see slow start/underfitting.

#### Tuning Process
A clean way to tune `warmup_steps` is to treat it like the **fade-in length** of the training signal and search for the shortest fade-in that avoids clipping.

**Step 0 — Freeze everything else**
- Fix: dataset preprocessing, max_seq_length/packing, optimizer, base LR, scheduler type, batch size, and max_steps/epochs.
- Use a fixed seed for comparability.

**Step 1 — Coarse sweep (ratio or steps)**
Pick a small grid:
- Ratios: **{0.00, 0.01, 0.03, 0.06, 0.10}**
- Or steps (short runs): **{0, 10, 25, 50, 100}**

Run each for the full training budget (or at least long enough to cover warmup + early learning).

**What to record**
- Train loss curve (first 200–500 optimizer steps are most diagnostic)
- Gradient norm / clipping frequency
- Any NaN/inf events
- Eval metric at fixed checkpoints (same global steps)

**Decision rule**
- If any run diverges or is noisy early: move upward.
- If all are stable: prefer the **smallest warmup** that reaches the best eval (or best final loss) — it wastes fewer steps.

**Step 2 — Local refinement**
Take the best value and refine around it:
- Example: if 0.03 wins, test **{0.02, 0.03, 0.04}** (or ±10–20 steps).

**Step 3 — Stress test for robustness**
- Repeat the final candidate with 2–3 different seeds (or slightly different data shuffles).
- If performance varies a lot between seeds, increase warmup slightly (it often improves stability).

**Common pattern**
- If you raise LR (or global batch), you usually need **more warmup**.
- If you shorten training (fewer steps), you usually need **less warmup** (or a small fixed integer).

### learning_rate

#### Definition
Think of **LoRA** as adding a *graphic equalizer* (EQ) on top of a finished song: the base model is the recorded track, and the LoRA adapters are the EQ sliders you’re allowed to touch.

**learning_rate** is **the global “how big a slider move” knob**:
- In gradient-based training, the optimizer computes a direction to move the trainable weights (for LoRA: the adapter matrices).
- **learning_rate scales the size of each update step** in parameter space (often the *peak* or *initial* step size, depending on your LR scheduler).
- With adaptive optimizers (e.g., Adam/AdamW), the learning rate is still the main multiplier on the final per-parameter update.

**EQ analogy:**
If gradients say “boost 2 kHz a little, cut 80 Hz a bit,” the learning rate decides whether you nudge the sliders by a *millimeter* (gentle) or *a whole notch* (aggressive) each beat.

---

#### Impact and Symptoms
Because it controls step size, learning_rate largely determines **stability**, **speed**, and **whether you land in a good minimum**.

**Too high (sliders jerk too far each beat)**
- Training loss is noisy, spikes, or diverges; you may see NaNs/inf.
- Validation loss worsens even as training loss sometimes drops (overshoot).
- Outputs become “overcooked”: weird style artifacts, repetition, brittle instruction-following.
- Sudden regressions after seeming improvement (catastrophic overshoot events).
- Strong sensitivity to random seed / batch order.

**Too low (sliders barely move)**
- Loss decreases painfully slowly; metrics plateau early.
- Model fails to adopt the desired style/format; “it didn’t learn.”
- You need many more steps/epochs to reach the same quality.

**Mismatch with schedule / batch / optimizer**
- No warmup + moderate/high LR → early instability (the “first 100 steps blow up” pattern).
- Increasing effective batch size without adjusting LR → learning slows (under-update).
- Excessive weight decay combined with low LR → adapters stay near-zero (muted EQ).

What you can *observe quickly*:
- Plot training loss vs. steps: high LR looks like **sawtooth spikes** or explosions; low LR looks **flat**.
- Sample generations every N steps: high LR yields sudden behavior flips; low LR yields near-baseline behavior for too long.

---

#### Simple Default
A solid starting point for **LoRA / QLoRA SFT with AdamW**:

- **learning_rate = 2e-4**

If you’re *full fine-tuning* (updating all model weights), a typical starting point is about **10× smaller**:
- **learning_rate ≈ 2e-5**

(Use these as first tries, not universal truths—dataset size/quality, sequence length, and effective batch size matter.)

---

#### Detailed Configuration
A best-practice way to set learning_rate *before* you do any large sweep:

1. **Decide what learning_rate means in your stack**
   - In Hugging Face-style trainers, `learning_rate` is commonly the **initial / peak LR** used by the scheduler.
   - If using cosine/linear decay with warmup, treat it as **peak LR after warmup** (most practical mental model).

2. **Pick an LR schedule that prevents early chaos**
   - Use **warmup** (often a small fraction of total steps) to ramp from 0 → peak LR.
   - Then either **constant**, **cosine**, or **linear decay** depending on preference and total steps.
   - If you are doing short runs (few thousand steps), constant-after-warmup is often perfectly fine.

3. **Anchor learning_rate to your effective batch size**
   - Effective batch size = (micro-batch) × (gradient_accumulation) × (#devices).
   - If you change effective batch size materially, consider a proportional LR adjustment as a first approximation:
     - Bigger batch → you can often use a somewhat bigger LR (but don’t assume perfect linear scaling for LLM SFT).

4. **Treat LoRA and full fine-tuning differently**
   - LoRA adapters are a small parameter subspace; they often tolerate (and need) **higher LR** than full fine-tuning.
   - If you use **LoRA+ / separate LR ratios** (different LR for A vs. B matrices), keep the base `learning_rate` as the anchor and adjust ratios cautiously.

5. **Add the stabilizers that make LR easier to choose**
   - **Gradient clipping** (max grad norm) reduces the chance that one bad batch destabilizes training.
   - Keep **optimizer betas** and **weight decay** at sane defaults initially; don’t change everything at once.
   - Confirm your data formatting (prompt template, label masking) is correct—many “LR problems” are actually data problems.

6. **Run a short sanity check**
   - 200–500 steps is often enough to see if you’re exploding (too high) or stuck (too low).
   - Sample generations periodically; watch for sudden degradation.

---

#### Tuning Process
A systematic sweep process that works well in practice:

1. **Bracket the stable range (fast LR range test)**
   - Do a short run where LR increases smoothly from very small → large.
   - Find the LR where loss starts to blow up; take ~10–30% of that as an upper bound.
   - This gives you a *problem-specific* search window.

2. **Coarse log sweep (pick the neighborhood)**
   - Try 5–7 values spaced by ~2×:
     - Example (LoRA): {5e-5, 1e-4, 2e-4, 4e-4, 8e-4}
   - Keep everything else fixed (seed, steps, schedule, batch, data).

3. **Choose the winner by validation + qualitative checks**
   - Prefer the LR that gives:
     - best validation metric / lowest val loss
     - stable training curve
     - consistently good samples (format adherence, reduced hallucination, task correctness)
   - Reject “lottery ticket” runs that look great only at one checkpoint but are unstable.

4. **Refine locally**
   - Sweep ±50% around the best coarse value:
     - If best was 2e-4 → try {1e-4, 1.5e-4, 2e-4, 2.5e-4, 3e-4}

5. **Stress-test**
   - Re-run the top 1–2 candidates with 2–3 different seeds.
   - If results vary wildly, LR may be too high (or data is too small/noisy).

6. **Lock it and move on**
   - Once LR is stable, only then start sweeping other knobs (rank, dropout, batch size, weight decay, scheduler shape).


### optim (SFTConfig)
#### Definition
`optim` selects **which optimizer algorithm** the Trainer uses to update *trainable parameters* during supervised fine-tuning (SFT). In `trl.SFTConfig`, the default is `adamw_torch_fused`.

**Graphic‑equalizer analogy (LoRA = EQ):**
Think of LoRA as a **graphic equalizer** where each trainable LoRA weight is an EQ slider shaping the “song” (the model’s behavior). The `optim` setting is **the technique your sound engineer uses to move the sliders**:
- Some techniques make *tiny, very stable* adjustments (conservative, smooth).
- Some add *momentum* (keep moving in the same direction if it’s working).
- Some adapt the step size per slider (different sliders move at different rates).
- Some store the “engineer’s notes” in lower precision or “page” them to save memory (8-bit/paged variants).

So: LoRA defines *what can be changed*; `optim` defines *how the change is applied each step*.

#### Impact and Symptoms
`optim` primarily affects **(1) stability**, **(2) speed/throughput**, and **(3) memory footprint**.

**1) Stability / convergence**
- AdamW-style optimizers (including fused / 8‑bit / paged variants) are the most common baseline for LLM SFT. In Transformers, optimizer selection is done by putting the optimizer name in `optim`.
- Symptoms `optim` is a poor match (or incompatible with your setup):
  - **Loss spikes / divergence** early (steps become too aggressive or numerically unstable).
  - **NaNs/Inf** in loss or gradients (often appears with mixed precision + incompatible fused setups).
  - **Loss oscillates** without trending down (step rule too “jittery” for your LR/schedule).
  - **Very slow loss decrease** even with a known-good LR (optimizer too conservative for your setup).

**2) Speed / wall-clock**
- “Fused” variants can be meaningfully faster on supported hardware because they fuse kernels and reduce overhead.
- Symptoms `optim` is non-optimal for speed:
  - **Tokens/sec is much lower** than expected for your GPU class and batch size.
  - GPU utilization is low while CPU is busy (optimizer step becomes the bottleneck).

**3) Memory**
- 8‑bit optimizers can reduce optimizer-state memory substantially (useful for full fine-tuning, bigger batches, or longer context).
- Symptoms `optim` is non-optimal for memory:
  - You’re **OOMing during optimizer.step()** (optimizer states don’t fit).
  - You can only run **tiny batch sizes** or can’t enable a longer context length.
  - Frequent **CUDA out-of-memory** right after backward/step even though forward fits.

**Practical note (LoRA/QLoRA):**
- With LoRA/QLoRA, you often train far fewer parameters, so optimizer-state memory is smaller than in full fine-tuning. In that regime, `optim` choice is often about **speed + stability** more than pure memory—unless you’re pushing very long context or large effective batch sizes.

#### Simple Default
A strong “first try” depends on your setup:

- **Default / best throughput on modern GPUs:** `adamw_torch_fused` (TRL’s current default).
  Use when you have a reasonably recent PyTorch + CUDA stack and you’re not hitting weird mixed-precision edge cases.

- **Fallback if fused causes issues:** `adamw_torch`
  Use when fused is unavailable or you see fused-specific errors with fp16/bf16, checkpointing, or unusual distributed setups.

- **If memory is the limiting factor (common on smaller GPUs):** `paged_adamw_8bit` or `adamw_bnb_8bit`
  Use when you’re OOMing due to optimizer state, or you need to unlock bigger batches/longer context.

#### Detailed Configuration
Use this “configure before you sweep” checklist to lock in a sensible `optim` choice.

**Step 1 — Classify your training regime**
1. **Full fine-tuning** (many trainable params): optimizer state is huge → prioritize memory-efficient variants early (8‑bit / paged) if you’re close to VRAM limits.
2. **Adapter fine-tuning (LoRA/QLoRA):** optimizer state is smaller → default to fused AdamW for speed unless you’re memory bound due to context/batch.

**Step 2 — Pick a baseline optimizer family**
- Start in the **AdamW family** unless you have a clear reason otherwise. It’s the most battle-tested baseline for SFT.
- Treat “optimizer family” as a *categorical* choice you change infrequently.

**Step 3 — Match the optimizer to your hardware + precision**
- If you’re on **GPU** with a modern stack → start with `adamw_torch_fused` (fastest baseline).
- If you’re running on constrained VRAM or want to push longer context → pick a bitsandbytes option (`adamw_bnb_8bit`, `paged_adamw_8bit`) provided your environment supports it.
- If you’re on CPU or special accelerators (XLA/NPU) → use the platform-specific optimizer names available in `OptimizerNames` (Transformers exposes many options).

**Step 4 — Avoid “silent overrides”**
In TRL/Transformers you can override `optim` in two other ways:
- `optimizers=(optimizer, scheduler)` (you fully supply both objects)
- `optimizer_cls_and_kwargs=(cls, kwargs)` (overrides `optim` + `optim_args`)

Make sure you’re not accidentally overriding `optim` elsewhere in your code.

**Step 5 — Do a short smoke test**
Before any expensive sweep:
- Run ~100–500 steps and log:
  - loss curve (should decrease smoothly)
  - grad norm / NaNs
  - tokens/sec
  - peak VRAM
- If fused → confirm no fp16/bf16 incompatibility errors.
- If 8‑bit/paged → confirm bitsandbytes is correctly installed and the optimizer initializes.

#### Tuning Process
Treat `optim` as a **small, structured categorical sweep**, not a giant hyperparameter grid.

**Phase 0 — Lock everything else**
Hold constant:
- dataset + sampling
- max_length / packing
- batch size (or effective batch via grad accumulation)
- LR schedule + warmup
- LoRA config (rank, alpha, target modules) if using adapters

**Phase 1 — Choose 2–3 candidates**
Good starting candidate set:
1. `adamw_torch_fused` (speed baseline; TRL default)
2. `adamw_torch` (stability fallback)
3. If memory-limited: `paged_adamw_8bit` or `adamw_bnb_8bit`

**Phase 2 — Short “screening” runs**
For each candidate:
- Run 1–3k steps (or ~5–10% of a full run).
- Compare:
  - stability (no NaNs, smooth loss)
  - speed (tokens/sec)
  - VRAM headroom

Eliminate anything that is unstable or far slower.

**Phase 3 — Pairwise LR retune for the winner(s)**
Optimizer choice and LR interact strongly.
- Take the top 1–2 optimizers and do a **small LR sweep** (e.g., ×0.5, ×1, ×2 around your baseline).
- Keep everything else fixed.
- Pick the combo with best validation/perplexity and most stable training.

**Phase 4 — Confirm with a full-length run**
Run to completion with the chosen optimizer + LR.
- Validate that improvements persist beyond the short window (some optimizers look good early and degrade later).

**Phase 5 — Only then consider “exotic” optimizers**
Transformers supports many specialized optimizer names (and new ones appear over time).
Only explore them after you have a stable AdamW baseline and a clear reason (e.g., you need lower memory than AdamW, or you’re doing a research comparison).

---

```text
Sources (URLs)
- TRL SFTTrainer / SFTConfig docs: https://huggingface.co/docs/trl/en/sft_trainer
- Transformers Optimizers guide: https://huggingface.co/docs/transformers/en/optimizers
- Transformers “Efficient training on GPU” guide: https://huggingface.co/docs/transformers/en/perf_train_gpu_one
- bitsandbytes 8-bit optimizers docs: https://huggingface.co/docs/bitsandbytes/en/optimizers
- Transformers issue listing OptimizerNames values (example): https://github.com/huggingface/transformers/issues/31651
```


### Parameter Name
**weight_decay** (in `trl.SFTConfig` / inherited from 🤗 Transformers `TrainingArguments`)

#### Definition
Think of your LoRA as a **graphic equalizer (EQ)** sitting on top of a finished song (the frozen base model).

- The **LoRA sliders** (the trainable adapter weights) boost or cut certain “frequencies” (behaviors) so the song matches your dataset.
- **`weight_decay`** is like adding a **gentle spring** to every slider: after each tiny adjustment, the spring tugs the slider back toward its neutral position.

In optimizer terms, `weight_decay` is the **decoupled weight decay coefficient** used by AdamW-style optimizers. Practically, each update step applies:
- the usual AdamW gradient-based update, **plus**
- a small **shrink** of the parameters proportional to their current value.

In Transformers/Trainer-style setups, decay is typically applied to **most weights**, while **biases and LayerNorm weights are excluded**.

#### Impact and Symptoms
What it changes:
- **Regularization strength**: discourages large parameter norms.
- **Generalization vs. fit** tradeoff: helps prevent memorization, but can also make the model “too conservative.”
- **Adapter magnitude (LoRA/QLoRA)**: because only adapters are trainable, `weight_decay` mainly controls how large your LoRA deltas become, i.e., how far you move from the base model.

Observable symptoms when it’s non-optimal:

**Too low (often 0.0 when you needed some decay)**
- Train loss keeps dropping while eval loss/metrics plateau or worsen.
- Outputs start to **mirror** training phrasing (memorization), brittle behavior on slightly reworded prompts.
- Strong improvements on in-domain prompts but **regression** on nearby domains (over-specialization).
- LoRA weight norms grow steadily; merged model deviates more than expected.

**Too high (over-regularized)**
- Train loss plateaus early; model struggles to learn the formatting or task constraints.
- Completions look **too close to the base model** (weak adaptation), “adapter collapse” (LoRA deltas remain tiny).
- You see a “bland safety blanket” effect: generic answers, less instruction adherence, lower task accuracy.
- Small dataset + high decay can look like **underfitting**: both train and eval metrics are poor.

Important interaction to remember:
- In common AdamW implementations, the *effective* decay pressure scales with the **learning rate**, roughly like `lr * weight_decay`. So if you change LR, you often need to re-think decay too.

#### Simple Default
A solid starting point for SFT with LoRA/QLoRA:

- **`weight_decay = 0.01`**

Why this is a good “first knob”:
- It’s small enough that you usually won’t destroy learning,
- but large enough to reduce overfitting on the typical SFT regime (smallish/medium datasets, a few epochs).

When to override immediately:
- If you have a **very large, diverse SFT dataset** (or you’re doing very few steps): try **0.0–0.001**.
- If you have **very limited data** (few thousand examples) or you see rapid overfitting: try **0.03–0.1**.

#### Detailed Configuration
Before you sweep `weight_decay`, lock down the surrounding “audio chain” so you’re tuning one knob at a time:

1. **Confirm what’s actually being decayed**
   - In full fine-tuning, most weights get decay, but **bias and LayerNorm weights are typically excluded**.
   - In LoRA/QLoRA, *only adapter parameters are trainable*, so decay mostly affects LoRA A/B matrices (and any trainable embeddings you enabled).

2. **Verify the optimizer family**
   - `weight_decay` assumes **AdamW-style decoupled decay** behavior.
   - If you switch optimizers (`optim=`), ensure it still supports the same semantics (e.g., AdamW, paged AdamW).

3. **Set a reliable evaluation protocol**
   - Use a **held-out validation set** that matches your deployment distribution.
   - Track at least one **task metric** (exact match, pass@k, rubric score) plus **eval loss**.
   - If you can, add a small **out-of-domain** eval slice to detect over-specialization.

4. **Add lightweight diagnostics**
   - Log: train vs eval loss curves, and (if possible) **LoRA parameter norm** over time.
   - Watch for: early divergence (overfit), or flat learning (underfit).

5. **Stabilize other regularizers first**
   - Keep **dropout** (including LoRA dropout) fixed.
   - Keep **num epochs / max_steps** fixed.
   - Keep **LR schedule + warmup** fixed.
   - Then tune `weight_decay`.

Rule of thumb:
- If you change **learning rate by a big factor**, revisit `weight_decay` because the “shrink per step” changes.

#### Tuning Process
A practical sweep recipe that works well for SFT:

**Phase 0 — sanity check**
- Run a short training (e.g., 2–5% of total steps) with `weight_decay=0.01`.
- Confirm: loss decreases, eval is finite, outputs change in the expected direction.

**Phase 1 — coarse log-scale sweep**
Test a small set spanning “none → strong”:
- `0.0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1`

Pick the winner based on:
- best validation metric (primary),
- plus secondary checks: stability, no obvious overfitting, acceptable “base skill” retention.

**Phase 2 — local refinement**
Around the best value `w*`, refine with 2–4 nearby points, e.g.:
- if `w* = 0.01`, try `0.005, 0.01, 0.02`
- if `w* = 0.03`, try `0.02, 0.03, 0.05`

**Phase 3 — couple with LR only if needed**
If you later change LR substantially (common during tuning), repeat a *small* decay sweep:
- keep the same grid shape, but narrower (e.g., ±3× around the current best),
- because the effective decay pressure changes with LR.

**Decision heuristics**
- If you see **train↓, eval↑ early** → increase `weight_decay`.
- If both **train and eval are bad** and the model barely adapts → decrease `weight_decay` (or raise LR / train longer).
- If adaptation is strong but you lose generality → increase `weight_decay` slightly (or reduce epochs / LoRA rank).


### lr_scheduler_type (SFTConfig)

#### Definition
`lr_scheduler_type` chooses **which learning-rate schedule** the TRL **SFTTrainer** uses while training—i.e., the *shape* of the learning rate over optimizer steps (including how it warms up and/or decays). In TRL’s `SFTConfig`, it defaults to `"linear"` and is passed through to the underlying Hugging Face `Trainer`, which builds the scheduler via `transformers.get_scheduler(...)` based on this name.

**LoRA-as-equalizer analogy (graphic EQ):**
- Think of your **LoRA adapter** as the **graphic equalizer**: it decides *which frequency bands* (model behaviors) get boosted or cut.
- `lr_scheduler_type` is the **automation curve for the master gain over time**: it decides *when* the EQ adjustments happen gently vs aggressively during the “song” (training run).
  - Warmup = fade-in (avoid a “pop” at the start).
  - Decay = fade-out (avoid over-driving late training).
  - Restarts/oscillations = intentional “pulses” in gain.

**Common values you can pass** (these are the scheduler names used by `TrainingArguments.lr_scheduler_type` and therefore by `SFTConfig`):
- `"linear"`
- `"cosine"`
- `"cosine_with_restarts"`
- `"polynomial"`
- `"constant"`
- `"constant_with_warmup"`
- `"inverse_sqrt"`
- `"reduce_lr_on_plateau"`
- `"cosine_with_min_lr"`
- `"warmup_stable_decay"`

> Note: Some schedules have *extra knobs* (e.g., `num_cycles`, `min_lr_rate`, `num_decay_steps`). In TRL/Transformers, those are supplied via `lr_scheduler_kwargs` (a sister argument in `SFTConfig`).

#### Impact and Symptoms
A learning-rate schedule controls the **effective step size** of every optimizer update. With LoRA/QLoRA SFT, the schedule often matters as much as the peak `learning_rate`, because adapter weights can overreact early and “freeze” late if LR decays too fast.

**What it changes in training behavior**
- **Stability in the first few hundred steps (warmup interaction):**
  - Schedules that warm up (explicitly or through a warmup ratio/steps) ramp LR up gradually → fewer loss spikes.
- **How quickly you stop learning (late-stage LR):**
  - Schedules that decay to ~0 quickly can “run out of LR” and stall.
  - Schedules with a floor (e.g., `cosine_with_min_lr`, WSD with `min_lr_ratio`) keep learning alive late.
- **How “exploratory” training feels mid-run:**
  - Cosine shapes tend to reduce LR more smoothly; restarts add periodic bursts that can help escape shallow minima but can also add noise.

**Observable symptoms it’s set non-optimally**
- **Too aggressive early LR (warmup too short for your peak LR / schedule too steep):**
  - Training loss spikes or becomes NaN/Inf early
  - Large, erratic `grad_norm` spikes
  - Sudden collapse in output quality (nonsense, repetition) after a few steps
- **LR decays too fast (schedule mismatched to run length):**
  - Loss improves early, then flatlines while you still have lots of steps left
  - Eval metrics stop improving much earlier than expected
  - “It trains, but feels like it stops learning halfway through”
- **LR stays high too long (constant / weak decay for your run):**
  - Training loss keeps dropping while eval plateaus or worsens (overfitting)
  - Output becomes overconfident, brittle, or style-copied from training set
- **Restarts are too punchy (cosine_with_restarts misused):**
  - Periodic oscillations in loss/eval that line up with restart boundaries
  - Instability right after each restart bump
- **reduce_lr_on_plateau misconfigured:**
  - LR never drops because the monitored metric is noisy / too infrequent
  - Or LR drops too early because eval metric is high variance

#### Simple Default
**Start with:** `lr_scheduler_type="linear"`

Why this is a good “first run” default:
- It is the **default in `SFTConfig`** and the most widely used baseline for Transformer fine-tuning.
- It is easy to reason about: warm up, then steadily reduce LR.
- It tends to be robust across short-to-medium SFT runs where you don’t yet know the best schedule.

If you want a *slightly more “LLM-SFT modern”* default (especially for longer runs or if you often see late-stage overfitting/forgetting), a strong alternative is:
- `lr_scheduler_type="cosine_with_min_lr"` with `lr_scheduler_kwargs={"min_lr_rate": 0.1}`

This keeps a non-trivial LR floor late training, which is frequently helpful for SFT where you still want gentle adaptation near the end.

#### Detailed Configuration
A best-practice way to configure `lr_scheduler_type` **before** doing sweeps:

1. **Compute your “true” training length in optimizer steps**
   - If using `max_steps`, that is your total.
   - If using epochs, compute:
     - `total_steps ≈ (num_examples / effective_batch_size) * num_epochs`
     - `effective_batch_size = per_device_train_batch_size * gradient_accumulation_steps * num_devices`
   - Reason: schedulers behave very differently at 500 vs 50,000 steps.

2. **Pick a schedule family based on run length + risk tolerance**
   - **Short SFT (≤ ~1–2k steps)**
     - Prefer: `"linear"` or `"constant_with_warmup"`
     - Avoid: aggressive decays to near-zero (they can end learning too early).
   - **Medium SFT (~2k–20k steps)**
     - Prefer: `"linear"` or `"cosine"`
   - **Long SFT (≥ ~20k steps) / “quality polishing”**
     - Prefer: `"cosine_with_min_lr"` or `"warmup_stable_decay"` (WSD)

3. **Configure warmup (even though it’s not *inside* lr_scheduler_type, it’s inseparable)**
   - Typical starting points:
     - LoRA/QLoRA SFT: `warmup_ratio=0.01–0.05` (often ~0.03)
     - Full fine-tune: `warmup_ratio=0.03–0.10` if you’re pushing LR
   - If you see early instability, increase warmup *before* changing scheduler type.

4. **Use `lr_scheduler_kwargs` for schedule-specific knobs**
   - Cosine:
     - `{"num_cycles": 0.5}` (default half-wave)
   - Cosine with min LR:
     - `{"min_lr_rate": 0.05}` or `{"min_lr_rate": 0.1}` (LR floor as a fraction of peak LR)
   - Cosine with restarts:
     - `{"num_cycles": 2}` or `3` (integer restarts; use sparingly)
   - Polynomial:
     - `{"power": 1.0, "lr_end": 1e-7}` (tune `power` first)
   - Inverse sqrt:
     - `{"timescale": <int>}` (defaults to warmup steps if omitted)
   - Warmup-stable-decay (WSD):
     - `{"num_decay_steps": ..., "num_stable_steps": ...,
        "warmup_type": "linear", "decay_type": "cosine",
        "min_lr_ratio": 0.05}`
   - Reduce-on-plateau:
     - Pass the usual `torch.optim.lr_scheduler.ReduceLROnPlateau` args (e.g., `factor`, `patience`, `threshold`, `min_lr`) and make sure your eval metric is reliable.

5. **Sanity-check the LR curve**
   - Log `learning_rate` each `logging_steps`.
   - Verify it ramps up and decays as expected, and that the decay isn’t “finished” halfway through training unless that’s intended.

**Concrete config example (common in modern SFT recipes):**
```python
from trl import SFTConfig

args = SFTConfig(
    learning_rate=2e-4,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine_with_min_lr",
    lr_scheduler_kwargs={"min_lr_rate": 0.1},
)
```

#### Tuning Process
A structured way to optimize `lr_scheduler_type` with minimal wasted compute:

1. **Lock everything else first**
   - Fix: model, dataset mixture, context length, packing strategy, optimizer (`adamw_torch` vs others), weight decay, batch size, gradient accumulation.
   - Use 1–2 seeds initially; don’t start with 5.

2. **Pilot run for stability**
   - 200–500 steps (or ~1–2% of total) is enough to detect:
     - early divergence
     - wild gradient norms
     - LR curve mistakes
   - If unstable: increase warmup_ratio (and/or reduce peak LR) before changing scheduler type.

3. **Coarse scheduler sweep (small grid)**
   Keep peak LR and warmup fixed. Compare:
   - `"linear"`
   - `"cosine"`
   - `"cosine_with_min_lr"` (e.g., min_lr_rate=0.05)
   - `"warmup_stable_decay"` (e.g., decay 30–50% of run, min_lr_ratio=0.05)

   **Select winners by:**
   - best eval metric at fixed compute budget
   - smoothness/stability (no spikes, no collapse)
   - better late-stage improvements (not just early)

4. **Fine sweep schedule-specific knobs**
   - If cosine wins:
     - `num_cycles ∈ {0.5, 1.0}`
   - If cosine_with_min_lr wins:
     - `min_lr_rate ∈ {0.02, 0.05, 0.1, 0.2}`
   - If WSD wins:
     - choose a **decay fraction**: `num_decay_steps / total_steps ∈ {0.2, 0.4, 0.6}`
     - set stable steps as the remainder (after warmup)
     - `min_lr_ratio ∈ {0.0, 0.02, 0.05, 0.1}`
   - If reduce_lr_on_plateau wins:
     - `patience ∈ {1, 2, 4}`, `factor ∈ {0.5, 0.2}`, and ensure eval cadence is frequent enough.

5. **Confirm with a full-length run + a second seed**
   - The “best” schedule can flip if it was just lucky on one seed.
   - Promote only schedules that are stable and consistently better.

**Sources used (for you to cross-check):**
- TRL `SFTConfig` signature showing `lr_scheduler_type` default `"linear"` and `lr_scheduler_kwargs` support:
  - https://huggingface.co/docs/trl/v0.20.0/en/sft_trainer
- Transformers scheduler name mapping and scheduler APIs (`get_scheduler`, `get_wsd_schedule`, cosine w/ min LR, etc.):
  - https://huggingface.co/docs/transformers/en/main_classes/optimizer_schedules
  - https://huggingface.co/docs/transformers/v4.51.1/en/main_classes/optimizer_schedules
- `lr_scheduler_kwargs` discussed as the way to pass schedule hyperparameters:
  - https://discuss.huggingface.co/t/hyperparameters-for-lr-scheduler-type-in-trainer-arguments/38469
- Example SFT recipe using `cosine_with_min_lr` + `min_lr_rate`:
  - https://cookbook.openai.com/articles/gpt-oss/fine-tune-transfomers
- Background on Warmup-Stable-Decay (WSD) popularity:
  - https://openreview.net/forum?id=QJfH0lBZ8d

