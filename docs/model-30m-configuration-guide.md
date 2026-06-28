# Zipformer-Small (~30M) — Detailed Configuration Guide

> **Scope:** This document covers every configurable parameter of the Zipformer-Small model,
> explains the reasoning behind each value, documents the constraints that must not be violated,
> shows how parameters interact with each other, and guides how to safely experiment with changes.
>
> **Single file to edit:** `ASR/zipformer/train.py` → function `add_model_arguments()` (line 133).

---

## Table of Contents

1. [Architecture at a Glance](#1-architecture-at-a-glance)
2. [Parameter Taxonomy](#2-parameter-taxonomy)
3. [Stack-Level Parameters (the core model size knobs)](#3-stack-level-parameters)
4. [Per-Head Attention Parameters](#4-per-head-attention-parameters)
5. [Training Regularization Parameters](#5-training-regularization-parameters)
6. [Loss Function Parameters](#6-loss-function-parameters)
7. [Decoder & Joiner Parameters](#7-decoder--joiner-parameters)
8. [Optional Heads (CTC, Attention Decoder)](#8-optional-heads)
9. [Streaming / Causal Parameters](#9-streaming--causal-parameters)
10. [Optimizer & Scheduler Parameters](#10-optimizer--scheduler-parameters)
11. [Constraint Rules (must not violate)](#11-constraint-rules)
12. [Parameter Interaction Map](#12-parameter-interaction-map)
13. [Complete Current Defaults (30M config)](#13-complete-current-defaults-30m-config)
14. [Size Variants Comparison](#14-size-variants-comparison)
15. [How to Safely Experiment](#15-how-to-safely-experiment)

---

## 1. Architecture at a Glance

The model is a **U-Net–shaped Zipformer2 encoder** paired with a **stateless RNN-T decoder** and a **joiner**:

```
Raw audio (16kHz WAV)
        │
        ▼
  [STFT / Fbank]  ← 80-dim Mel filterbank, 25ms/10ms → 100Hz sequence
        │
        ▼
  [Conv2dSubsampling]  ← 3×Conv2d + ConvNeXt → 50Hz, dim=192
        │
        ▼  50Hz
  ┌─────────────────────────────────────────────────────────┐
  │  Stack 1 (50Hz,  d=192, 2 blocks)                       │
  │  Stack 2 (25Hz,  d=256, 2 blocks)  ← downsample ×2      │
  │  Stack 3 (12.5Hz,d=256, 2 blocks)  ← downsample ×4      │
  │  Stack 4 (6.25Hz,d=256, 2 blocks)  ← downsample ×8      │
  │  Stack 5 (12.5Hz,d=256, 2 blocks)  ← upsample ×4        │
  │  Stack 6 (25Hz,  d=256, 2 blocks)  ← upsample ×2        │
  └─────────────────────────────────────────────────────────┘
        │
        ▼  25Hz (output_downsampling_factor=2)
  [encoder output: (T/4, B, 256)]
        │                          ▲
        ├───────────────────────── │ ──── encoder_proj: Linear(256→512)
        │                          │
  [Decoder]                  [Joiner]  ──── output_linear: Linear(512→vocab_size=2000)
  Embedding(2000, 512)
  + Conv1d(context=2)
```

**The final output sequence rate is 25Hz** (one frame every 40ms), because:
- Conv2dSubsampling reduces 100Hz → 50Hz (factor 2)
- `output_downsampling_factor=2` reduces 50Hz → 25Hz (factor 2)
- Combined: every 4th input frame is one output frame

---

## 2. Parameter Taxonomy

Parameters fall into four categories:

| Category | Effect on params? | Effect on memory? | Risk of breaking? |
|----------|:-----------------:|:-----------------:|:-----------------:|
| **Stack dims / layers** | ✅ Major | ✅ Major | ⚠️ Medium |
| **Attention head dims** | ✅ Medium | ✅ Medium | ⚠️ Constraints apply |
| **Training regularization** | ❌ None | ❌ None | 🟢 Safe to tune |
| **Loss scales** | ❌ None | ❌ None | 🟢 Safe to tune |

---

## 3. Stack-Level Parameters

These are the **primary knobs that control model size**. All 6 values correspond to Stacks 1–6.

---

### 3.1 `--num-encoder-layers`

```
Current: "2,2,2,2,2,2"
Type:    comma-separated int tuple (one per stack)
Range:   [1, ∞)  — practical max ~6 before diminishing returns
```

**What it controls:** The number of `Zipformer2EncoderLayer` blocks inside each stack.

**How parameters scale:** Each layer adds one full block of parameters. The cost per block for stack `i` is roughly:
```
~3d² (NonlinAttn) + 6d×ff (FeedForward) + 6d² (Conv×2) + ... ≈ 9d² + 6d×ff
```

**Per-stack block cost (approximate):**

| Stack | d | ff | Cost/block | Layers | Total |
|-------|---|----|------------|--------|-------|
| 1 | 192 | 512 | ~1.0M | 2 | ~2.0M |
| 2 | 256 | 768 | ~1.9M | 2 | ~3.8M |
| 3 | 256 | 768 | ~1.9M | 2 | ~3.8M |
| 4 | 256 | 768 | ~2.0M | 2 | ~4.0M |
| 5 | 256 | 768 | ~1.9M | 2 | ~3.8M |
| 6 | 256 | 768 | ~1.9M | 2 | ~3.8M |

**Effect on training:** More layers → more expressive + more stable with `ScaledAdam`, but also slower per epoch and more GPU memory. For low-resource settings (≤ 50h labeled), adding layers in outer stacks (1, 6) can help more than in the bottleneck (stack 4).

**Warmup interaction:** Each stack's warmup window is `[warmup_batches*(i+1)/(N+1), warmup_batches*(i+2)/(N+1)]`. More stacks = each stack warms up faster within the total window.

---

### 3.2 `--encoder-dim`

```
Current: "192,256,256,256,256,256"
Type:    comma-separated int tuple (one per stack)
Constraint: encoder_unmasked_dim[i] <= encoder_dim[i]  ← hard assert, will crash if violated
```

**What it controls:** The embedding/hidden dimension `d` of each stack. This is the single most impactful parameter for model size — cost scales as `O(d²)`.

**How `convert_num_channels` works:** Between stacks, the Zipformer top-level loop calls `convert_num_channels(x, encoder_dim[i])`. If `encoder_dim[i]` > current `x.shape[-1]`, it **zero-pads** the extra dims. If smaller, it **truncates**. This is how different stack dims chain together without learnable projections.

**Joiner coupling:** The joiner's `encoder_proj` input dim is `max(encoder_dim)`. Currently `max(192,256,256,256,256,256)=256`. Changing any stack above 256 increases joiner size. Changing the max dimension changes the joiner input.

**Stack 1 is special:** `encoder_dim[0]` must equal the output dim of `Conv2dSubsampling` (controlled separately in `get_encoder_embed()`). Both are currently `192`. If you change `encoder_dim[0]`, you **must** also verify the subsampling output dim matches.

**Memory note:** Activation memory during training scales as `O(T × B × d)`, so stack 1 (highest frame rate, 50Hz) benefits most from a small `d=192`.

---

### 3.3 `--feedforward-dim`

```
Current: "512,768,768,768,768,768"
Type:    comma-separated int tuple (one per stack)
```

**What it controls:** The **middle** FF hidden dimension inside each block. There are actually 3 FF modules per block with different sizes:
- **FF1 (first):** hidden = `3/4 × feedforward_dim[i]`
- **FF2 (middle):** hidden = `feedforward_dim[i]`  ← this is what you set
- **FF3 (last):** hidden = `5/4 × feedforward_dim[i]`

So for stack 2 (`ff=768`):
```
FF1: d=256, hidden=576   → params: 2×256×576 + bias ≈ 295K
FF2: d=256, hidden=768   → params: 2×256×768 + bias ≈ 393K
FF3: d=256, hidden=960   → params: 2×256×960 + bias ≈ 491K
```
Combined FF contribution per block = ~1.18M out of ~1.9M total per block (**62% of block cost**).

**Ratio guidance:** A good ratio is `feedforward_dim ≈ 3× encoder_dim`. The current config uses:
- Stack 1: 512/192 = 2.67×
- Stacks 2-6: 768/256 = 3.0×

---

### 3.4 `--num-heads`

```
Current: "4,4,4,8,4,4"
Type:    comma-separated int tuple (one per stack)
Constraint: encoder_dim[i] must be divisible by num_heads[i]
Constraint: encoder_dim[i] >= num_heads[i] × (query_head_dim + pos_head_dim + value_head_dim)
```

**What it controls:** Number of attention heads per stack. Stack 4 uses 8 heads (highest temporal resolution → needs richer attention patterns to capture long-range dependencies at 6.25Hz).

**Head budget check (must satisfy):**
```
For stack 1: 4 × (32 + 4 + 12) = 4 × 48 = 192 ≤ encoder_dim[0]=192  ✅ (exactly at limit)
For stacks 2-6: 4 × 48 = 192 ≤ encoder_dim=256  ✅
For stack 4: 8 × 48 = 384 ≤ encoder_dim=256  ❌ VIOLATION!
```

Wait — Stack 4 has `8 heads × 48 = 384 > d=256`. How? Because `MHAW` uses a projection:
- Query: `d × (num_heads × query_head_dim)` = `256 × (8×32) = 256×256`
- The input projection is `d → num_heads × (query_head_dim + pos_head_dim)` = `256 → 8×36 = 288`

This is an **overcomplete projection** and is valid — attention weights are computed in a space larger than the embedding. The constraint is actually in `RelPositionMultiheadAttentionWeights` which uses an `in_proj: Linear(d, num_heads*(query_head_dim + pos_head_dim))`. With d=256 and num_heads=8: `256 → 8×36=288`. This is fine.

**Effect on compute:** MHAW cost scales as `O(d × H × qd × T²)` per block. Doubling heads from 4 to 8 roughly doubles attention compute.

---

### 3.5 `--downsampling-factor`

```
Current: "1,2,4,8,4,2"
Type:    comma-separated int tuple (one per stack)
Constraint: Must form a valid U-Net pattern (go down, then come back up)
```

**What it controls:** The temporal downsampling ratio for each stack **relative to the Conv2dSubsampling output (50Hz)**. So stack 4 with factor 8 operates at 50/8 = 6.25Hz.

**Do NOT change this** unless you're redesigning the U-Net topology. It has no effect on parameter count (SimpleDownsample has only `downsample` bias params), but it fundamentally changes what the model learns.

**Layer drop rate interaction:** The final layer drop rate for stack `i` is:
```python
final_layerdrop_rate = 0.035 * (downsampling_factor[i] ** 0.5)
```
Higher downsampling → higher layer drop probability → stronger regularization in bottleneck stacks. Stack 4 (ds=8): `0.035 × √8 ≈ 0.099` = ~10% drop rate.

---

### 3.6 `--encoder-unmasked-dim`

```
Current: "192,192,256,256,256,256"
Type:    comma-separated int tuple (one per stack)
Hard constraint: encoder_unmasked_dim[i] <= encoder_dim[i]  (assertion in Zipformer2.__init__)
```

**What it controls:** The "always-visible" portion of each stack's embedding during **training-time feature masking** (not SpecAugment). The masking scheme:
- Dims `[0 : unmasked_dim]` → always visible (never zeroed)
- Dims `[unmasked_dim : unmasked_dim + (d - unmasked_dim)//2]` → zeroed on ~12.5% of sequences
- Dims `[unmasked_dim + (d - unmasked_dim)//2 : d]` → zeroed on ~25% of sequences (double-masked)

This forces the model to learn redundant representations across the full embedding dimension, not just rely on the first channels.

**Recommended setting:** Set to the same as `encoder_dim` for maximum capacity, or to `encoder_dim - 64` for aggressive masking. Do not set to 0.

**Current values explained:**
- Stack 1: `unmasked=192 = encoder_dim=192` → no masking (stack 1 is bottleneck-free, it's the input)
- Stack 2: `unmasked=192 < encoder_dim=256` → 64 dims are masked
- Stacks 3-6: `unmasked=256 = encoder_dim=256` → full masking disabled for these stacks

> **Note:** Setting all values equal to `encoder_dim` is safe and simply disables the feature masking regularization.

---

### 3.7 `--cnn-module-kernel`

```
Current: "31,31,15,15,15,31"
Type:    comma-separated int tuple (one per stack)
Constraint: Must be ODD (assertion in ConvolutionModule.__init__)
```

**What it controls:** The kernel size of the depthwise Conv1d inside each `ConvolutionModule` (there are 2 per block).

**What kernel size means:** A kernel of size `k` at frame rate `f` covers a receptive field of `k/f` seconds:
- Stack 1 (50Hz, k=31): 31/50 = **620ms** receptive field per conv layer
- Stack 4 (6.25Hz, k=15): 15/6.25 = **2.4 seconds** receptive field per conv layer

**Parameter cost:** Depthwise conv uses `d × k` params (groups=d). Cost is negligible compared to linear layers:
- Stack 4: `256 × 15 = 3,840` params (×2 convs per block = 7,680)

**Safe to change:** Can increase for more context or decrease for faster inference. Always keep odd.

---

## 4. Per-Head Attention Parameters

These are **constant across all stacks** (single scalar values) and change the attention mechanism's expressiveness and cost.

---

### 4.1 `--query-head-dim`

```
Current: 32
Type:    int (or comma-separated per-stack)
Effect on params: O(d × H × qd) for MHAW in_proj
```

**What it controls:** The dimension of Q and K vectors per attention head. The attention score matrix is computed as `QKᵀ / √query_head_dim`.

**Effect on compute:** Dot-product attention cost scales as `O(T² × H × qd)`. Reducing `query_head_dim` from 32 to 24 reduces attention compute by 25%.

**Typical values:** 24–64. The upstream icefall Large uses `query_head_dim=32` (same as here).

---

### 4.2 `--value-head-dim`

```
Current: 12
Type:    int (or comma-separated per-stack)
Effect on params: O(d × H × vd) for SelfAttention in_proj and out_proj
```

**What it controls:** The dimension of V vectors per attention head. The output of attention is `AV` where `V` has dim `H × value_head_dim`.

**Coupling:** `value_head_dim` is deliberately small (12 vs. 32 for query) — this is Zipformer's design for efficiency. The SA output has dim `H × 12`, which is then projected back to `d` via `out_proj`. With `H=4, vd=12`: SA output = 48-dim → projected to 256-dim.

---

### 4.3 `--pos-head-dim`

```
Current: 4
Type:    int (or comma-separated per-stack)
Effect on params: Linear(pos_dim=48, num_heads × pos_head_dim) in MHAW
```

**What it controls:** The dimension of the relative positional encoding projection per head. Small by design — positional information is a subtle signal.

**Total positional params per stack:** `pos_dim × H × pos_head_dim = 48 × 4 × 4 = 768`

---

### 4.4 `--pos-dim`

```
Current: 48
Type:    int (single, shared across all stacks)
Effect on params: CompactRelPositionalEncoding (no learnable params — fixed sinusoid) + Linear(pos_dim, H×pos_head_dim) in each MHAW
```

**What it controls:** The dimensionality of the compact relative positional encoding vector fed to MHAW. The `CompactRelPositionalEncoding` buffer is computed from atan + sinusoidal functions — no gradient flows to it.

**Do not change** unless you understand the positional encoding math. Keep at 48.

---

## 5. Training Regularization Parameters

These have **zero effect on parameter count or model structure**. They only affect training dynamics.

---

### 5.1 `--dropout` (implicit via `ScheduledFloat`)

Not a CLI argument — hardcoded in `Zipformer2.__init__()` as:
```python
dropout = ScheduledFloat((0.0, 0.3), (20000.0, 0.1))
```
Starts at 0.3, anneals to 0.1 over 20,000 training steps. Applied inside `FeedforwardModule` and `Zipformer2Encoder` layer drop.

---

### 5.2 `--enable-spec-aug`

```
Location: asr_datamodule.py
Default: True (via train.sh)
```
SpecAugment: randomly masks time steps and frequency bands in the fbank input. Standard augmentation for ASR. For very small datasets (<10h), consider reducing `time_mask_param` and `freq_mask_param` in `asr_datamodule.py`.

---

### 5.3 `--seed`

```
Current: 42
```
Controls torch, numpy, and Python random seeds for reproducibility. Change to run different random trials.

---

### 5.4 `--warmup-batches` (implicit)

Hardcoded at `4000.0` batches in `Zipformer2.__init__`. Controls the warmup window during which each encoder stack has elevated bypass (i.e., acts closer to identity). Larger values = more conservative warmup. The actual schedule per stack:
```
Stack i warms up from: 4000 × (i+1)/7  to  4000 × (i+2)/7
Stack 1: batches 571 → 1142
Stack 4: batches 2285 → 2857
Stack 6: batches 4000 → ... (end of warmup period)
```

---

## 6. Loss Function Parameters

---

### 6.1 `--simple-loss-scale`

```
Current: 0.5
Range:   [0.0, 1.0]
```
The RNN-T loss has two components:
1. **Simple loss** — computed with a simple additive joiner (fast, used to determine pruning ranges)
2. **Pruned loss** — the real RNN-T loss with k2 lattice pruning

The final loss = `pruned_loss + simple_loss_scale × simple_loss`. Setting to 0 disables the simple loss (not recommended — it's needed for stable pruning range estimation in early training).

---

### 6.2 `--prune-range`

```
Current: 5
```
How many symbols around the diagonal to retain when computing the pruned RNN-T loss. Larger range = more accurate loss but more memory and compute. Range 5 is standard.

---

### 6.3 `--lm-scale`

```
Current: 0.25
```
Label smoothing scale applied to the decoder (prediction network) output distribution during RNN-T loss. Higher values give the decoder more influence on the loss gradient.

---

### 6.4 `--am-scale`

```
Current: 0.0
```
Scale for acoustic model (encoder) output smoothing. Usually kept at 0.

---

### 6.5 `--ctc-loss-scale`

```
Current: 0.2
```
Only active when `--use-ctc True`. Adds `ctc_loss_scale × CTC_loss` to the total loss. CTC provides a useful auxiliary objective during early training.

---

### 6.6 `--attention-decoder-loss-scale`

```
Current: 0.8
```
Only active when `--use-attention-decoder True`. Scale for the cross-entropy loss from the attention decoder head.

---

## 7. Decoder & Joiner Parameters

---

### 7.1 `--decoder-dim`

```
Current: 512
Type:    int
Effect on params: Embedding(vocab_size × decoder_dim) + Conv1d
```

**What it controls:** The embedding dimension of the stateless decoder (prediction network). This affects:
- `Embedding(2000, 512)` = 1,024,000 params
- `Conv1d(512, 512, kernel=2, groups=128)` = 131,072 params

**Coupling:** `decoder_dim` must equal `joiner_dim` for the `decoder_proj` inside the joiner to work without a mismatch. Current: both are 512.

**To reduce:** You can try `decoder_dim=256` to save ~512K params, but you must also change the `decoder_proj` in the joiner. The recommended change is: if `decoder_dim ≠ joiner_dim`, the joiner's `decoder_proj = Linear(decoder_dim, joiner_dim)` handles it automatically. So they can be different.

---

### 7.2 `--joiner-dim`

```
Current: 512
Type:    int
Effect on params: 
  encoder_proj: Linear(max(encoder_dim)=256, 512) = 131,584
  decoder_proj: Linear(512, 512)                  = 262,656
  output_linear: Linear(512, 2000)                = 1,026,000
  Total joiner: ~1,420K params
```

**What it controls:** The hidden dimension of the joiner network. The joiner adds encoder and decoder projections then passes through `tanh` and `output_linear`.

**Coupling with encoder:** `encoder_proj` input dim = `max(encoder_dim)`. Currently `max=256`. If you increase any stack's `encoder_dim` above 256, the joiner input automatically grows.

**To reduce:** Setting `joiner_dim=256` saves ~500K params and reduces `output_linear` from 1M to 512K (halving vocabulary projection cost). This is a good tradeoff for a small model.

---

### 7.3 `--context-size`

```
Current: 2
Options: 1 or 2
```
`1` = bigram decoder (embedding only, no conv). `2` = trigram decoder (embedding + Conv1d over 2 previous tokens). Almost always use 2 — the parameter cost is small and it significantly helps recognition of common word sequences.

---

## 8. Optional Heads

---

### 8.1 `--use-transducer` / `--use-ctc`

```
use-transducer: True  (default, main head)
use-ctc:        False (auxiliary head)
```

You can enable both simultaneously for multi-task training. CTC provides a stronger gradient signal early in training (no blanks issue) while transducer gives better final accuracy. If both are enabled, the total loss is:
```
total_loss = pruned_rnnt_loss + simple_loss_scale×simple_rnnt + ctc_loss_scale×ctc_loss
```

---

### 8.2 `--use-attention-decoder`

```
Current: False
```
Adds a 6-layer Transformer cross-attention decoder on top of encoder outputs. Parameters:
- `attention_decoder_dim=512`
- `attention_decoder_num_layers=6`
- `attention_decoder_attention_dim=512`
- `attention_decoder_num_heads=8`
- `attention_decoder_feedforward_dim=2048`

This adds ~30M params on its own — effectively doubling the model. **Do not enable for the 30M config** unless you have a very large dataset.

---

## 9. Streaming / Causal Parameters

---

### 9.1 `--causal`

```
Current: False
```
If `True`, `ConvolutionModule` uses `ChunkCausalDepthwiseConv1d` instead of symmetric padding. Enables streaming inference. Small WER degradation (~0.5% relative) vs. non-causal. Enable only if you need real-time streaming inference.

---

### 9.2 `--chunk-size`

```
Current: "16,32,64,-1"  (only relevant if causal=True)
```
Chunk sizes in frames at 50Hz. `-1` means full context (non-streaming). During training, a random chunk size is sampled per batch, which trains the model to work at all chunk sizes.
- `16` frames = 320ms chunks (very low latency, higher WER)
- `64` frames = 1280ms chunks (lower WER, higher latency)

---

### 9.3 `--left-context-frames`

```
Current: "64,128,256,-1"  (only relevant if causal=True)
```
How many previous frames the model can attend to within its chunk window. `-1` = unlimited left context. For streaming, set to `256` (5.12 seconds) for a good tradeoff.

---

## 10. Optimizer & Scheduler Parameters

---

### 10.1 `--base-lr`

```
Current: 0.045
```
The peak learning rate for `ScaledAdam`. **Do not use standard values like 1e-3** — `ScaledAdam` operates in a different scale space. The `0.045` default is well-tuned for this architecture. If you see loss divergence, try `0.03`; if training is slow, try `0.06`.

---

### 10.2 `--lr-batches` and `--lr-epochs`

```
lr-batches: 7500
lr-epochs:  3.5
```
`Eden` scheduler: LR decays as `1 / sqrt(max(lr_batches_so_far / lr_batches, epoch / lr_epochs))`. These control the "pivot point" after which LR starts decaying. Increasing them slows decay (useful for large datasets).

---

### 10.3 `--ref-duration`

```
Current: 600  (seconds of audio per reference batch)
```
Used to normalize `ScheduledFloat` step counts across different batch sizes/GPU counts. If you change `--max-duration` significantly, adjust this to match. Keeps dropout schedules consistent.

---

### 10.4 `--num-epochs`

```
Current: 30  (in train.py) — typically set to 300 in train.sh
```
Total training epochs. For 50h labeled data: 50–100 epochs is typically sufficient. For <10h: 200+ epochs.

---

### 10.5 `--max-duration`

```
Location: ASR/scripts/train.sh
Default: 1000 seconds per batch
```
Maximum total audio duration per training batch. Increase for more stable gradients (requires more GPU memory). For 24GB GPU, 1000–1200s is a safe range with the 30M model.

---

## 11. Constraint Rules

These are **hard constraints** — violating them will either crash or produce incorrect models.

| # | Rule | Consequence of violation |
|---|------|--------------------------|
| 1 | `encoder_unmasked_dim[i] <= encoder_dim[i]` for all i | `AssertionError` in `Zipformer2.__init__` |
| 2 | `cnn_module_kernel[i]` must be **odd** | `AssertionError` in `ConvolutionModule.__init__` |
| 3 | `encoder_dim[0]` must match `Conv2dSubsampling` output dim | Silent shape error or crash in first forward pass |
| 4 | `len(all stack tuples)` must all be equal | `AssertionError` in `_to_tuple()` |
| 5 | `num_heads[i] >= 4` (Zipformer minimum) | May degrade severely; no hard assert but behaviors undefined |
| 6 | `output_downsampling_factor` must be exactly 2 | `AssertionError` in forward pass (line 356 of zipformer.py) |
| 7 | `downsampling_factor` must be powers of 2 for `SimpleDownsample` | Incorrect temporal alignment |
| 8 | `joiner_dim` must be > 0 and match `output_linear` output | Shape mismatch in joiner forward |

---

## 12. Parameter Interaction Map

```
┌──────────────────────────────────────────────────────────────┐
│ encoder_dim[i] ──────────────────────────────────────────────┤
│   │                                                          │
│   ├──→ must be ≥ encoder_unmasked_dim[i]                    │
│   ├──→ encoder_dim[0] must match Conv2dSubsampling output   │
│   └──→ max(encoder_dim) ──→ joiner.encoder_proj input dim   │
│                                                              │
│ num_heads[i] ────────────────────────────────────────────────┤
│   └──→ MHAW projection: d × H × (qd + pd)                  │
│                                                              │
│ feedforward_dim[i] ──────────────────────────────────────────┤
│   ├──→ FF1 hidden = 3/4 × ff                                │
│   ├──→ FF2 hidden = ff                                       │
│   └──→ FF3 hidden = 5/4 × ff                                │
│                                                              │
│ decoder_dim ─────────────────────────────────────────────────┤
│   └──→ joiner.decoder_proj input dim                        │
│                                                              │
│ joiner_dim ──────────────────────────────────────────────────┤
│   ├──→ joiner.encoder_proj output dim                       │
│   ├──→ joiner.decoder_proj output dim                       │
│   └──→ joiner.output_linear input dim → vocab_size          │
└──────────────────────────────────────────────────────────────┘
```

---

## 13. Complete Current Defaults (30M config)

Copy-paste reference of the exact defaults currently set in `ASR/zipformer/train.py`:

### Architecture (in `add_model_arguments()`)

```python
--num-encoder-layers      "2,2,2,2,2,2"
--downsampling-factor      "1,2,4,8,4,2"
--feedforward-dim          "512,768,768,768,768,768"
--num-heads                "4,4,4,8,4,4"
--encoder-dim              "192,256,256,256,256,256"
--query-head-dim           "32"
--value-head-dim           "12"
--pos-head-dim             "4"
--pos-dim                  48
--encoder-unmasked-dim     "192,192,256,256,256,256"
--cnn-module-kernel        "31,31,15,15,15,31"
--decoder-dim              512
--joiner-dim               512
--attention-decoder-dim    512      # unused (use-attention-decoder=False)
--attention-decoder-num-layers 6    # unused
--attention-decoder-attention-dim 512  # unused
--attention-decoder-num-heads 8     # unused
--attention-decoder-feedforward-dim 2048  # unused
--causal                   False
--chunk-size               "16,32,64,-1"   # unused (causal=False)
--left-context-frames      "64,128,256,-1" # unused (causal=False)
--use-transducer           True
--use-ctc                  False
--use-attention-decoder    False
```

### Training (in `get_parser()`)

```python
--world-size           1       # set to 4 in train.sh
--num-epochs           30      # set to 300 in train.sh
--base-lr              0.045
--lr-batches           7500
--lr-epochs            3.5
--ref-duration         600
--context-size         2
--prune-range          5
--lm-scale             0.25
--am-scale             0.0
--simple-loss-scale    0.5
--ctc-loss-scale       0.2
--attention-decoder-loss-scale  0.8
--seed                 42
--save-every-n         4000
--keep-last-k          30
--average-period       200
--use-fp16             False
--bpe-model            "data/lang_bpe_500/bpe.model"
--exp-dir              "zipformer/exp"
```

---

## 14. Size Variants Comparison

| Variant | encoder_dim | layers | ff_dim | Total Params | Notes |
|---------|-------------|--------|--------|-------------|-------|
| **Zipformer-Small (current)** | **192,256,256,256,256,256** | **2,2,2,2,2,2** | **512,768,768,768,768,768** | **~30M** | **This config** |
| Zipformer-Tiny | 192,192,192,192,192,192 | 2,2,2,2,2,2 | 512,512,512,512,512,512 | ~12M | Very fast, moderate WER |
| Zipformer-Original (was in repo) | 192,256,384,512,384,256 | 2,2,3,4,3,2 | 512,768,1024,1536,1024,768 | ~66M | Before this change |
| Zipformer-Large (icefall) | 192,256,512,512,512,256 | 2,4,4,8,4,4 | 768,1024,1536,2048,1536,1024 | ~80M+ | Full-data training |
| Zipformer-Mini | 192,256,256,256,256,256 | 2,2,2,2,2,2 | 512,512,512,512,512,512 | ~20M | Further reduced FF |

---

## 15. How to Safely Experiment

### Step-by-step process for any architecture change:

**1. Edit defaults in `train.py → add_model_arguments()`**
   ```python
   # Only these 4 lines (for size changes):
   --num-encoder-layers
   --encoder-dim
   --feedforward-dim
   --encoder-unmasked-dim
   ```

**2. Verify constraints before training:**
   ```bash
   cd ASR
   python -c "
   from zipformer.train import get_params, get_model, add_model_arguments
   import argparse
   p = argparse.ArgumentParser()
   add_model_arguments(p)
   params = p.parse_args([])
   from icefall.utils import AttributeDict
   params = AttributeDict(vars(params))
   params.feature_dim = 80
   params.vocab_size = 2000
   model = get_model(params)
   total = sum(p.numel() for p in model.parameters())
   print(f'Total parameters: {total/1e6:.1f}M')
   "
   ```

**3. Quick sanity forward pass:**
   ```bash
   python -c "
   import torch
   from zipformer.train import get_params, get_model, add_model_arguments
   import argparse
   p = argparse.ArgumentParser()
   add_model_arguments(p)
   params = p.parse_args([])
   from icefall.utils import AttributeDict
   params = AttributeDict(vars(params))
   params.feature_dim = 80
   params.vocab_size = 2000
   model = get_model(params)
   model.eval()
   # Fake a batch: 2 utterances, 200 frames each, 80 fbank dims
   x = torch.randn(2, 200, 80)
   x_lens = torch.tensor([200, 180])
   enc_out, enc_lens = model.encoder(x.permute(1,0,2), x_lens)
   print('Encoder output shape:', enc_out.shape)  # expect (T/4, 2, 256)
   print('SUCCESS')
   "
   ```

**4. Check memory on target GPU:**
   ```bash
   # Rule of thumb: 30M model at max_duration=1000 needs ~8GB GPU RAM
   # For max_duration=2000: ~16GB
   # Monitor with: nvidia-smi dmon -s mu
   ```

**5. Watch for these warning signs in early training (first 500 batches):**
   - `loss > 100`: check `--base-lr` (try halving it)
   - `loss = inf/nan`: reduce `--max-duration` or enable `--inf-check True`
   - `grad_norm > 100`: normal in first 100 batches, should drop after warmup

**6. Checkpointing during experiments:**
   Always use a different `--exp-dir` for each architecture variant:
   ```bash
   --exp-dir zipformer/exp_30m_small
   --exp-dir zipformer/exp_20m_mini
   ```
   This prevents checkpoint conflicts.

---

> **Summary:** For a ~30M model, the current defaults are already set correctly. The only reason to touch `train.py` again is if you want to go smaller (reduce dims/layers) or larger (increase dims/layers). All other parameters in this document are training hyperparameters that can be tuned independently without affecting model architecture.
