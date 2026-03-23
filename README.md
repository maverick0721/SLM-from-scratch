# 🧠 Training a GPT-Style Language Model from Scratch  
### End-to-End Transformer Implementation, Pretraining, and Autoregressive Generation

> 🚀 Implemented a GPT-style decoder-only Transformer entirely from scratch in PyTorch and trained it on TinyStories.  
> 🧠 Built custom attention, FlashAttention fallback, LR scheduling, mixed precision training, and gradient accumulation.  
> 📈 Demonstrated full-stack LLM engineering — from dataset tokenization to optimized autoregressive generation.

---

# 1. Overview

This project implements and trains a **GPT-style decoder-only Transformer** from first principles using PyTorch.

Unlike high-level frameworks, this implementation builds:

- Custom LayerNorm
- Causal Self-Attention
- MLP blocks
- Transformer stack
- Autoregressive sampling
- Mixed precision training
- Cosine LR scheduling with warmup
- Gradient accumulation
- Manual dataset tokenization and memory-mapped batching

The model is trained on:

**Dataset:** `roneneldan/TinyStories`

---

# ✅ Latest Validated Smoke Run (March 23, 2026)

This repository now includes a validated smoke-mode run with the updated notebook pipeline and exported artifacts.

## Smoke Run Settings

| Parameter | Value |
|------------|--------|
| Dataset source | TinyStories (`train[:500]`, `validation[:200]`) |
| Layers | 2 |
| Heads | 2 |
| Embedding Size | 96 |
| Dropout | 0.2 |
| Max Iterations | 2500 |
| Eval Interval | 100 |
| Learning Rate | 5e-5 |
| Min LR | 1e-5 |
| Batch Size | 16 |
| Gradient Accumulation | 8 |

## Latest Metrics

| Metric | Value |
|------------|--------|
| Best validation loss | 8.8081 |
| Last validation loss | 8.8081 |
| Last training loss | 8.7717 |
| Evaluation points | 24 |
| Overfitting flag | false |

## Run Artifacts

- Run summary JSON: `artifacts/run_summary.json`
- Best checkpoint: `best_model_params.pt`

## Training Loop Stability Fix

The scheduler step order warning has been fixed by stepping the LR scheduler only when the optimizer steps during gradient accumulation.

---

# 2. Architecture

## Model Configuration

| Parameter | Value |
|------------|--------|
| Layers | 6 |
| Heads | 6 |
| Embedding Size | 384 |
| Context Length | 128 |
| Vocabulary Size | 50257 |
| Dropout | 0.1 |
| Weight Tying | Enabled |

This corresponds to a ~30M parameter GPT-style model.

---

## Transformer Block Structure

Each block contains:

```
LayerNorm
→ Causal Self-Attention
→ Residual
→ LayerNorm
→ MLP (GELU)
→ Residual
```

### Attention Implementation

- Uses `scaled_dot_product_attention` if available (Flash attention backend)
- Falls back to manual masked attention if unavailable
- Strictly causal masking enforced
- Multi-head projection via linear layers

---

# 3. Training Pipeline

## Dataset Processing

Steps:

1. Load TinyStories
2. Tokenize using GPT-2 tokenizer (`tiktoken`)
3. Convert tokens into contiguous `.bin` memory-mapped files
4. Implement custom `get_batch()` loader
5. Efficient CPU → GPU pin memory transfers

This avoids HuggingFace Trainer abstractions and builds raw pretraining pipeline manually.

---

## Optimization Strategy

### Optimizer

```
AdamW
β1 = 0.9
β2 = 0.95
Weight decay = 0.1
```

### Learning Rate Schedule

```
Linear Warmup → Cosine Decay
```

- Warmup steps: 1000
- Total iterations: 20000
- Gradient accumulation: 32
- Mixed precision training (fp16 / bf16)

### Additional Stability Techniques

- Gradient clipping (0.5)
- Automatic mixed precision
- Loss scaling via GradScaler
- Best model checkpointing

---

# 4. Training Configuration

| Hyperparameter | Value |
|---------------|--------|
| Max Iterations | 20000 |
| Batch Size | 32 |
| Gradient Accumulation | 32 |
| Effective Batch Size | 1024 |
| Context Length | 128 |
| LR | 1e-4 |
| Min LR | 5e-4 |
| Eval Interval | 500 steps |

---

# 5. Generation

After training:

```python
sentence = "A little girl went to the woods"
context = torch.tensor(enc.encode_ordinary(sentence)).unsqueeze(0)
y = model.generate(context, 300)
```

The model performs:

- Autoregressive token-by-token sampling
- Optional top-k filtering
- Temperature scaling
- Context truncation to block size

---

# 6. Core Engineering Highlights

## 1️⃣ Full Transformer From Scratch

- No HuggingFace model classes
- Manual implementation of:
  - Attention
  - Residual pathways
  - Weight tying
  - Causal masking

---

## 2️⃣ Efficient Dataset Engineering

- Memory-mapped binary storage
- Parallel tokenization (`num_proc=8`)
- Sharded writes
- Zero-copy batch slicing

This mimics large-scale pretraining pipelines used in production.

---

## 3️⃣ Mixed Precision + Accumulation

Implemented:

- `torch.amp.autocast`
- `GradScaler`
- Gradient accumulation
- Custom scheduler chaining

This mirrors techniques used in large-scale LLM training.

---

# 7. Training Curve

Training and validation losses are logged and visualized to verify:

- Stable convergence
- No divergence
- No overfitting spikes

---

# 8. System Design Principles

This project demonstrates understanding of:

- Transformer internals
- Autoregressive modeling
- Optimization stability
- GPU efficiency
- Data pipeline engineering
- Memory management
- LR scheduling strategies
- Causal attention masking

---

# 9. Why This Matters

Training from scratch builds deep intuition about:

- Attention scaling behavior
- Tokenization effects
- Gradient flow stability
- Overfitting patterns
- Capacity vs data tradeoffs
- Learning rate scheduling impact
- Memory constraints

This level of understanding is critical for:

- Foundation model research
- Scaling laws experiments
- Optimization research
- Systems-level AI engineering

---

# 10. Future Extensions

- Rotary positional embeddings (RoPE)
- RMSNorm instead of LayerNorm
- SwiGLU instead of GELU
- Multi-query attention
- FlashAttention v2
- Fused optimizers
- Scaling to 350M+ parameters
- Curriculum learning
- Distributed training (DDP)

---

# 11. Summary

This project demonstrates full-stack language model engineering:

- Designed a decoder-only Transformer from first principles.
- Built a scalable tokenization and batching pipeline using memory-mapped binaries.
- Implemented modern optimization techniques including warmup + cosine decay, mixed precision training, and gradient accumulation.
- Trained and evaluated a GPT-style model end-to-end with autoregressive generation.

---

# 13. Technical Stack

- PyTorch
- tiktoken
- HuggingFace Datasets
- NumPy
- CUDA
- Mixed Precision (AMP)

---

# 14. Project Scope

This is not a wrapper project.

It is a ground-up implementation of a Transformer pretraining pipeline.

It demonstrates:

- Deep architecture-level understanding
- Optimization-level reasoning
- Systems-level awareness
- Research-level implementation maturity
