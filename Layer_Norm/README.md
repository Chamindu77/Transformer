# Layer Normalization in Transformers — Theory & Notebook

This repository contains a theory-focused notebook and supporting material that explain the role of **Layer Normalization** inside Transformer architectures. It is intended for researchers, students, and engineers who want a compact but rigorous walkthrough of why layer norm matters, how it is applied in Transformer blocks, and an interactive Jupyter notebook that explores behavior and experiments.

---

## Contents

- `Layer_Norm_Transformers.ipynb` — interactive Jupyter notebook with visualizations, experiments, and code exploring layer norm variants in Transformer-style blocks.
- `README.md` — this file (theory summary, how to run the notebook, references).

---

## Short summary

Layer Normalization (LayerNorm) stabilizes the training of deep neural networks by normalizing activations **per example** across the features (channels) dimension. In Transformer models LayerNorm is typically applied either **before** (pre-norm) or **after** (post-norm) residual sublayers (self-attention and feed-forward). The choice affects gradient flow, training stability, and maximum achievable depth without specialized techniques.

---

## Key concepts

### 1. What LayerNorm does
Given input vector \(x \in \mathbb{R}^d\) for one example,

- mean: \(\mu = \frac{1}{d} \sum_{i=1}^{d} x_i\)
- variance: \(\sigma^2 = \frac{1}{d} \sum_{i=1}^{d} (x_i - \mu)^2\)
- normalized output: \(\hat{x}_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}}\)
- final output: \(y_i = \gamma \hat{x}_i + \beta\) where \(\gamma,\beta \in \mathbb{R}^d\) are learned scale and shift parameters.

LayerNorm is computed per-token (per-example) and across the feature dimension, unlike BatchNorm which computes stats across the batch.

### 2. Pre-norm vs Post-norm Transformers
- **Post-norm (original Transformer)**: apply attention / feed-forward, then add residual, then LayerNorm. This sometimes suffers from gradient instability for very deep Transformers.
- **Pre-norm**: apply LayerNorm first, then sublayer (attention or FFN), then residual add. Pre-norm typically improves training stability and allows deeper models to converge without learning rate warmup tricks.

### 3. Why it helps
- Reduces internal covariate shift across features for each token.
- Improves gradient flow, especially when used as pre-norm, because gradients pass through an additive identity path with normalized inputs.
- Decouples per-example feature scale from the rest of the network, making optimization easier.

---
