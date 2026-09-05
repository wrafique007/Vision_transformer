# Vision Transformer (ViT) — PyTorch Implementation

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)](https://pytorch.org/)
[![arXiv](https://img.shields.io/badge/arXiv-2010.11929-b31b1b.svg)](https://arxiv.org/abs/2010.11929)

A complete, from‑scratch PyTorch implementation of the **Vision Transformer (ViT)** as described in the paper
**[An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/pdf/2010.11929)**
by Dosovitskiy et al. (Google Research, 2020).

This repository provides a clean, modular, and configurable ViT model, along with a ready‑to‑use training pipeline on the **Food‑101** dataset. It is designed for educational understanding, quick experimentation, and as a foundation for custom vision‑transformer projects.

---

## Table of Contents

1. [Key Features](#key-features)
2. [Architecture Overview](#architecture-overview)
   - [Patch Embedding](#patch-embedding)
   - [Class Token & Positional Embedding](#class-token--positional-embedding)
   - [Transformer Encoder](#transformer-encoder)
   - [Multi‑Head Self‑Attention](#multi-head-self-attention)
   - [MLP Block](#mlp-block)
   - [Classification Head](#classification-head)
3. [Configuration & Hyperparameters](#configuration--hyperparameters)
4. [Dataset](#dataset)
5. [Installation](#installation)
6. [Project Structure](#project-structure)

---

## Key Features

- **Full ViT implementation** — All components (patch embedding, self‑attention, MLP, class token) built from scratch in PyTorch.
- **Modular design** — Each block (`PatchEmbedding`, `MultiHeadAttention`, `MLPBlock`, `TransformerBlock`, `ViT`) is isolated for easy modification and reuse.
- **Configurable architecture** — Adjust embedding dimensions, number of layers, heads, patch size, MLP ratio, and dropout via a central dictionary.
- **Pre‑configured for Food‑101** — Training script loads and preprocesses the Food‑101 dataset using Hugging Face `datasets`.
- **Training pipeline** — Includes cross‑entropy with label smoothing, AdamW optimizer, checkpoint saving, and progress bars.
- **Research‑grade details** — Pre‑layer normalization, residual connections, GELU activations, and learnable positional embeddings match the original paper.

---

## Architecture Overview

The model follows the original ViT design, summarised below.

### Patch Embedding
Images are split into non‑overlapping patches of size `patch_size × patch_size` (e.g., 16×16). A convolutional layer with kernel and stride equal to `patch_size` projects each patch to an embedding vector of dimension `embedding_dim` (768 in the base model). This produces a sequence of `N = (H/patch_size) × (W/patch_size)` patch embeddings.

### Class Token & Positional Embedding
A learnable `[class]` token is prepended to the patch sequence; its final representation serves as the image‑level feature for classification. Learnable positional embeddings are added to the entire sequence to retain spatial information, followed by a dropout layer.

### Transformer Encoder
The encoder comprises `num_of_layers` identical blocks. Each block applies **pre‑layer normalisation** (before both attention and MLP), a residual connection after each sub‑layer, and uses GELU activation in the MLP.

### Multi‑Head Self‑Attention
The implementation:
- Splits the input into `num_of_heads` heads.
- Computes scaled dot‑product attention: `Attention(Q,K,V) = softmax(QKᵀ / √d_k) V`.
- Concatenates heads and projects back to `embedding_dim`.
- Includes dropout for attention weights and the output projection.

### MLP Block
A two‑layer feed‑forward network with an expansion factor (typically 4× the embedding dimension). The first layer expands to `mlp_dim`, applies GELU, then a second layer projects back to `embedding_dim`. Dropout is applied after each linear layer.

### Classification Head
The final representation of the `[class]` token is extracted, passed through a LayerNorm, and fed into a single linear layer to produce logits for `num_of_classes`.

---

## Configuration & Hyperparameters

All settings are defined in the `get_config()` function inside `vision_transformer.py`:

| Parameter          | Default  | Description                                 |
|--------------------|----------|---------------------------------------------|
| `lr`               | 1e-9     | Learning rate (conservative for stability)  |
| `batch_size`       | 16       | Number of images per batch                  |
| `label_smoothing`  | 0.1      | Label smoothing factor for cross‑entropy   |
| `channels`         | 3        | Input image channels (RGB)                  |
| `patch_size`       | 16       | Patch size (e.g., 16×16)                    |
| `embedding_dim`    | 768      | Transformer hidden dimension (D)            |
| `mlp_dim`          | 3072     | MLP hidden dimension (4×D)                  |
| `num_of_heads`     | 12       | Number of self‑attention heads              |
| `num_of_layers`    | 12       | Number of transformer encoder blocks        |
| `num_of_classes`   | 101      | Output classes (Food‑101)                   |
| `epochs`           | 3        | Number of training epochs (change as needed)|
| `file_name`        | `"vision_transformer_model_{0}.pt"` | Checkpoint filename pattern |

> **Tip**: To train a smaller variant (e.g., ViT‑Tiny), reduce `embedding_dim` (e.g., to 192), `mlp_dim` (768), `num_of_heads` (3), and `num_of_layers` (6). This reduces memory usage and speeds up training.

---

## Dataset

The training script uses the **[Food‑101](https://huggingface.co/datasets/ethz/food101)** dataset, which contains 101 food categories (e.g., pizza, sushi, steak) with 750 training and 250 test images per class.

**Preprocessing**:
- Resize images to 256 pixels on the shorter edge.
- Center crop to 224×224 (standard for ViT).
- Convert to a PyTorch tensor (values are automatically scaled to [0,1] by `ToTensor`).

The dataset is loaded via Hugging Face’s `datasets` library with the `"food101"` configuration. The training split is used for training.

---

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/wrafique007/Vision_transformer.git
   cd Vision_transformer

## Project Structure

```
Vision_transformer
├── init.py # Makes the directory a package
├── vision_transformer.py # Core model: PatchEmbedding, MHSA, MLPBlock,
│ # TransformerBlock, ViT, and get_config()
├── train.py # Training script: loads Food-101, runs training
│ # loop, and saves checkpoints
├── requirements.txt # Python dependencies
└── README.md # This document
```

