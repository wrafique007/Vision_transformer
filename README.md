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
   - [Multi‑Head Self‑Attention (MHSA)](#multihead-self-attention-mhsa)
   - [MLP Block](#mlp-block)
   - [Classification Head](#classification-head)
3. [Configuration & Hyperparameters](#configuration--hyperparameters)
4. [Dataset](#dataset)
5. [Installation](#installation)
6. [Training](#training)
   - [Command‑Line Usage](#commandline-usage)
   - [Training Progress & Checkpoints](#training-progress--checkpoints)
7. [Inference](#inference)
   - [Loading a Saved Model](#loading-a-saved-model)
   - [Single‑Image Prediction](#singleimage-prediction)
8. [Project Structure](#project-structure)
9. [Future Improvements](#future-improvements)
10. [References](#references)
11. [License](#license)

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
