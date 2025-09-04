# Generative-GNN-for-ATR

This repository contains my end-to-end workflow for exploring **graph reliability** using
a **conditional generative graph neural network (VAE)**. It covers the full pipeline:
from generating graphs and computing reliability metrics, to training a model that can
generate **new graphs of arbitrary size** while controlling for reliability properties.

---

## Objectives

- **Enumerate graphs** and study their **all-terminal reliability (ATR)**.
- **Identify isomorphic graphs** and deduplicate to create a canonical dataset.
- **Compute ATR across all probabilities** \(p ∈ (0.01, 0.99)\) for each graph.
- **Summarize reliability metrics**: ATR min/mean/max, λ-optimal, t-optimal.
- **Train a conditional VAE (GNN)** to learn structure–reliability relationships.
- **Extend to size-agnostic generation**: generate graphs for *any* number of nodes `n`.
- **Enable ATR evaluation** on generated graphs (exact for small, Monte Carlo for large).
- Provide an open, reproducible pipeline for graph reliability research.

---

## Repository Structure

Generative-GNN-for-Graph-Reliability
- notebooks - generative_gnn_1.ipynb # Main notebook: data prep, model training, generation
- src - # ATR computation utilities, notebooks used for visualizations
- data - #generated datasets used for training
- figures - # Generated plots & visualizations

---

## Workflow

### 1. **Graph Generation**
- Generated **all connected graphs on 8 nodes** with edge counts ranging from 7 → 28.
- Eliminated duplicates by checking **graph isomorphism** (NetworkX canonicalization).
- Final dataset: ~11,000 unique graphs.

### 2. **ATR Calculation**
- For each graph, calculated **All-Terminal Reliability (ATR)**:
  - \(ATR(p)\) = probability the graph remains connected if each edge survives independently with probability \(p\).
  - Computed for **p = 0.01, 0.02, …, 0.99**.
- Stored ATR values across all p in CSV files.
- Summarized per-graph reliability with:
  - **atr_min / atr_mean / atr_max**
  - **λ-optimal** and **t-optimal** metrics.

### 3. **Dataset Assembly**
- Combined graph structure, ATR surfaces, and reliability summaries.
- Prepared a dataset suitable for PyTorch Geometric:
  - Each graph = `Data` object with nodes, edges, and reliability label.

### 4. **Conditional VAE (GNN)**
- Encoder: GCN layers → global mean pooling → latent distribution (μ, logσ²).
- Decoder: Reconstructs edge probabilities for 8-node graphs.
- Conditioning: Graph label (e.g., high vs low reliability).
- Loss = BCE (edge reconstruction) + KL divergence (+ optional aux classification).

### 5. **Size-Agnostic Decoder**
- Extended the model with a **new decoder**:
  - Takes latent vector + condition.
  - Samples **n node embeddings** and scores all pairs with a shared MLP.
  - Generates graphs of arbitrary node counts (not just 8).
- Added connectivity enforcement (MST backbone) and density control.

### 6. **Graph Generation**
- Generate graphs under desired conditions:
  - **Reliability label** (e.g., “high ATR” vs “low ATR”).
  - **Target node count n** (e.g., 6, 12, …).
  - **Target density** (approximate edge sparsity).
- Visualize with NetworkX spring layouts.

### 7. **ATR Evaluation of Generated Graphs**
- Implemented exact ATR (subset enumeration) for small graphs (≤22 edges).
- Implemented Monte Carlo ATR estimation for larger graphs.
- Compute ATR curves across p ∈ [0.1, 0.95].
- Summarize with atr_min / atr_mean / atr_max for generated graphs.

---

## Future Work
- This model should be used as a starting point for analyzing and predicting all-terminal-reliability.
- Training should be improved with multi-size augmentation
- Model should be able to easily scale to larger node counts efficiently and accurately
