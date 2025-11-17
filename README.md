<div align="center">
  <h1>
    <bold> Model-Stitching Challenge </bold>
  </h1>
  <p><strong>Sapienza University of Rome</strong></p>
  <p><em>Advanced Machine Learning </em></p>
  <p>
    <img src="https://img.shields.io/badge/Python-3.9+-blue.svg" alt="Python"/>
    <img src="https://img.shields.io/badge/License-Academic-green.svg" alt="License"/>
  </p>
</div>

---

## Overview

This repository contains our solution for the kaggle competition for *Advanced Machine Learning* course for the master degree in [Data Science](https://corsidilaurea.uniroma1.it/it/course/33519). 

---

## Task

Our challenge was to solve an image–text retrieval task, where the goal is to generate caption embeddings that maximize the Mean Reciprocal Rank (MRR) when matched against ground-truth image embeddings, 
while also keeping the model as lightweight and efficient as possible.

## Dataset Structure

For this challenge we work with two specific [datasets](https://github.com/Flavio-Mangione/AML-Challenge-Model-Stitching/tree/main/Data): 

- `test_clean.npz` 
- `train.npz`.

The train file consists of 125k captions associated with 25k unique images, while the test_clean file contains 1500 captions used for the inference task and for scoring in the Kaggle competition.

---

## Evaluation 

Model performance is measured using `Mean Reciprocal Rank` (MRR). For each test caption, its predicted embedding is compared against all gallery image embeddings, in batches of 100, and the rank of the correct image is used to compute the reciprocal rank.

---

## Repository Structure

```
├── Data/                           # Dataset folder
│   ├── test_clean.npz      
│   └── train.npz
├── Utils and Functions/               
│   ├── metrics.py   # metrics definitions                     
│   └── eval_2.py    # evaluation function for the validation
├── Challenge_Notebook.ipynb         # Notebook for the submission
├── README.md
```

## Model 

The model used is a translator that maps 1024-dimensional text embeddings into the 1536-dimensional [DINOv2](https://huggingface.co/facebook/dinov2-giant) image-embedding space. It uses a two-block encoder with LayerNorm, GELU, and dropout for regularization, followed by a decoder that reconstructs the target embedding dimension. A learnable temperature parameter (logit_scale) is included for contrastive alignment. The full architecture consists of approximately the number of trainable parameters reported below

