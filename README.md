# BreastMR-FM v1.0: A Foundational DCE-MRI Model for Breast Tumor Characterization and Outcome Prediction

This repository provides the implementation for:

- Training a foundation model  
- Extracting feature embeddings using a trained or pretrained checkpoint  

---

# 📦 Dataset

## Download Dataset (MAMAMIA)

The dataset can be downloaded from the Synapse platform:

https://www.synapse.org/Synapse:syn60868042/wiki/628716

---

# 🚀 1. Train the Foundation Model

Once the dataset is prepared, run:

```bash
CUDA_VISIBLE_DEVICES=0 python3 foundation_model_training.py
```

---

# 🔍 2. Generate Feature Embeddings

You can either:

- Use your trained checkpoint  
or  
- Download pretrained weights  

---

## Option A: Download Pretrained Weights

Download pretrained weights from:

https://drive.google.com/file/d/1GIQHGRnC1GVqkKdVYfyxcwm_sPZ11ahv/view?usp=share_link

Place the downloaded `.ckpt` file in your checkpoint directory.

---

## Run Feature Extraction

```bash
python3 generate_features.py \
    --ckpt_path /path/to/checkpoints/epoch=99-step=46200.ckpt \
    --data_dir /path/to/test/duke \
    --output_dir /path/to/features/duke
```
