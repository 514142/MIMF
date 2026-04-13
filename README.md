📌 Overview
This paper proposes a lightweight & resource-efficient multi-focus image fusion (MFIF) framework for IoT edge vision systems.
Lightweight feature extraction with depthwise separable convolutions
Dual attention (CSE / SCSE) for focus-aware feature recalibration
Parameter-free fusion decision using spatial frequency & gradient cues (SF-GE)
Automated Bayesian hyperparameter optimization (Optuna)
Superior efficiency-performance trade-off for resource-constrained devices
🧩 Framework Architecture
Lightweight Feature Extraction
Attention-Guided Feature Recalibration
Edge-Aware Fusion Decision (SF + GE)
Decision Map Refinement (Morphology + Guided Filtering)
Bayesian Hyperparameter Optimization
🧪 Experimental Settings
GPU: NVIDIA GeForce RTX 4090 (32GB RAM)
Python 3.8, PyTorch 1.10.1, CUDA 11.3
Datasets: Lytro (20 pairs), MFFW (13 pairs)
Evaluation metrics: AG, EN, SSIM, CC, PSNR, RMSE
📂 Datasets
Lytro Dataset
MFFW Dataset
RealMFF Dataset (710 pairs, large-scale real-world multi-focus dataset)
📊 Main Results
Achieves state-of-the-art fusion quality with significantly reduced model size
Strong generalization on complex and texture-rich scenes
Suitable for real-time IoT / edge deployment
🚀 Quick Start
bash
运行
# Clone repo
git clone https://github.com/your-username/your-repo.git
cd your-repo

# Install dependencies
pip install -r requirements.txt

# Run test
python test.py --dataset Lytro
python test.py --dataset MFFW

# Bayesian hyperparameter search
python bayesian_optimization.py
