# TRACE: Transient Chaotic Evolution Network for Time Series Forecasting

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg?style=plastic)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-green.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12%2B-red.svg)](https://pytorch.org/)

> **TRACE** embeds continuous-time chaotic oscillators into a neural forecasting architecture, replacing learned recurrent layers with a parameter-free physical dynamical core. The transient evolution of the coupled chaotic system autonomously performs nonlinear temporal processing, achieving competitive accuracy with substantially fewer parameters.

## 📖 Overview

Traditional deep forecasting models rely on stacked black-box layers to learn temporal dependencies, inevitably facing a trade-off between parametric efficiency and representational capacity. TRACE takes a fundamentally different approach: instead of learning nonlinearity, it harnesses the native dynamics of chaotic oscillators as a computational substrate.

Key idea: each time step is embedded as the initial state of a chaotic oscillator, and a causal coupling topology governs inter-node interactions. The system evolves autonomously under a fixed ODE, naturally performing expansion and contraction that separate or merge temporal features—analogous to what deep networks learn, but without any trainable parameters in the core.

## 🔧 Key Features

- **Parameter-Free Chaotic Core**: Offloads nonlinear temporal processing to a fixed ODE (Lorenz, Rössler, etc.)
- **Causal Coupling Topology**: Enforces temporal causality via an exponential decay coupling matrix
- **Geometric Stretch-and-Fold**: Global phase-space contraction filters noise while local chaotic expansion enriches representations
- **Spectral Modal Filtering**: The coupling acts as a frequency-dependent filter in the graph spectral domain
- **Lightweight & Interpretable**: Only a shallow encoder and decoder are trained; information flow is traceable through phase-space trajectories

## 📁 Repository Structure

```
TRACE/
├── data_provider/          # Data loading and preprocessing (dataset classes, dataloaders)
├── datasets/               # Raw or processed benchmark datasets (e.g., ETTh1, ETTm1, Weather, Electricity)
├── experiments/            # Configuration files and experiment launchers (YAML/JSON configs, SLURM scripts)
├── layers/                 # Core modules: chaotic ODE layer, coupling topology, encoder/decoder blocks
├── model/                  # Main TRACE model definition and parameter initialization
├── scripts/                # Utility bash scripts for training, evaluation, and visualization
├── utils/                  # Helper functions (metrics, logger, visualization, ODE solvers)
├── run.py                  # Main entry point for training and testing
└── README.md               # This file
```

## 🚀 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/TRACE.git
   cd TRACE
   ```

2. **Create a virtual environment (optional but recommended)**
   ```bash
   conda create -n trace python=3.8
   conda activate trace
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
   Core requirements:
   Python >= 3.8, PyTorch >= 1.12, (Optional) torchdiffeq – for adaptive ODE solver comparison

## 📊 Quick Start

### Data Preparation
Place your time series datasets (CSV format) under `datasets/` from [Datasets](https://drive.google.com/file/d/1l51QsKvQPcqILT3DwfjCgx8Dsg2rpjot/view?usp=drive_link). The code expects standard forecasting benchmarks (e.g., ETTh1, ETTm1, Weather, ILI). You can also implement a custom data loader in `data_provider/`.

### Training a Model
All running scripts are in `scripts`.
```bash
bash scripts/TRACE/ETTh1.sh
```

## 🙏 Acknowledgements

We thank the following open-source contributions that made this work possible:

- **Dataset**: The benchmark time series datasets used in this project are provided by [this data source](https://drive.google.com/file/d/1l51QsKvQPcqILT3DwfjCgx8Dsg2rpjot/view?usp=drive_link). We are grateful for making this collection publicly available.

- **Baseline Implementations**: Our experimental comparisons leverage code from the following repositories:
  - [Time-Series-Library](https://github.com/thuml/Time-Series-Library.git) – a comprehensive collection of time series forecasting models.
  - [interPDN](https://github.com/leonardokong486/interPDN.git) – for the interPDN baseline.
  - [FACT](https://github.com/wanghq21/FACT.git) – for the FACT baseline.
  - [FITS](https://github.com/VEWOXIC/FITS.git) - for the FITS baseline.
  - [MixLinear](https://github.com/aitianma/MixLinear.git) - for the MixLinear baseline.
  - [SparseTSF](https://github.com/lss-1138/SparseTSF.git) - for the SparseTSF baseline.

We sincerely appreciate the efforts of all researchers who open-sourced their code and data, which greatly benefited the reproducibility and fairness of our evaluation.

## 📝 Citation

If you find TRACE useful for your research, please cite our paper:

```bibtex
@article{trace2026,
  title={Harnessing Chaos for Time Series Forecasting},
  author={},
  journal={},
  year={2026}
}
```



