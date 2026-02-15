# Trust Your Neighbors: Multimodal Patient Retrieval for TBI Prognosis

[![IEEE JBHI](https://img.shields.io/badge/IEEE%20JBHI-2025-blue)](https://doi.org/10.1109/JBHI.2025.3622508)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

Official implementation of "Trust Your Neighbors: Multimodal Patient Retrieval for TBI Prognosis" published in IEEE Journal of Biomedical and Health Informatics.

![Figure 1: Multimodal Retrieval Framework](Figures/fig1.png)

## Overview

This repository contains the code for a novel multimodal AI approach to traumatic brain injury (TBI) prognosis that combines retrieval-augmented classification with example-based explainability. Our method leverages both clinical data and CT imaging to retrieve similar patient cases and improve prognostic predictions.

**Key Features:**
- Multimodal embedding generation from clinical time-series and CT imaging
- Retrieval-augmented classification (RAC) framework
- Graph Neural Network (GNN) based patient similarity modeling
- RPEC (Retrieval-augmented Prognostic Ensemble Classifier) architecture
- Example-based explainable AI for clinical decision support

## Repository Structure

```
Trust-Your-Neighbors/
├── important_data/          # Data preprocessing and organization
├── models/                  # Model architectures
│   └── RPEC.py             # RPEC model implementation
├── RAC/                     # Retrieval Augmented Classification
│   ├── gnn_training.py     # Graph Neural Network training
│   ├── model_training_RPEC.py  # RPEC training script
│   ├── rac_utils.py        # RAC utility functions
│   └── rac_utils_retrieval.py  # Retrieval utilities
├── MMDataset.py            # Multimodal dataset implementation
├── model_training.py       # Main multimodal embedding training
├── run_params.py           # Training parameters and configuration
├── final_utils.py          # General utility functions
└── utils.py                # Helper functions
```

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/Trust-Your-Neighbors.git
cd Trust-Your-Neighbors

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### 1. Multimodal Embedding Training

Train the multimodal embedding model that combines clinical time-series data and CT imaging:

```bash
python model_training.py
```

This script:
- References `utils.py`, `MMDataset.py`, and model architectures in `models/`
- Creates unified patient representations from heterogeneous data sources
- Saves trained embeddings for downstream tasks

### 2. Retrieval-Augmented Classification

#### Training RPEC Model

Train the Retrieval-augmented Prognostic Ensemble Classifier:

```bash
cd RAC
python model_training_RPEC.py
```

The RPEC model architecture is defined in `models/RPEC.py` and leverages retrieved similar patients to enhance classification performance.

#### Training Graph Neural Network

Train the GNN for patient similarity modeling:

```bash
cd RAC
python gnn_training.py
```

The GNN learns to model relationships between patients in the embedding space for improved retrieval.

### Configuration

Modify training parameters in `run_params.py`:
- Learning rates
- Batch sizes
- Model hyperparameters
- Data paths
- Training epochs

## Data

> **Note**: Due to compliance requirements with our hospital institution, we are not able to share the training dataset at this time. However, we plan to open source the dataset later this year!

The expected data format includes:
- Clinical time-series data (vital signs, lab values, etc.)
- CT imaging data
- Patient outcome labels

## Model Architecture

Our approach consists of three main components:

1. **Multimodal Encoder**: Learns joint representations from clinical and imaging data
2. **Retrieval Module**: Identifies similar patients from the training set using learned embeddings
3. **RPEC Classifier**: Combines retrieved patient information with query patient data for prognosis

## Citation

If you use this code or methodology in your research, please cite our paper:

```bibtex
@ARTICLE{11283634,
  author={Manjunath, Pranav and Lerner, Brian and Dunn, Timothy W.},
  journal={IEEE Journal of Biomedical and Health Informatics}, 
  title={Trust Your Neighbors: Multimodal Patient Retrieval for TBI Prognosis}, 
  year={2025},
  volume={29},
  number={12},
  pages={8775-8782},
  keywords={Hospitals;Soft sensors;Computed tomography;Predictive models;Radiology;Time measurement;Prognostics and health management;Brain injuries;Resilience;Artificial intelligence;Multimodal sensors;Multimodal AI;Example-based XAI;Traumatic Brain Injury},
  doi={10.1109/JBHI.2025.3622508}
}
```

## Paper

Read the full paper: [IEEE Journal of Biomedical and Health Informatics](https://doi.org/10.1109/JBHI.2025.3622508)


**Disclaimer**: This is research code. Clinical validation is required before any clinical deployment.
