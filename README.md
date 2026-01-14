# MGCN-CL: A Parallel Multi-Scale Graph Convolutional Network with Contrastive Self-Attention for Automated Lithology Identification in Complex Reservoirs

This repository contains the PyTorch implementation of the **MGCN-CL** framework, as described in the methodology section of the associated research paper. MGCN-CL is a novel deep learning architecture designed for high-precision lithology identification from well-logging signals.



##  Framework Architecture

The framework consists of four integrated modules:

1.  **Multi-scale Information Extraction**:
    *   **Signal Processing**: Adapts Continuous Wavelet Transform (CWT) to generate high-fidelity Time-Frequency maps.
    *   **Dynamic Graph Construction**: Partitions the spectral map into $P=8$ frequency blocks (nodes) and establishes connectivity based on Cosine Distance (threshold $\kappa=0.6$) to model spectral topology.

2.  **Parallel Multi-scale GCN**:
    *   **Scale-1 Branch**: Single GCN layer for local, high-frequency textures.
    *   **Scale-2 Branch**: Cascaded GCN layers (2-hop) for global semantic dependencies.
    *   **Raw Feature Branch**: Linear projection acting as a "Petrophysical Anchor" to preserve original signal properties.

3.  **Contrastive Self-Attention Pooling**:
    *   **Dual-Pathway**: Computes Constructive ($S^+$) and Inhibitory ($S^-$) attention scores.
    *   **Contrastive Learning**: Maximizes the divergence between beneficial and detrimental node distributions using Jensen-Shannon Divergence.
    *   **Node Selection**: Retains the most significant nodes based on $S^+$ scores.

4.  **Global Readout & Classification**:
    *   **Fusion**: Concatenates Global Mean and Global Max pooling outputs.
    *   **Classifier**: 4-layer MLP with Residual Blocks for stable gradient flow.

## 📂 Project Structure

```bash
MGCN_CL_Paper_Reproduction/
├── data/
│   ├── builder.py        
│   └── dataset.py        
├── models/
│   └── mgcn.py          # MGCN_CL Network Architecture & ResidualMLP
├── utils/
│   └── loss.py          # Custom Loss Function (Cross-Entropy + Contrastive)
├── train.py             
├── requirements.txt      
└── README.md           
```

## 🚀 Getting Started

### Prerequisites

*   Python 3.8+
*   PyTorch >= 1.9.0
*   PyTorch Geometric >= 2.0.0
*   SciPy
*   NumPy
*   Pandas

### Installation

1.  Clone the repository:
    ```bash
    git clone  https://github.com/logginglearn/MGCN-CL
    cd MGCN-CL
    ```

2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

### Usage

**Training**:
To train the model on the provided test dataset (or your own data), run:

```bash
python train.py
```
 
 

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.
