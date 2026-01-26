# Temporal-Aware Multi-Task Transformer for Finance

A PyTorch implementation of a Transformer-based model designed for financial time-series and textual data. This project features temporal-aware embeddings (Time2Vec) and a multi-task learning architecture to predict market returns, volatility, events, and regimes simultaneously.

## 🚀 Key Features

*   **Temporal Awareness**: Uses `Time2Vec` for continuous time encoding and `CalendarEmbeddings` for discrete time features (Hour, Day, Weekday, Month).
*   **Multi-Modal Fusion**: Incorporates a `Variable Selection Network` (inspired by TFT) to dynamically weigh numerical (price/volume) and textual (news embeddings) inputs.
*   **Multi-Task Learning**: Shared Transformer backbone with 4 specialized heads:
    *   **Return**: Price movement prediction (Regression).
    *   **Volatility**: Market risk estimation (Regression, Softplus activation).
    *   **Event Impact**: News significance score (Sigmoid activation).
    *   **Regime**: Market state classification (Bull/Bear/Shock).
*   **Synthetic Data Pipeline**: Built-in generator to create realistic financial sequences with injected "shocks" and correlated text embeddings.

## 🛠 Setup Instructions

### 1. Conda Environment (Recommended)

```bash
# Create a new conda environment
conda create -n tft-finance python=3.10 -y
conda activate tft-finance

# Install dependencies
pip install -r finance_transformer/requirements.txt
```

### 2. Standard Pip install

```bash
pip install torch pandas numpy scikit-learn
```

## 📈 How to Run

### Generate Data & Train
The `main.py` script handles data generation, training, and evaluation.

```bash
python finance_transformer/main.py
```

Upon running, the script will:
1.  Generate 2,000 synthetic samples.
2.  Save raw sequence data to `finance_transformer/data/synthetic_data.csv`.
3.  Execute a 10-epoch training loop.
4.  Print sample predictions vs. ground truth for all 4 tasks.

## 📂 Project Structure

```
.
├── README.md
├── finance_transformer/
│   ├── main.py              # Entry point: Training loop and evaluation
│   ├── requirements.txt     # Python dependencies
│   ├── data/
│   │   ├── generator.py     # Synthetic data generation logic
│   │   └── synthetic_data.csv # Generated data (after first run)
│   ├── model/
│   │   ├── transformer.py   # Main model assembly
│   │   ├── embeddings.py    # Time2Vec and Calendar embeddings
│   │   ├── layers.py        # GRN and Variable Selection logic
│   │   └── heads.py         # Multi-task output heads
│   └── docs/
│       ├── architecture.md  # Detailed explanation of temporal/multi-task concepts
│       └── plan.md          # Original implementation plan
```

## 📖 Documentation

For a deeper dive into the methodology:
- [Architecture Concepts](finance_transformer/docs/architecture.md): Understanding Temporal Awareness and GRNs.
- [Implementation Plan](finance_transformer/docs/plan.md): The technical roadmap used to build this project.
