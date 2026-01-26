# Walkthrough - Temporal Multi-Task Transformer for Finance

I have successfully built and trained a **Temporal-Aware Multi-Task Transformer** tailored for financial data.

## 1. Key Components Implemented

### Data Generation (`data/generator.py`)
Since no dataset was provided, I created a robust synthetic generator that produces:
*   **Sequential Data**: Price returns, Volume changes.
*   **Temporal Features**: Hour, Day, Month.
*   **Text Context**: 768-dim embeddings simulating news vectors.
*   **Multi-Task Labels**: Return (Regression), Volatility (Regression), Event Score (Regression), Regime (Classification).

### Model Architecture (`model/`)

#### A. Temporal Awareness (`embeddings.py`)
*   **`Time2Vec`**: A learnable continuous time encoding that captures periodic patterns ($sin(\omega t + \phi)$) and linear trends.
*   **`TimeFeatureEmbedding`**: Explicitly learns representations for "Hour of Day", "Monday vs Friday", etc.

#### B. Feature Selection (`layers.py`)
*   **`GatedResidualNetwork` (GRN)**: A key component from the Temporal Fusion Transformer (TFT). It enables the model to apply non-linear processing only where needed.
*   **`VariableSelectionNetwork`**: It weighs the importance of Numeric vs. Text vs. Time features dynamically for each time step.

#### C. Multi-Task Heads (`heads.py`)
Instead of a single output, the model projects the latent state into 4 heads:
1.  **Return Head**: Predicting future price movement.
2.  **Volatility Head**: Predicting market risk (ensured positive via Softplus).
3.  **Event Head**: Detecting if a significant news event is impacting the market.
4.  **Regime Head**: Classifying the market state (e.g., Stable vs. Volatile).

## 2. Verification Results

I trained the model for 10 epochs on 2000 synthetic samples.

### Training Log
```
Epoch 1/10 | Total Loss: 0.7275 | Reg Loss: 0.6061
...
Epoch 10/10 | Total Loss: 0.1260 | Reg Loss: 0.0729
```
The model successfully converged, reducing the total loss significantly.

### Prediction Sample
```
True Vol: [0.0135 0.0131 0.0119]
Pred Vol: [0.0199 0.0185 0.0156]
```
The model is learning to track volatility and discern regimes.

## 3. How to Run It
1.  **Install Requirements**:
    ```bash
    pip install -r finance_transformer/requirements.txt
    ```
2.  **Run Training**:
    ```bash
    python3 finance_transformer/main.py
    ```

## 4. Next Steps for You
*   **Real Data**: Replace `generate_financial_data` with a real `Dataset` loading from CSV/API.
*   **Text Encoder**: Replace random embeddings with a real BERT/RoBERTa model fine-tuned on financial news.
*   **Hyperparameter Tuning**: Tune `d_model`, `seq_len`, and loss weights.
