# Temporal-Aware Multi-Task Transformers in Finance

## 1. Making Transformers "Temporal Aware"

Standard Transformers use Positional Encodings to understand order, but they don't inherently understand *time* (e.g., gaps between data points, seasonality, or calendar effects). For finance, "sequence order" isn't enough; we need "temporal context".

### A. Time Features & Embeddings
Instead of just $t=1, 2, 3$, we feed explicit time features into the model.
*   **Calendar Features**: Day of week, Month of year, Day of month, Hour of day.
*   **Embeddings**: These indices are passed through `Embedding` layers. This helps the model learn that "Monday" often behaves differently from "Friday".

### B. Time2Vec (Continuous Time Encoding)
For continuous time, we use a learnable sine-cosine representation.
*   **Formula**: $\text{Time2Vec}(t) = [\sin(\omega_1 t + \phi_1), \dots, \sin(\omega_k t + \phi_k), \omega_0 t + \phi_0]$
*   Captures periodic patterns (cycles) agnostic of specific calendar dates.

### C. Gating Mechanisms (Gated Residual Network - GRN)
Inspired by the **Temporal Fusion Transformer (TFT)**.
*   Not all inputs are relevant at all times.
*   **GRN** is a non-linear layer with a gate (sigmoid) that selects which features to pass to the Transformer.
*   It allows the model to selectively ignore or prioritize features (e.g., "Ignore news sentiment for this timestep, focus on volume").

---

## 2. Multi-Task Architecture Design

We use a **Hard Parameter Sharing** approach.
*   **Shared Encoder**: A Transformer backbone that learns a rich representation of the market state.
*   **Task-Specific Heads**: Separate dense layers that project the shared representation to different outputs.

### The Heads

1.  **Return Prediction**: Predicts future price movement (MSE Loss).
2.  **Volatility Prediction**: Predicts rolling standard deviation (MSE Loss, Softplus activation).
3.  **Event Impact Score**: Predicts news significance (Sigmoid activation).
4.  **Regime Classification**: Identifies market states like Bull or Bear (Cross-Entropy Loss).
