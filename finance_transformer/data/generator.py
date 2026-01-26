import numpy as np
import pandas as pd
import torch
from typing import Tuple, Dict

def generate_financial_data(
    num_samples: int = 1000, 
    seq_len: int = 60, 
    d_model: int = 64, 
    text_dim: int = 768,
    save_path: str = "finance_transformer/data"
) -> Dict[str, torch.Tensor]:
    """
    Generates synthetic financial data with temporal features and multi-task labels.
    
    Args:
        num_samples: Number of time steps in the total dataset.
        seq_len: Length of the lookback window.
        d_model: Dimension of numerical features (implied, not directly used here but good context).
        text_dim: Dimension of the text embeddings (e.g. BERT output).
        
    Returns:
        Dictionary containing tensors:
        - numerics: (N, seq_len, 2) [Price_pct, Volume_pct]
        - time: (N, seq_len, 4) [Month, Day, Weekday, Hour]
        - text: (N, seq_len, text_dim)
        - labels:
            - return: (N, 1)
            - volatility: (N, 1)
            - event: (N, 1)
            - regime: (N, 1) (Class index)
    """
    print(f"Generating {num_samples} samples of synthetic data...")
    
    # 1. Timeline
    # Hourly data for 'num_samples' + 'seq_len' to allow for windowing
    total_len = num_samples + seq_len + 1 
    dates = pd.date_range(start="2023-01-01", periods=total_len, freq="h")
    
    # 2. Price & Volume (Random Walk & Noise)
    np.random.seed(42)
    # Drift + Volatility
    returns = np.random.normal(loc=0.0001, scale=0.01, size=total_len)
    price = 100 * np.cumprod(1 + returns)
    
    # Volume: Log-normal with some spikes
    volume = np.random.lognormal(mean=10, sigma=1, size=total_len)
    
    # 3. Latent "Event" State
    # Random "shocks" that affect price and text
    event_shocks = np.random.choice([0, 1], size=total_len, p=[0.95, 0.05])
    
    # Modify return/volatility when shock happens
    returns = returns + (event_shocks * np.random.normal(0, 0.05, size=total_len))
    volatility_proxy = pd.Series(returns).rolling(window=10).std().fillna(0).values
    
    # 4. Text Embeddings
    # Base noise
    text = np.random.normal(0, 1, size=(total_len, text_dim))
    # Add signal correlates with events
    # If event_shock[t] == 1, add a specific "event vector" to text
    event_vector = np.random.normal(loc=2, scale=0.5, size=(1, text_dim))
    text += (event_shocks[:, None] * event_vector)
    
    # Normalize features
    price_pct = pd.Series(price).pct_change().fillna(0).values
    vol_pct = pd.Series(volume).pct_change().fillna(0).values
    
    # --- SAVE RAW DATA BEGIN ---
    if save_path:
        import os
        os.makedirs(save_path, exist_ok=True)
        
        # 1. Create DataFrame for Inspectable Data
        # We calculate the Regime for the whole sequence to save it
        regime_labels = [1 if v > 0.015 else 0 for v in volatility_proxy]
        
        df = pd.DataFrame({
            "Date": dates,
            "Price": price,
            "Volume": volume,
            "Return": returns,
            "Volatility": volatility_proxy,
            "Event_Shock": event_shocks,
            "Regime_Label": regime_labels
        })
        
        csv_path = os.path.join(save_path, "synthetic_data.csv")
        df.to_csv(csv_path, index=False)
        print(f"Raw data saved to: {csv_path}")
        
        # 2. Save Text Embeddings (Optional, large file)
        npy_path = os.path.join(save_path, "text_embeddings.npy")
        np.save(npy_path, text)
        print(f"Text embeddings saved to: {npy_path}")
    # --- SAVE RAW DATA END ---
    
    # Prepare Windows
    X_num = []
    X_time = []
    X_text = []
    Y_ret = []
    Y_vol = []
    Y_evt = []
    Y_reg = []
    
    for i in range(num_samples):
        # Input Window: [i : i+seq_len]
        # Target: [i+seq_len] (Next step prediction)
        
        # Numeric Features: Price Changes, Volume Changes
        # Shape: (seq_len, 2)
        win_price = price_pct[i : i+seq_len]
        win_vol = vol_pct[i : i+seq_len]
        X_num.append(np.stack([win_price, win_vol], axis=1))
        
        # Time Features: Month, Day, Weekday, Hour
        # Shape: (seq_len, 4)
        win_dates = dates[i : i+seq_len]
        t_feats = np.stack([
            win_dates.month,
            win_dates.day,
            win_dates.dayofweek,
            win_dates.hour
        ], axis=1)
        X_time.append(t_feats)
        
        # Text Features
        X_text.append(text[i : i+seq_len])
        
        # Targets (at step i+seq_len)
        target_idx = i + seq_len
        target_ret = returns[target_idx]
        target_vol = volatility_proxy[target_idx]
        target_evt = event_shocks[target_idx]
        
        # Regime: 0=Low Vol, 1=High Vol
        # Simple thresholding logic for label
        target_reg = 1 if target_vol > 0.015 else 0
        
        Y_ret.append(target_ret)
        Y_vol.append(target_vol)
        Y_evt.append(target_evt)
        Y_reg.append(target_reg)
        
    # Convert to Tensors
    data = {
        "x_num": torch.FloatTensor(np.array(X_num)),      # (B, T, 2)
        "x_time": torch.LongTensor(np.array(X_time)),     # (B, T, 4)
        "x_text": torch.FloatTensor(np.array(X_text)),    # (B, T, 768)
        "y_ret": torch.FloatTensor(np.array(Y_ret)).unsqueeze(-1),   # (B, 1)
        "y_vol": torch.FloatTensor(np.array(Y_vol)).unsqueeze(-1),   # (B, 1)
        # Event is usually binary classification or regression signal.
        # Let's treat it as a Score (0.0 to 1.0) for MSE/BCE.
        "y_evt": torch.FloatTensor(np.array(Y_evt)).unsqueeze(-1),   # (B, 1)
        "y_reg": torch.LongTensor(np.array(Y_reg))        # (B, ) -> for CrossEntropy
    }
    
    print(f"Data generated. Shapes:")
    print(f"  Numerics: {data['x_num'].shape}")
    print(f"  Time:     {data['x_time'].shape}")
    print(f"  Text:     {data['x_text'].shape}")
    
    return data

if __name__ == "__main__":
    generate_financial_data(num_samples=100)
