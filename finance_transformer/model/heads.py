import torch
import torch.nn as nn

class MultiTaskHeads(nn.Module):
    def __init__(self, d_model, num_regimes=2):
        super(MultiTaskHeads, self).__init__()
        
        # 1. Return Prediction (Regression)
        # Predicts next step return.
        self.return_head = nn.Linear(d_model, 1)
        
        # 2. Volatility Prediction (Regression, Positive)
        # Predicts next step volatility.
        self.vol_head = nn.Sequential(
            nn.Linear(d_model, 1),
            nn.Softplus() # Ensures output > 0
        )
        
        # 3. Event Impact Score (Regression/Classification 0-1)
        # Predicts if an event is occurring or its magnitude.
        self.event_head = nn.Sequential(
            nn.Linear(d_model, 1),
            nn.Sigmoid() 
        )
        
        # 4. Regime Classification
        # Predicts market regime (e.g. Bull/Bear/Shock).
        self.regime_head = nn.Linear(d_model, num_regimes)
        
    def forward(self, x):
        # x: (Batch, SeqLen, d_model) -> We usually take the last step for prediction
        # or we can predict sequence-to-sequence.
        # Let's assume we want to predict the target for the *next* step based on history.
        
        # Pooling: We usually take the embedding of the last time step T as the summary.
        # x_last: (Batch, d_model)
        x_last = x[:, -1, :]
        
        ret_pred = self.return_head(x_last)
        vol_pred = self.vol_head(x_last)
        evt_pred = self.event_head(x_last)
        reg_pred = self.regime_head(x_last)
        
        return {
            'return': ret_pred,
            'volatility': vol_pred,
            'event': evt_pred,
            'regime': reg_pred
        }
