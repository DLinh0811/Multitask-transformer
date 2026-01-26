import torch
import torch.nn as nn
import math

class Time2Vec(nn.Module):
    """
    Time2Vec: Learning a Vector Representation of Time.
    Paper: https://arxiv.org/abs/1907.05321
    
    Captures both periodic and non-periodic patterns.
    - term 0: linear (w0 * t + phi0) -> non-periodic / trend
    - term 1..k: periodic (sin(wi * t + phii))
    """
    def __init__(self, output_dim=16):
        super(Time2Vec, self).__init__()
        self.output_dim = output_dim
        
        # We assume input time is scalar (1D) per step
        self.w0 = nn.Parameter(torch.randn(1, 1))
        self.phi0 = nn.Parameter(torch.randn(1, 1))
        
        self.w = nn.Parameter(torch.randn(1, output_dim - 1))
        self.phi = nn.Parameter(torch.randn(1, output_dim - 1))
        
    def forward(self, t):
        """
        Args:
            t: Tensor of shape (Batch, SeqLen, 1) or (Batch, SeqLen)
        Returns:
            Tensor of shape (Batch, SeqLen, output_dim)
        """
        if t.dim() == 2:
            t = t.unsqueeze(-1) # (B, T, 1)
            
        # 1. Non-periodic component: w0 * t + phi0
        v_linear = self.w0 * t + self.phi0 # Broadcasting works
        
        # 2. Periodic component: sin(w * t + phi)
        v_periodic = torch.sin(t @ self.w.unsqueeze(0) + self.phi) # t: (B,T,1), w: (1, K) -> (B,T, K)
        
        return torch.cat([v_linear, v_periodic], dim=-1)


class TimeFeatureEmbedding(nn.Module):
    """
    Embeds categorical time features (Month, Day, Weekday, Hour).
    """
    def __init__(self, d_model=16):
        super(TimeFeatureEmbedding, self).__init__()
        # Cardinalities
        self.embed_month = nn.Embedding(13, d_model)  # 1-12
        self.embed_day = nn.Embedding(32, d_model)    # 1-31
        self.embed_weekday = nn.Embedding(7, d_model) # 0-6
        self.embed_hour = nn.Embedding(24, d_model)   # 0-23
        
    def forward(self, x_time):
        """
        Args:
            x_time: (Batch, SeqLen, 4) -> [Month, Day, Weekday, Hour]
        Returns:
            (Batch, SeqLen, 4 * d_model)
        """
        months = self.embed_month(x_time[:, :, 0])
        days = self.embed_day(x_time[:, :, 1])
        weekdays = self.embed_weekday(x_time[:, :, 2])
        hours = self.embed_hour(x_time[:, :, 3])
        
        return torch.cat([months, days, weekdays, hours], dim=-1)
