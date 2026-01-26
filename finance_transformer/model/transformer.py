import torch
import torch.nn as nn
from .embeddings import Time2Vec, TimeFeatureEmbedding
from .layers import VariableSelectionNetwork
from .heads import MultiTaskHeads

class TemporalMultiTaskModel(nn.Module):
    def __init__(
        self, 
        d_numeric=2, 
        d_text=768, 
        d_model=64, 
        nhead=4, 
        num_layers=2, 
        dropout=0.1
    ):
        super(TemporalMultiTaskModel, self).__init__()
        
        # 1. Feature Encoders
        # -------------------
        # Numeric: Price, Vol -> Linear -> d_model
        self.num_encoder = nn.Linear(d_numeric, d_model)
        
        # Text: BERT Emb -> Linear -> d_model
        self.text_encoder = nn.Linear(d_text, d_model)
        
        # Time: 
        #   (a) Time2Vec (Continuous)
        #   (b) Calendar (Discrete)
        # We combine them.
        self.time2vec = Time2Vec(output_dim=d_model) # (B, T, d_model)
        self.calendar_encoder = TimeFeatureEmbedding(d_model=d_model // 4) 
        # Calendar outputs 4 features * (d_model/4) = d_model
        
        # 2. Variable Selection (Fusion)
        # ------------------------------
        # Inputs to selection: [Numeric, Text, ContinuousTime, CalendarTime]
        # All projected to d_model first.
        self.vsn = VariableSelectionNetwork(input_dims=[d_model]*4, hidden_dim=d_model)
        
        # 3. Transformer Backbone
        # -----------------------
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 4. Multi-Task Heads
        # -------------------
        self.heads = MultiTaskHeads(d_model=d_model)
        
    def forward(self, x_num, x_time, x_text):
        """
        x_num: (B, T, 2)
        x_time: (B, T, 4)
        x_text: (B, T, 768)
        """
        # A. Feature Encoding
        e_num = self.num_encoder(x_num)
        e_text = self.text_encoder(x_text)
        
        # Time processing
        # x_time[:,:,3] is Hour (0-23). We use that for Time2Vec as "continuous" proxy for this synthetic data
        # In real data, you'd pass a float timestamp.
        t_cont = x_time[:, :, 3].float() 
        e_t2v = self.time2vec(t_cont)
        e_cal = self.calendar_encoder(x_time)
        
        # B. Variable Selection
        # Combine all features dynamically
        # text acts as a context prior here
        fused_features, weights = self.vsn([e_num, e_text, e_t2v, e_cal])
        
        # C. Transformer
        # (B, T, d_model)
        context = self.transformer(fused_features)
        
        # D. Heads
        # Predictions based on the last state
        outputs = self.heads(context)
        
        return outputs, weights
