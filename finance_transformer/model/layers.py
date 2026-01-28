import torch
import torch.nn as nn

class GatedResidualNetwork(nn.Module):
    """
    GRN: Gated Residual Network.
    A key component of Temporal Fusion Transformers (TFT).
    Logic: Input -> Dense -> ELU -> Dense -> GLU -> Add -> Norm
    """
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.1):
        super(GatedResidualNetwork, self).__init__()
        
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.elu = nn.ELU()
        self.layer2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Gating mechanism
        # Gated Linear Unit (GLU) roughly: x * sigmoid(gate_weights)
        self.gate = nn.Linear(output_dim, output_dim * 2) 
        # Note: Standard GLU usually takes 2*dim input and splits. 
        # Here we follow TFT implementation style often seen:
        # We output 2*dim from gate layer, then GLU reduces it.
        
        self.norm = nn.LayerNorm(output_dim)
        
        # Residual connection if dims match
        if input_dim != output_dim:
            self.skip = nn.Linear(input_dim, output_dim)
        else:
            self.skip = nn.Identity()
            
    def forward(self, x):
        # x: (B, T, input_dim)
        residual = self.skip(x)
        
        x = self.layer1(x)
        x = self.elu(x)
        x = self.layer2(x)
        x = self.dropout(x)
        
        # Gating
        gate_out = self.gate(x) # (B, T, 2*out)
        val, gate = torch.chunk(gate_out, 2, dim=-1)
        x = val * torch.sigmoid(gate)
        
        return self.norm(x + residual)

class VariableSelectionNetwork(nn.Module):
    """
    Selects relevant features using GRNs. 
    Not explicitly requested but very useful for 'Text should modulate, not dominate'.
    We can treat Text, Time, and Price as 3 variable groups.
    """
    def __init__(self, input_dims, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.input_dims = input_dims # List of dims for each variable
        
        # Per-variable GRNs
        self.variable_grns = nn.ModuleList([
            GatedResidualNetwork(d, hidden_dim, hidden_dim) 
            for d in input_dims
        ])
        
        # Weighting GRN (calculates weights for each variable)
        # Inputs flattened or concatenated? 
        # Simplified: We learn a weight based on the flattened concatenation of transformed features
        # Or more robustly: A global context vector. 
        
        
        self.weight_network = GatedResidualNetwork(hidden_dim * len(input_dims), hidden_dim, len(input_dims))
        
    def forward(self, inputs):
        """
        inputs: List of tensors [(B, T, d1), (B, T, d2), ...]
        """
        # 1. Transform each variable
        transformed = [grn(x) for grn, x in zip(self.variable_grns, inputs)]
        # Stack: (B, T, NumVars, Hidden)
        stacked = torch.stack(transformed, dim=2)
        
        # 2. Calculate Weights
        # Flatten input to weight network
        flat = torch.cat(transformed, dim=-1) # (B, T, NumVars*Hidden)
        weights = self.weight_network(flat)   # (B, T, NumVars)
        weights = torch.softmax(weights, dim=-1).unsqueeze(-1) # (B, T, NumVars, 1)
        
        # 3. Weighted Sum
        weighted_sum = torch.sum(stacked * weights, dim=2) # (B, T, Hidden)
        
        return weighted_sum, weights
