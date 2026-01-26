import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from data.generator import generate_financial_data
from model.transformer import TemporalMultiTaskModel

def train():
    # 1. Hyperparameters
    BATCH_SIZE = 32
    EPOCHS = 10
    LR = 0.001
    SEQ_LEN = 30
    
    # 2. Data Preparation
    print("--- 1. Generating Data ---")
    data = generate_financial_data(num_samples=2000, seq_len=SEQ_LEN)
    
    dataset = TensorDataset(
        data['x_num'], 
        data['x_time'], 
        data['x_text'],
        data['y_ret'], 
        data['y_vol'], 
        data['y_evt'], 
        data['y_reg']
    )
    
    # 80/20 Split
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_ds, test_ds = torch.utils.data.random_split(dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)
    
    # 3. Model Initialization
    print("\n--- 2. Initializing Model ---")
    model = TemporalMultiTaskModel(
        d_numeric=2,
        d_text=768,
        d_model=64,
        nhead=4,
        num_layers=2,
        dropout=0.1
    )
    
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    # Loss Functions
    mse_loss = nn.MSELoss()
    ce_loss = nn.CrossEntropyLoss()
    
    # 4. Training Loop
    print("\n--- 3. Starting Training ---")
    model.train()
    
    for epoch in range(EPOCHS):
        total_loss = 0
        l_ret_sum = 0
        l_vol_sum = 0
        l_evt_sum = 0
        l_reg_sum = 0
        
        for batch in train_loader:
            x_n, x_t, x_txt, y_r, y_v, y_e, y_reg_label = batch
            
            optimizer.zero_grad()
            
            outputs, weights = model(x_n, x_t, x_txt)
            
            # Calculate Multi-Task Loss
            # 1. Return (MSE)
            loss_ret = mse_loss(outputs['return'], y_r)
            
            # 2. Volatility (MSE)
            loss_vol = mse_loss(outputs['volatility'], y_v)
            
            # 3. Event (MSE for Score)
            loss_evt = mse_loss(outputs['event'], y_e)
            
            # 4. Regime (CrossEntropy)
            # Flatten outputs: (B, 2) -> Target (B)
            loss_reg = ce_loss(outputs['regime'], y_reg_label)
            
            # Weighted Sum (Simple equal weights for now, can be tuned)
            loss = loss_ret + loss_vol + loss_evt + loss_reg
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            l_ret_sum += loss_ret.item()
            l_vol_sum += loss_vol.item()
            l_evt_sum += loss_evt.item()
            l_reg_sum += loss_reg.item()
            
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{EPOCHS} | Total Loss: {avg_loss:.4f} | "
              f"Ret: {l_ret_sum/len(train_loader):.4f} | "
              f"Vol: {l_vol_sum/len(train_loader):.4f} | "
              f"Evt: {l_evt_sum/len(train_loader):.4f} | "
              f"Reg: {l_reg_sum/len(train_loader):.4f}")
        
    print("\n--- 4. Training Complete ---")
    
    # 5. Quick Verification on Test Set
    model.eval()
    with torch.no_grad():
        x_n, x_t, x_txt, y_r, y_v, y_e, y_reg_label = next(iter(test_loader))
        outputs, _ = model(x_n, x_t, x_txt)
        
        print("\nPrediction Samples (First 3 in batch):")
        print("True Return:", y_r[:3].flatten().numpy())
        print("Pred Return:", outputs['return'][:3].flatten().numpy())
        print("-" * 20)
        print("True Vol:", y_v[:3].flatten().numpy())
        print("Pred Vol:", outputs['volatility'][:3].flatten().numpy())
        print("-" * 20)
        print("True Event:", y_e[:3].flatten().numpy())
        print("Pred Event:", outputs['event'][:3].flatten().numpy())
        print("-" * 20)
        print("True Regime:", y_reg_label[:3].numpy())
        print("Pred Regime:", torch.argmax(outputs['regime'][:3], dim=1).numpy())

if __name__ == "__main__":
    train()
