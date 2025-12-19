"""
Behavioral Cloning (BC) Fine-Tuning (Phase 17 Step B)

Fine-tunes the PPO agent using expert data from Shield interventions.
This "shocks" the agent out of the "Safe Death" local minimum by
forcing it to learn the Shield's "Rescue" behavior.
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from stable_baselines3 import PPO
from torch.utils.data import DataLoader, TensorDataset

# Configuration
MODEL_PATH = "ppo_finetuned_v08"
DATA_PATH = "expert_data_shield_harvest.csv"
OUTPUT_PATH = "ppo_cloned_v09"
EPOCHS = 50
BATCH_SIZE = 64
LR = 1e-4

# Action Mapping (Must match ml_controller.py)
# From ml_controller.py action_map: {0: "increase_batch", 1: "reduce_batch", 2: "defer_niw", 3: "no_action"}
ACTION_MAP = {
    'increase_batch': 0,
    'reduce_batch': 1,
    'defer_niw': 2,
    'no_action': 3
}


def train_bc():
    print("="*60)
    print("🧠 BEHAVIORAL CLONING FINE-TUNING")
    print(f"   Source Model: {MODEL_PATH}")
    print(f"   Expert Data: {DATA_PATH}")
    print(f"   Output: {OUTPUT_PATH}")
    print("="*60)
    
    # 1. Load the "broken" agent
    print(f"\n🔄 Loading agent from {MODEL_PATH}...")
    model = PPO.load(MODEL_PATH)
    policy = model.policy
    device = model.device
    print(f"   Device: {device}")
    
    # 2. Load and Preprocess Expert Data
    print(f"\n📂 Loading expert data from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    print(f"   Total samples: {len(df)}")
    print(f"   Action distribution:\n{df['expert_action'].value_counts()}")
    
    # Reconstruct Observations
    # Must match the observation vector in ml_controller.py:
    # obs = [queue_iw, queue_niw, queue_velocity, gpu_utilization, kv_util, 
    #        gpu_count, batch_size, pending_gpus]
    obs_data = []
    actions = []
    
    for _, row in df.iterrows():
        # Build observation vector (8 elements matching ml_controller.py)
        obs = [
            row['queue_depth_iw'] / 50.0,       # normalized queue IW
            row['queue_depth_niw'] / 50.0,      # normalized queue NIW
            row['queue_velocity'] / 10.0,        # normalized velocity
            row['gpu_utilization'],              # already 0-1
            row['kv_utilization'],               # already 0-1
            row['gpu_count'] / 8.0,              # normalized GPU count
            row['batch_size'] / 32.0,            # normalized batch size
            0.0                                   # pending_gpus placeholder
        ]
        
        # Map string action to int index
        action_str = row['expert_action'].lower().strip()
        act_idx = ACTION_MAP.get(action_str)
        
        if act_idx is not None:
            obs_data.append(obs)
            actions.append(act_idx)
    
    # Convert to Tensors
    X = torch.tensor(np.array(obs_data), dtype=torch.float32).to(device)
    y = torch.tensor(np.array(actions), dtype=torch.long).to(device)
    
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    print(f"\n🎯 Starting BC Training on {len(dataset)} samples...")
    print(f"   Epochs: {EPOCHS}, Batch Size: {BATCH_SIZE}, LR: {LR}")
    
    # 3. Supervised Training Loop
    optimizer = optim.Adam(policy.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()
    
    policy.train()
    
    best_acc = 0.0
    
    for epoch in range(EPOCHS):
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_obs, batch_acts in loader:
            optimizer.zero_grad()
            
            # Forward pass through SB3 policy
            # For MlpPolicy with discrete actions, get_distribution returns Categorical
            dist = policy.get_distribution(batch_obs)
            logits = dist.distribution.logits
            
            loss = loss_fn(logits, batch_acts)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Accuracy metric
            preds = torch.argmax(logits, dim=1)
            correct += (preds == batch_acts).sum().item()
            total += batch_acts.size(0)
        
        acc = correct / total
        avg_loss = total_loss / len(loader)
        
        if acc > best_acc:
            best_acc = acc
        
        if epoch % 5 == 0 or epoch == EPOCHS - 1:
            print(f"   Epoch {epoch:3d}/{EPOCHS} | Loss: {avg_loss:.4f} | Accuracy: {acc*100:.1f}%")
    
    # 4. Save the BC-Finetuned Model
    print(f"\n💾 Saving BC-finetuned agent to {OUTPUT_PATH}.zip...")
    model.save(OUTPUT_PATH)
    
    print("\n" + "="*60)
    print("✅ BEHAVIORAL CLONING COMPLETE!")
    print(f"   Best Accuracy: {best_acc*100:.1f}%")
    print(f"   Model saved to: {OUTPUT_PATH}.zip")
    print("   The agent now has the 'Rescue' reflex installed.")
    print("="*60)


if __name__ == "__main__":
    train_bc()
