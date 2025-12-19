"""Pretrain PPO with Behavior Cloning (Imitation Warmstart).

This script performs supervised pretraining on the PPO network using
expert data from training_data.jsonl. This initializes the policy
in a region where DEFER_NIW is already probable, solving entropy collapse.
"""

import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from llm_env_v08 import LLMServerEnvV08


# Action name to index mapping (must match environment)
ACTION_MAP = {
    "no_action": 0,
    "increase_batch": 1,
    "reduce_batch": 2,
    "scale_out": 3,
    "scale_in": 4,
    "defer_niw": 5
}


def load_expert_data(filepath: str = "training_data_sawtooth_v2.jsonl"):
    """Load expert demonstrations and filter for critical moments."""
    observations = []
    actions = []
    
    with open(filepath, "r") as f:
        for line in f:
            entry = json.loads(line)
            state = entry['state']
            action = entry['action']
            
            # Build observation vector (must match llm_env_v08)
            obs = [
                state.get('queue_iw', 0),
                state.get('queue_niw', 0),
                state.get('velocity', 0),
                state.get('gpu_util', 50),
                state.get('kv_util', 0.1),  # Use ACTUAL kv_util from expert data!
                state.get('gpu_count', 1),
                state.get('batch_size', 8),
                0  # pending_gpus
            ]
            observations.append(obs)
            actions.append(ACTION_MAP.get(action, 0))
    
    return np.array(observations, dtype=np.float32), np.array(actions, dtype=np.int64)


def pretrain_ppo(epochs: int = 100, lr: float = 1e-3):
    """Pretrain PPO policy network with Behavior Cloning."""
    print("=" * 60)
    print("🎓 IMITATION WARMSTART - Pretraining PPO")
    print("=" * 60)
    
    # 1. Load expert data
    print("\n1. Loading expert data...")
    observations, actions = load_expert_data()
    print(f"   Loaded {len(observations)} demonstrations")
    
    # Count action distribution in expert data
    unique, counts = np.unique(actions, return_counts=True)
    action_names = ["NO_OP", "INC_BATCH", "DEC_BATCH", "SCALE_OUT", "SCALE_IN", "DEFER_NIW"]
    print("\n   Expert action distribution:")
    for u, c in zip(unique, counts):
        pct = c / len(actions) * 100
        print(f"     {action_names[u]}: {c} ({pct:.1f}%)")
    
    # 2. Filter for critical moments (high velocity = where DEFER matters)
    print("\n2. Filtering critical moments (velocity > 0)...")
    velocity_mask = observations[:, 2] > 0  # velocity column
    critical_obs = observations[velocity_mask]
    critical_actions = actions[velocity_mask]
    print(f"   Critical moments: {len(critical_obs)} ({len(critical_obs)/len(observations)*100:.1f}%)")
    
    if len(critical_obs) < 100:
        print("   Warning: Few critical moments. Using all data.")
        critical_obs = observations
        critical_actions = actions
    
    # 3. Create fresh PPO agent
    print("\n3. Creating PPO agent...")
    env = LLMServerEnvV08()
    env = Monitor(env)
    agent = PPO(
        "MlpPolicy",
        env,
        verbose=0,
        learning_rate=3e-4,
        ent_coef=0.05  # Keep some entropy for exploration
    )
    
    # 4. Access policy network components
    policy = agent.policy
    optimizer = optim.Adam(policy.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    
    # Convert to tensors
    obs_tensor = torch.as_tensor(critical_obs, dtype=torch.float32)
    action_tensor = torch.as_tensor(critical_actions, dtype=torch.long)
    
    # 5. Supervised training loop (Behavior Cloning)
    print(f"\n4. Training for {epochs} epochs...")
    policy.train()
    
    batch_size = 256
    n_batches = len(critical_obs) // batch_size
    
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        
        # Shuffle
        perm = torch.randperm(len(obs_tensor))
        obs_shuffled = obs_tensor[perm]
        act_shuffled = action_tensor[perm]
        
        for i in range(n_batches):
            start = i * batch_size
            end = start + batch_size
            
            obs_batch = obs_shuffled[start:end]
            act_batch = act_shuffled[start:end]
            
            # Forward pass through policy
            distribution = policy.get_distribution(obs_batch)
            logits = distribution.distribution.logits
            
            # Calculate loss
            loss = loss_fn(logits, act_batch)
            
            # Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Accuracy
            pred = logits.argmax(dim=1)
            correct += (pred == act_batch).sum().item()
        
        avg_loss = total_loss / max(1, n_batches)
        accuracy = correct / (n_batches * batch_size) * 100
        
        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"   Epoch {epoch:3d}: Loss={avg_loss:.4f}, Accuracy={accuracy:.1f}%")
    
    # 6. Save warmstarted agent
    save_path = "ppo_warmstarted_v08"
    agent.save(save_path)
    print(f"\n✅ Agent saved to: {save_path}.zip")
    
    # 7. Quick evaluation
    print("\n5. Quick evaluation...")
    eval_env = LLMServerEnvV08()
    obs, _ = eval_env.reset()
    
    action_counts = {}
    for step in range(100):
        action, _ = agent.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = eval_env.step(action)
        
        name = action_names[action]
        action_counts[name] = action_counts.get(name, 0) + 1
        
        if terminated:
            break
    
    print("   Action distribution (100 steps):")
    for action, count in sorted(action_counts.items(), key=lambda x: -x[1]):
        print(f"     {action}: {count}%")
    
    return agent


if __name__ == "__main__":
    pretrain_ppo(epochs=100)
