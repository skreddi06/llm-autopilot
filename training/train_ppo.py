"""PPO Training Script for LLM Autopilot RL Agent.

Uses Stable Baselines3 to train a PPO agent on the LLMServerEnv.
The agent learns to balance throughput, latency, and cost.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from llm_env import LLMServerEnv
import numpy as np


def train_ppo(total_timesteps: int = 100000, save_path: str = "ppo_autopilot_v07"):
    """Train PPO agent on LLM serving environment."""
    print("=" * 60)
    print("🤖 PPO TRAINING - LLM Autopilot v0.7")
    print("=" * 60)
    
    # 1. Create Environment
    print("\n1. Creating environment...")
    env = LLMServerEnv()
    env = Monitor(env)  # Wrap for logging
    
    # 2. Create PPO Agent
    print("\n2. Initializing PPO agent...")
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=0.0003,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2
        # tensorboard_log disabled to avoid dependency
    )
    
    print(f"\n   Policy network: {model.policy}")
    
    # 3. Train
    print(f"\n3. Training for {total_timesteps} timesteps...")
    print("   (This may take a few minutes)\n")
    
    model.learn(
        total_timesteps=total_timesteps,
        progress_bar=False  # Disabled to avoid tqdm/rich dependency
    )
    
    # 4. Save Model
    model.save(save_path)
    print(f"\n✅ Model saved to: {save_path}.zip")
    
    # 5. Quick Evaluation
    print("\n" + "=" * 60)
    print("📊 QUICK EVALUATION")
    print("=" * 60)
    
    eval_env = LLMServerEnv()
    obs, _ = eval_env.reset()
    
    total_reward = 0
    action_counts = {}
    action_names = ["NO_OP", "INC_BATCH", "DEC_BATCH", "SCALE_OUT", "SCALE_IN", "DEFER_NIW"]
    
    for step in range(300):  # One full episode
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = eval_env.step(action)
        total_reward += reward
        
        name = action_names[action]
        action_counts[name] = action_counts.get(name, 0) + 1
        
        if terminated:
            break
    
    print(f"\nEpisode Results:")
    print(f"  Total Reward: {total_reward:.2f}")
    print(f"  Final GPUs: {eval_env.num_gpus}")
    print(f"  Final Batch: {eval_env.batch_size}")
    print(f"  Latency Violations: {eval_env.total_latency_violations}")
    
    print(f"\nAction Distribution:")
    for action, count in sorted(action_counts.items(), key=lambda x: -x[1]):
        pct = count / 300 * 100
        print(f"  {action}: {count} ({pct:.1f}%)")
    
    # Check if agent learned DEFER_NIW
    defer_pct = action_counts.get("DEFER_NIW", 0) / 300 * 100
    if defer_pct > 5:
        print(f"\n🎉 EMERGENT BEHAVIOR: Agent learned DEFER_NIW ({defer_pct:.1f}%)")
    else:
        print(f"\n⚠️ Agent hasn't learned DEFER_NIW yet ({defer_pct:.1f}%)")
    
    return model


if __name__ == "__main__":
    # Check for gymnasium and stable-baselines3
    try:
        import gymnasium
        import stable_baselines3
        print(f"gymnasium version: {gymnasium.__version__}")
        print(f"stable_baselines3 version: {stable_baselines3.__version__}")
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Install with: pip install gymnasium stable-baselines3 shimmy")
        sys.exit(1)
    
    train_ppo(total_timesteps=50000)  # Reduced for faster demo
