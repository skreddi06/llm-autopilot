"""PPO Training Script v0.8 - Stress Test Edition

Trains with:
- LLMServerEnvV08 (stress scenarios)
- Longer training (100k timesteps)
- Tracks DEFER_NIW emergence
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from llm_env_v08 import LLMServerEnvV08
import numpy as np


def train_ppo_v08(total_timesteps: int = 100000):
    print("=" * 60)
    print("🔥 PPO TRAINING v0.8 - STRESS TEST EDITION")
    print("=" * 60)
    print("\nKey changes:")
    print("  • Provisioning lag: SCALE_OUT delayed 10 steps")
    print("  • Queue acceleration penalty (squared)")
    print("  • NIW interference cost")
    print("  • 40% surge probability")
    print("  • Surge survival bonus (+10)")
    print()
    
    # Create environment
    env = LLMServerEnvV08()
    env = Monitor(env)
    
    # Create PPO with tuned hyperparameters for long-horizon learning
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=0.0001,  # Lower LR for stability
        n_steps=4096,          # Longer rollouts for credit assignment
        batch_size=128,
        n_epochs=10,
        gamma=0.995,           # Higher gamma for long-term rewards
        gae_lambda=0.98,       # Higher GAE for variance reduction
        clip_range=0.2,
        ent_coef=0.1,          # HIGH entropy for exploration of DEFER_NIW
    )
    
    print(f"Training for {total_timesteps} timesteps...")
    print("(Looking for DEFER_NIW emergence)\n")
    
    model.learn(total_timesteps=total_timesteps, progress_bar=False)
    
    # Save
    model.save("ppo_autopilot_v08")
    print(f"\n✅ Model saved to: ppo_autopilot_v08.zip")
    
    # Evaluate
    print("\n" + "=" * 60)
    print("📊 EVALUATION")
    print("=" * 60)
    
    eval_env = LLMServerEnvV08()
    obs, _ = eval_env.reset()
    
    total_reward = 0
    action_counts = {}
    action_names = ["NO_OP", "INC_BATCH", "DEC_BATCH", "SCALE_OUT", "SCALE_IN", "DEFER_NIW"]
    
    for step in range(500):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = eval_env.step(action)
        total_reward += reward
        
        name = action_names[action]
        action_counts[name] = action_counts.get(name, 0) + 1
        
        if terminated:
            break
    
    stats = eval_env.get_episode_stats()
    
    print(f"\nEpisode Results:")
    print(f"  Total Reward: {total_reward:.2f}")
    print(f"  Latency Violations: {stats['latency_violations']}")
    print(f"  Surges: {stats['surges']}")
    print(f"  Survived: {stats['survived_surges']}")
    print(f"  Survival Rate: {stats['survival_rate']*100:.1f}%")
    print(f"  Final GPUs: {stats['final_gpus']}")
    print(f"  Final Batch: {stats['final_batch']}")
    
    print(f"\nAction Distribution:")
    for action, count in sorted(action_counts.items(), key=lambda x: -x[1]):
        pct = count / 500 * 100
        print(f"  {action}: {count} ({pct:.1f}%)")
    
    # Check for DEFER_NIW emergence
    defer_pct = action_counts.get("DEFER_NIW", 0) / 500 * 100
    print("\n" + "=" * 60)
    if defer_pct >= 5:
        print(f"🎉 SUCCESS: DEFER_NIW EMERGED! ({defer_pct:.1f}%)")
        print("   Agent learned preventive control!")
    elif defer_pct > 0:
        print(f"🔶 PROGRESS: DEFER_NIW starting to emerge ({defer_pct:.1f}%)")
        print("   May need more training time")
    else:
        print(f"⚠️ DEFER_NIW not yet emergent ({defer_pct:.1f}%)")
        print("   Reward shaping may need further tuning")
    
    return model


if __name__ == "__main__":
    train_ppo_v08(total_timesteps=100000)
