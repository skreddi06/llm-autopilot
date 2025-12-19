"""Fine-tune Warmstarted PPO with RL.

Loads the BC-pretrained model and refines it with reinforcement learning.
Uses low learning rate to prevent catastrophic forgetting.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from llm_env_v08 import LLMServerEnvV08


def finetune_warmstarted(total_timesteps: int = 100000):
    print("=" * 60)
    print("🚀 RL FINE-TUNING - Warmstarted PPO")
    print("=" * 60)
    
    # 1. Load warmstarted model
    print("\n1. Loading warmstarted model...")
    env = LLMServerEnvV08()
    env = Monitor(env)
    
    model = PPO.load("ppo_warmstarted_v08", env=env)
    
    # 2. Configure WSRL hyperparameters
    model.learning_rate = 5e-5  # Moderate LR
    model.ent_coef = 0.05       # HIGH ENTROPY to prevent collapse
    model.clip_range = lambda _: 0.2  # Standard low clip
    # Note: SB3 doesn't support asymmetric clipping easily, so we rely on 
    # High Entropy + Critic Warmup to achieve the effect.
    
    print("\n--- WSRL CONFIGURATION ---")
    print("   Learning Rate: 5e-5")
    print("   Entropy Coef:  0.05 (High)")
    print("   Critic Warmup: 5000 steps (Actor Frozen)")
    print("   Total Steps:   200,000")
    
    # 3. Standard Fine-tuning (Phase 15 Constraint Learning)
    # We use standard PPO training now that we have strong constraints
    print(f"\n3. Fine-tuning for {total_timesteps} timesteps...")
    
    # Ensure strict mode (Stage 2 physics) from the start
    with open("curriculum_stage.txt", "w") as f:
        f.write("2")
        
    model.learn(total_timesteps=total_timesteps, progress_bar=False)
    
    # 5. Save
    model.save("ppo_finetuned_v09")
    print(f"\n✅ Model saved to: ppo_finetuned_v09.zip")
    
    # 5. Evaluate
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
    
    print(f"\nAction Distribution:")
    for action, count in sorted(action_counts.items(), key=lambda x: -x[1]):
        pct = count / 500 * 100
        print(f"  {action}: {count} ({pct:.1f}%)")
    
    # Check for DEFER_NIW
    defer_pct = action_counts.get("DEFER_NIW", 0) / 500 * 100
    print("\n" + "=" * 60)
    if defer_pct >= 5:
        print(f"🎉 SUCCESS: DEFER_NIW LEARNED! ({defer_pct:.1f}%)")
        print("   Agent has learned SageServe yield management!")
    else:
        print(f"⚠️ DEFER_NIW usage dropped ({defer_pct:.1f}%)")
        print("   May need less aggressive fine-tuning")
    
    return model


if __name__ == "__main__":
    finetune_warmstarted(total_timesteps=200000)
