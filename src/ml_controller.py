"""ML Controller - Deploys trained Student Policy with Safety Wrapper.

Loads the trained Random Forest model and provides inference with
deterministic safety guardrails to prevent catastrophic actions.
"""

import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from dataclasses import dataclass

from models import Metrics, Decision, Action


@dataclass
class MLControllerConfig:
    """Configuration for ML Controller safety limits."""
    min_gpus: int = 1
    max_gpus: int = 16
    min_batch_size: int = 1
    max_batch_size: int = 64
    max_queue_before_panic: int = 500



class MLController:
    """ML-based controller using trained PPO Policy with Safety Shield (Hybrid).
    
    The Shield ensures liveness (queue draining) while PPO optimizes capacity.
    """
    
    def __init__(self, model_path: str = "ppo_finetuned_v08", 
                 config: MLControllerConfig = None):
        """Load PPO model and initialize safety limits."""
        from stable_baselines3 import PPO
        try:
             self.model = PPO.load(model_path)
             print(f"Loaded PPO model: {model_path}")
        except Exception as e:
             print(f"Error loading PPO model: {e}")
             self.model = None

        self.config = config or MLControllerConfig()
        
        # Track state for safety checks
        self.current_batch_size = 8
        self.current_gpu_count = 1
        self.decision_count = 0
        self.safety_overrides = 0
        
        # PPO Action Map
        self.action_map = {
            0: "no_action",
            1: "increase_batch",
            2: "reduce_batch",
            3: "scale_out",
            4: "scale_in",
            5: "defer_niw"
        }
    
    def update_config(self, batch_size: int = None, gpu_count: int = None):
        """Update internal state tracking."""
        if batch_size is not None:
            self.current_batch_size = batch_size
        if gpu_count is not None:
            self.current_gpu_count = gpu_count
    
    def make_decision(self, metrics: Metrics) -> Decision:
        """Make a decision using Hybrid Shielded Logic."""
        self.decision_count += 1
        
        queue_iw = getattr(metrics, 'queue_depth_iw', metrics.queue_depth)
        queue_niw = getattr(metrics, 'queue_depth_niw', 0)
        niw_in_flight = getattr(metrics, 'niw_in_flight', 0)
        
        # Calculate approximate kv_util (Physics Approximation)
        # Matches logic in llm_env_v08.py
        running_iw = min(queue_iw, self.current_batch_size * self.current_gpu_count)
        total_running = running_iw + niw_in_flight
        kv_util = min(1.0, total_running / (self.config.max_gpus * 16))
        
        # === 1. THE SHIELD (Deterministic Liveness & Safety) ===
        
        # RULE A: EMERGENCY DRAIN (Extreme Queue)
        # If Queue is extremely high, force max throughput regardless of memory
        if queue_iw > 50:
            self.safety_overrides += 1
            if self.current_batch_size >= self.config.max_batch_size:
                 return Decision(Action.NO_ACTION, reason="Shield: Emergency - Max batch reached")
            
            return Decision(
                Action.INCREASE_BATCH,
                reason=f"Shield: EMERGENCY Queue={queue_iw} Mem={kv_util:.2f}",
                batch_size=min(self.config.max_batch_size, self.current_batch_size + 4),
                gpu_count=self.current_gpu_count
            )
        
        # RULE B: RESCUE (Sawtooth Upstroke)
        # If Queue is high AND Memory is safe (extended to 85%), FORCE INC_BATCH.
        # This breaks the "Safe Death" loop where agent refuses to scale.
        if queue_iw > 10 and kv_util < 0.85:
            # Shield Intervention
            self.safety_overrides += 1
            if self.current_batch_size >= self.config.max_batch_size:
                 return Decision(Action.NO_ACTION, reason="Shield: Max batch reached")
            
            return Decision(
                Action.INCREASE_BATCH,
                reason=f"Shield: RESCUE Queue={queue_iw} Mem={kv_util:.2f}",
                batch_size=min(self.config.max_batch_size, self.current_batch_size + 2),
                gpu_count=self.current_gpu_count
            )

        # RULE C: PANIC (Memory Safety - Only if >95%)
        # If Memory is critical, FORCE DEFER or DEC_BATCH
        if kv_util > 0.95:
            self.safety_overrides += 1
            return Decision(
                Action.DEFER_NIW,
                reason=f"Shield: PANIC Memory={kv_util:.2f}",
                batch_size=self.current_batch_size,
                gpu_count=self.current_gpu_count
            )

        # === 2. THE RL AGENT (Stochastic Optimization) ===
        if self.model is None:
            return Decision(Action.NO_ACTION, reason="No Model")

        # Build Observation Vector (Must match training env)
        obs = np.array([
            queue_iw,
            queue_niw,
            metrics.queue_velocity,
            metrics.gpu_utilization,
            kv_util,
            self.current_gpu_count,
            self.current_batch_size,
            0 # pending_gpus placeholder
        ], dtype=np.float32)

        action_idx, _ = self.model.predict(obs, deterministic=True)
        if hasattr(action_idx, 'item'):
            action_idx = int(action_idx.item())
        action_name = self.action_map[action_idx]
        
        # Map to Decision
        if action_name == "increase_batch":
             return Decision(
                 Action.INCREASE_BATCH, 
                 reason="RL: increase_batch",
                 batch_size=min(self.config.max_batch_size, self.current_batch_size + 2),
                 gpu_count=self.current_gpu_count
             )
        elif action_name == "reduce_batch":
             return Decision(
                 Action.REDUCE_BATCH, 
                 reason="RL: reduce_batch",
                 batch_size=max(1, self.current_batch_size - 1),
                 gpu_count=self.current_gpu_count
             )
        elif action_name == "scale_out":
             return Decision(
                 Action.SCALE_OUT, 
                 reason="RL: scale_out",
                 batch_size=self.current_batch_size,
                 gpu_count=min(self.config.max_gpus, self.current_gpu_count + 1)
             )
        elif action_name == "scale_in":
             return Decision(
                 Action.SCALE_IN, 
                 reason="RL: scale_in",
                 batch_size=self.current_batch_size,
                 gpu_count=max(1, self.current_gpu_count - 1)
             )
        elif action_name == "defer_niw":
             return Decision(
                 Action.DEFER_NIW, 
                 reason="RL: defer_niw",
                 batch_size=self.current_batch_size,
                 gpu_count=self.current_gpu_count
             )
        else:
             return Decision(Action.NO_ACTION, reason="RL: no_action")

    def get_stats(self) -> Dict[str, Any]:
        """Return controller statistics."""
        return {
            "decision_count": self.decision_count,
            "safety_overrides": self.safety_overrides,
            "override_rate": self.safety_overrides / max(1, self.decision_count)
        }
