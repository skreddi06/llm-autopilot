"""LLM Server Gym Environment v0.8 - Stress Test Edition

Key changes from v0.7:
1. PROVISIONING LAG: SCALE_OUT takes 10+ steps to activate (simulating cold start)
2. QUEUE ACCELERATION PENALTY: Squared penalty for rising queue
3. NIW INTERFERENCE COST: Penalty when NIW blocks IW during stress
4. LONGER EPISODES: 500 steps for better credit assignment
5. MORE FREQUENT SURGES: 40% surge probability to force learning

These changes force the agent to discover DEFER_NIW as a survival mechanism.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random


class LLMServerEnvV08(gym.Env):
    """v0.8 Stress Test environment - forces preventive control learning."""
    
    metadata = {"render_modes": ["human"]}
    
    def __init__(self, render_mode=None):
        super().__init__()
        
        self.action_space = spaces.Discrete(6)
        self.observation_space = spaces.Box(
            low=np.array([0, 0, -100, 0, 0, 1, 1, 0]),  # Added pending_gpus
            high=np.array([1000, 1000, 100, 100, 1, 16, 64, 16]),
            dtype=np.float32
        )
        
        # v0.8: Longer episodes for credit assignment
        self.sim_duration = 500
        self.slo_target_ms = 200
        
        # v0.8b: SCARCITY CONSTRAINT - force DEFER_NIW discovery
        self.MAX_GPUS = 6  # Moderate cap - enough pressure but survivable
        
        # v0.8b: Provisioning lag (simulates cold start)
        self.PROVISIONING_LAG = 30  # Steps to activate new GPU (severe cold start)
        
        # State
        self.current_time = 0
        self.num_gpus = 1
        self.pending_gpus = 0  # GPUs being provisioned (not yet active)
        self.gpu_activation_time = []  # When each pending GPU activates
        self.batch_size = 8
        self.queue_iw = 0
        self.queue_niw = 0
        self.prev_queue = 0
        self.prev_velocity = 0  # For acceleration calculation
        self.niw_in_flight = 0
        self.kv_util = 0.0
        
        # Tracking
        self.total_throughput = 0
        self.total_latency_violations = 0
        self.episode_actions = []
        self.surge_count = 0
        self.survived_surges = 0
        
        self.render_mode = render_mode
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.current_time = 0
        self.num_gpus = 1
        self.pending_gpus = 0
        self.gpu_activation_time = []
        self.batch_size = 8
        self.queue_iw = 5
        self.queue_niw = 2
        self.prev_queue = 5
        self.prev_velocity = 0
        self.niw_in_flight = 5  # v0.8: More NIW to create contention
        self.kv_util = 0.1
        
        self.total_throughput = 0
        self.total_latency_violations = 0
        self.episode_actions = []
        self.surge_count = 0
        self.survived_surges = 0
        
        return self._get_obs(), {}
    
    def _get_obs(self):
        queue_velocity = self.queue_iw - self.prev_queue
        gpu_util = min(100, self.batch_size * 10 + self.queue_iw * 0.5)
        
        return np.array([
            self.queue_iw,
            self.queue_niw,
            queue_velocity,
            gpu_util,
            self.kv_util,
            self.num_gpus,
            self.batch_size,
            self.pending_gpus  # v0.8: Agent sees pending GPUs
        ], dtype=np.float32)
    
    def step(self, action_idx):
        self.episode_actions.append(action_idx)
        
        # === 1. CHECK GPU PROVISIONING ===
        # v0.8: Activate GPUs that finished provisioning
        activated = 0
        remaining = []
        for activation_time in self.gpu_activation_time:
            if self.current_time >= activation_time:
                activated += 1
            else:
                remaining.append(activation_time)
        self.gpu_activation_time = remaining
        self.num_gpus += activated
        self.pending_gpus -= activated
        
        # === 2. APPLY ACTION ===
        self._apply_action(action_idx)
        
        # === 3. INJECT TRAFFIC ===
        # v0.8: 40% surge probability (up from 20%)
        is_surge = random.random() < 0.4
        if is_surge:
            iw_arrivals = random.randint(40, 80)  # Bigger surges
            self.surge_count += 1
        else:
            iw_arrivals = random.randint(5, 15)
        
        # NIW: MORE AGGRESSIVE to create real contention (50% of steps)
        niw_arrivals = random.randint(3, 8) if random.random() < 0.5 else 0
        
        self.queue_iw += iw_arrivals
        self.queue_niw += niw_arrivals
        
        # === 4. PROCESS REQUESTS ===
        # v0.8: NIW in_flight reduces effective capacity (interference)
        effective_capacity = self.batch_size * self.num_gpus * 0.5
        interference_factor = 1.0 - (self.niw_in_flight * 0.05)  # Each NIW steals 5%
        effective_capacity *= max(0.5, interference_factor)
        
        processed = min(self.queue_iw + self.queue_niw, effective_capacity)
        
        iw_processed = min(self.queue_iw, processed)
        self.queue_iw = max(0, self.queue_iw - iw_processed)
        
        niw_processed = min(self.queue_niw, processed - iw_processed)
        self.queue_niw = max(0, self.queue_niw - niw_processed)
        
        self.niw_in_flight = max(0, min(15, self.niw_in_flight + niw_arrivals - niw_processed))
        
        # === 5. CALCULATE METRICS ===
        queue_velocity = self.queue_iw - self.prev_queue
        queue_acceleration = queue_velocity - self.prev_velocity
        self.prev_velocity = queue_velocity
        self.prev_queue = self.queue_iw
        
        ttft = 50 + self.queue_iw * 5
        throughput = effective_capacity * 0.8
        
        self.total_throughput += throughput
        
        # === 6. CALCULATE REWARD (v0.8 SHAPED) ===
        
        # A: Throughput (good)
        r_throughput = throughput * 0.01
        
        # B: Latency violation (bad) - INCREASED PENALTY
        violation = max(0, ttft - self.slo_target_ms)
        r_latency = -2.0 * (violation / 100.0)  # v0.8: 2x penalty
        if violation > 0:
            self.total_latency_violations += 1
        
        # C: v0.8 CRITICAL - QUADRATIC GPU COST (prevents "throw GPUs at it")
        # GPUs are VERY expensive - agent must find another way
        r_cost = -2.0 * (self.num_gpus ** 1.5)  # Quadratic scaling!
        
        # D: v0.8 NEW - QUEUE ACCELERATION PENALTY (squared)
        # Punishes "queue is rising faster" - forces preventive action
        r_accel = -0.3 * (max(0, queue_acceleration) ** 2)
        
        # E: v0.8 NEW - NIW INTERFERENCE COST
        # Punishes having NIW in flight when queue is rising
        if queue_velocity > 0 and self.niw_in_flight > 0:
            r_interference = -1.5 * self.niw_in_flight
        else:
            r_interference = 0
        
        # F: v0.8 NEW - SURGE SURVIVAL BONUS
        # Big reward for handling a surge without violation
        r_survival = 0
        if is_surge and violation == 0:
            r_survival = +10.0
            self.survived_surges += 1
        
        # G: Memory crash (catastrophic)
        # Fix v0.9: KV Cache depends on RUNNING requests (Batch + NIW), not Queue!
        running_iw = min(self.queue_iw, self.batch_size * self.num_gpus)
        total_running = running_iw + self.niw_in_flight
        
        # CURRICULUM LEARNING (Phase 14)
        # Check for 'curriculum_stage.txt'
        try:
            with open("curriculum_stage.txt", "r") as f:
                stage = int(f.read().strip())
        except:
            stage = 2  # Default to hard mode
            
        # Capacity model
        # soft_cap = 80% usage
        base_util = total_running / (self.MAX_GPUS * 16)
        
        if stage == 1:
            # Stage 1: "Easy Mode" - Memory is cheap!
            # Reduce utilized % by 50% so they rarely crash, encouraging INC_BATCH
            self.kv_util = min(0.9, base_util * 0.5) 
        else:
            # Stage 2: "Hard Mode" - Real Physics
            self.kv_util = min(1.0, base_util)

        # Add a penalty if queue is huge (CPU RAM stress) but rare to crash
        if self.kv_util >= 1.0:
            return self._get_obs(), -1000, True, False, {"crash": True}
        
        # H: v0.8c - DEFER_NIW BONUS REMOVED
        # (It was causing the agent to spam DEFER to farm rewards while ignoring the queue)
        r_defer_bonus = 0.0
        
        # I: v0.8c - PUNISH SCALE_OUT when NIW could be deferred
        # Force agent to exhaust cheaper options first (SageServe principle)
        r_scaleout_penalty = 0
        if action_idx == 3 and self.niw_in_flight > 0 and queue_velocity > 0:
            r_scaleout_penalty = -30.0  # "Why scale when you could defer?"
        
        # J: v0.8d - ACTION PARADOX FIX (Phase 13)
        # Penalize DEC_BATCH during surges - this is suicidal behavior
        # Reward INC_BATCH during surges - this drains the queue
        r_action_paradox = 0
        if queue_velocity > 5:  # Queue is growing fast
            if action_idx == 2:  # DEC_BATCH during surge = death spiral
                r_action_paradox = -100.0  # MASSIVE penalty - this kills you
            elif action_idx == 1:  # INC_BATCH during surge = survival
                r_action_paradox = +30.0  # Reward for correct behavior
        
        # K: v0.9 - PHASE 14: CLEAR & FILL COMBO REWARD
        # State-dependent: INC_BATCH when memory safe, DEFER when memory critical
        r_combo = 0
        
        # Condition A: Queue high but memory safe (< 60% KV usage)
        if self.queue_iw > 10 and self.kv_util < 0.6:
            if action_idx == 1:  # INC_BATCH
                r_combo = +80.0  # Big reward
            elif action_idx == 5:  # DEFER_NIW
                r_combo = -40.0
                
        # Condition B: Memory dangerous (> 80% KV usage)
        if self.kv_util > 0.8:
            if action_idx == 5:
                r_combo = +30.0
            elif action_idx == 1:
                r_combo = -60.0

        # L: v0.95 - PHASE 15: CONSTRAINT ENFORCEMENT (Action Masking -> Teacher Forcing)
        # "Soft Masking" failed (ignored penalty). "Hard Termination" failed (instant death).
        # Solution: "Teacher Forcing". If agent fails, we TAKE the wheel, but give low reward.
        r_constraint = 0
        if self.queue_iw > 10 and self.kv_util < 0.6:
            if action_idx in [0, 5]:  # NO_OP or DEFER
                # OVERRIDE ACTION to INC_BATCH
                action_idx = 1
                r_constraint = -10.0  # Small penalty for needing help
                # But we do NOT terminate. We let the 'good result' of INC_BATCH happen.
                # Gradient: Doing it yourself (+80) > Needing help (-10).
                # This ensures the state trajectory (Queue Draining) is experienced!

        reward = r_throughput + r_latency + r_cost + r_accel + r_interference + r_survival + r_defer_bonus + r_scaleout_penalty + r_action_paradox + r_combo + r_constraint

        # L: v0.95 - PHASE 15: CONSTRAINT ENFORCEMENT (Action Masking)
        # Force the agent to work when safe.
        # If Queue > 10 and Memory < 60%, DEFER/NO_OP ARE FORBIDDEN.
        r_constraint = 0
        if self.queue_iw > 10 and self.kv_util < 0.6:
            if action_idx in [0, 5]:  # NO_OP or DEFER
                r_constraint = -1000.0  # "Action Mask" via penalty
                # We do NOT return True for terminated, we just punish heavily.
                # This makes these actions strictly dominated by any other action.

        reward = r_throughput + r_latency + r_cost + r_accel + r_interference + r_survival + r_defer_bonus + r_scaleout_penalty + r_action_paradox + r_combo + r_constraint
        
        reward = r_throughput + r_latency + r_cost + r_accel + r_interference + r_survival + r_defer_bonus + r_scaleout_penalty + r_action_paradox + r_combo
        
        # === 7. TERMINATION ===
        self.current_time += 1
        terminated = self.current_time >= self.sim_duration
        
        info = {
            "ttft": ttft,
            "throughput": throughput,
            "queue_velocity": queue_velocity,
            "queue_accel": queue_acceleration,
            "niw_interference": r_interference,
            "is_surge": is_surge,
            "violation": violation > 0
        }
        
        return self._get_obs(), reward, terminated, False, info
    
    def _apply_action(self, action_idx):
        if action_idx == 0:  # NO_OP
            pass
        elif action_idx == 1:  # INCREASE_BATCH
            self.batch_size = min(64, self.batch_size + 4)
        elif action_idx == 2:  # DECREASE_BATCH
            self.batch_size = max(1, self.batch_size - 2)
        elif action_idx == 3:  # SCALE_OUT
            # v0.8b: PROVISIONING LAG + HARD CAP
            if self.num_gpus + self.pending_gpus < self.MAX_GPUS:
                self.pending_gpus += 1
                self.gpu_activation_time.append(self.current_time + self.PROVISIONING_LAG)
        elif action_idx == 4:  # SCALE_IN
            if self.num_gpus > 1:
                self.num_gpus -= 1
        elif action_idx == 5:  # DEFER_NIW
            # v0.8: More aggressive deferral (5 instead of 3)
            deferred = min(5, self.niw_in_flight)
            self.niw_in_flight -= deferred
            self.queue_niw += deferred
    
    def get_episode_stats(self):
        action_counts = {}
        action_names = ["NO_OP", "INC_BATCH", "DEC_BATCH", "SCALE_OUT", "SCALE_IN", "DEFER_NIW"]
        for a in self.episode_actions:
            name = action_names[a]
            action_counts[name] = action_counts.get(name, 0) + 1
        
        return {
            "total_throughput": self.total_throughput,
            "latency_violations": self.total_latency_violations,
            "action_counts": action_counts,
            "final_gpus": self.num_gpus,
            "final_batch": self.batch_size,
            "surges": self.surge_count,
            "survived_surges": self.survived_surges,
            "survival_rate": self.survived_surges / max(1, self.surge_count)
        }


if __name__ == "__main__":
    print("Testing LLMServerEnvV08 (Stress Test)...")
    env = LLMServerEnvV08()
    obs, _ = env.reset()
    print(f"Obs shape: {obs.shape}")
    
    total_reward = 0
    for _ in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated:
            break
    
    print(f"Total reward: {total_reward:.2f}")
    print(f"Stats: {env.get_episode_stats()}")
