"""
Phase 18: BC Agent Verification

Tests whether the BC-trained agent (ppo_cloned_v09) can correctly
handle critical states without Shield intervention.

Goal: >90% Autonomy Rate (Agent matches Shield behavior independently)
"""

import asyncio
import numpy as np
import pandas as pd
import time
from fake_llm_server import FakeLLMServer, InFlightRequest
from ml_controller import MLController, MLControllerConfig
from actuator import Actuator
from models import Metrics, Action, Decision
from stable_baselines3 import PPO

# Configuration
MODEL_PATH = "ppo_cloned_v09"  # The BC-trained agent
DURATION = 1000
SURGE_START = 200
SURGE_INTENSITY = 3.0
SURGE_DURATION = 100


class VerificationController:
    """
    A modified controller that lets the Agent predict FIRST.
    The Shield only overrides if the Agent is wrong in a critical state.
    Tracks "autonomy rate" - how often agent matches shield behavior.
    """
    
    def __init__(self, model_path: str):
        self.model = PPO.load(model_path)
        print(f"Loaded verification model: {model_path}")
        
        # Match ml_controller.py action mapping
        self.action_map = {0: "increase_batch", 1: "reduce_batch", 2: "defer_niw", 3: "no_action"}
        
        # Stats
        self.agent_correct = 0
        self.shield_overrides = 0
        self.total_critical = 0
        self.total_decisions = 0
        
        # Config
        self.current_batch_size = 8
        self.current_gpu_count = 1
        self.max_batch_size = 32
        self.max_gpus = 8
    
    def get_action(self, metrics: Metrics):
        """Get action with autonomy tracking."""
        self.total_decisions += 1
        
        # Calculate state
        queue_iw = getattr(metrics, 'queue_depth_iw', metrics.queue_depth)
        queue_niw = getattr(metrics, 'queue_depth_niw', 0)
        niw_in_flight = getattr(metrics, 'niw_in_flight', 0)
        
        # KV utilization approximation
        running_iw = min(queue_iw, self.current_batch_size * self.current_gpu_count)
        total_running = running_iw + niw_in_flight
        kv_util = min(1.0, total_running / (self.max_gpus * 16))
        
        # Build observation (must match training)
        obs = np.array([
            queue_iw / 50.0,
            queue_niw / 50.0,
            metrics.queue_velocity / 10.0,
            metrics.gpu_utilization / 100.0,  # Convert from 0-100 to 0-1
            kv_util,
            self.current_gpu_count / 8.0,
            self.current_batch_size / 32.0,
            0.0  # pending_gpus placeholder
        ], dtype=np.float32)
        
        # 1. Ask the Agent FIRST (Autonomous check)
        rl_action_idx, _ = self.model.predict(obs, deterministic=True)
        if hasattr(rl_action_idx, 'item'):
            rl_action_idx = int(rl_action_idx.item())
        rl_action_name = self.action_map[rl_action_idx]
        
        # 2. Check Shield Conditions (The "Correct Answer")
        shield_triggered = False
        shield_action = None
        
        # EMERGENCY: Extreme queue
        if queue_iw > 50:
            shield_triggered = True
            shield_action = "increase_batch"
        # RESCUE: High Queue + Available Memory
        elif queue_iw > 10 and kv_util < 0.85:
            shield_triggered = True
            shield_action = "increase_batch"
        # PANIC: Critical memory
        elif kv_util > 0.95:
            shield_triggered = True
            shield_action = "defer_niw"
        
        # 3. Determine Outcome
        if shield_triggered:
            self.total_critical += 1
            
            if rl_action_name == shield_action:
                self.agent_correct += 1
                return {
                    'action_name': rl_action_name,
                    'source': 'RL_AGENT_CORRECT',
                    'reason': f'Agent matched Shield (Queue={queue_iw}, Mem={kv_util:.2f})',
                    'action_idx': rl_action_idx
                }
            else:
                self.shield_overrides += 1
                # Return shield action (override)
                shield_idx = {v: k for k, v in self.action_map.items()}[shield_action]
                return {
                    'action_name': shield_action,
                    'source': 'SHIELD_OVERRIDE',
                    'reason': f'Shield corrected Agent (wanted {rl_action_name})',
                    'action_idx': shield_idx
                }
        
        # Normal operation (non-critical states)
        return {
            'action_name': rl_action_name,
            'source': 'RL_AGENT',
            'reason': 'Normal operation',
            'action_idx': rl_action_idx
        }
    
    def get_autonomy_rate(self):
        if self.total_critical == 0:
            return 0.0
        return (self.agent_correct / self.total_critical) * 100


async def run_verification():
    print("="*60)
    print("🔬 PHASE 18: BC AGENT VERIFICATION")
    print(f"   Model: {MODEL_PATH}")
    print(f"   Goal: >90% Autonomy Rate")
    print("="*60)
    
    # Initialize
    server = FakeLLMServer()
    actuator = Actuator("http://localhost:8000")
    controller = VerificationController(model_path=MODEL_PATH)
    
    # Start server
    runner = await server.start()
    server.auto_load_enabled = True
    
    metrics_history = []
    
    print("\n📊 Starting Verification Run...")
    
    try:
        for step in range(DURATION):
            # Inject Surge at designated time
            if step == SURGE_START:
                print(f"\n⚠️  SURGE INJECTED at step {step}")
                server.inject_surge(SURGE_INTENSITY, SURGE_DURATION)
                # Also manually inject requests
                for i in range(100):
                    req = InFlightRequest(
                        id=server.request_counter + i,
                        state='queued',
                        arrival_time=time.time(),
                        priority=0
                    )
                    req.prompt_tokens = 128
                    req.max_tokens = 128
                    req.prefill_tokens_processed = 0
                    req.generated_tokens = 0
                    server.queued_requests.append(req)
                server.request_counter += 100
            
            # Get metrics
            q_depth = len(server.queued_requests)
            q_iw = len([r for r in server.queued_requests if r.priority == 0])
            q_niw = len([r for r in server.queued_requests if r.priority > 0])
            active_reqs = len(server.prefilling_requests) + len(server.decoding_requests)
            capacity = server.config.batch_size * server.config.gpu_count
            gpu_util = min(100.0, (active_reqs / max(1, capacity)) * 100)
            kv_util = server.kv_utilization if hasattr(server, 'kv_utilization') else 0.0
            
            current_metrics = Metrics(
                ttft_ms=0.0,
                inter_token_latency_ms=0.0,
                prefill_latency_ms=0.0,
                decode_latency_ms=0.0,
                gpu_utilization=gpu_util,
                memory_efficiency=kv_util * 100,
                gpu_balance_index=1.0,
                comm_bubble_ratio=0.0,
                speculative_factor=0.0,
                queue_depth=q_depth,
                timestamp=time.time(),
                queue_velocity=0.0,
                queue_depth_iw=q_iw,
                queue_depth_niw=q_niw,
                niw_in_flight=active_reqs
            )
            
            # Get decision
            decision = controller.get_action(current_metrics)
            
            # Apply decision
            await actuator.apply_decision(Decision(
                action=Action(decision['action_name']),
                reason=decision['reason']
            ))
            
            # Log
            metrics_history.append({
                'step': step,
                'queue': q_depth,
                'source': decision['source'],
                'action': decision['action_name']
            })
            
            # Progress
            if step % 100 == 0:
                autonomy = controller.get_autonomy_rate()
                print(f"   Step {step:4d} | Queue: {q_depth:4d} | Action: {decision['action_name']:15s} | Source: {decision['source']:20s} | Autonomy: {autonomy:.1f}%")
            
            await asyncio.sleep(0.01)
    
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted")
    
    finally:
        await server.stop()
        await runner.cleanup()
        await actuator.cleanup()
    
    # Final Analysis
    print("\n" + "="*60)
    print("📊 VERIFICATION RESULTS")
    print("="*60)
    
    df = pd.DataFrame(metrics_history)
    
    final_queue = df['queue'].iloc[-1] if len(df) > 0 else 0
    survived = final_queue < 500
    
    print(f"   Total Decisions: {controller.total_decisions}")
    print(f"   Critical States: {controller.total_critical}")
    print(f"   Agent Correct: {controller.agent_correct}")
    print(f"   Shield Overrides: {controller.shield_overrides}")
    print(f"   AUTONOMY RATE: {controller.get_autonomy_rate():.1f}%")
    print(f"   Final Queue: {final_queue}")
    print(f"   Survived: {'✅ YES' if survived else '❌ NO'}")
    
    # Action distribution
    print("\n   Action Distribution:")
    print(df['action'].value_counts().to_string())
    
    print("\n   Source Distribution:")
    print(df['source'].value_counts().to_string())
    
    if controller.get_autonomy_rate() >= 90:
        print("\n🎉 SUCCESS! Agent has learned autonomous 'Rescue' behavior!")
    else:
        print("\n⚠️  Agent still needs Shield assistance. More training required.")
    
    print("="*60)


if __name__ == "__main__":
    asyncio.run(run_verification())
