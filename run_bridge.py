"""
Reality Bridge Orchestrator (Phase v0.9)

Connects the HybridController (ppo_cloned_v09) to a vLLM-compatible endpoint.
Runs the full control loop: Metrics → Decision → Admission Control.
"""

import asyncio
import time
import numpy as np
from typing import Optional
from dataclasses import dataclass

from vllm_client import VLLMClient, VLLMConfig
from ml_controller import MLController
from models import Metrics, Decision, Action


@dataclass
class BridgeConfig:
    """Configuration for the Reality Bridge."""
    model_path: str = "ppo_cloned_v09"
    vllm_url: str = "http://localhost:8000"
    control_interval_sec: float = 0.5
    load_rps: float = 5.0  # Requests per second to generate
    duration_sec: int = 60
    surge_at_sec: int = 20
    surge_rps: float = 20.0
    surge_duration_sec: int = 10


async def generate_load(vllm: VLLMClient, rps: float, duration: float):
    """Background task to generate load at specified RPS."""
    interval = 1.0 / max(0.1, rps)
    end_time = time.time() + duration
    
    while time.time() < end_time:
        # Submit request (fire-and-forget style)
        asyncio.create_task(vllm.submit_request(
            prompt="Generate a detailed explanation of quantum computing.",
            max_tokens=64
        ))
        await asyncio.sleep(interval)


async def run_bridge(config: Optional[BridgeConfig] = None):
    """Main orchestrator: connects Controller to vLLM."""
    config = config or BridgeConfig()
    
    print("="*60)
    print("🌉 REALITY BRIDGE: Controller ↔ vLLM Integration")
    print("="*60)
    print(f"   Model: {config.model_path}")
    print(f"   vLLM: {config.vllm_url}")
    print(f"   Duration: {config.duration_sec}s")
    print(f"   Base Load: {config.load_rps} RPS")
    print(f"   Surge: {config.surge_rps} RPS at t={config.surge_at_sec}s")
    print("="*60)
    
    # Initialize
    vllm_config = VLLMConfig(base_url=config.vllm_url)
    vllm = VLLMClient(vllm_config)
    controller = MLController(model_path=config.model_path)
    
    await vllm.start()
    
    # Stats tracking
    history = []
    start_time = time.time()
    surge_active = False
    load_task: Optional[asyncio.Task] = None
    
    print("\n📊 Control Loop Started...")
    print(f"{'Time':>6s} | {'Queue':>5s} | {'KV%':>5s} | {'Concur':>6s} | {'Action':>15s} | {'Source':>12s}")
    print("-"*70)
    
    try:
        # Start baseline load
        load_task = asyncio.create_task(generate_load(vllm, config.load_rps, config.duration_sec))
        
        while (elapsed := time.time() - start_time) < config.duration_sec:
            # Check for surge trigger
            if not surge_active and elapsed >= config.surge_at_sec:
                surge_active = True
                print(f"\n⚠️  SURGE ACTIVATED: {config.surge_rps} RPS for {config.surge_duration_sec}s\n")
                asyncio.create_task(generate_load(vllm, config.surge_rps, config.surge_duration_sec))
            
            # === CONTROL LOOP ===
            
            # 1. OBSERVE: Get metrics from vLLM
            metrics = vllm.get_metrics()
            raw = vllm.get_raw_metrics()
            
            # 2. DECIDE: Controller makes decision
            decision = controller.make_decision(metrics)
            
            # 3. ACT: Apply to admission control
            vllm.apply_decision(decision)
            
            # Log
            source = "SHIELD" if "shield" in decision.reason.lower() else "AGENT"
            print(f"{elapsed:6.1f}s | {raw.num_requests_waiting:5d} | {raw.gpu_cache_usage_perc*100:5.1f} | {vllm.current_concurrency:6d} | {decision.action.value:>15s} | {source:>12s}")
            
            history.append({
                'time': elapsed,
                'queue': raw.num_requests_waiting,
                'kv_cache': raw.gpu_cache_usage_perc,
                'concurrency': vllm.current_concurrency,
                'action': decision.action.value,
                'source': source
            })
            
            await asyncio.sleep(config.control_interval_sec)
    
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
    
    finally:
        if load_task:
            load_task.cancel()
        await vllm.stop()
    
    # Final Report
    print("\n" + "="*60)
    print("📊 REALITY BRIDGE RESULTS")
    print("="*60)
    
    stats = vllm.get_stats()
    
    print(f"   Duration: {elapsed:.1f}s")
    print(f"   Requests Submitted: {stats['requests_submitted']}")
    print(f"   Requests Completed: {stats['requests_completed']}")
    print(f"   Requests Throttled: {stats['requests_throttled']}")
    print(f"   Final Concurrency: {stats['concurrency']}")
    
    # Controller stats (handle missing keys gracefully)
    total_decisions = getattr(controller, 'decision_count', 0)
    safety_overrides = getattr(controller, 'safety_overrides', 0)
    print(f"   Controller Decisions: {total_decisions}")
    print(f"   Shield Overrides: {safety_overrides}")
    
    # Calculate autonomy
    if total_decisions > 0:
        override_rate = (safety_overrides / total_decisions) * 100
        autonomy = 100 - override_rate
        print(f"   Autonomy Rate: {autonomy:.1f}%")
    
    print("="*60)
    
    return history


if __name__ == "__main__":
    asyncio.run(run_bridge())
