"""Comparative Benchmark - All Controller Generations

Pits the three generations of controllers against each other:
- Reactive (v0.4): Simple threshold-based rules
- Predictive (v0.5): Expert policy with SageServe heuristics
- PPO RL (v0.7): Learned policy from reward optimization

All run on the same deterministic load surge scenario.
"""

import sys
import os
import time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataclasses import dataclass
from typing import Dict, List, Any
import numpy as np

from controller import Controller as ReactiveController
from controller_v2 import PredictiveController
from models import Metrics, Action


@dataclass
class BenchmarkResult:
    """Results from a controller benchmark run."""
    controller_name: str
    max_queue: int
    avg_ttft: float
    max_ttft: float
    latency_violations: int
    scale_outs: int
    defer_niw: int
    final_gpus: int
    final_batch: int
    total_actions: int


def generate_deterministic_scenario():
    """Generate deterministic load surge for fair comparison."""
    steps = []
    
    # Warm-up: 30s at 10 RPS
    for t in range(0, 30, 1):
        steps.append({'time': t, 'rps': 10, 'niw': 3})
    
    # Ramp up: 10s from 10 to 50 RPS
    for t in range(30, 40, 1):
        rps = 10 + (t - 30) * 4
        steps.append({'time': t, 'rps': rps, 'niw': 5})
    
    # Sustained spike: 30s at 50 RPS with NIW pressure
    for t in range(40, 70, 1):
        steps.append({'time': t, 'rps': 50, 'niw': 8})
    
    # Ramp down: 10s from 50 to 10 RPS
    for t in range(70, 80, 1):
        rps = 50 - (t - 70) * 4
        steps.append({'time': t, 'rps': rps, 'niw': 5})
    
    # Recovery: 20s at 10 RPS
    for t in range(80, 100, 1):
        steps.append({'time': t, 'rps': 10, 'niw': 2})
    
    return steps


def run_reactive_controller(scenario: List[Dict]) -> BenchmarkResult:
    """Run reactive controller (v0.4)."""
    controller = ReactiveController()
    return _run_controller(controller, "Reactive (v0.4)", scenario, is_reactive=True)


def run_predictive_controller(scenario: List[Dict]) -> BenchmarkResult:
    """Run predictive controller (v0.5)."""
    controller = PredictiveController()
    return _run_controller(controller, "Predictive (v0.5)", scenario, is_reactive=False)


def run_ppo_controller(scenario: List[Dict]) -> BenchmarkResult:
    """Run PPO RL controller (v0.7)."""
    try:
        from stable_baselines3 import PPO
        agent = PPO.load("ppo_autopilot_v07")
    except Exception as e:
        print(f"Failed to load PPO model: {e}")
        return None
    
    return _run_ppo(agent, "PPO RL (v0.7)", scenario)


def _run_controller(controller, name: str, scenario: List[Dict], is_reactive: bool) -> BenchmarkResult:
    """Run a standard controller through the scenario."""
    batch_size = 8
    gpu_count = 1
    queue = 5
    prev_queue = 5
    niw_in_flight = 3
    
    max_queue = 0
    ttfts = []
    violations = 0
    scale_outs = 0
    defer_niw = 0
    total_actions = 0
    
    if hasattr(controller, 'update_config'):
        controller.update_config(batch_size, gpu_count)
    
    for step in scenario:
        # Simulate queue dynamics
        incoming = step['rps']
        capacity = batch_size * gpu_count * 0.5
        processed = min(queue + incoming, capacity)
        queue = max(0, queue + incoming - processed)
        
        velocity = queue - prev_queue
        prev_queue = queue
        
        ttft = 50 + queue * 5
        gpu_util = min(100, batch_size * 10 + queue * 0.5)
        
        metrics = Metrics(
            ttft_ms=ttft,
            inter_token_latency_ms=15.0,
            prefill_latency_ms=100.0,
            decode_latency_ms=15.0,
            gpu_utilization=gpu_util,
            memory_efficiency=0.8,
            gpu_balance_index=0.9,
            comm_bubble_ratio=0.1,
            speculative_factor=0.3,
            queue_depth=int(queue),
            timestamp=time.time(),
            queue_velocity=velocity,
            queue_depth_iw=int(queue),
            queue_depth_niw=step['niw'],
            niw_in_flight=niw_in_flight
        )
        
        # Get decision
        decision = controller.make_decision(metrics)
        
        # Apply decision
        if decision and decision.action != Action.NO_ACTION:
            total_actions += 1
            if decision.batch_size:
                batch_size = decision.batch_size
            if decision.gpu_count:
                gpu_count = decision.gpu_count
            
            if decision.action == Action.SCALE_OUT:
                scale_outs += 1
            elif decision.action == Action.DEFER_NIW:
                defer_niw += 1
                niw_in_flight = max(0, niw_in_flight - 2)
        
        if hasattr(controller, 'update_config'):
            controller.update_config(batch_size, gpu_count)
        
        # Track metrics
        max_queue = max(max_queue, queue)
        ttfts.append(ttft)
        if ttft > 200:
            violations += 1
    
    return BenchmarkResult(
        controller_name=name,
        max_queue=int(max_queue),
        avg_ttft=np.mean(ttfts),
        max_ttft=max(ttfts),
        latency_violations=violations,
        scale_outs=scale_outs,
        defer_niw=defer_niw,
        final_gpus=gpu_count,
        final_batch=batch_size,
        total_actions=total_actions
    )


def _run_ppo(agent, name: str, scenario: List[Dict]) -> BenchmarkResult:
    """Run PPO agent through the scenario."""
    batch_size = 8
    gpu_count = 1
    queue = 5
    prev_queue = 5
    niw_in_flight = 3
    
    max_queue = 0
    ttfts = []
    violations = 0
    scale_outs = 0
    defer_niw = 0
    total_actions = 0
    
    action_names = ["NO_OP", "INC_BATCH", "DEC_BATCH", "SCALE_OUT", "SCALE_IN", "DEFER_NIW"]
    
    for step in scenario:
        # Simulate queue dynamics
        incoming = step['rps']
        capacity = batch_size * gpu_count * 0.5
        processed = min(queue + incoming, capacity)
        queue = max(0, queue + incoming - processed)
        
        velocity = queue - prev_queue
        prev_queue = queue
        
        ttft = 50 + queue * 5
        gpu_util = min(100, batch_size * 10 + queue * 0.5)
        kv_util = min(1.0, queue * 0.01)
        
        # Create observation vector (must match llm_env.py)
        obs = np.array([
            queue,           # queue_iw
            step['niw'],     # queue_niw
            velocity,        # velocity
            gpu_util,        # gpu_util
            kv_util,         # kv_util
            gpu_count,       # num_gpus
            batch_size       # batch_size
        ], dtype=np.float32)
        
        # Get PPO action
        action_idx, _ = agent.predict(obs, deterministic=True)
        action_idx = int(action_idx)
        
        # Apply action
        if action_idx != 0:  # Not NO_OP
            total_actions += 1
            
            if action_idx == 1:  # INC_BATCH
                batch_size = min(64, batch_size + 4)
            elif action_idx == 2:  # DEC_BATCH
                batch_size = max(1, batch_size - 2)
            elif action_idx == 3:  # SCALE_OUT
                gpu_count = min(16, gpu_count + 1)
                scale_outs += 1
            elif action_idx == 4:  # SCALE_IN
                gpu_count = max(1, gpu_count - 1)
            elif action_idx == 5:  # DEFER_NIW
                niw_in_flight = max(0, niw_in_flight - 2)
                defer_niw += 1
        
        # Track metrics
        max_queue = max(max_queue, queue)
        ttfts.append(ttft)
        if ttft > 200:
            violations += 1
    
    return BenchmarkResult(
        controller_name=name,
        max_queue=int(max_queue),
        avg_ttft=np.mean(ttfts),
        max_ttft=max(ttfts),
        latency_violations=violations,
        scale_outs=scale_outs,
        defer_niw=defer_niw,
        final_gpus=gpu_count,
        final_batch=batch_size,
        total_actions=total_actions
    )


def print_results(results: List[BenchmarkResult]):
    """Print comparison table."""
    print("\n" + "=" * 80)
    print("COMPARATIVE BENCHMARK RESULTS")
    print("=" * 80)
    print(f"\n{'Metric':<20} ", end="")
    for r in results:
        print(f"{r.controller_name:<18} ", end="")
    print()
    print("-" * 80)
    
    metrics = [
        ("Max Queue", lambda r: r.max_queue),
        ("Avg TTFT (ms)", lambda r: f"{r.avg_ttft:.0f}"),
        ("Max TTFT (ms)", lambda r: f"{r.max_ttft:.0f}"),
        ("SLA Violations", lambda r: r.latency_violations),
        ("Scale Outs", lambda r: r.scale_outs),
        ("Defer NIW", lambda r: r.defer_niw),
        ("Final GPUs", lambda r: r.final_gpus),
        ("Final Batch", lambda r: r.final_batch),
        ("Total Actions", lambda r: r.total_actions),
    ]
    
    for name, fn in metrics:
        print(f"{name:<20} ", end="")
        for r in results:
            val = fn(r)
            print(f"{val:<18} ", end="")
        print()
    
    # Determine winner
    print("\n" + "=" * 80)
    print("WINNER ANALYSIS")
    print("=" * 80)
    
    # Queue control
    min_queue = min(r.max_queue for r in results)
    queue_winner = [r.controller_name for r in results if r.max_queue == min_queue]
    print(f"Best Queue Control: {', '.join(queue_winner)} (max queue = {min_queue})")
    
    # Cost efficiency
    min_gpus = min(r.final_gpus for r in results)
    cost_winner = [r.controller_name for r in results if r.final_gpus == min_gpus]
    print(f"Most Cost Efficient: {', '.join(cost_winner)} (GPUs = {min_gpus})")
    
    # SageServe behavior
    defer_leaders = [r for r in results if r.defer_niw > 0]
    if defer_leaders:
        max_defer = max(r.defer_niw for r in defer_leaders)
        sageserve = [r.controller_name for r in results if r.defer_niw == max_defer]
        print(f"SageServe Leader: {', '.join(sageserve)} ({max_defer} DEFER_NIW)")
    else:
        print("SageServe Leader: None (no DEFER_NIW actions)")


def main():
    print("=" * 80)
    print("🏁 FINAL BENCHMARK: Reactive vs Predictive vs PPO")
    print("=" * 80)
    print("\nScenario: Deterministic Load Surge (10→50→10 RPS, 100s)")
    
    scenario = generate_deterministic_scenario()
    results = []
    
    # Run reactive
    print("\n1. Running Reactive Controller (v0.4)...")
    try:
        r1 = run_reactive_controller(scenario)
        results.append(r1)
        print(f"   ✓ Max Queue: {r1.max_queue}, Violations: {r1.latency_violations}")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
    
    # Run predictive
    print("\n2. Running Predictive Controller (v0.5)...")
    try:
        r2 = run_predictive_controller(scenario)
        results.append(r2)
        print(f"   ✓ Max Queue: {r2.max_queue}, Violations: {r2.latency_violations}")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
    
    # Run PPO
    print("\n3. Running PPO RL Controller (v0.7)...")
    try:
        r3 = run_ppo_controller(scenario)
        if r3:
            results.append(r3)
            print(f"   ✓ Max Queue: {r3.max_queue}, Violations: {r3.latency_violations}")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
    
    # Print comparison
    if results:
        print_results(results)
    else:
        print("\n⚠️ No controllers completed successfully")


if __name__ == "__main__":
    main()
