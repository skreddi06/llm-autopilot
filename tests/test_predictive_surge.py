"""Load Surge Test Scenario

Compares Reactive vs Predictive controller behavior during a traffic spike.

Scenario:
- Stable 10 RPS for 60s
- Spike to 50 RPS for 30s
- Return to 10 RPS

Expected:
- Reactive: Latency spikes, scale-out happens AFTER queue builds
- Predictive: Detects positive velocity early, scales BEFORE queue explodes
"""

import sys
import os
import time
import asyncio
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dataclasses import dataclass
from typing import List
from predictor import LoadPredictor
from controller_v2 import PredictiveController
from controller import Controller  # v0.4 reactive controller
from models import Metrics, Decision, Action


@dataclass
class ScenarioStep:
    """A step in the load scenario."""
    time_sec: float
    arrival_rate: float  # RPS


def generate_surge_scenario() -> List[ScenarioStep]:
    """Generate the load surge scenario.
    
    - 0-60s: 10 RPS (stable)
    - 60-90s: 50 RPS (spike)
    - 90-120s: 10 RPS (recovery)
    """
    steps = []
    
    # Stable period
    for t in range(0, 60, 2):
        steps.append(ScenarioStep(time_sec=t, arrival_rate=10))
    
    # Ramp up (gradual over 5s)
    for t, rate in [(60, 20), (62, 35), (64, 50)]:
        steps.append(ScenarioStep(time_sec=t, arrival_rate=rate))
    
    # Sustained spike
    for t in range(66, 90, 2):
        steps.append(ScenarioStep(time_sec=t, arrival_rate=50))
    
    # Ramp down
    for t, rate in [(90, 35), (92, 20), (94, 10)]:
        steps.append(ScenarioStep(time_sec=t, arrival_rate=rate))
    
    # Recovery
    for t in range(96, 120, 2):
        steps.append(ScenarioStep(time_sec=t, arrival_rate=10))
    
    return steps


def simulate_metrics(step: ScenarioStep, prev_queue: int, batch_size: int, gpu_count: int) -> tuple:
    """Simulate metrics based on arrival rate and config.
    
    Returns: (Metrics, new_queue, new_batch, new_gpu)
    """
    import time as t
    
    # Throughput capacity (rough model)
    throughput_capacity = batch_size * gpu_count * 0.5  # RPS
    
    # Queue dynamics
    incoming = step.arrival_rate * 2  # 2s interval
    completed = min(prev_queue + incoming, throughput_capacity * 2)
    new_queue = max(0, prev_queue + incoming - completed)
    
    # Queue velocity
    queue_velocity = (new_queue - prev_queue) / 2  # per second
    
    # TTFT (simplified: scales with queue)
    base_ttft = 50  # ms
    queue_factor = new_queue * 5  # 5ms per queued request
    ttft = base_ttft + queue_factor
    
    # GPU utilization
    gpu_util = min(100, (batch_size * 10) + (new_queue * 0.5))
    
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
        queue_depth=int(new_queue),
        timestamp=t.time(),
        queue_velocity=queue_velocity,
    )
    
    return metrics, int(new_queue), batch_size, gpu_count


async def run_controller_scenario(controller, scenario: List[ScenarioStep], name: str):
    """Run a controller through the scenario and collect metrics."""
    print(f"\n{'='*60}")
    print(f"Running: {name}")
    print(f"{'='*60}")
    
    # Initial state
    batch_size = 8
    gpu_count = 1
    queue = 5
    
    # Controller needs config for v0.4
    if hasattr(controller, 'update_config'):
        from models import ServerConfig
        controller.update_config(ServerConfig(batch_size=batch_size, gpu_count=gpu_count))
    
    results = []
    decisions = []
    scale_out_times = []
    
    for step in scenario:
        # Simulate metrics
        metrics, queue, batch_size, gpu_count = simulate_metrics(step, queue, batch_size, gpu_count)
        
        # Get controller decision
        decision = controller.make_decision(metrics)
        decisions.append(decision)
        
        # Apply controller decision
        if decision.batch_size:
            batch_size = decision.batch_size
        if decision.gpu_count:
            gpu_count = decision.gpu_count
            
        # Update config for v0.4 controller
        if hasattr(controller, 'update_config'):
            from models import ServerConfig
            controller.update_config(ServerConfig(batch_size=batch_size, gpu_count=gpu_count))
        
        # Track scale-out decisions
        if decision.action == Action.SCALE_OUT:
            scale_out_times.append(step.time_sec)
        
        results.append({
            'time': step.time_sec,
            'arrival': step.arrival_rate,
            'queue': metrics.queue_depth,
            'ttft': metrics.ttft_ms,
            'batch': batch_size,
            'gpus': gpu_count,
            'decision': decision.action.value if decision else 'none'
        })
        
        # Print key events
        if step.time_sec >= 58 and step.time_sec <= 100:
            action_str = f"→ {decision.action.value}" if decision.action != Action.NO_ACTION else ""
            print(f"  T={step.time_sec:3.0f}s | RPS={step.arrival_rate:2.0f} | Queue={metrics.queue_depth:3.0f} | TTFT={metrics.ttft_ms:6.0f}ms | {action_str}")
    
    # Summary
    max_queue = max(r['queue'] for r in results)
    max_ttft = max(r['ttft'] for r in results)
    avg_ttft_during_spike = sum(r['ttft'] for r in results if 60 <= r['time'] <= 90) / max(1, len([r for r in results if 60 <= r['time'] <= 90]))
    
    print(f"\n  Summary:")
    print(f"    Max Queue: {max_queue}")
    print(f"    Max TTFT: {max_ttft:.0f}ms")
    print(f"    Avg TTFT during spike: {avg_ttft_during_spike:.0f}ms")
    print(f"    Scale-out decisions: {len(scale_out_times)}")
    if scale_out_times:
        print(f"    First scale-out at: T={scale_out_times[0]}s")
    
    return {
        'name': name,
        'max_queue': max_queue,
        'max_ttft': max_ttft,
        'avg_spike_ttft': avg_ttft_during_spike,
        'scale_outs': len(scale_out_times),
        'first_scale_out': scale_out_times[0] if scale_out_times else None
    }


async def main():
    print("=" * 60)
    print("LOAD SURGE TEST: Reactive vs Predictive Controller")
    print("=" * 60)
    print("\nScenario: 10 RPS → 50 RPS spike → 10 RPS")
    
    scenario = generate_surge_scenario()
    
    # Run with Reactive Controller
    try:
        from controller import Controller
        reactive = Controller()
        reactive_results = await run_controller_scenario(reactive, scenario, "Reactive Controller (v0.4)")
    except Exception as e:
        print(f"Note: Reactive controller not available ({e})")
        reactive_results = None
    
    # Run with Predictive Controller
    predictive = PredictiveController()
    predictive_results = await run_controller_scenario(predictive, scenario, "Predictive Controller (v0.5-α)")
    
    # Comparison
    print("\n" + "=" * 60)
    print("COMPARISON RESULTS")
    print("=" * 60)
    
    if reactive_results:
        print(f"\n{'Metric':<25} {'Reactive':<15} {'Predictive':<15} {'Winner'}")
        print("-" * 60)
        
        queue_winner = "Predictive" if predictive_results['max_queue'] < reactive_results['max_queue'] else "Reactive"
        ttft_winner = "Predictive" if predictive_results['avg_spike_ttft'] < reactive_results['avg_spike_ttft'] else "Reactive"
        
        print(f"{'Max Queue':<25} {reactive_results['max_queue']:<15} {predictive_results['max_queue']:<15} {queue_winner}")
        print(f"{'Avg TTFT (spike)':<25} {reactive_results['avg_spike_ttft']:<15.0f} {predictive_results['avg_spike_ttft']:<15.0f} {ttft_winner}")
        print(f"{'First Scale-out':<25} {reactive_results['first_scale_out'] or 'N/A':<15} {predictive_results['first_scale_out'] or 'N/A':<15}")
        
        if predictive_results['avg_spike_ttft'] < reactive_results['avg_spike_ttft']:
            improvement = ((reactive_results['avg_spike_ttft'] - predictive_results['avg_spike_ttft']) / reactive_results['avg_spike_ttft']) * 100
            print(f"\n✅ Predictive Controller reduced spike TTFT by {improvement:.1f}%")
    else:
        print("\nPredictive Controller Results:")
        print(f"  Max Queue: {predictive_results['max_queue']}")
        print(f"  Avg TTFT (spike): {predictive_results['avg_spike_ttft']:.0f}ms")


if __name__ == "__main__":
    asyncio.run(main())
