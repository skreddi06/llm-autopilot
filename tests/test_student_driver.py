"""Student Driver Test - Turing Test for ML Controller.

Runs the Student (ML Controller) through the same load surge scenario
and compares performance to the Teacher (PredictiveController).
"""

import sys
import os
import time
import asyncio

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dataclasses import dataclass
from typing import List
from ml_controller import MLController
from models import Metrics, Decision, Action
from benchmarks import SCENARIOS, get_scenario_by_name


def generate_surge_scenario():
    """Generate same load surge as teacher test: 10→50→10 RPS."""
    steps = []
    
    # Stable period
    for t in range(0, 60, 2):
        steps.append({'time': t, 'rps': 10})
    
    # Ramp up
    for t, rate in [(60, 20), (62, 35), (64, 50)]:
        steps.append({'time': t, 'rps': rate})
    
    # Sustained spike
    for t in range(66, 90, 2):
        steps.append({'time': t, 'rps': 50})
    
    # Ramp down
    for t, rate in [(90, 35), (92, 20), (94, 10)]:
        steps.append({'time': t, 'rps': rate})
    
    # Recovery
    for t in range(96, 120, 2):
        steps.append({'time': t, 'rps': 10})
    
    return steps


async def run_student_eval():
    """Run Student Driver through load surge test."""
    print("=" * 60)
    print("🤖 STUDENT DRIVER TEST (Turing Test)")
    print("=" * 60)
    print("\nScenario: Same as Teacher test (10→50→10 RPS)")
    print("Objective: Max queue < 300, matching Teacher performance\n")
    
    # Initialize
    student = MLController("student_policy.pkl")
    scenario = generate_surge_scenario()
    
    # State tracking
    batch_size = 8
    gpu_count = 1
    queue = 5
    prev_queue = 5
    
    history = {
        'queue': [],
        'ttft': [],
        'actions': []
    }
    
    action_counts = {}
    
    for step in scenario:
        # Simulate metrics
        incoming = step['rps'] * 2
        capacity = batch_size * gpu_count * 0.5 * 2
        completed = min(queue + incoming, capacity)
        queue = max(0, queue + incoming - completed)
        
        velocity = (queue - prev_queue) / 2
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
            queue_depth_niw=0,
            niw_in_flight=0
        )
        
        # Student decision
        student.update_config(batch_size, gpu_count)
        decision = student.make_decision(metrics)
        
        # Apply decision
        if decision.batch_size:
            batch_size = decision.batch_size
        if decision.gpu_count:
            gpu_count = decision.gpu_count
        
        # Track history
        history['queue'].append(queue)
        history['ttft'].append(ttft)
        
        action_name = decision.action.value
        action_counts[action_name] = action_counts.get(action_name, 0) + 1
        
        if decision.action != Action.NO_ACTION:
            history['actions'].append((step['time'], action_name))
        
        # Print key moments
        if step['time'] >= 58 and step['time'] <= 100:
            action_str = f"→ {action_name}" if decision.action != Action.NO_ACTION else ""
            print(f"  T={step['time']:3d}s | RPS={step['rps']:2d} | Queue={queue:5.0f} | {action_str}")
    
    # Calculate results
    max_queue = max(history['queue'])
    avg_ttft_spike = sum(h for h, s in zip(history['ttft'], scenario) if 60 <= s['time'] <= 90) / max(1, sum(1 for s in scenario if 60 <= s['time'] <= 90))
    
    stats = student.get_stats()
    
    # Print results
    print("\n" + "=" * 60)
    print("📊 STUDENT REPORT CARD")
    print("=" * 60)
    
    print(f"\nPerformance:")
    print(f"  Max Queue: {max_queue:.0f} (Teacher was ~209)")
    print(f"  Avg TTFT (spike): {avg_ttft_spike:.0f}ms")
    
    print(f"\nActions taken:")
    for action, count in sorted(action_counts.items(), key=lambda x: -x[1]):
        print(f"  {action}: {count}")
    
    print(f"\nSafety Stats:")
    print(f"  Total decisions: {stats['decision_count']}")
    print(f"  Safety overrides: {stats['safety_overrides']}")
    print(f"  Override rate: {stats['override_rate']*100:.1f}%")
    
    # Grade
    print("\n" + "=" * 60)
    if max_queue < 300:
        print("✅ PASS: Student handled the surge without collapse!")
        print("   Ready for v0.7 Reinforcement Learning")
        return True
    else:
        print("❌ FAIL: Student lost control of the queue")
        print("   Need more training data or feature engineering")
        return False


if __name__ == "__main__":
    result = asyncio.run(run_student_eval())
    sys.exit(0 if result else 1)
