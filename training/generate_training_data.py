"""Training Data Generator for RL Agent.

Generates the "Golden Dataset" by running the Expert Controller (PredictiveController)
against mixed IW/NIW traffic scenarios.

Output: training_data.jsonl with State-Action-Reward tuples
"""

import json
import random
import time
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dataclasses import asdict
from typing import Dict, Any
from fake_llm_server import FakeLLMServer, InFlightRequest, ServerConfig
from controller_v2 import PredictiveController
from models import Metrics, Action
from benchmarks import SCENARIOS, TrafficScenario


# Configuration
OUTPUT_FILE = "training_data.jsonl"
SIMULATION_HOURS = 48  # Accelerated time (runs in seconds)
TICKS_PER_HOUR = 200   # Increased: 200 ticks = 1 simulated hour (~9600+ ticks total)


def create_metrics_from_server(server: FakeLLMServer, queue_velocity: float) -> Metrics:
    """Create Metrics object from server state."""
    # Count IW vs NIW
    iw_count = sum(1 for r in server.queued_requests if r.priority == 0)
    niw_count = sum(1 for r in server.queued_requests if r.priority >= 1)
    niw_in_flight = sum(1 for r in (server.prefilling_requests + server.decoding_requests) if r.priority >= 1)
    
    # Estimate TTFT from queue depth
    queue_depth = len(server.queued_requests)
    ttft = 50 + queue_depth * 5  # Simplified model
    
    # GPU utilization from batch size
    gpu_util = min(100, server.config.batch_size * 10 + queue_depth * 0.5)
    
    return Metrics(
        ttft_ms=ttft,
        inter_token_latency_ms=15.0,
        prefill_latency_ms=100.0,
        decode_latency_ms=15.0,
        gpu_utilization=gpu_util,
        memory_efficiency=0.8,
        gpu_balance_index=0.9,
        comm_bubble_ratio=0.1,
        speculative_factor=0.3,
        queue_depth=queue_depth,
        timestamp=time.time(),
        queue_velocity=queue_velocity,
        queue_depth_iw=iw_count,
        queue_depth_niw=niw_count,
        niw_in_flight=niw_in_flight
    )


def run_data_generation():
    """Run the data generation loop."""
    print("=" * 60)
    print("TRAINING DATA GENERATOR")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Simulated hours: {SIMULATION_HOURS}")
    print(f"  Ticks per hour: {TICKS_PER_HOUR}")
    print(f"  Output file: {OUTPUT_FILE}")
    print()
    
    # Initialize server and controller
    server_config = ServerConfig(batch_size=8, gpu_count=1)
    server = FakeLLMServer(server_config)
    controller = PredictiveController()
    
    # Statistics
    stats = {
        "total_ticks": 0,
        "total_requests": 0,
        "iw_requests": 0,
        "niw_requests": 0,
        "actions": {},
        "avg_queue_depth": 0,
        "max_queue_depth": 0
    }
    
    request_id = 0
    current_time = 0.0
    prev_queue_depth = 0
    
    with open(OUTPUT_FILE, "w") as f:
        for hour in range(SIMULATION_HOURS):
            # Pick a scenario for this hour
            scenario = random.choice(SCENARIOS)
            print(f"Hour {hour:2d}: {scenario.name} ({scenario.description})")
            
            for tick in range(TICKS_PER_HOUR):
                # 1. Calculate current arrival rate
                tick_in_scenario = (tick / TICKS_PER_HOUR) * scenario.duration_sec
                iw_rps = scenario.rate_function(tick_in_scenario)
                
                # 2. Inject IW traffic
                iw_to_inject = max(0, int(iw_rps * 0.5))  # Scale down for simulation
                for _ in range(iw_to_inject):
                    req = InFlightRequest(
                        id=request_id,
                        state='queued',
                        arrival_time=current_time,
                        prompt_tokens=scenario.prompt_len_function(),
                        target_tokens=scenario.output_len_function(),
                        priority=0  # Interactive
                    )
                    server.queued_requests.append(req)
                    request_id += 1
                    stats["iw_requests"] += 1
                
                # 3. Inject NIW background traffic (20% chance per tick)
                if random.random() < 0.2:
                    req = InFlightRequest(
                        id=request_id,
                        state='queued',
                        arrival_time=current_time,
                        prompt_tokens=2048,
                        target_tokens=512,
                        priority=1,  # Non-Interactive
                        deadline_sec=3600  # 1 hour deadline
                    )
                    server.queued_requests.append(req)
                    request_id += 1
                    stats["niw_requests"] += 1
                
                # 4. Calculate queue velocity
                current_queue = len(server.queued_requests)
                queue_velocity = (current_queue - prev_queue_depth) / max(1.0, 1.0)
                prev_queue_depth = current_queue
                
                # 5. Create metrics
                metrics = create_metrics_from_server(server, queue_velocity)
                
                # 6. Get controller decision
                controller.update_config(server.config.batch_size, server.config.gpu_count)
                decision = controller.make_decision(metrics)
                
                # 7. Apply decision to server
                action_type = decision.action.value
                if decision.action == Action.SCALE_OUT:
                    server.config = ServerConfig(
                        batch_size=server.config.batch_size,
                        gpu_count=min(8, server.config.gpu_count + 1)
                    )
                elif decision.action == Action.DECREASE_BATCH if hasattr(Action, 'DECREASE_BATCH') else decision.action == Action.REDUCE_BATCH:
                    server.config = ServerConfig(
                        batch_size=max(1, server.config.batch_size - 1),
                        gpu_count=server.config.gpu_count
                    )
                elif decision.action == Action.INCREASE_BATCH:
                    server.config = ServerConfig(
                        batch_size=min(32, server.config.batch_size + 2),
                        gpu_count=server.config.gpu_count
                    )
                elif decision.action == Action.DEFER_NIW:
                    # Remove NIW from processing (defer)
                    deferred = [r for r in server.queued_requests if r.priority >= 1][:3]
                    for r in deferred:
                        r.state = 'deferred'
                
                # 8. Simulate processing (remove completed requests)
                capacity = server.config.batch_size * server.config.gpu_count
                to_remove = min(capacity, len(server.queued_requests))
                server.queued_requests = server.queued_requests[to_remove:]
                
                # 9. Calculate reward signal
                throughput = capacity * 0.8  # Simulated throughput
                latency_penalty = max(0, metrics.ttft_ms - 200)  # Penalty above SLA
                gpu_cost = server.config.gpu_count * 0.1  # Cost per GPU-tick
                
                reward = (throughput * 0.1) - (latency_penalty * 0.01) - (gpu_cost * 1.0)
                
                # 10. Log state-action-reward
                log_entry = {
                    "time": current_time,
                    "hour": hour,
                    "scenario": scenario.name,
                    "state": {
                        "queue_iw": metrics.queue_depth_iw,
                        "queue_niw": metrics.queue_depth_niw,
                        "velocity": queue_velocity,
                        "gpu_util": metrics.gpu_utilization,
                        "batch_size": server.config.batch_size,
                        "gpu_count": server.config.gpu_count,
                        "niw_in_flight": metrics.niw_in_flight
                    },
                    "action": action_type,
                    "reward": {
                        "total": reward,
                        "throughput": throughput,
                        "latency_penalty": latency_penalty,
                        "gpu_cost": gpu_cost
                    }
                }
                f.write(json.dumps(log_entry) + "\n")
                
                # Update stats
                stats["total_ticks"] += 1
                stats["total_requests"] = request_id
                stats["avg_queue_depth"] += current_queue
                stats["max_queue_depth"] = max(stats["max_queue_depth"], current_queue)
                stats["actions"][action_type] = stats["actions"].get(action_type, 0) + 1
                
                current_time += 1.0
    
    # Print summary
    print("\n" + "=" * 60)
    print("GENERATION COMPLETE")
    print("=" * 60)
    
    stats["avg_queue_depth"] /= max(1, stats["total_ticks"])
    
    print(f"\nStatistics:")
    print(f"  Total ticks: {stats['total_ticks']}")
    print(f"  Total requests: {stats['total_requests']}")
    print(f"  IW requests: {stats['iw_requests']}")
    print(f"  NIW requests: {stats['niw_requests']}")
    print(f"  Avg queue depth: {stats['avg_queue_depth']:.1f}")
    print(f"  Max queue depth: {stats['max_queue_depth']}")
    
    print(f"\nAction distribution:")
    for action, count in sorted(stats["actions"].items(), key=lambda x: -x[1]):
        pct = (count / stats["total_ticks"]) * 100
        print(f"  {action}: {count} ({pct:.1f}%)")
    
    print(f"\nOutput saved to: {OUTPUT_FILE}")
    

if __name__ == "__main__":
    run_data_generation()
