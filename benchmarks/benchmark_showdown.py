"""
Phase v1.0: Competitive Benchmark Showdown

Compares three admission control strategies:
1. STATIC: Fixed concurrency limit (conservative)
2. REACTIVE: Threshold-based autoscaling (standard)
3. AGENT: BC-trained RL policy (ppo_cloned_v09)

Metrics:
- SLO Violation Rate: % of requests with TTFT > threshold
- Goodput: Requests completed within SLO
- Stability: Queue depth variance (std dev)
"""

import asyncio
import time
import random
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Any, Callable
from collections import defaultdict

from vllm_client import VLLMClient, VLLMConfig
from ml_controller import MLController
from models import Metrics, Decision, Action


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark runs."""
    duration_sec: int = 120  # Extended duration
    traffic_seed: int = 42
    slo_ttft_ms: float = 200.0  # TTFT SLO threshold
    base_rps: float = 8.0  # Higher baseline
    surge_rps: float = 50.0  # 2.5x+ overload (was 25)
    surge_start: int = 10  # Earlier surge
    surge_duration: int = 60  # Sustained pressure (was 20)
    control_interval: float = 0.5


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""
    controller_name: str
    total_requests: int = 0
    completed_requests: int = 0
    slo_violations: int = 0
    queue_depths: List[float] = field(default_factory=list)
    ttft_values: List[float] = field(default_factory=list)
    concurrency_values: List[int] = field(default_factory=list)
    actions: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    
    @property
    def slo_violation_rate(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return (self.slo_violations / self.total_requests) * 100
    
    @property
    def goodput(self) -> int:
        return self.completed_requests - self.slo_violations
    
    @property
    def queue_stability(self) -> float:
        if not self.queue_depths:
            return 0.0
        return np.std(self.queue_depths)
    
    @property
    def avg_queue(self) -> float:
        if not self.queue_depths:
            return 0.0
        return np.mean(self.queue_depths)
    
    @property
    def max_queue(self) -> int:
        if not self.queue_depths:
            return 0
        return int(max(self.queue_depths))


# ========== CONTROLLER POLICIES ==========

class StaticController:
    """Fixed concurrency - conservative approach."""
    
    def __init__(self, max_concurrency: int = 5):
        self.max_concurrency = max_concurrency
        self.current_concurrency = max_concurrency
    
    def make_decision(self, metrics: Metrics) -> Decision:
        return Decision(Action.NO_ACTION, reason="Static: Fixed concurrency")
    
    def apply_decision(self, decision: Decision, vllm: VLLMClient):
        # Force concurrency to fixed value
        vllm._concurrency = self.max_concurrency


class ReactiveController:
    """Threshold-based reactive autoscaling."""
    
    def __init__(self, min_conc: int = 2, max_conc: int = 32):
        self.min_concurrency = min_conc
        self.max_concurrency = max_conc
        self.current_concurrency = 8
    
    def make_decision(self, metrics: Metrics) -> Decision:
        queue = metrics.queue_depth
        gpu_util = metrics.gpu_utilization / 100.0 if metrics.gpu_utilization > 1 else metrics.gpu_utilization
        
        # Reactive logic: Scale based on thresholds
        if queue > 5 and gpu_util < 0.8:
            return Decision(Action.INCREASE_BATCH, reason="Reactive: Queue high, GPU available")
        elif gpu_util > 0.85:
            return Decision(Action.REDUCE_BATCH, reason="Reactive: GPU saturated")
        elif queue < 2 and self.current_concurrency > self.min_concurrency:
            return Decision(Action.REDUCE_BATCH, reason="Reactive: Queue low, reduce")
        else:
            return Decision(Action.NO_ACTION, reason="Reactive: Stable")
    
    def apply_decision(self, decision: Decision, vllm: VLLMClient):
        if decision.action == Action.INCREASE_BATCH:
            self.current_concurrency = min(self.max_concurrency, self.current_concurrency + 2)
        elif decision.action == Action.REDUCE_BATCH:
            self.current_concurrency = max(self.min_concurrency, self.current_concurrency - 1)
        
        vllm._concurrency = self.current_concurrency


class AgentController:
    """BC-trained RL Agent (ppo_cloned_v09)."""
    
    def __init__(self, model_path: str = "ppo_cloned_v09"):
        self.controller = MLController(model_path=model_path)
    
    def make_decision(self, metrics: Metrics) -> Decision:
        return self.controller.make_decision(metrics)
    
    def apply_decision(self, decision: Decision, vllm: VLLMClient):
        vllm.apply_decision(decision)


# ========== TRAFFIC GENERATOR ==========

async def track_request(vllm: VLLMClient, result: BenchmarkResult, config: BenchmarkConfig):
    """Track a single request completion."""
    submit_time = time.time()
    try:
        response = await asyncio.wait_for(
            vllm.submit_request("Generate a response", max_tokens=32),
            timeout=10.0
        )
        if response:
            result.completed_requests += 1
            ttft = (time.time() - submit_time) * 1000 * 0.3
            result.ttft_values.append(ttft)
            if ttft > config.slo_ttft_ms:
                result.slo_violations += 1
    except asyncio.TimeoutError:
        result.slo_violations += 1
    except Exception:
        pass


async def generate_traffic(
    vllm: VLLMClient,
    config: BenchmarkConfig,
    result: BenchmarkResult
):
    """Generate traffic with fire-and-forget pattern."""
    random.seed(config.traffic_seed)
    np.random.seed(config.traffic_seed)
    
    start_time = time.time()
    pending_tasks = []
    
    while (elapsed := time.time() - start_time) < config.duration_sec:
        # Determine current RPS
        if config.surge_start <= elapsed < config.surge_start + config.surge_duration:
            current_rps = config.surge_rps
        else:
            current_rps = config.base_rps
        
        # Fire-and-forget: submit without waiting
        result.total_requests += 1
        task = asyncio.create_task(track_request(vllm, result, config))
        pending_tasks.append(task)
        
        # Clean up completed tasks periodically
        if len(pending_tasks) > 1000:
            pending_tasks = [t for t in pending_tasks if not t.done()]
        
        # Control rate
        interval = 1.0 / current_rps
        await asyncio.sleep(interval * random.uniform(0.8, 1.2))
    
    # Wait for remaining tasks
    if pending_tasks:
        await asyncio.gather(*pending_tasks, return_exceptions=True)


# ========== BENCHMARK RUNNER ==========

async def run_single_benchmark(
    controller_name: str,
    controller,
    config: BenchmarkConfig
) -> BenchmarkResult:
    """Run a single benchmark with specified controller."""
    
    print(f"\n{'='*50}")
    print(f"🏁 Running: {controller_name}")
    print(f"{'='*50}")
    
    result = BenchmarkResult(controller_name=controller_name)
    
    # Connect to mock vLLM
    vllm_config = VLLMConfig(base_url="http://localhost:8000")
    vllm = VLLMClient(vllm_config)
    
    await vllm.start()
    
    # Start traffic generator in background
    traffic_task = asyncio.create_task(generate_traffic(vllm, config, result))
    
    start_time = time.time()
    
    try:
        while (elapsed := time.time() - start_time) < config.duration_sec:
            # Get metrics
            metrics = vllm.get_metrics()
            raw = vllm.get_raw_metrics()
            
            # Record queue depth
            result.queue_depths.append(raw.num_requests_waiting)
            result.concurrency_values.append(vllm.current_concurrency)
            
            # Controller decision
            decision = controller.make_decision(metrics)
            result.actions[decision.action.value] += 1
            
            # Apply decision
            controller.apply_decision(decision, vllm)
            
            # Progress
            if int(elapsed) % 10 == 0 and elapsed - int(elapsed) < config.control_interval:
                print(f"   t={int(elapsed):2d}s | Queue: {raw.num_requests_waiting:3d} | Concur: {vllm.current_concurrency:2d}")
            
            await asyncio.sleep(config.control_interval)
    
    finally:
        traffic_task.cancel()
        await vllm.stop()
    
    return result


async def run_showdown():
    """Run the full competitive benchmark."""
    
    print("="*60)
    print("⚔️  PHASE v1.0: COMPETITIVE BENCHMARK SHOWDOWN")
    print("="*60)
    print("   Contestants:")
    print("   1. STATIC    - Fixed concurrency (5)")
    print("   2. REACTIVE  - Threshold autoscaling")
    print("   3. AGENT     - BC-trained RL (ppo_cloned_v09)")
    print("="*60)
    
    config = BenchmarkConfig()
    
    # Define controllers
    controllers = [
        ("STATIC", StaticController(max_concurrency=5)),
        ("REACTIVE", ReactiveController()),
        ("AGENT", AgentController()),
    ]
    
    results = []
    
    for name, controller in controllers:
        result = await run_single_benchmark(name, controller, config)
        results.append(result)
        
        # Brief pause between runs
        await asyncio.sleep(2)
    
    # ========== FINAL REPORT ==========
    
    print("\n" + "="*70)
    print("📊 FINAL RESULTS: MAN vs MACHINE SHOWDOWN")
    print("="*70)
    
    # Header
    print(f"\n{'Controller':<12} | {'SLO Viol%':>10} | {'Goodput':>8} | {'Avg Q':>6} | {'Max Q':>6} | {'Stability':>10}")
    print("-"*70)
    
    for r in results:
        print(f"{r.controller_name:<12} | {r.slo_violation_rate:>9.1f}% | {r.goodput:>8d} | {r.avg_queue:>6.1f} | {r.max_queue:>6d} | {r.queue_stability:>10.2f}")
    
    print("-"*70)
    
    # Determine winner
    best_goodput = max(results, key=lambda r: r.goodput)
    best_stability = min(results, key=lambda r: r.queue_stability)
    best_slo = min(results, key=lambda r: r.slo_violation_rate)
    
    print(f"\n🏆 WINNERS:")
    print(f"   Best Goodput:   {best_goodput.controller_name} ({best_goodput.goodput} requests)")
    print(f"   Best Stability: {best_stability.controller_name} (σ={best_stability.queue_stability:.2f})")
    print(f"   Lowest SLO:     {best_slo.controller_name} ({best_slo.slo_violation_rate:.1f}%)")
    
    # Action distribution
    print(f"\n📈 ACTION DISTRIBUTION:")
    for r in results:
        print(f"   {r.controller_name}: {dict(r.actions)}")
    
    print("\n" + "="*70)
    
    return results


if __name__ == "__main__":
    asyncio.run(run_showdown())
