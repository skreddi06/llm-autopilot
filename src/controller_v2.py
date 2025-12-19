"""
v0.5-α Predictive Controller
Anticipatory control using queue velocity, acceleration, and TTFT slope forecasting.
Replaces reactive decision-making with predictive trend analysis and confidence gating.

Phase 6: Integrated with LoadPredictor for arrival rate forecasting.
"""

import asyncio
from collections import deque
from dataclasses import dataclass
from typing import List, Tuple, Optional
import statistics

from models import Metrics, Decision, Action
from predictor import LoadPredictor, PredictionResult


@dataclass
class ControllerState:
    """Internal state for predictive controller."""
    mode: str = "safe"  # safe, throughput_optimized, latency_optimized
    confidence_history: deque = None  # Track decision confidence over time
    prev_queue_velocity: float = 0.0
    consecutive_stable_samples: int = 0
    
    def __post_init__(self):
        if self.confidence_history is None:
            self.confidence_history = deque(maxlen=10)


class PredictiveController:
    """
    v0.5-α Predictive Controller with:
    - Queue velocity & acceleration tracking
    - TTFT slope forecasting (10s ahead)
    - Confidence gating (requires 2/3 agreement from last 3 samples)
    - Emergency circuit breaker (PANIC modes)
    - Safe fallback to reactive control
    """
    
    def __init__(self, slo_latency_ms: int = 600):
        self.slo_latency_ms = slo_latency_ms
        self.history = deque(maxlen=10)  # Last 10 metrics samples
        self.state = ControllerState()
        
        # Circuit breaker thresholds
        self.PANIC_QUEUE_THRESHOLD = 500
        self.PANIC_TTFT_MULTIPLIER = 5  # 5x SLO = emergency
        self.PANIC_GPU_THRESHOLD = 95
        self.PANIC_VELOCITY_THRESHOLD = 10  # requests/sec growth rate
        
        # Confidence gating parameters
        self.CONFIDENCE_THRESHOLD = 0.7
        self.CONSENSUS_WINDOW = 3  # Require 2/3 agreement
        self.MIN_HISTORY_FOR_PREDICTION = 3
        
        # Phase 6: Load Predictor for arrival rate forecasting
        self.load_predictor = LoadPredictor(window_size=30, surge_velocity_threshold=5.0)
        self.last_prediction: Optional[PredictionResult] = None
        
        # Capacity headroom (SageServe heuristic)
        self.CAPACITY_HEADROOM_TARGET = 0.20  # 20% spare capacity target
        
        # Internal config tracking
        self.current_batch_size = 8
        self.current_gpu_count = 1
    
    def update_config(self, config_or_batch=None, gpu_count: int = None):
        """Update internal config tracking.
        
        Accepts either:
        - ServerConfig object: update_config(config)
        - Individual values: update_config(batch_size=8, gpu_count=2)
        """
        from models import ServerConfig
        if isinstance(config_or_batch, ServerConfig):
            self.current_batch_size = config_or_batch.batch_size
            self.current_gpu_count = config_or_batch.gpu_count
        else:
            if config_or_batch is not None:
                self.current_batch_size = config_or_batch
            if gpu_count is not None:
                self.current_gpu_count = gpu_count
        
    def compute_trends(self, metrics: Metrics) -> Tuple[float, float, float]:
        """
        Compute predictive signals from metrics history.
        
        Returns:
            (ttft_slope, queue_velocity, queue_acceleration)
            - ttft_slope: ms/sample - positive means latency increasing
            - queue_velocity: requests/sec from metrics
            - queue_acceleration: change in velocity (requests/sec²)
        """
        if len(self.history) < 2:
            return 0.0, metrics.queue_velocity, 0.0
        
        # TTFT slope via linear regression
        ttft_values = [m.ttft_ms for m in self.history]
        if len(ttft_values) >= 2:
            # Simple slope: (last - first) / samples
            ttft_slope = (ttft_values[-1] - ttft_values[0]) / len(ttft_values)
        else:
            ttft_slope = 0.0
        
        # Queue velocity from current metrics
        queue_velocity = metrics.queue_velocity
        
        # Queue acceleration (change in velocity)
        queue_acceleration = queue_velocity - self.state.prev_queue_velocity
        self.state.prev_queue_velocity = queue_velocity
        
        return ttft_slope, queue_velocity, queue_acceleration
    
    def forecast_ttft(self, ttft_slope: float, current_ttft: float, steps_ahead: int = 5) -> float:
        """
        Forecast TTFT N samples ahead using linear extrapolation.
        
        Args:
            ttft_slope: Rate of TTFT change (ms/sample)
            current_ttft: Current TTFT value
            steps_ahead: Number of samples to forecast (default 5 = 10s ahead at 2s intervals)
        
        Returns:
            Forecasted TTFT in ms
        """
        return current_ttft + (ttft_slope * steps_ahead)
    
    def estimate_confidence(self, ttft_slope: float, queue_velocity: float, queue_acceleration: float) -> float:
        """
        Estimate confidence in predictions based on signal stability.
        
        High confidence when:
        - History buffer is full (10 samples)
        - Trends are consistent (low variance)
        - Signals agree in direction
        
        Returns:
            Confidence score 0.0-1.0
        """
        if len(self.history) < self.MIN_HISTORY_FOR_PREDICTION:
            return 0.0
        
        confidence = 0.5  # Base confidence
        
        # History depth bonus
        history_factor = len(self.history) / 10.0
        confidence += 0.2 * history_factor
        
        # Signal agreement bonus (velocity and acceleration aligned)
        if (queue_velocity > 0 and queue_acceleration > 0) or \
           (queue_velocity < 0 and queue_acceleration < 0):
            confidence += 0.2
        
        # TTFT stability check
        if len(self.history) >= 5:
            recent_ttft = [m.ttft_ms for m in list(self.history)[-5:]]
            ttft_variance = statistics.variance(recent_ttft) if len(recent_ttft) > 1 else 0
            if ttft_variance < 100:  # Low variance = stable
                confidence += 0.1
        
        return min(confidence, 1.0)
    
    def check_emergency_conditions(self, metrics: Metrics, queue_velocity: float) -> Optional[Decision]:
        """
        Circuit breaker: bypass prediction if emergency detected.
        
        Emergency conditions (PANIC modes):
        - PANIC_RESET: Queue > 500 → instant reset
        - PANIC_DRAIN: TTFT > 5x SLO → aggressive batch reduction
        - PANIC_SCALE: GPU > 95% AND velocity > 10 → immediate scale-out
        
        Returns:
            Emergency Decision if triggered, None otherwise
        """
        # PANIC_RESET: Massive queue backlog
        if metrics.queue_depth > self.PANIC_QUEUE_THRESHOLD:
            return Decision(
                action=Action.REDUCE_BATCH,
                reason=f"PANIC_RESET: Queue={metrics.queue_depth} > {self.PANIC_QUEUE_THRESHOLD}",
                batch_size=1,  # Emergency minimum
                gpu_count=self.current_gpu_count
            )
        
        # PANIC_DRAIN: Extreme latency violation
        if metrics.ttft_ms > self.slo_latency_ms * self.PANIC_TTFT_MULTIPLIER:
            return Decision(
                action=Action.REDUCE_BATCH,
                reason=f"PANIC_DRAIN: TTFT={metrics.ttft_ms:.0f}ms > {self.slo_latency_ms * self.PANIC_TTFT_MULTIPLIER}ms",
                batch_size=max(1, self.current_batch_size // 2),
                gpu_count=self.current_gpu_count
            )
        
        # PANIC_SCALE: GPU saturation with accelerating queue
        # Phase 7: Even in PANIC, try to defer NIW first
        if metrics.gpu_utilization > self.PANIC_GPU_THRESHOLD and queue_velocity > self.PANIC_VELOCITY_THRESHOLD:
            niw_available = getattr(metrics, 'niw_in_flight', 0)
            if niw_available > 0:
                return Decision(
                    action=Action.DEFER_NIW,
                    reason=f"PANIC_DEFER: GPU={metrics.gpu_utilization:.1f}%, deferring {niw_available} NIW first",
                    batch_size=self.current_batch_size,
                    gpu_count=self.current_gpu_count
                )
            else:
                return Decision(
                    action=Action.SCALE_OUT,
                    reason=f"PANIC_SCALE: GPU={metrics.gpu_utilization:.1f}% + velocity={queue_velocity:.1f}, no NIW",
                    batch_size=self.current_batch_size,
                    gpu_count=min(8, self.current_gpu_count + 1)
                )
        
        return None
    
    def update_mode(self, metrics: Metrics, queue_velocity: float):
        """
        Update controller mode based on system state.
        Mirrors v0.4 mode logic but incorporates predictive signals.
        """
        # Latency-optimized: SLO breach or trending toward breach
        if metrics.ttft_ms > self.slo_latency_ms:
            self.state.mode = "latency_optimized"
        # Throughput-optimized: High queue but capacity available
        elif metrics.queue_depth > 50 and metrics.gpu_utilization < 85:
            self.state.mode = "throughput_optimized"
        # Safe: Stable operation
        elif metrics.ttft_ms < self.slo_latency_ms * 0.8 and metrics.queue_depth < 20:
            self.state.mode = "safe"
            self.state.consecutive_stable_samples += 1
        else:
            # Keep current mode if transitioning
            pass
    
    def make_predictive_decision(self, metrics: Metrics, ttft_slope: float, 
                                 queue_velocity: float, queue_acceleration: float,
                                 confidence: float) -> Decision:
        """
        Core predictive decision logic.
        
        Key differences from v0.4 reactive controller:
        1. Acts on forecasted TTFT, not current TTFT
        2. Pre-scales based on queue velocity + acceleration
        3. Avoids "Action Paradox" by considering throughput impact
        4. Confidence-gated: requires high confidence for aggressive actions
        """
        # Forecast TTFT 10s ahead (5 samples * 2s interval)
        forecasted_ttft = self.forecast_ttft(ttft_slope, metrics.ttft_ms, steps_ahead=5)
        
        # Low confidence: fall back to conservative action
        if confidence < self.CONFIDENCE_THRESHOLD:
            if metrics.ttft_ms > self.slo_latency_ms * 1.2:
                return Decision(
                    action=Action.REDUCE_BATCH,
                    reason=f"Low confidence ({confidence:.2f}), reactive fallback",
                    batch_size=max(1, self.current_batch_size - 1),
                    gpu_count=self.current_gpu_count
                )
            return Decision(action=Action.NO_ACTION, reason="Low confidence, holding steady")
        
        # Pre-emptive scale-out: queue accelerating upward
        # Phase 7: Defer NIW first before scaling out
        if queue_velocity > 5 and queue_acceleration > 0 and metrics.gpu_utilization > 75:
            niw_available = getattr(metrics, 'niw_in_flight', 0)
            if niw_available > 0:
                return Decision(
                    action=Action.DEFER_NIW,
                    reason=f"Predictive: Queue accelerating (vel={queue_velocity:.1f}), deferring {niw_available} NIW first",
                    batch_size=self.current_batch_size,
                    gpu_count=self.current_gpu_count
                )
            else:
                return Decision(
                    action=Action.SCALE_OUT,
                    reason=f"Predictive scale-out: vel={queue_velocity:.1f}, accel={queue_acceleration:.1f}, no NIW to defer",
                    batch_size=self.current_batch_size,
                    gpu_count=min(8, self.current_gpu_count + 1)
                )
        
        # Pre-emptive batch increase: forecasted TTFT safe + capacity available
        if forecasted_ttft < self.slo_latency_ms * 0.7 and \
           metrics.queue_depth < 10 and \
           metrics.gpu_utilization < 60 and \
           queue_velocity < 2:
            return Decision(
                action=Action.INCREASE_BATCH,
                reason=f"Predictive throughput boost: forecast={forecasted_ttft:.0f}ms safe",
                batch_size=min(32, self.current_batch_size + 2),
                gpu_count=self.current_gpu_count
            )
        
        # Pre-emptive batch reduction: forecasted SLO breach
        if forecasted_ttft > self.slo_latency_ms and ttft_slope > 0:
            return Decision(
                action=Action.REDUCE_BATCH,
                reason=f"Predictive batch reduction: forecast={forecasted_ttft:.0f}ms > SLO",
                batch_size=max(1, self.current_batch_size - 1),
                gpu_count=self.current_gpu_count
            )
        
        # Speculative decoding enable: latency mode + low GPU
        if self.state.mode == "latency_optimized" and metrics.gpu_utilization < 70:
            return Decision(
                action=Action.ENABLE_SPECULATIVE_DECODE,
                reason=f"Latency mode: enable speculation (GPU={metrics.gpu_utilization:.1f}%)"
            )
        
        # Load rebalancing: throughput mode + memory inefficiency
        if self.state.mode == "throughput_optimized" and metrics.memory_efficiency < 0.7:
            return Decision(
                action=Action.REBALANCE_LOAD,
                reason=f"Throughput mode: rebalance (mem_eff={metrics.memory_efficiency:.2f})"
            )
        
        # Stable operation
        return Decision(action=Action.NO_ACTION, reason=f"Predictive steady state (confidence={confidence:.2f})")
    
    def requires_consensus(self, decision: Decision) -> bool:
        """
        Check if decision requires consensus from recent history.
        
        Aggressive actions (SCALE_OUT, REDUCE_BATCH to minimum) require
        2/3 agreement from last 3 decisions to prevent oscillation.
        """
        if decision.action in [Action.SCALE_OUT, Action.REDUCE_BATCH]:
            if decision.batch_size and decision.batch_size <= 2:
                return True
        return False
    
    def check_consensus(self, decision: Decision) -> bool:
        """
        Verify consensus from recent decision history.
        
        Returns:
            True if consensus met (2/3 recent decisions agree), False otherwise
        """
        if len(self.state.confidence_history) < self.CONSENSUS_WINDOW:
            return False
        
        recent_decisions = list(self.state.confidence_history)[-self.CONSENSUS_WINDOW:]
        agreement_count = sum(1 for d in recent_decisions if d.action == decision.action)
        
        return agreement_count >= 2  # 2 out of 3
    
    def make_decision(self, metrics: Metrics) -> Decision:
        """
        Main decision entry point - orchestrates predictive control flow.
        
        Flow:
        1. Add metrics to history
        2. Check emergency circuit breaker
        3. Compute predictive trends
        4. Estimate confidence
        5. Update mode
        6. Make predictive decision
        7. Apply consensus gating if needed
        8. Return final decision
        """
        # Add to history
        self.history.append(metrics)
        
        # Compute trends
        ttft_slope, queue_velocity, queue_acceleration = self.compute_trends(metrics)
        
        # Emergency bypass
        emergency_decision = self.check_emergency_conditions(metrics, queue_velocity)
        if emergency_decision:
            return emergency_decision
        
        # === PHASE 6: ARRIVAL RATE FORECASTING (SageServe) ===
        # Record current arrival rate and get prediction
        current_rps = metrics.throughput if hasattr(metrics, 'throughput') else 0
        self.load_predictor.record(current_rps)
        prediction = self.load_predictor.predict(horizon_sec=10)
        self.last_prediction = prediction
        
        # Capacity headroom check (SageServe heuristic)
        # If forecast exceeds capacity, pre-scale BEFORE queue builds
        current_capacity = self.current_batch_size * self.current_gpu_count * 0.5  # Rough RPS capacity estimate
        forecasted_load = prediction.predicted_rps
        capacity_headroom = (current_capacity - forecasted_load) / max(1, current_capacity)
        
        # === PHASE 7: SAGESERVE YIELD MANAGEMENT ===
        # If capacity deficit detected:
        # 1. FIRST: Try to defer NIW (reclaim resources for free)
        # 2. ONLY IF NIW exhausted: Scale out (pay for new resources)
        
        niw_available = getattr(metrics, 'niw_in_flight', 0)
        
        if capacity_headroom < self.CAPACITY_HEADROOM_TARGET:
            if prediction.surge_detected:
                # Check if we can defer NIW before scaling
                if niw_available > 0:
                    return Decision(
                        action=Action.DEFER_NIW,
                        reason=f"SageServe: Capacity crunch, deferring {niw_available} NIW jobs before scaling",
                        batch_size=self.current_batch_size,
                        gpu_count=self.current_gpu_count
                    )
                else:
                    # No NIW to defer, must scale out
                    return Decision(
                        action=Action.SCALE_OUT,
                        reason=f"SageServe: Forecast {forecasted_load:.1f} RPS > capacity, no NIW to defer",
                        batch_size=self.current_batch_size,
                        gpu_count=min(8, self.current_gpu_count + 1)
                    )
        
        # Pre-emptive batch increase: surplus capacity
        if capacity_headroom > 0.40 and prediction.confidence > 0.6:
            return Decision(
                action=Action.INCREASE_BATCH,
                reason=f"SageServe: Headroom {capacity_headroom:.0%} > 40%, increasing batch",
                batch_size=min(32, self.current_batch_size + 2),
                gpu_count=self.current_gpu_count
            )
        
        # Estimate confidence
        confidence = self.estimate_confidence(ttft_slope, queue_velocity, queue_acceleration)
        
        # Update mode
        self.update_mode(metrics, queue_velocity)
        
        # Make predictive decision
        decision = self.make_predictive_decision(
            metrics, ttft_slope, queue_velocity, queue_acceleration, confidence
        )
        
        # Consensus gating for aggressive actions
        if self.requires_consensus(decision):
            if not self.check_consensus(decision):
                decision = Decision(
                    action=Action.NO_ACTION,
                    reason=f"Consensus required but not met for {decision.action.value}"
                )
        
        # Track decision for consensus
        self.state.confidence_history.append(decision)
        
        return decision
    
    @property
    def mode(self) -> str:
        """Expose current mode for logging/dashboard."""
        return self.state.mode
