"""Brain/Controller that makes decisions based on metrics."""

import logging
from typing import Optional
from models import Metrics, Decision, Action, ServerConfig
import collections

logger = logging.getLogger(__name__)


class Controller:
    """The brain that makes autopilot decisions."""
    
    def __init__(
        self,
        slo_latency_ms: float = 600.0,
        min_batch_size: int = 1,
        max_batch_size: int = 32,
        min_gpu_count: int = 1,
        max_gpu_count: int = 8,
        danger_gpu_threshold: float = 90.0,
        stability_window: int = 3
    ):
        self.slo_latency_ms = slo_latency_ms
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.min_gpu_count = min_gpu_count
        self.max_gpu_count = max_gpu_count
        self.danger_gpu_threshold = danger_gpu_threshold
        self.stability_window = stability_window
        self.recent_metrics = collections.deque(maxlen=stability_window)
        
        # Current state
        self.current_config: Optional[ServerConfig] = None
        self.mode = "safe"  # safe, danger, breach
    
    def update_config(self, config: ServerConfig):
        """Update the current server configuration."""
        self.current_config = config
    
    def make_decision(self, metrics: Optional[Metrics]) -> Decision:
        """Make a decision based on current metrics."""
        if not metrics:
            return Decision(
                action=Action.NO_ACTION,
                reason="No metrics available"
            )

        if not self.current_config:
            return Decision(
                action=Action.NO_ACTION,
                reason="No server configuration known"
            )

        # Track recent metrics for stability
        self.recent_metrics.append(metrics)

        # Predictive smoothing
        avg_ttft = sum(m.ttft_ms for m in self.recent_metrics) / len(self.recent_metrics)
        avg_gpu = sum(m.gpu_utilization for m in self.recent_metrics) / len(self.recent_metrics)

        # Mode switching
        if metrics.queue_depth > 50 and avg_gpu < 85:
            self.mode = "throughput_optimized"
        elif metrics.ttft_ms > self.slo_latency_ms:
            self.mode = "latency_optimized"
        else:
            self.mode = "safe"

        # Decision logic based on mode
        if self.mode == "throughput_optimized":
            if metrics.memory_efficiency < 0.7:
                return Decision(
                    action=Action.REBALANCE_LOAD,
                    reason="Memory efficiency too low, rebalancing load"
                )
            elif metrics.gpu_balance_index > 0.9:
                return Decision(
                    action=Action.INCREASE_BATCH,
                    reason="GPU balance optimal, increasing batch size"
                )
        elif self.mode == "latency_optimized":
            if metrics.ttft_ms > self.slo_latency_ms * 1.2:
                if metrics.speculative_factor < 0.5:
                    return Decision(
                        action=Action.ENABLE_SPECULATIVE_DECODE,
                        reason="Latency too high, enabling speculative decoding"
                    )
                else:
                    return Decision(
                        action=Action.REDUCE_BATCH,
                        reason="Latency too high, reducing batch size"
                    )

        # Default no action
        return Decision(
            action=Action.NO_ACTION,
            reason=(
                f"Stable: TTFT={metrics.ttft_ms:.1f}ms, "
                f"GPU={metrics.gpu_utilization:.1f}%, "
                f"Batch={self.current_config.batch_size}, GPUs={self.current_config.gpu_count}"
            )
        )

