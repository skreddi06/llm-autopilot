"""Data structures for the LLM Autopilot system."""

from dataclasses import dataclass
from typing import Optional
from enum import Enum


class Action(Enum):
    """Actions that the controller can decide to take."""
    NO_ACTION = "no_action"
    REDUCE_BATCH = "reduce_batch"
    INCREASE_BATCH = "increase_batch"
    SCALE_OUT = "scale_out"
    SCALE_IN = "scale_in"
    ENABLE_SPECULATIVE_DECODE = "enable_speculative_decode"
    REBALANCE_LOAD = "rebalance_load"
    ENABLE_OVERLAP_MODE = "enable_overlap_mode"
    DEFER_NIW = "defer_niw"  # Phase 7: Pause batch jobs to free capacity


@dataclass
class Metrics:
    """Metrics collected from the LLM server."""
    ttft_ms: float  # Time-to-first-token in milliseconds
    inter_token_latency_ms: float  # Inter-token latency in milliseconds
    prefill_latency_ms: float  # Prefill latency in milliseconds
    decode_latency_ms: float  # Decode latency in milliseconds
    gpu_utilization: float  # GPU utilization percentage (0-100)
    memory_efficiency: float  # Memory efficiency (0-1)
    gpu_balance_index: float  # Balance index across GPUs (0-1)
    comm_bubble_ratio: float  # Communication bubble ratio (0-1)
    speculative_factor: float  # Speculative decoding factor (0-1)
    queue_depth: int  # Number of pending requests
    timestamp: float  # Unix timestamp when metrics were collected
    queue_velocity: float = 0.0  # Velocity of the queue (requests per second)
    
    # Phase 7: IW/NIW queue tracking
    queue_depth_iw: int = 0   # Interactive workload queue depth
    queue_depth_niw: int = 0  # Non-interactive workload queue depth
    niw_in_flight: int = 0    # NIW requests currently being processed (deferable)


@dataclass
class ServerConfig:
    """Configuration of the LLM server."""
    batch_size: int
    gpu_count: int


@dataclass
class Decision:
    """Decision made by the controller."""
    action: Action
    batch_size: Optional[int] = None
    gpu_count: Optional[int] = None
    reason: str = ""

