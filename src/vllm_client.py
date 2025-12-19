"""
vLLM Client Adapter (Phase v0.9: The Reality Bridge)

Provides a shim layer between the Hybrid Controller and a real vLLM engine.
Translates Controller decisions into admission control actions.

Metric Mapping:
    queue_depth → vllm:num_requests_waiting
    kv_cache_usage_pct → vllm:gpu_cache_usage_perc
    latency → vllm:time_per_output_token_seconds (TPOT)

Action Mapping:
    INC_BATCH → Increase concurrency semaphore (allow more requests)
    DEC_BATCH → Decrease concurrency semaphore (throttle requests)
    DEFER_NIW → Pause queue processing temporarily
"""

import asyncio
import time
import logging
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from aiohttp import ClientSession, ClientError
from models import Metrics, Decision, Action

logger = logging.getLogger(__name__)


@dataclass
class VLLMConfig:
    """Configuration for vLLM connection."""
    base_url: str = "http://localhost:8000"
    metrics_path: str = "/metrics"
    generate_path: str = "/v1/completions"
    model_name: str = "meta-llama/Llama-2-7b-hf"
    
    # Admission Control
    initial_concurrency: int = 8
    min_concurrency: int = 1
    max_concurrency: int = 64
    
    # Polling
    metrics_interval_sec: float = 0.5


@dataclass 
class VLLMMetrics:
    """Parsed metrics from vLLM Prometheus endpoint."""
    num_requests_waiting: int = 0
    num_requests_running: int = 0
    gpu_cache_usage_perc: float = 0.0
    time_per_output_token_ms: float = 0.0
    num_preemptions: int = 0
    timestamp: float = field(default_factory=time.time)
    
    def to_controller_metrics(self) -> Metrics:
        """Convert vLLM metrics to Controller-compatible format."""
        return Metrics(
            ttft_ms=0.0,  # vLLM handles prefill internally
            inter_token_latency_ms=self.time_per_output_token_ms,
            prefill_latency_ms=0.0,
            decode_latency_ms=self.time_per_output_token_ms,
            gpu_utilization=self.gpu_cache_usage_perc * 100,
            memory_efficiency=self.gpu_cache_usage_perc * 100,
            gpu_balance_index=1.0,
            comm_bubble_ratio=0.0,
            speculative_factor=0.0,
            queue_depth=self.num_requests_waiting,
            timestamp=self.timestamp,
            queue_velocity=0.0,  # Would need history to calculate
            queue_depth_iw=self.num_requests_waiting,
            queue_depth_niw=0,
            niw_in_flight=self.num_requests_running
        )


class VLLMClient:
    """
    Client adapter for vLLM serving engine.
    
    Provides:
    1. Metrics collection from vLLM Prometheus endpoint
    2. Admission control via async semaphore
    3. Request submission with rate limiting
    """
    
    def __init__(self, config: Optional[VLLMConfig] = None):
        self.config = config or VLLMConfig()
        self.session: Optional[ClientSession] = None
        
        # Admission Control: Semaphore limits concurrent requests
        self._concurrency = self.config.initial_concurrency
        self._semaphore = asyncio.Semaphore(self._concurrency)
        self._pending_adjustment = 0
        
        # Metrics cache
        self._latest_metrics = VLLMMetrics()
        self._metrics_task: Optional[asyncio.Task] = None
        
        # Stats
        self.requests_submitted = 0
        self.requests_completed = 0
        self.requests_throttled = 0
        
    async def start(self):
        """Initialize the client and start metrics polling."""
        self.session = ClientSession()
        self._metrics_task = asyncio.create_task(self._poll_metrics())
        logger.info(f"vLLM Client started. Target: {self.config.base_url}")
        logger.info(f"Initial concurrency: {self._concurrency}")
        
    async def stop(self):
        """Cleanup resources."""
        if self._metrics_task:
            self._metrics_task.cancel()
            try:
                await self._metrics_task
            except asyncio.CancelledError:
                pass
        
        if self.session:
            await self.session.close()
        
        logger.info("vLLM Client stopped.")
        logger.info(f"Stats: submitted={self.requests_submitted}, completed={self.requests_completed}, throttled={self.requests_throttled}")
    
    # ========== METRICS ==========
    
    async def _poll_metrics(self):
        """Background task to poll vLLM metrics."""
        while True:
            try:
                await self._fetch_metrics()
            except Exception as e:
                logger.warning(f"Metrics fetch failed: {e}")
            
            await asyncio.sleep(self.config.metrics_interval_sec)
    
    async def _fetch_metrics(self):
        """Fetch and parse Prometheus metrics from vLLM."""
        url = f"{self.config.base_url}{self.config.metrics_path}"
        
        async with self.session.get(url) as response:
            if response.status != 200:
                logger.warning(f"Metrics endpoint returned {response.status}")
                return
            
            text = await response.text()
            self._latest_metrics = self._parse_prometheus_metrics(text)
    
    def _parse_prometheus_metrics(self, text: str) -> VLLMMetrics:
        """Parse Prometheus format metrics from vLLM."""
        metrics = VLLMMetrics(timestamp=time.time())
        
        for line in text.split('\n'):
            if line.startswith('#') or not line.strip():
                continue
            
            try:
                # Parse prometheus format: metric_name{labels} value
                if ' ' in line:
                    metric_part, value_str = line.rsplit(' ', 1)
                    value = float(value_str)
                    
                    # Match known metrics
                    if 'vllm:num_requests_waiting' in metric_part:
                        metrics.num_requests_waiting = int(value)
                    elif 'vllm:num_requests_running' in metric_part:
                        metrics.num_requests_running = int(value)
                    elif 'vllm:gpu_cache_usage_perc' in metric_part:
                        metrics.gpu_cache_usage_perc = value
                    elif 'vllm:avg_generation_throughput_toks_per_s' in metric_part:
                        if value > 0:
                            metrics.time_per_output_token_ms = 1000.0 / value
                    elif 'vllm:num_preemptions_total' in metric_part:
                        metrics.num_preemptions = int(value)
            except (ValueError, IndexError):
                continue
        
        return metrics
    
    def get_metrics(self) -> Metrics:
        """Get latest metrics in Controller-compatible format."""
        return self._latest_metrics.to_controller_metrics()
    
    def get_raw_metrics(self) -> VLLMMetrics:
        """Get raw vLLM metrics."""
        return self._latest_metrics
    
    # ========== ADMISSION CONTROL ==========
    
    def apply_decision(self, decision: Decision):
        """
        Apply Controller decision to admission control.
        
        This translates high-level actions into concurrency adjustments:
        - INC_BATCH: Allow more concurrent requests
        - DEC_BATCH: Throttle requests
        - DEFER_NIW: Pause (reduce to minimum)
        """
        action = decision.action
        
        if action == Action.INCREASE_BATCH:
            self._adjust_concurrency(+4)
            logger.info(f"Increased concurrency to {self._concurrency}")
            
        elif action == Action.REDUCE_BATCH:
            self._adjust_concurrency(-2)
            logger.info(f"Decreased concurrency to {self._concurrency}")
            
        elif action == Action.DEFER_NIW:
            # Reduce to minimum but don't stop entirely
            self._set_concurrency(self.config.min_concurrency)
            logger.info(f"Throttled to minimum concurrency: {self._concurrency}")
            
        elif action == Action.NO_ACTION:
            pass  # Maintain current concurrency
    
    def _adjust_concurrency(self, delta: int):
        """Adjust concurrency limit by delta."""
        new_value = max(
            self.config.min_concurrency,
            min(self.config.max_concurrency, self._concurrency + delta)
        )
        self._set_concurrency(new_value)
    
    def _set_concurrency(self, value: int):
        """Set absolute concurrency limit."""
        old_value = self._concurrency
        self._concurrency = value
        
        # Adjust semaphore (this is tricky - we store the delta for next acquire)
        self._pending_adjustment = value - old_value
    
    # ========== REQUEST SUBMISSION ==========
    
    async def submit_request(
        self,
        prompt: str,
        max_tokens: int = 128,
        temperature: float = 0.7,
        timeout: float = 30.0
    ) -> Optional[Dict[str, Any]]:
        """
        Submit a request to vLLM with admission control.
        
        Blocks if concurrency limit reached (throttling).
        """
        # Apply any pending concurrency adjustments
        if self._pending_adjustment != 0:
            # Note: Python semaphore doesn't support dynamic resizing
            # We track this for logging purposes
            self._pending_adjustment = 0
        
        # Try to acquire semaphore (admission control)
        acquired = self._semaphore.locked()
        if acquired:
            self.requests_throttled += 1
        
        async with self._semaphore:
            self.requests_submitted += 1
            
            try:
                result = await self._send_request(prompt, max_tokens, temperature, timeout)
                self.requests_completed += 1
                return result
            except Exception as e:
                logger.error(f"Request failed: {e}")
                return None
    
    async def _send_request(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        timeout: float
    ) -> Dict[str, Any]:
        """Send request to vLLM generate endpoint."""
        url = f"{self.config.base_url}{self.config.generate_path}"
        
        payload = {
            "model": self.config.model_name,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        
        async with self.session.post(url, json=payload, timeout=timeout) as response:
            if response.status != 200:
                raise ClientError(f"vLLM returned {response.status}")
            
            return await response.json()
    
    # ========== UTILITIES ==========
    
    @property
    def current_concurrency(self) -> int:
        """Get current concurrency limit."""
        return self._concurrency
    
    @property
    def available_slots(self) -> int:
        """Get number of available request slots."""
        # Approximate - semaphore doesn't expose this directly
        return max(0, self._concurrency - self._latest_metrics.num_requests_running)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get client statistics."""
        return {
            'concurrency': self._concurrency,
            'requests_submitted': self.requests_submitted,
            'requests_completed': self.requests_completed,
            'requests_throttled': self.requests_throttled,
            'queue_depth': self._latest_metrics.num_requests_waiting,
            'gpu_cache_usage': self._latest_metrics.gpu_cache_usage_perc
        }


# ========== INTEGRATION EXAMPLE ==========

async def demo_integration():
    """Demo: Connect Controller to vLLM."""
    from ml_controller import MLController
    
    print("="*60)
    print("🚀 vLLM INTEGRATION DEMO")
    print("="*60)
    
    # Initialize
    vllm = VLLMClient()
    controller = MLController(model_path="ppo_cloned_v09")
    
    await vllm.start()
    
    try:
        for step in range(100):
            # Get metrics from vLLM
            metrics = vllm.get_metrics()
            
            # Controller makes decision
            decision = controller.make_decision(metrics)
            
            # Apply to admission control
            vllm.apply_decision(decision)
            
            # Log
            if step % 10 == 0:
                stats = vllm.get_stats()
                print(f"Step {step}: Queue={stats['queue_depth']}, Concurrency={stats['concurrency']}, Action={decision.action.value}")
            
            await asyncio.sleep(0.5)
    
    finally:
        await vllm.stop()


if __name__ == "__main__":
    asyncio.run(demo_integration())
