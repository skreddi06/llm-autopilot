"""Collector that polls metrics from the LLM server."""

import asyncio
import time
import logging
from typing import Optional, List
from aiohttp import ClientSession, ClientError, web
from models import Metrics

logger = logging.getLogger(__name__)


class Collector:
    """Collects metrics from the LLM server."""
    
    def __init__(self, server_url: str = "http://localhost:8000", poll_interval: float = 1.0):
        self.server_url = server_url
        self.poll_interval = poll_interval
        self.metrics_history: List[Metrics] = []
        self.current_metrics: Optional[Metrics] = None
        self.running = False
        self.session: Optional[ClientSession] = None
    
    async def _fetch_metrics(self) -> Optional[Metrics]:
        """Fetch metrics from the server."""
        try:
            async with self.session.get(f"{self.server_url}/metrics") as response:
                if response.status == 200:
                    data = await response.json()
                    
                    # Calculate queue velocity from history
                    queue_velocity = 0.0
                    if len(self.metrics_history) > 0:
                        prev_metrics = self.metrics_history[-1]
                        time_delta = data["timestamp"] - prev_metrics.timestamp
                        if time_delta > 0:
                            queue_delta = data["queue_depth"] - prev_metrics.queue_depth
                            queue_velocity = queue_delta / time_delta
                    
                    metrics = Metrics(
                        ttft_ms=float(data.get("ttft_ms", 0.0)),
                        inter_token_latency_ms=float(data.get("inter_token_latency_ms", 0.0)),
                        prefill_latency_ms=float(data.get("prefill_latency_ms", 0.0)),
                        decode_latency_ms=float(data.get("decode_latency_ms", 0.0)),
                        gpu_utilization=float(data.get("gpu_utilization", 0.0)),
                        memory_efficiency=float(data.get("memory_efficiency", 1.0)),
                        gpu_balance_index=float(data.get("gpu_balance_index", 1.0)),
                        comm_bubble_ratio=float(data.get("comm_bubble_ratio", 0.0)),
                        speculative_factor=float(data.get("speculative_factor", 0.0)),
                        queue_depth=int(data.get("queue_depth", 0)),
                        timestamp=float(data.get("timestamp", time.time()))
                    )
                    
                    return metrics
                else:
                    logger.warning(f"Failed to fetch metrics: HTTP {response.status}")
                    return None
        except ClientError as e:
            logger.error(f"Error fetching metrics: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error fetching metrics: {e}")
            return None
    
    async def collect_loop(self):
        """Main collection loop."""
        self.running = True
        self.session = ClientSession()
        
        logger.info(f"Collector started, polling {self.server_url} every {self.poll_interval}s")
        
        try:
            while self.running:
                metrics = await self._fetch_metrics()
                
                if metrics:
                    self.current_metrics = metrics
                    self.metrics_history.append(metrics)
                    
                    # Keep only last 100 metrics to avoid memory issues
                    if len(self.metrics_history) > 100:
                        self.metrics_history = self.metrics_history[-100:]
                    
                    logger.debug(f"Collected metrics: TTFT={metrics.ttft_ms:.1f}ms, "
                               f"GPU={metrics.gpu_utilization:.1f}%, "
                               f"Queue={metrics.queue_depth}, "
                               f"Velocity={metrics.queue_velocity:.2f}")
                
                await asyncio.sleep(self.poll_interval)
        except asyncio.CancelledError:
            logger.info("Collector loop cancelled")
        finally:
            if self.session:
                await self.session.close()
            self.running = False
            logger.info("Collector stopped")
    
    def get_current_metrics(self) -> Optional[Metrics]:
        """Get the most recent metrics."""
        return self.current_metrics
    
    def get_metrics_history(self, n: int = 10) -> List[Metrics]:
        """Get the last n metrics."""
        return self.metrics_history[-n:] if self.metrics_history else []
    
    def stop(self):
        """Stop the collector."""
        self.running = False

    def get_moving_average(self, field: str, n: int = 3) -> Optional[float]:
        """Calculate the moving average of a specific metric field over the last n metrics."""
        if len(self.metrics_history) < n:
            return None

        values = [getattr(m, field, None) for m in self.metrics_history[-n:]]
        if None in values:
            return None

        return sum(values) / len(values)

    def get_trend(self, field: str, n: int = 3) -> Optional[float]:
        """Calculate the trend (difference between latest and average) for a specific metric field."""
        if len(self.metrics_history) < n:
            return None

        latest_value = getattr(self.metrics_history[-1], field, None)
        moving_avg = self.get_moving_average(field, n)

        if latest_value is None or moving_avg is None:
            return None

        return latest_value - moving_avg

    async def live_metrics_handler(self, request):
        """HTTP handler to serve the latest metrics as JSON."""
        if self.current_metrics is None:
            return web.json_response({"error": "No metrics available yet."}, status=503)

        return web.json_response(self.current_metrics.__dict__)

    async def last_decision_handler(self, request):
        """HTTP handler to serve the last decision as JSON."""
        if hasattr(self, 'last_decision') and self.last_decision is not None:
            return web.json_response({
                "action": self.last_decision.action.value,
                "batch_size": self.last_decision.batch_size,
                "gpu_count": self.last_decision.gpu_count,
                "reason": self.last_decision.reason,
                "mode": getattr(self, 'current_mode', 'unknown')
            })
        return web.json_response({"error": "No decision available yet."}, status=503)

    def update_last_decision(self, decision, mode="unknown"):
        """Update the last decision for the /last_decision endpoint."""
        self.last_decision = decision
        self.current_mode = mode

    async def start_http_server(self, host="0.0.0.0", port=8080):
        """Start an HTTP server to expose live metrics."""
        self._app = web.Application()
        self._app.router.add_get("/live_metrics", self.live_metrics_handler)
        self._app.router.add_get("/last_decision", self.last_decision_handler)
        self._runner = web.AppRunner(self._app)
        await self._runner.setup()
        self._site = web.TCPSite(self._runner, host, port)
        await self._site.start()
        logger.info(f"Live metrics server running at http://{host}:{port}/live_metrics")

    async def stop_http_server(self):
        """Gracefully stop the live metrics server."""
        try:
            if hasattr(self, "_runner") and self._runner:
                await self._runner.cleanup()
                logger.info("Live metrics server stopped.")
        except Exception as e:
            logger.warning(f"Error shutting down live metrics server: {e}")

