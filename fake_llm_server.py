"""Fake LLM server that simulates inference with configurable metrics."""

import asyncio
import time
import json
import random
from typing import Dict, Any
from aiohttp import web
from dataclasses import asdict
from models import ServerConfig, Metrics


class FakeLLMServer:
    """Simulates an LLM inference server with realistic behavior."""
    
    def __init__(self, host: str = "localhost", port: int = 8000):
        self.host = host
        self.port = port
        self.config = ServerConfig(batch_size=4, gpu_count=1)
        
        # Internal state
        self.queue_depth = 0
        self.request_rate = 0.0  # requests per second
        self.last_request_time = time.time()
        self.processing_requests = 0
        
        # Base latencies (in milliseconds)
        self.base_ttft = 200.0
        self.base_inter_token = 50.0
        
        # Processing parameters
        self.processing_rate = 0.0  # requests processed per second
        self.auto_load_task = None
        self.auto_load_enabled = True
        
    def _calculate_metrics(self) -> Metrics:
        """Calculate current metrics based on configuration and load."""
        current_time = time.time()

        # --- simulate latency growth ---
        ttft_ms = self.base_ttft + (self.config.batch_size ** 1.5) * 60.0 + self.queue_depth * 40.0
        inter_token_ms = self.base_inter_token + (self.config.batch_size - 1) * 6.0

        # Prefill and decode latencies
        prefill_latency_ms = ttft_ms * 0.6  # 60% of TTFT
        decode_latency_ms = ttft_ms * 0.4  # 40% of TTFT

        # --- simulate load buildup ---
        self.processing_rate = (self.config.batch_size * self.config.gpu_count) / 8.0
        load_factor = (self.processing_requests * 2 + self.queue_depth * 0.8) / max(1, self.config.batch_size * self.config.gpu_count)
        gpu_utilization = min(100.0, load_factor * 120.0)

        # Memory efficiency and speculative decoding
        memory_efficiency = max(0.6, 1.0 - (self.queue_depth / 200))  # Decreases with queue depth
        speculative_factor = min(1.0, 0.5 + (self.config.batch_size / 32))  # Increases with batch size

        # GPU balance and communication bubble
        gpu_balance_index = max(0.0, 1.0 - abs(self.config.gpu_count - 4) / 4.0)  # Optimal at 4 GPUs
        comm_bubble_ratio = min(0.2, self.queue_depth / 500)  # Higher queue depth increases comm bubble

        # --- queue dynamics ---
        if self.request_rate > self.processing_rate:
            self.queue_depth += int((self.request_rate - self.processing_rate) * 5)
        else:
            self.queue_depth = max(0, self.queue_depth - 1)

        # If GPU saturates, latency balloons
        if gpu_utilization > 85:
            ttft_ms *= 1.4
            inter_token_ms *= 1.2

        queue_velocity = (self.request_rate - self.processing_rate) / 5.0

        return Metrics(
            ttft_ms=ttft_ms,
            inter_token_latency_ms=inter_token_ms,
            prefill_latency_ms=prefill_latency_ms,
            decode_latency_ms=decode_latency_ms,
            gpu_utilization=gpu_utilization,
            memory_efficiency=memory_efficiency,
            gpu_balance_index=gpu_balance_index,
            comm_bubble_ratio=comm_bubble_ratio,
            speculative_factor=speculative_factor,
            queue_depth=self.queue_depth,
            timestamp=current_time
        )

    async def handle_inference(self, request: web.Request) -> web.Response:
        """Handle inference requests."""
        self.processing_requests += 1
        self.request_rate = 1.0 / max(0.1, time.time() - self.last_request_time)
        self.last_request_time = time.time()

        # Simulate processing time
        await asyncio.sleep(0.4)

        self.processing_requests = max(0, self.processing_requests - 1)

        return web.json_response({
            "status": "success",
            "tokens": ["Hello", "world"]
        })
    
    async def handle_metrics(self, request: web.Request) -> web.Response:
        """Return current metrics."""
        metrics = self._calculate_metrics()
        return web.json_response(asdict(metrics))
    
    async def handle_configure(self, request: web.Request) -> web.Response:
        """Accept configuration changes."""
        try:
            data = await request.json()
            
            if "batch_size" in data:
                new_batch_size = int(data["batch_size"])
                if 1 <= new_batch_size <= 32:
                    self.config.batch_size = new_batch_size
            
            if "gpu_count" in data:
                new_gpu_count = int(data["gpu_count"])
                if 1 <= new_gpu_count <= 8:
                    self.config.gpu_count = new_gpu_count
            
            return web.json_response({
                "status": "success",
                "config": asdict(self.config)
            })
        except Exception as e:
            return web.json_response({
                "status": "error",
                "message": str(e)
            }, status=400)
    
    async def handle_get_config(self, request: web.Request) -> web.Response:
        """Return current configuration."""
        return web.json_response(asdict(self.config))
    
    async def handle_auto_load(self, request: web.Request) -> web.Response:
        """Toggle auto-load generation on or off."""
        data = await request.json()
        self.auto_load_enabled = bool(data.get("enabled", True))
        return web.json_response({"auto_load": self.auto_load_enabled})

    def create_app(self) -> web.Application:
        """Create the aiohttp application."""
        app = web.Application()
        app.router.add_post("/inference", self.handle_inference)
        app.router.add_get("/metrics", self.handle_metrics)
        app.router.add_post("/configure", self.handle_configure)
        app.router.add_get("/config", self.handle_get_config)
        app.router.add_post("/auto_load", self.handle_auto_load)  # New endpoint
        return app
    
    async def _generate_load(self):
        """Simulate fluctuating traffic automatically."""
        while self.auto_load_enabled:
            # Simulate random bursts of traffic
            burst = random.randint(5, 20)  # Random burst size
            interval = random.uniform(0.05, 0.2)  # Random interval between requests

            for _ in range(burst):
                if self.queue_depth < 100:  # Prevent runaway
                    self.request_rate += 1.0 * (1 - min(self.queue_depth / 100, 0.9))
                await asyncio.sleep(interval)

            # Cooldown period
            await asyncio.sleep(random.uniform(1.0, 3.0))

            # Decay request_rate naturally
            self.request_rate = max(0.0, self.request_rate * 0.7)  # Slowly decays

    async def start(self):
        """Start the server."""
        app = self.create_app()
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, self.host, self.port)
        await site.start()
        print(f"Fake LLM Server started on http://{self.host}:{self.port}")

        # Start auto-load generator
        self.auto_load_task = asyncio.create_task(self._generate_load())
        return runner

    async def stop(self):
        """Stop the server and auto-load generator."""
        self.auto_load_enabled = False
        if self.auto_load_task:
            await self.auto_load_task
        print("Auto-load generator stopped.")


async def main():
    """Run the fake LLM server standalone."""
    server = FakeLLMServer()
    runner = await server.start()
    
    try:
        # Keep running
        await asyncio.Event().wait()
    except KeyboardInterrupt:
        print("\nShutting down server...")
    finally:
        await runner.cleanup()


if __name__ == "__main__":
    asyncio.run(main())

