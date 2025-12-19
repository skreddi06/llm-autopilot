"""
Mock vLLM Server (Phase v0.9: Integration Testing)

Simulates a vLLM instance by exposing:
- /metrics (Prometheus format)
- /v1/completions (OpenAI-compatible generate)

Physics: KV cache grows with active requests, queue grows when overloaded.
"""

from fastapi import FastAPI
from fastapi.responses import PlainTextResponse
import asyncio
import random
import uvicorn
import time

app = FastAPI(title="Mock vLLM Server")

# Simulation State
state = {
    "num_requests_waiting": 0,
    "num_requests_running": 0,
    "gpu_cache_usage_perc": 0.1,
    "total_processed": 0,
    "preemptions": 0,
    "start_time": time.time()
}

# Physics Constants
MAX_CONCURRENT = 32  # Simulated GPU capacity
KV_PER_REQUEST = 0.03  # KV cache usage per concurrent request


@app.post("/v1/completions")
async def generate(prompt: str = "Hello", max_tokens: int = 128):
    """Simulates handling a completion request with realistic physics."""
    
    # Admission: Request enters queue
    state["num_requests_waiting"] += 1
    
    # Wait for slot (simulated scheduling)
    if state["num_requests_running"] >= MAX_CONCURRENT:
        await asyncio.sleep(0.1 + state["num_requests_waiting"] * 0.02)
    else:
        await asyncio.sleep(0.02)
    
    # Move from waiting to running
    state["num_requests_waiting"] = max(0, state["num_requests_waiting"] - 1)
    state["num_requests_running"] += 1
    
    # KV Cache grows with running requests
    state["gpu_cache_usage_perc"] = min(0.99, state["num_requests_running"] * KV_PER_REQUEST)
    
    # Check for preemption (memory pressure)
    if state["gpu_cache_usage_perc"] > 0.9:
        state["preemptions"] += 1
    
    # Simulate Processing Time - moderate speed
    base_time = 0.1 + (max_tokens / 500)
    load_factor = 1.0 + (state["num_requests_running"] / MAX_CONCURRENT)
    processing_time = base_time * load_factor * random.uniform(0.9, 1.1)
    
    await asyncio.sleep(processing_time)
    
    # Cleanup
    state["num_requests_running"] = max(0, state["num_requests_running"] - 1)
    state["gpu_cache_usage_perc"] = max(0.05, state["num_requests_running"] * KV_PER_REQUEST)
    state["total_processed"] += 1
    
    return {
        "id": f"cmpl-{state['total_processed']}",
        "choices": [{"text": f"Mock response to: {prompt[:20]}...", "index": 0}],
        "usage": {"prompt_tokens": len(prompt.split()), "completion_tokens": max_tokens}
    }


@app.get("/metrics", response_class=PlainTextResponse)
async def metrics():
    """Exposes vLLM-compatible Prometheus metrics."""
    uptime = time.time() - state["start_time"]
    throughput = state["total_processed"] / max(1, uptime)
    
    return f"""# HELP vllm:num_requests_waiting Number of requests waiting in queue.
# TYPE vllm:num_requests_waiting gauge
vllm:num_requests_waiting {state['num_requests_waiting']}

# HELP vllm:num_requests_running Number of requests currently being processed.
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running {state['num_requests_running']}

# HELP vllm:gpu_cache_usage_perc GPU KV-cache usage (0.0 to 1.0).
# TYPE vllm:gpu_cache_usage_perc gauge
vllm:gpu_cache_usage_perc {state['gpu_cache_usage_perc']:.4f}

# HELP vllm:num_preemptions_total Total number of request preemptions due to memory pressure.
# TYPE vllm:num_preemptions_total counter
vllm:num_preemptions_total {state['preemptions']}

# HELP vllm:avg_generation_throughput_toks_per_s Average tokens generated per second.
# TYPE vllm:avg_generation_throughput_toks_per_s gauge
vllm:avg_generation_throughput_toks_per_s {throughput * 100:.2f}
"""


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "processed": state["total_processed"]}


@app.get("/stats")
async def stats():
    """Debug stats endpoint."""
    return state


if __name__ == "__main__":
    print("🚀 Starting Mock vLLM Server on http://localhost:8000")
    print("   Endpoints: /v1/completions, /metrics, /health")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")
