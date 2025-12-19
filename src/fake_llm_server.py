"""Fake LLM server with physics-grounded simulation.

v0.5 Phases:
- Phase 1: Conservation Law (queued + prefilling + decoding + completed = submitted)
- Phase 2: Dual-Phase Physics (prefill=compute, decode=memory)
- Phase 3: KV Cache Memory Bottleneck (18GB limit, swap/thrash penalty)
- Phase 4: Compute Saturation (roofline: max of memory and compute bound)
- Phase 5: Continuous Batching & Chunked Prefill (token budget, iteration-level scheduling)
"""

import asyncio
import time
import random
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field, asdict
from aiohttp import web
from models import ServerConfig, Metrics


@dataclass
class InFlightRequest:
    """Tracks a single request through its lifecycle with token-level state."""
    id: int
    state: str  # 'queued', 'prefilling', 'decoding', 'completed', 'deferred'
    arrival_time: float
    prefill_start: float = 0.0
    decode_start: float = 0.0
    completion_time: float = 0.0
    
    # Phase 3: Token tracking for KV cache memory
    prompt_tokens: int = 512      # Tokens in input prompt
    generated_tokens: int = 0     # Tokens generated so far (grows during decode)
    target_tokens: int = 128      # Target output length
    
    # Phase 5: Chunked prefill tracking
    prefill_tokens_processed: int = 0  # Tokens prefilled so far (grows in chunks)
    
    # Phase 7: Workload tiers (SageServe)
    priority: int = 0             # 0=Interactive (IW), 1=Non-Interactive (NIW)
    deadline_sec: float = float('inf')  # Soft deadline for NIW (seconds from arrival)
    
    @property
    def is_interactive(self) -> bool:
        """True if this is an Interactive Workload (chatbot, low latency)."""
        return self.priority == 0
    
    @property
    def is_batch(self) -> bool:
        """True if this is a Non-Interactive Workload (batch job, deferable)."""
        return self.priority >= 1
    
    @property
    def prefill_complete(self) -> bool:
        """True if all prompt tokens have been prefilled."""
        return self.prefill_tokens_processed >= self.prompt_tokens
    
    @property
    def prefill_remaining(self) -> int:
        """Tokens still needing prefill."""
        return max(0, self.prompt_tokens - self.prefill_tokens_processed)
    
    @property
    def total_tokens(self) -> int:
        """Total tokens in KV cache (prefilled prompt + generated)."""
        return self.prefill_tokens_processed + self.generated_tokens


class FakeLLMServer:
    """Simulates an LLM inference server with physics-grounded behavior.
    
    Phase 1: Conservation Law
    - Enforces: queued + prefilling + decoding + completed = submitted
    - Queue drain rate = throughput
    - Queue growth rate = arrival rate - throughput
    """
    
    def __init__(self, host: str = "localhost", port: int = 8000):
        self.host = host
        self.port = port
        self.config = ServerConfig(batch_size=4, gpu_count=1)
        
        # === CONSERVATION LAW STATE ===
        # These four counts must sum to total_submitted at all times
        self.request_counter = 0          # Auto-incrementing ID
        self.total_submitted = 0          # Total requests ever received
        self.total_submitted = 0          # Total requests ever received
        self.total_submitted = 0          # Total requests ever received
        self.completed_count = 0          # Total finished

        # Surge Control (Phase 15 Verification)
        self.surge_multiplier = 1.0
        self.surge_remaining_steps = 0
        
        # Stateful Physics Metrics (Phase 15 Fix)
        self.gpu_utilization = 0.0
        self.kv_utilization = 0.0

        # Surge Control (Phase 15 Verification)
        self.surge_multiplier = 1.0
        self.surge_remaining_steps = 0
        
        # Request pools by state
        self.queued_requests: List[InFlightRequest] = []
        self.prefilling_requests: List[InFlightRequest] = []
        self.decoding_requests: List[InFlightRequest] = []
        
        # === PHASE 2: CALIBRATED PHYSICS CONSTANTS ===
        # Source: NanoFlow Table 2 (LLaMA-2 70B, 8x A100) + Databricks benchmarks
        
        # Prefill: Compute-bound (processes all prompt tokens in parallel)
        self.prefill_tokens_per_sec = 3000.0  # A100/7B baseline from vLLM benchmarks
        self.avg_prompt_tokens = 512          # Average prompt length
        
        # Decode: Memory-bound (loads KV cache for each token generation)
        self.decode_base_step_ms = 15.0       # Base latency per decode step (low batch)
        self.decode_batch_penalty_exp = 0.5   # sqrt penalty for memory contention
        self.avg_output_tokens = 128          # Average output length
        
        # Hardware reference (A100-40GB)
        self.a100_memory_bw_gbs = 1555        # GB/s peak bandwidth
        self.a100_compute_tflops = 312        # FP16 peak
        
        # Databricks throughput baselines (for validation)
        self.static_batch_1_rps = 0.9         # req/sec at batch=1
        self.static_batch_16_rps = 8.0        # req/sec at batch=16
        self.static_batch_64_rps = 33.0       # req/sec at batch=64
        
        # === PHASE 3: KV CACHE MEMORY CONSTANTS ===
        # Source: Databricks formula + A100-40GB specs
        
        # Hardware: A100-40GB
        self.gpu_memory_bytes = 40 * 1024 * 1024 * 1024  # 40 GB
        self.model_weights_bytes = 14 * 1024 * 1024 * 1024  # 7B params × 2 bytes (FP16) = 14 GB
        self.memory_overhead_ratio = 0.20  # 20% for activations/workspace
        
        # Available KV cache memory per GPU
        overhead = self.gpu_memory_bytes * self.memory_overhead_ratio
        self.available_kv_memory_bytes = self.gpu_memory_bytes - self.model_weights_bytes - overhead
        # ~18 GB for 1 GPU
        
        # KV cache cost per token
        # Using smaller value for 70B with GQA (Grouped Query Attention)
        # GQA reduces KV cache by factor of n_heads/n_kv_heads
        # 70B with GQA: ~0.125 MB per token (vs 0.5MB without GQA)
        # This allows more requests before memory wall, revealing plateau zone
        self.kv_bytes_per_token = 128 * 1024  # 0.125 MB (GQA-optimized)
        
        # Swap/thrash penalty when memory exceeded
        self.memory_swap_penalty_ms = 300.0  # 300ms latency penalty
        self.memory_congested = False  # Current congestion state
        
        # === PHASE 4: COMPUTE SATURATION CONSTANTS ===
        # Source: Roofline model from NanoFlow/Databricks
        
        # A100 Compute Capacity
        self.a100_peak_tflops = 312          # FP16 peak
        self.effective_utilization = 0.60    # 60% realistic utilization (kernel overheads)
        self.effective_tflops = self.a100_peak_tflops * self.effective_utilization  # ~180 TFLOPS
        
        # Model compute cost (70B parameters - matches NanoFlow Table 2)
        # FLOPs per token ≈ 2 × parameters (multiply-accumulate)
        # 70B model: 2 × 70e9 = 140 GFLOPs per token
        # This ensures we hit compute ceiling BEFORE memory wall for demonstration
        self.flops_per_token = 140e9          # 140 GFLOPs per token for 70B model
        
        # Saturation threshold (penalty kicks in above this)
        self.compute_saturation_threshold = 0.80  # 80% utilization
        self.compute_saturation_factor = 5.0      # Penalty multiplier
        
        # Current compute utilization (updated each step)
        self.current_compute_utilization = 0.0
        
        # === PHASE 5: CONTINUOUS BATCHING CONSTANTS ===
        # Source: Anyscale, vLLM iteration-level scheduling
        
        # Token budget per iteration (limits compute per step)
        # 16k tokens allows mixing decode + partial prefill
        self.max_tokens_per_batch = 16384  # 16k tokens
        
        # Chunked prefill size (processes large prompts in chunks)
        # Prevents head-of-line blocking from massive prompts
        self.prefill_chunk_size = 512  # tokens per chunk
        
        # Scheduling mode
        self.continuous_batching_enabled = True  # False = static batching
        
        # Scheduling metrics (updated each step)
        self.step_stall_count = 0           # Steps where requests were blocked
        self.total_step_count = 0           # Total simulation steps
        self.decode_latencies = []          # For jitter calculation
        self.batch_occupancies = []         # For utilization tracking
        
        # === LOAD GENERATION ===
        self.arrival_rate = 0.0           # Requests per second (measured)
        self.last_arrival_time = time.time()
        self.auto_load_enabled = True
        self.auto_load_task = None
        
        # === SIMULATION LOOP ===
        self.simulation_task = None
        self.simulation_interval = 0.1    # 100ms ticks
        
        # === METRICS TRACKING ===
        self.last_metrics_time = time.time()
        self.prev_queue_depth = 0
        
    # ========== CONSERVATION LAW HELPERS ==========
    
    @property
    def queue_depth(self) -> int:
        """Number of requests waiting in queue."""
        return len(self.queued_requests)
    
    @property
    def prefilling_count(self) -> int:
        """Number of requests currently prefilling."""
        return len(self.prefilling_requests)
    
    @property
    def decoding_count(self) -> int:
        """Number of requests currently decoding."""
        return len(self.decoding_requests)
    
    def verify_conservation(self) -> bool:
        """Verify conservation law: queued + prefilling + decoding + completed = submitted."""
        actual_sum = self.queue_depth + self.prefilling_count + self.decoding_count + self.completed_count
        return actual_sum == self.total_submitted
    
    def get_conservation_stats(self) -> Dict[str, int]:
        """Return conservation law breakdown for debugging."""
        return {
            "queued": self.queue_depth,
            "prefilling": self.prefilling_count,
            "decoding": self.decoding_count,
            "completed": self.completed_count,
            "submitted": self.total_submitted,
            "conservation_valid": self.verify_conservation()
        }
    
    # ========== PHASE 2: PHYSICS CALCULATIONS ==========
    
    def calculate_prefill_latency_ms(self) -> float:
        """Calculate prefill latency (compute-bound).
        
        Prefill processes all prompt tokens in parallel.
        Scales linearly with token count, inversely with GPU count.
        
        Formula: prefill_latency = total_prompt_tokens / tokens_per_second
        
        Note: Unlike decode, prefill does NOT degrade significantly with
        batch size until compute ceiling (Phase 4 saturation).
        """
        tokens_per_sec = self.prefill_tokens_per_sec * self.config.gpu_count
        prefill_ms = (self.avg_prompt_tokens / tokens_per_sec) * 1000
        return prefill_ms
    
    def calculate_decode_latency_ms(self) -> float:
        """Calculate decode latency per token (memory-bound).
        
        Decode loads KV cache for each token generation.
        Memory bandwidth is the bottleneck, causing sqrt degradation with batch.
        
        Formula: decode_latency = base_step_time * (1 + penalty * batch^0.5)
        
        This models: larger batch = more KV cache reads = memory contention.
        """
        import math
        batch = self.config.batch_size
        gpus = self.config.gpu_count
        
        # Base decode time (scales inversely with GPU count)
        base_ms = self.decode_base_step_ms / gpus
        
        # Memory contention penalty: sqrt(batch) scaling
        # At batch=1: penalty=0, at batch=16: penalty ~3, at batch=64: penalty ~7
        penalty = (math.sqrt(batch) - 1) * 0.3  # 30% penalty per sqrt unit
        
        decode_ms = base_ms * (1 + penalty)
        return decode_ms
    
    def calculate_throughput_rps(self) -> float:
        """Calculate current throughput based on batch size and GPU count.
        
        Phase 2: Uses Databricks baselines for interpolation.
        Throughput increases with batch (amortizes weight loading).
        """
        batch = self.config.batch_size
        gpus = self.config.gpu_count
        
        # Interpolate from Databricks static batching data
        if batch <= 1:
            throughput_per_gpu = self.static_batch_1_rps
        elif batch <= 16:
            # Linear interpolation: batch=1 → 0.9, batch=16 → 8.0
            t = (batch - 1) / 15
            throughput_per_gpu = self.static_batch_1_rps + t * (self.static_batch_16_rps - self.static_batch_1_rps)
        else:
            # Sublinear scaling above 16: batch=64 → 33.0
            t = (batch - 16) / 48
            throughput_per_gpu = self.static_batch_16_rps + t * (self.static_batch_64_rps - self.static_batch_16_rps)
        
        return throughput_per_gpu * gpus
    
    # ========== PHASE 3: MEMORY CALCULATIONS ==========
    
    def calculate_kv_memory_usage(self) -> int:
        """Calculate total KV cache memory usage in bytes.
        
        Sums tokens across all active requests (prefilling + decoding).
        Each token costs ~0.5 MB in KV cache.
        """
        total_tokens = 0
        
        # Prefilling requests have prompt tokens in cache
        for req in self.prefilling_requests:
            total_tokens += req.prompt_tokens
        
        # Decoding requests have prompt + generated tokens
        for req in self.decoding_requests:
            total_tokens += req.total_tokens
        
        return total_tokens * self.kv_bytes_per_token
    
    def get_available_kv_memory(self) -> int:
        """Total KV memory available across all GPUs."""
        return self.available_kv_memory_bytes * self.config.gpu_count
    
    def get_memory_utilization(self) -> float:
        """Memory utilization as percentage (0-100)."""
        used = self.calculate_kv_memory_usage()
        available = self.get_available_kv_memory()
        return min(100.0, (used / available) * 100) if available > 0 else 100.0
    
    def check_memory_congestion(self) -> bool:
        """Check if memory is congested (usage > available).
        
        Returns True if system should apply swap penalty.
        """
        used = self.calculate_kv_memory_usage()
        available = self.get_available_kv_memory()
        self.memory_congested = used > available
        return self.memory_congested
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Return memory statistics for debugging/metrics."""
        used = self.calculate_kv_memory_usage()
        available = self.get_available_kv_memory()
        return {
            "kv_memory_used_mb": used / (1024 * 1024),
            "kv_memory_available_mb": available / (1024 * 1024),
            "memory_utilization_pct": self.get_memory_utilization(),
            "memory_congested": self.memory_congested,
            "total_active_tokens": sum(r.total_tokens for r in self.decoding_requests) + \
                                   sum(r.prompt_tokens for r in self.prefilling_requests)
        }
    
    # ========== PHASE 4: COMPUTE SATURATION ==========
    
    def calculate_compute_flops(self) -> float:
        """Calculate FLOPs needed for current batch.
        
        FLOPs ≈ 2 × parameters × active_tokens (for multiply-accumulate)
        """
        # Active tokens = batch size (one token per request per step)
        active_batch = self.decoding_count + self.prefilling_count
        return self.flops_per_token * max(1, active_batch)
    
    def get_compute_utilization(self) -> float:
        """Compute utilization as fraction (0.0-1.0+)."""
        flops_needed = self.calculate_compute_flops()
        max_flops = self.effective_tflops * 1e12 * self.config.gpu_count
        return flops_needed / max_flops if max_flops > 0 else 1.0
    
    def calculate_roofline_latency_ms(self, batch_size: int = None) -> float:
        """Calculate step latency using roofline model.
        
        The roofline model takes the MAXIMUM of:
        1. Memory-bound latency (loading KV cache from HBM)
        2. Compute-bound latency (matrix multiply throughput)
        
        Plus non-linear saturation penalty when utilization > 80%.
        """
        import math
        
        if batch_size is None:
            batch_size = self.config.batch_size
        gpus = self.config.gpu_count
        
        # === 1. MEMORY COMPONENT (Phase 2 logic) ===
        # Base overhead + sqrt penalty for KV cache contention
        base_ms = self.decode_base_step_ms / gpus
        memory_penalty = (math.sqrt(batch_size) - 1) * 0.3
        t_memory_ms = base_ms * (1 + memory_penalty)
        
        # === 2. COMPUTE COMPONENT (Phase 4 logic) ===
        # FLOPs needed for this batch
        flops_needed = self.flops_per_token * batch_size
        # Time = Work / Rate (convert TFLOPS to FLOPS)
        max_flops_per_sec = self.effective_tflops * 1e12 * gpus
        t_compute_ms = (flops_needed / max_flops_per_sec) * 1000
        
        # === 3. ROOFLINE: MAX of constraints ===
        step_latency = max(t_memory_ms, t_compute_ms)
        
        # === 4. SATURATION PENALTY (The "Squeeze") ===
        # As we approach 100% utilization, GPU warp schedulers jam
        compute_utilization = flops_needed / max_flops_per_sec
        self.current_compute_utilization = compute_utilization
        
        if compute_utilization > self.compute_saturation_threshold:
            # Exponential drag near 100%
            excess = compute_utilization - self.compute_saturation_threshold
            penalty_factor = 1.0 + excess * self.compute_saturation_factor
            step_latency *= penalty_factor
        
        return step_latency
    
    def get_compute_stats(self) -> Dict[str, Any]:
        """Return compute statistics for debugging/metrics."""
        flops_needed = self.calculate_compute_flops()
        max_flops = self.effective_tflops * 1e12 * self.config.gpu_count
        return {
            "flops_needed_gflops": flops_needed / 1e9,
            "max_flops_tflops": max_flops / 1e12,
            "compute_utilization_pct": self.get_compute_utilization() * 100,
            "compute_saturated": self.get_compute_utilization() > self.compute_saturation_threshold
        }
    
    # ========== PHASE 5: SCHEDULING METRICS ==========
    
    def get_scheduling_stats(self) -> Dict[str, Any]:
        """Return scheduling metrics for debugging/analysis."""
        import statistics
        
        # Stall ratio: % of steps where requests were blocked
        stall_ratio = self.step_stall_count / max(1, self.total_step_count)
        
        # Decode jitter: standard deviation of decode latencies
        decode_jitter = 0.0
        if len(self.decode_latencies) > 1:
            decode_jitter = statistics.stdev(self.decode_latencies)
        
        # Average batch occupancy
        avg_occupancy = 0.0
        if self.batch_occupancies:
            avg_occupancy = sum(self.batch_occupancies) / len(self.batch_occupancies)
        
        return {
            "stall_ratio_pct": stall_ratio * 100,
            "decode_jitter_ms": decode_jitter,
            "avg_batch_occupancy_pct": avg_occupancy * 100,
            "total_steps": self.total_step_count,
            "stall_steps": self.step_stall_count,
            "continuous_batching_enabled": self.continuous_batching_enabled
        }
    
    # ========== SIMULATION LOOP ==========
    
    async def simulation_loop(self):
        """Main simulation tick - processes requests through lifecycle.
        
        Phase 5: Continuous Batching with Token Budget
        
        Each tick:
        1. Schedule: Allocate token budget (decode first, then chunked prefill)
        2. Process: Generate decode tokens, process prefill chunks
        3. Complete: Move finished prefills to decode, finished decodes to completed
        4. Track metrics: stall_ratio, batch_occupancy
        """
        while True:
            current_time = time.time()
            self.total_step_count += 1
            
            # === PHASE 3: CHECK MEMORY CONGESTION ===
            swap_penalty_factor = 1.0
            if self.check_memory_congestion():
                swap_penalty_factor = 1.0 + (self.memory_swap_penalty_ms / 100.0)
            
            # === PHASE 5: TOKEN BUDGET SCHEDULING ===
            if self.continuous_batching_enabled:
                # Token budget for this iteration
                remaining_budget = self.max_tokens_per_batch
                tokens_scheduled_decode = 0
                tokens_scheduled_prefill = 0
                
                # PRIORITY 1: Allocate budget to decode requests (1 token per request)
                # Decode requests have priority to minimize latency jitter
                decode_tokens = min(len(self.decoding_requests), remaining_budget)
                tokens_scheduled_decode = decode_tokens
                remaining_budget -= decode_tokens
                
                # PRIORITY 2: Allocate remaining budget to prefill (chunked)
                # Process existing prefilling requests first
                for req in self.prefilling_requests:
                    if remaining_budget <= 0:
                        break
                    if not req.prefill_complete:
                        # Process up to chunk_size or remaining prompt
                        tokens_to_process = min(self.prefill_chunk_size, 
                                               req.prefill_remaining, 
                                               remaining_budget)
                        req.prefill_tokens_processed += tokens_to_process
                        tokens_scheduled_prefill += tokens_to_process
                        remaining_budget -= tokens_to_process
                
                # PRIORITY 3: Pull new requests from queue and start chunked prefill
                memory_headroom = self.get_available_kv_memory() - self.calculate_kv_memory_usage()
                
                stalled = False
                while remaining_budget > 0 and self.queued_requests:
                    # Check memory for new request
                    next_req = self.queued_requests[0]
                    req_memory_cost = min(self.prefill_chunk_size, next_req.prompt_tokens) * self.kv_bytes_per_token
                    
                    if req_memory_cost > memory_headroom:
                        stalled = True
                        break
                    
                    # Start prefilling this request
                    req = self.queued_requests.pop(0)
                    req.state = 'prefilling'
                    req.prefill_start = current_time
                    
                    # Process first chunk
                    tokens_to_process = min(self.prefill_chunk_size, 
                                           req.prompt_tokens, 
                                           remaining_budget)
                    req.prefill_tokens_processed = tokens_to_process
                    tokens_scheduled_prefill += tokens_to_process
                    remaining_budget -= tokens_to_process
                    memory_headroom -= tokens_to_process * self.kv_bytes_per_token
                    
                    self.prefilling_requests.append(req)
                
                if stalled:
                    self.step_stall_count += 1
                
                # Track batch occupancy
                total_tokens = tokens_scheduled_decode + tokens_scheduled_prefill
                occupancy = total_tokens / self.max_tokens_per_batch
                self.batch_occupancies.append(occupancy)
            
            else:
                # STATIC BATCHING (Phase 4 behavior for comparison)
                # No token budget - just batch slot limits
                pass  # Use old logic below
            
            # === PHASE 4: ROOFLINE LATENCY ===
            step_latency_ms = self.calculate_roofline_latency_ms() * swap_penalty_factor
            
            # === DECODE: Generate tokens ===
            tokens_per_tick = max(1, int(self.simulation_interval * 1000 / step_latency_ms))
            for req in self.decoding_requests:
                if req.generated_tokens < req.target_tokens:
                    req.generated_tokens = min(req.target_tokens, req.generated_tokens + tokens_per_tick)
                    # Track decode latency for jitter calculation
                    if self.continuous_batching_enabled:
                        self.decode_latencies.append(step_latency_ms)
            
            # === COMPLETE: Move finished decodes to completed ===
            decode_duration_sec = (step_latency_ms * self.avg_output_tokens) / 1000
            for req in self.decoding_requests[:]:
                if req.generated_tokens >= req.target_tokens:
                    req.state = 'completed'
                    req.completion_time = current_time
                    self.decoding_requests.remove(req)
                    self.completed_count += 1
            
            # === TRANSITION: Move completed prefills to decode ===
            for req in self.prefilling_requests[:]:
                if req.prefill_complete:
                    req.state = 'decoding'
                    req.decode_start = current_time
                    req.generated_tokens = 0
                    self.prefilling_requests.remove(req)
                    self.decoding_requests.append(req)
            
            # === VERIFY CONSERVATION LAW ===
            if not self.verify_conservation():
                print(f"[WARN] Conservation law violated! {self.get_conservation_stats()}")
            
            # Update stateful metrics (Phase 15 Fix)
            self._calculate_metrics()
            
            # Update stateful metrics (Phase 15 Fix)
            self._calculate_metrics()

            await asyncio.sleep(self.simulation_interval)
    
    # ========== METRICS ==========
    
    def _calculate_metrics(self) -> Metrics:
        """Calculate current metrics based on conservation law state."""
        current_time = time.time()
        
        # Queue velocity (requests per second change)
        time_delta = max(0.1, current_time - self.last_metrics_time)
        queue_velocity = (self.queue_depth - self.prev_queue_depth) / time_delta
        self.prev_queue_depth = self.queue_depth
        self.last_metrics_time = current_time
        
        # Throughput
        throughput_rps = self.calculate_throughput_rps()
        
        # GPU utilization: based on active slots vs capacity
        active_slots = self.prefilling_count + self.decoding_count
        capacity = self.config.batch_size * self.config.gpu_count
        gpu_util_val = min(100.0, (active_slots / max(1, capacity)) * 100)
        
        # PERSIST STATE (The Fix)
        # We store these so the Simulation Loop can use them for saturation penalties
        self.gpu_utilization = gpu_util_val / 100.0 # Store as 0-1 float
        
        # KV Utilization
        # Approx: Total tokens vs Available
        # (This is rough, real physics tracks headroom precisely in simulation_loop)
        # We'll use a derived value here for the Controller to see
        total_tokens = sum([r.total_tokens for r in self.prefilling_requests + self.decoding_requests])
        max_tokens = self.available_kv_memory_bytes / self.kv_bytes_per_token
        self.kv_utilization = min(1.0, total_tokens / max(1, max_tokens * self.config.gpu_count))

        # Return Metrics object
        # Note: We return gpu_utilization as 0-100 for Metrics class compatibility

        
        # === PHASE 2: Physics-based latency ===
        prefill_latency_ms = self.calculate_prefill_latency_ms()
        decode_latency_per_token_ms = self.calculate_decode_latency_ms()
        
        # TTFT = queue wait + prefill time
        avg_queue_wait_ms = 0.0
        if self.queued_requests:
            wait_times = [current_time - r.arrival_time for r in self.queued_requests]
            avg_queue_wait_ms = (sum(wait_times) / len(wait_times)) * 1000
        
        ttft_ms = avg_queue_wait_ms + prefill_latency_ms
        
        # Total decode latency for full response
        total_decode_ms = decode_latency_per_token_ms * self.avg_output_tokens
        
        # Memory efficiency (placeholder for Phase 3)
        memory_efficiency = 1.0 - (self.queue_depth / 200) if self.queue_depth < 200 else 0.5
        
        return Metrics(
            ttft_ms=ttft_ms,
            inter_token_latency_ms=decode_latency_per_token_ms,
            prefill_latency_ms=prefill_latency_ms,
            decode_latency_ms=total_decode_ms,
            gpu_utilization=gpu_util_val,
            memory_efficiency=max(0.5, memory_efficiency),
            gpu_balance_index=1.0,  # Placeholder
            comm_bubble_ratio=0.0,  # Placeholder
            speculative_factor=0.0,  # Placeholder
            queue_depth=self.queue_depth,
            timestamp=current_time,
            queue_velocity=queue_velocity
        )
    
    # ========== HTTP HANDLERS ==========
    
    async def handle_inference(self, request: web.Request) -> web.Response:
        """Handle inference request - adds to queue."""
        current_time = time.time()
        
        # Update arrival rate measurement
        time_since_last = current_time - self.last_arrival_time
        if time_since_last > 0.01:  # Avoid division by zero
            self.arrival_rate = 0.8 * self.arrival_rate + 0.2 * (1.0 / time_since_last)
        self.last_arrival_time = current_time
        
        # Create new request and add to queue
        self.request_counter += 1
        self.total_submitted += 1
        
        new_request = InFlightRequest(
            id=self.request_counter,
            state='queued',
            arrival_time=current_time
        )
        self.queued_requests.append(new_request)
        
        # Simulate processing delay (for the HTTP response)
        await asyncio.sleep(0.01)
        
        return web.json_response({
            "status": "queued",
            "request_id": new_request.id,
            "queue_position": self.queue_depth
        })
    
    async def handle_metrics(self, request: web.Request) -> web.Response:
        """Return current metrics with conservation law stats."""
        metrics = self._calculate_metrics()
        response = asdict(metrics)
        
        # Add conservation law stats
        response["conservation"] = self.get_conservation_stats()
        response["throughput_rps"] = self.calculate_throughput_rps()
        response["prefilling_count"] = self.prefilling_count
        response["decoding_count"] = self.decoding_count
        
        return web.json_response(response)
    
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
        """Toggle auto-load generation."""
        if request.method == "GET":
            return web.json_response({"auto_load": self.auto_load_enabled})
        
        data = await request.json()
        self.auto_load_enabled = bool(data.get("enabled", True))
        return web.json_response({"auto_load": self.auto_load_enabled})
    
    async def handle_reset(self, request: web.Request) -> web.Response:
        """Reset all state - clears queues and counters."""
        # Clear all request pools
        self.queued_requests.clear()
        self.prefilling_requests.clear()
        self.decoding_requests.clear()
        
        # Reset counters
        self.total_submitted = 0
        self.completed_count = 0
        self.arrival_rate = 0.0
        
        return web.json_response({
            "status": "reset",
            "conservation": self.get_conservation_stats()
        })
    
    async def handle_conservation(self, request: web.Request) -> web.Response:
        """Debug endpoint: return conservation law state."""
        return web.json_response(self.get_conservation_stats())

    def inject_surge(self, intensity: float, duration: int):
        """
        Multiplies traffic arrival rate by 'intensity' for 'duration' steps.
        Used to test if the Shield kicks in during sudden load.
        """
        self.surge_multiplier = intensity
        self.surge_remaining_steps = duration
        print(f"\n⚠️  INJECTING SURGE: {intensity}x load for {duration} steps!")
    
    # ========== LOAD GENERATION ==========
    
    async def _generate_load(self):
        """Simulate fluctuating traffic."""
        while self.auto_load_enabled:
            # Generate burst proportional to throughput capacity
            throughput = self.calculate_throughput_rps()
            
            # Apply Surge Multiplier
            current_multiplier = self.surge_multiplier
            if self.surge_remaining_steps > 0:
                self.surge_remaining_steps -= 1
            else:
                self.surge_multiplier = 1.0
                
            # Target slightly above throughput to create queue pressure
            target_rps = throughput * 1.2 * current_multiplier
            burst_size = max(1, int(target_rps * 2))
            
            for _ in range(burst_size):
                if not self.auto_load_enabled:
                    break
                # Submit request
                self.request_counter += 1
                self.total_submitted += 1
                new_request = InFlightRequest(
                    id=self.request_counter,
                    state='queued',
                    arrival_time=time.time()
                )
                self.queued_requests.append(new_request)
                await asyncio.sleep(random.uniform(0.1, 0.3))
            
            if not self.auto_load_enabled:
                break
            
            # Cooldown
            await asyncio.sleep(random.uniform(2.0, 4.0))
    
    # ========== SERVER LIFECYCLE ==========
    
    def create_app(self) -> web.Application:
        """Create the aiohttp application."""
        app = web.Application()
        app.router.add_post("/inference", self.handle_inference)
        app.router.add_get("/metrics", self.handle_metrics)
        app.router.add_post("/configure", self.handle_configure)
        app.router.add_get("/config", self.handle_get_config)
        app.router.add_get("/auto_load", self.handle_auto_load)
        app.router.add_post("/auto_load", self.handle_auto_load)
        app.router.add_post("/reset", self.handle_reset)
        app.router.add_get("/conservation", self.handle_conservation)
        return app
    
    async def start(self):
        """Start the server and simulation loop."""
        app = self.create_app()
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, self.host, self.port)
        await site.start()
        print(f"Fake LLM Server (v0.5 Phase 1) started on http://{self.host}:{self.port}")
        
        # Start simulation loop
        self.simulation_task = asyncio.create_task(self.simulation_loop())
        
        # Start auto-load generator
        self.auto_load_task = asyncio.create_task(self._generate_load())
        
        return runner
    
    async def stop(self):
        """Stop the server."""
        self.auto_load_enabled = False
        if self.auto_load_task:
            self.auto_load_task.cancel()
            try:
                await self.auto_load_task
            except asyncio.CancelledError:
                pass
        
        if self.simulation_task:
            self.simulation_task.cancel()
            try:
                await self.simulation_task
            except asyncio.CancelledError:
                pass
        
        print("Fake LLM Server stopped.")


# Alias for backward compatibility
MetricsCollector = None  # Collector is separate


async def main():
    """Run the fake LLM server standalone."""
    server = FakeLLMServer()
    runner = await server.start()
    
    try:
        await asyncio.Event().wait()
    except KeyboardInterrupt:
        print("\nShutting down server...")
    finally:
        await runner.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
