# Calibrated LLM Inference Simulator - Physical Model Specification

## Design Philosophy

**Current Problem**: `fake_llm_server.py` uses arbitrary formulas (`ttft = 200 + batch^1.5 * 60`) that don't reflect real LLM inference physics, leading to:
- Controllers learning simulator artifacts, not reality
- Actions with unclear causal impact (Action Paradox)
- Queue dynamics that violate conservation of requests

**Goal**: Build a physically grounded simulator that:
1. Models real bottlenecks (prefill compute, decode memory bandwidth, KV cache)
2. Makes actions causally meaningful (speculative decode reduces latency by measurable %, not magic)
3. Conserves requests (queue + processing + completed = total submitted)
4. Produces metrics consistent with production LLM serving (vLLM, TensorRT-LLM, etc.)

---

## Core Physics Model

### 1. Request Lifecycle

Every request goes through:
```
[Submitted] → [Queued] → [Prefill] → [Decode] → [Completed]
```

**State Variables**:
- `queue_depth`: Requests waiting to start prefill
- `prefilling_requests`: Currently in prefill phase
- `decoding_requests`: Currently in decode phase (active batch)
- `completed_requests`: Total finished (for rate calculation)

**Conservation Law** (critical):
```
queue_depth + prefilling_requests + decoding_requests + completed_requests = total_submitted
```

### 2. Prefill Phase (Compute-Bound)

**Purpose**: Process input tokens, generate KV cache.

**Latency Model**:
```python
prefill_time_ms = (prompt_length / tokens_per_second_prefill) * 1000

tokens_per_second_prefill = base_prefill_tps * gpu_count * (1 - saturation_penalty)
```

**Saturation Penalty** (GPU utilization):
```python
# Based on real vLLM behavior: throughput drops ~30% when util > 85%
if gpu_utilization > 85:
    saturation_penalty = 0.15 * ((gpu_utilization - 85) / 15)  # Linear 85→100% = 0→15% penalty
elif gpu_utilization > 95:
    saturation_penalty = 0.15 + 0.20 * ((gpu_utilization - 95) / 5)  # Severe 95→100% = +20%
else:
    saturation_penalty = 0
```

**Realistic Base Values** (A100 GPU, 7B model):
```python
base_prefill_tps = 3000  # tokens/sec per GPU
avg_prompt_length = 512  # tokens (configurable per request)
```

**TTFT (Time To First Token)**:
```python
ttft_ms = queue_wait_time_ms + prefill_time_ms
```

### 3. Decode Phase (Memory-Bandwidth-Bound)

**Purpose**: Generate output tokens autoregressively.

**Latency Model**:
```python
decode_time_per_token_ms = 1000 / tokens_per_second_decode

tokens_per_second_decode = base_decode_tps * gpu_count / batch_size^0.5
```

**Why `batch_size^0.5` penalty?**
- Decode is memory-bandwidth limited (reading KV cache)
- Larger batch = more KV cache reads = contention
- Real vLLM data shows ~sqrt scaling penalty, not linear

**Realistic Base Values**:
```python
base_decode_tps = 200  # tokens/sec per GPU (much slower than prefill)
avg_output_length = 128  # tokens per response
```

**Total Decode Latency**:
```python
total_decode_ms = decode_time_per_token_ms * avg_output_length
```

### 4. Batch Scheduling

**Dynamic Batching** (continuous batching like vLLM):
```python
# Batch slots available (limited by memory)
max_batch_size = config.batch_size  # Controller sets this

# Requests we can prefill this cycle
available_slots = max_batch_size - decoding_requests

# Pull from queue
requests_to_prefill = min(queue_depth, available_slots)
```

**Iteration Loop** (every 100ms):
```python
1. Finish any completed decode sequences → move to completed
2. Start prefilling new requests from queue (up to available_slots)
3. Continue decoding active batch
4. Update metrics (GPU util, queue, TTFT)
```

### 5. GPU Utilization Model

**Compute Utilization**:
```python
# Prefill dominates compute (matrix multiplies)
prefill_compute_load = (prefilling_requests * avg_prompt_length) / (base_prefill_tps * gpu_count)

# Decode uses compute too, but less
decode_compute_load = (decoding_requests * 1) / (base_decode_tps * gpu_count)

gpu_utilization = min(100, (prefill_compute_load + decode_compute_load * 0.3) * 100)
```

**Memory Utilization**:
```python
# KV cache per request (critical in decode)
kv_cache_per_request_mb = (avg_prompt_length + avg_output_length) * 0.5  # ~0.5MB per token (7B model)

total_kv_cache_mb = (decoding_requests + prefilling_requests) * kv_cache_per_request_mb
gpu_memory_total_mb = 40960 * gpu_count  # A100 80GB

memory_utilization = (total_kv_cache_mb / gpu_memory_total_mb) * 100
```

### 6. Queue Dynamics (Conservation-Respecting)

**Queue Growth**:
```python
# Incoming request rate (from auto-load or external)
arrival_rate = request_rate  # requests/sec

queue_delta_in = arrival_rate * time_delta_sec
```

**Queue Drain**:
```python
# Drain = requests we can pull into prefill
drain_rate = available_slots / iteration_interval_sec

queue_delta_out = min(queue_depth, drain_rate * time_delta_sec)
```

**Net Queue Change**:
```python
queue_depth += queue_delta_in - queue_delta_out
```

**Critical**: No magic multipliers. Queue is a FIFO buffer.

---

## Action Semantics (Causal Impact)

### INCREASE_BATCH / REDUCE_BATCH

**Effect**:
```python
# Larger batch = more concurrent decode = higher throughput BUT higher latency per token
batch_size = new_value

# Impacts:
# 1. Decode latency increases (memory contention)
decode_time_per_token_ms *= sqrt(batch_size / old_batch_size)

# 2. More slots available = faster queue drain
available_slots = batch_size - decoding_requests  # Can pull more from queue

# 3. GPU memory pressure increases
memory_utilization recalculated
```

**Trade-off**:
- ↑ batch → ↑ throughput, ↓ per-request latency (if queue is deep)
- ↓ batch → ↓ throughput, ↑ per-request latency (if queue is shallow)

**When it helps**:
- Increase batch when queue > 20 AND memory < 80% (maximize throughput)
- Reduce batch when TTFT > SLO AND queue < 10 (minimize latency)

### SCALE_OUT / SCALE_IN

**Effect**:
```python
gpu_count = new_value

# Impacts:
# 1. Linear increase in prefill throughput
tokens_per_second_prefill *= (new_gpu_count / old_gpu_count)

# 2. Linear increase in decode capacity
tokens_per_second_decode *= (new_gpu_count / old_gpu_count)

# 3. Cross-GPU communication overhead (TP: Tensor Parallel)
if gpu_count > 1:
    communication_overhead_ms = 5 * log2(gpu_count)  # All-reduce latency
    prefill_time_ms += communication_overhead_ms
    decode_time_per_token_ms += communication_overhead_ms / 10  # Less impact in decode
```

**Trade-off**:
- More GPUs = more capacity, but diminishing returns due to TP overhead
- 1→2 GPUs: ~1.9x throughput (5% overhead)
- 4→8 GPUs: ~1.7x throughput (15% overhead)

**When it helps**:
- Scale out when GPU > 85% AND queue growing (need more capacity)
- Scale in when GPU < 40% AND queue stable (wasting resources)

### ENABLE_SPECULATIVE_DECODE

**Effect**:
```python
# Speculative decoding: draft model generates K tokens, verify with main model
# Reduces decode steps by speculation_factor if drafts accepted

if speculative_decode_enabled:
    speculation_factor = 0.6  # Real vLLM reports ~40% speedup (1/0.6 = 1.67x)
    effective_decode_steps = avg_output_length * speculation_factor
else:
    effective_decode_steps = avg_output_length

total_decode_ms = decode_time_per_token_ms * effective_decode_steps
```

**Trade-off**:
- Reduces decode latency by 40% (matches real benchmarks)
- Increases GPU compute by ~20% (running draft model)
- Only beneficial when decode is bottleneck (high batch, GPU < 90%)

**When it helps**:
- Enable when TTFT > SLO AND decoding_requests > 10 AND GPU < 90%
- Disable when GPU > 90% (overhead outweighs benefit)

### REBALANCE_LOAD

**Effect**:
```python
# In multi-GPU: balance KV cache across GPUs to avoid hotspots
# Simulates pipeline parallelism rebalancing

if gpu_count > 1:
    memory_imbalance_factor = max(per_gpu_memory_utilization) / avg_memory_utilization
    
    if memory_imbalance_factor > 1.3:  # 30% imbalance
        # Rebalance cost: migrate KV cache (takes time)
        rebalance_latency_ms = 200 * gpu_count  # Migration overhead
        
        # Benefit: reduce hotspot penalty
        memory_efficiency = 1 / memory_imbalance_factor  # Was 0.77, now → 1.0
```

**Trade-off**:
- Costs 200ms * gpu_count (one-time migration)
- Improves memory efficiency from ~0.7 to ~0.95
- Only useful when imbalance detected

**When it helps**:
- Multi-GPU AND memory_efficiency < 0.8 AND no active rebalance

---

## Metrics Exposed (Match Real Systems)

```python
{
    "ttft_ms": queue_wait_time_ms + prefill_time_ms,  # What user sees
    "tpot_ms": decode_time_per_token_ms,              # Time per output token
    "e2e_latency_ms": ttft_ms + total_decode_ms,      # Full request latency
    "throughput_rps": completed_requests / uptime_sec,
    "queue_depth": queue_depth,
    "queue_velocity": (queue_depth - prev_queue_depth) / time_delta,  # req/sec
    "gpu_utilization": gpu_utilization,               # % compute
    "memory_utilization": memory_utilization,         # % KV cache
    "memory_efficiency": 1.0 / memory_imbalance_factor,
    "batch_size": config.batch_size,
    "gpu_count": config.gpu_count,
    "prefilling_count": prefilling_requests,
    "decoding_count": decoding_requests
}
```

---

## Load Generation (Deterministic)

**Requirement**: Same seed → same arrivals.

```python
class LoadGenerator:
    def __init__(self, seed=42, pattern="poisson"):
        self.rng = random.Random(seed)
        self.pattern = pattern
    
    def generate_arrival_times(self, duration_sec, target_rps):
        """Generate deterministic arrival schedule."""
        if self.pattern == "poisson":
            # Exponential inter-arrival times
            times = []
            t = 0
            while t < duration_sec:
                interval = self.rng.expovariate(target_rps)
                t += interval
                times.append(t)
            return times
        
        elif self.pattern == "burst":
            # 5s quiet, 10s burst, repeat
            times = []
            for cycle_start in range(0, duration_sec, 15):
                # Burst at t=5-15 in each cycle
                for t in range(5, 15):
                    for _ in range(int(target_rps * 3)):  # 3x rate
                        times.append(cycle_start + t + self.rng.uniform(0, 1))
            return [t for t in times if t < duration_sec]
        
        elif self.pattern == "ramp":
            # Linear ramp from 0 to 2*target_rps
            times = []
            t = 0
            while t < duration_sec:
                current_rps = (t / duration_sec) * 2 * target_rps
                interval = self.rng.expovariate(max(0.1, current_rps))
                t += interval
                times.append(t)
            return times
```

---

## Validation Against Reality

**Sources to calibrate against**:
1. **vLLM benchmarks**: https://github.com/vllm-project/vllm/tree/main/benchmarks
   - Prefill: ~3000 tok/s (A100, 7B)
   - Decode: ~150-200 tok/s (batch=8)
   - Saturation curve at 85-95% GPU

2. **TensorRT-LLM**: Similar numbers, slight edge in decode

3. **Our constraints**:
   - TTFT should hit ~600ms with queue=0, batch=4, gpu=1 (baseline)
   - TTFT should degrade linearly with queue up to saturation
   - Scaling 1→2 GPUs should give ~1.9x throughput (not 2.0x)

**Calibration Test**:
```python
# Scenario: Steady load at capacity
config = {"batch_size": 8, "gpu_count": 1}
arrival_rate = 12 rps  # Near saturation for 1 GPU

# Expected steady-state (after warmup):
assert 500 < avg_ttft_ms < 700        # Matches SLO target
assert 60 < avg_tpot_ms < 80          # Decode reasonable
assert 0 < queue_depth < 20           # Slight queue, not runaway
assert 80 < gpu_utilization < 90      # Near saturation
```

---

## Implementation Checklist

- [ ] Separate prefill/decode state tracking
- [ ] Conservation law enforced (queue + active + completed = submitted)
- [ ] GPU saturation penalty (nonlinear at 85%+)
- [ ] Batch size affects decode latency (sqrt penalty)
- [ ] Cross-GPU communication overhead (scale_out cost)
- [ ] Speculative decode reduces decode steps by 40%
- [ ] Rebalance has migration cost + efficiency gain
- [ ] Deterministic load generator (seed-based)
- [ ] Metrics match vLLM output format
- [ ] Calibration test passes (steady-state at capacity)

---

## Questions for Review

1. **Prefill/decode split**: Is `tokens_per_second_prefill = 3000` and `tokens_per_second_decode = 200` realistic for A100 + 7B model?

2. **Saturation model**: Should GPU penalty be steeper above 95%? (I have 15% at 85%, +20% at 95%)

3. **Batch scaling**: Using `sqrt(batch_size)` for decode penalty - is this consistent with vLLM's continuous batching behavior?

4. **Speculative decode**: 40% speedup (speculation_factor=0.6) matches vLLM benchmarks, but should compute overhead be higher than 20%?

5. **Queue conservation**: Does the drain model (`available_slots / iteration_interval`) correctly model vLLM's preemptive scheduling?

6. **Action semantics**: Are there critical actions I'm missing? (PagedAttention tuning? KV cache compression?)

---

## What This Enables

Once implemented:
- **v0.4 baseline re-run**: See if reactive controller still has Action Paradox
- **Load replay**: Same scenario → same results → measure improvements
- **Controller comparison**: Predictive vs reactive on identical, realistic load
- **RL safety**: Agent learns policies that transfer to real systems

**No more learning simulator artifacts.** Every decision trains on physics that matches production.
