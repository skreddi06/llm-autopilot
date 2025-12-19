# v0.5-α Predictive Controller - Quick Start

## What's New in v0.5-α

The predictive controller introduces **anticipatory decision-making** to prevent SLO violations before they occur, addressing the fundamental limitation of v0.4's reactive approach.

### Key Improvements Over v0.4

| Feature | v0.4 Reactive | v0.5-α Predictive |
|---------|---------------|-------------------|
| **Decision Basis** | Current TTFT | Forecasted TTFT (10s ahead) |
| **Queue Handling** | React after buildup | Pre-scale on velocity increase |
| **Action Gating** | Immediate application | Confidence + consensus gating |
| **Emergency Mode** | None | Circuit breaker (PANIC modes) |
| **Oscillation Control** | None | 2/3 consensus for aggressive actions |

### Architecture Overview

```
Metrics History (10 samples)
        ↓
Trend Analysis:
  - TTFT slope (linear regression)
  - Queue velocity (from collector)
  - Queue acceleration (Δvelocity)
        ↓
Confidence Estimation (0.0-1.0)
        ↓
Emergency Check → [PANIC modes bypass]
        ↓
Predictive Decision:
  - Forecast TTFT 10s ahead
  - Pre-scale on velocity trends
  - Consider throughput impact
        ↓
Consensus Gating (2/3 agreement)
        ↓
Apply Decision
```

---

## Installation & Setup

### Prerequisites

Same as v0.4:
```bash
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
pip install -r requirements.txt
```

No additional dependencies required.

---

## Running v0.5-α

### Option 1: Predictive Controller Only

```bash
# Start autopilot with predictive controller
python run_autopilot_v2.py

# In separate terminal: Start dashboard
streamlit run dashboard.py
```

**Logs Generated**:
- `decision_log_v2_predictive.jsonl` - Predictive decisions (APPLIED)
- `decision_log_v2_predictive.txt` - Human-readable log

### Option 2: Side-by-Side Comparison Mode (Recommended)

```bash
# Edit run_autopilot_v2.py line 221:
# comparison_mode=True  # Already default

python run_autopilot_v2.py
```

**Logs Generated**:
- `decision_log_v2_predictive.jsonl` - Predictive decisions (APPLIED)
- `decision_log_v2_reactive.jsonl` - Reactive decisions (SHADOW MODE)
- `decision_log_v2_comparison.jsonl` - Divergence tracking

**Benefits**:
- See both controllers' decisions side-by-side
- Measure improvement quantitatively
- Identify where predictive helps most

---

## Testing the Predictive Features

### 1. Quick Smoke Test

```bash
# Terminal 1: Start v0.5-α
python run_autopilot_v2.py

# Terminal 2: Generate gradual load
for i in {1..50}; do 
  curl -X POST http://localhost:8000/inference & 
  sleep 0.5
done

# Watch for: "🔀 DIVERGENCE" messages showing different decisions
```

**Expected**: Predictive scales out ~5-10s before reactive.

### 2. Emergency Circuit Breaker Test

```bash
# Massive spike
for i in {1..300}; do 
  curl -X POST http://localhost:8000/inference & 
done

# Watch for: "PANIC_RESET" or "PANIC_DRAIN" in logs
```

**Expected**: Immediate batch reduction to minimum, bypassing confidence checks.

### 3. Comprehensive Validation

Follow the full test suite:
```bash
# See V05_ALPHA_TESTING_GUIDE.md for 5 test scenarios
cat V05_ALPHA_TESTING_GUIDE.md
```

---

## Understanding the Logs

### Predictive Decision Format

```json
{
  "timestamp": "2025-12-14T10:30:45",
  "decision": {
    "action": "SCALE_OUT",
    "reason": "Predictive scale-out: vel=8.3, accel=2.1",
    "batch_size": 4,
    "gpu_count": 2
  },
  "metrics": {
    "ttft_ms": 550,
    "queue_depth": 35,
    "queue_velocity": 8.3,
    "gpu_utilization": 78
  },
  "mode": "throughput_optimized"
}
```

**Key Fields**:
- `queue_velocity`: Requests/sec change rate (from collector)
- `reason`: Shows why action taken (includes forecast values)
- `mode`: Current controller mode (safe/throughput/latency)

### Detecting Predictive Actions

```bash
# Pre-emptive scale-outs (before SLO breach)
grep "Predictive scale-out" decision_log_v2_predictive.jsonl

# Forecasted TTFT decisions
grep "forecast=" decision_log_v2_predictive.jsonl

# Emergency bypasses
grep "PANIC" decision_log_v2_predictive.jsonl

# Consensus blocks
grep "Consensus required but not met" decision_log_v2_predictive.jsonl
```

---

## Configuration Parameters

Edit `controller_v2.py` to tune behavior:

```python
class PredictiveController:
    def __init__(self, slo_latency_ms: int = 600):
        # Circuit breaker thresholds
        self.PANIC_QUEUE_THRESHOLD = 500
        self.PANIC_TTFT_MULTIPLIER = 5  # 5x SLO
        self.PANIC_GPU_THRESHOLD = 95
        self.PANIC_VELOCITY_THRESHOLD = 10
        
        # Confidence gating
        self.CONFIDENCE_THRESHOLD = 0.7
        self.CONSENSUS_WINDOW = 3
        self.MIN_HISTORY_FOR_PREDICTION = 3
```

**Common Adjustments**:
- **More aggressive**: Lower `CONFIDENCE_THRESHOLD` to 0.5
- **More stable**: Increase `CONSENSUS_WINDOW` to 5
- **Faster response**: Lower `PANIC_QUEUE_THRESHOLD` to 200

---

## Analyzing Results

### Generate Comparison Report

```bash
# After running tests with comparison_mode=True
python << 'EOF'
import json
from collections import defaultdict

stats = {'predictive': defaultdict(int), 'reactive': defaultdict(int)}

for log, key in [('decision_log_v2_predictive.jsonl', 'predictive'),
                 ('decision_log_v2_reactive.jsonl', 'reactive')]:
    with open(log) as f:
        for line in f:
            data = json.loads(line)
            stats[key][data['decision']['action']] += 1
            if data['metrics']['ttft_ms'] > 600:
                stats[key]['slo_violations'] += 1

print(f"SLO Violations: Predictive={stats['predictive']['slo_violations']} "
      f"Reactive={stats['reactive']['slo_violations']}")
print(f"SCALE_OUT: Predictive={stats['predictive']['SCALE_OUT']} "
      f"Reactive={stats['reactive']['SCALE_OUT']}")
EOF
```

### Visualize in Dashboard

The existing dashboard works with v0.5-α (fetches from collector's `/live_metrics` and `/last_decision`). Both controllers update the same endpoints, so dashboard shows whichever is active.

---

## Troubleshooting

### "Low confidence, holding steady" in all decisions

**Cause**: Insufficient history (< 3 samples).

**Solution**: Wait 6-10 seconds for buffer to fill.

### No divergence between controllers

**Cause**: Load too low to trigger predictive features.

**Solution**: Increase load intensity or enable auto-load:
```bash
curl -X POST http://localhost:8000/auto_load -d '{"enabled": true}'
```

### Emergency modes triggering constantly

**Cause**: Thresholds too aggressive for current setup.

**Solution**: Increase thresholds in `controller_v2.py`:
```python
self.PANIC_QUEUE_THRESHOLD = 1000  # Was 500
self.PANIC_TTFT_MULTIPLIER = 10    # Was 5
```

---

## Next Steps

1. **Run Validation Suite**: Execute all 5 tests in `V05_ALPHA_TESTING_GUIDE.md`
2. **Archive Baseline**: Save logs to `validation_data/v05_alpha/`
3. **Tune Parameters**: Adjust thresholds based on test results
4. **Proceed to v0.5-β**: Supervised RL training on collected data

---

## File Reference

| File | Purpose |
|------|---------|
| `controller_v2.py` | Predictive controller implementation |
| `run_autopilot_v2.py` | Dual-controller orchestrator |
| `V05_ALPHA_TESTING_GUIDE.md` | Comprehensive test scenarios |
| `decision_log_v2_predictive.jsonl` | Applied decisions log |
| `decision_log_v2_reactive.jsonl` | Shadow mode decisions log |
| `decision_log_v2_comparison.jsonl` | Divergence tracking log |

---

## Questions?

- **How does forecasting work?** Linear extrapolation: `forecast = current_ttft + (slope × steps_ahead)`
- **What is confidence gating?** Requires 70% confidence before aggressive actions
- **What is consensus gating?** Requires 2/3 recent decisions to agree for batch=1 reductions
- **When do PANIC modes trigger?** Queue > 500, TTFT > 3000ms, or GPU > 95% + velocity > 10
- **Can I disable comparison mode?** Yes, set `comparison_mode=False` in `run_autopilot_v2.py` line 221

---

**Ready to test?** Start with:
```bash
python run_autopilot_v2.py
```

Then follow Test 1 in `V05_ALPHA_TESTING_GUIDE.md` for baseline validation.
