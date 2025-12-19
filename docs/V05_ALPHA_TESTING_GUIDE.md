# v0.5-α Predictive Controller Testing Guide

## Overview

This guide validates the v0.5-α predictive controller against the v0.4 reactive baseline. The predictive controller uses queue velocity, acceleration, and TTFT slope forecasting to make anticipatory decisions.

## Key Features to Test

### 1. **Predictive Pre-scaling**
- Controller should scale out *before* queue causes SLO breach
- Compare: v0.4 scales after TTFT spike, v0.5-α scales during velocity increase

### 2. **Confidence Gating**
- Low confidence (< 0.7) → conservative fallback actions
- High confidence → aggressive predictive actions
- Requires 2/3 consensus for batch reduction to minimum

### 3. **Emergency Circuit Breaker**
- PANIC_RESET: Queue > 500
- PANIC_DRAIN: TTFT > 3000ms (5x SLO)
- PANIC_SCALE: GPU > 95% AND velocity > 10

### 4. **TTFT Forecasting**
- 10s-ahead prediction (5 samples × 2s interval)
- Pre-emptive batch reduction when forecast > SLO

### 5. **Action Paradox Resolution**
- v0.4: REDUCE_BATCH during overload worsens throughput
- v0.5-α: Should consider throughput impact in decision

---

## Test Scenarios

### Test 1: Baseline Comparison (No Load)

**Purpose**: Verify both controllers behave identically in stable conditions.

```bash
# Terminal 1: Start v0.5-α autopilot
python run_autopilot_v2.py

# Terminal 2: Start dashboard
streamlit run dashboard.py

# Terminal 3: Disable auto-load
curl -X POST http://localhost:8000/auto_load -d '{"enabled": false}'

# Wait 60 seconds, observe metrics
```

**Expected Outcome**:
- Both controllers: NO_ACTION
- TTFT: < 600ms
- Queue: 0-5
- Mode: safe
- Agreement: 100%

**Metrics to Capture**:
```bash
# Count decisions in last 60s
grep -c "NO_ACTION" decision_log_v2_predictive.jsonl
grep -c "NO_ACTION" decision_log_v2_reactive.jsonl
```

---

### Test 2: Gradual Load Ramp (Predictive Pre-scaling)

**Purpose**: Test if predictive controller scales out earlier than reactive.

```bash
# Reset system
curl -X POST http://localhost:8000/reset
curl -X POST http://localhost:8000/auto_load -d '{"enabled": false}'

# Generate gradual load increase
for i in {1..100}; do 
  curl -X POST http://localhost:8000/inference & 
  sleep 0.5
done
```

**Expected Outcome**:
- **v0.5-α Predictive**: SCALE_OUT when queue velocity > 5 (before TTFT breach)
- **v0.4 Reactive**: SCALE_OUT after TTFT > 720ms
- **Key Metric**: Time to SCALE_OUT action (predictive should be 5-10s earlier)

**Metrics to Capture**:
```bash
# Time to first SCALE_OUT
grep "SCALE_OUT" decision_log_v2_predictive.jsonl | head -1 | jq '.timestamp'
grep "SCALE_OUT" decision_log_v2_reactive.jsonl | head -1 | jq '.timestamp'

# Max TTFT during ramp
grep "ttft_ms" decision_log_v2_predictive.jsonl | jq '.metrics.ttft_ms' | sort -n | tail -1
```

---

### Test 3: Sudden Load Spike (Emergency Circuit Breaker)

**Purpose**: Verify emergency bypass under extreme load.

```bash
# Reset system
curl -X POST http://localhost:8000/reset
curl -X POST http://localhost:8000/auto_load -d '{"enabled": false}'

# Sudden spike
for i in {1..200}; do 
  curl -X POST http://localhost:8000/inference & 
done

# Wait 30 seconds
```

**Expected Outcome**:
- Queue spikes to 100-200
- **Emergency Trigger**: PANIC_DRAIN or PANIC_RESET
- Action: REDUCE_BATCH to minimum (batch=1)
- Bypasses confidence gating

**Metrics to Capture**:
```bash
# Check for PANIC triggers
grep "PANIC" decision_log_v2_predictive.jsonl

# Max queue depth
grep "queue_depth" decision_log_v2_predictive.jsonl | jq '.metrics.queue_depth' | sort -n | tail -1

# Recovery time (queue back to < 10)
grep "queue_depth" decision_log_v2_predictive.jsonl | jq 'select(.metrics.queue_depth < 10)' | head -1 | jq '.timestamp'
```

---

### Test 4: Oscillation Prevention (Consensus Gating)

**Purpose**: Verify consensus requirement prevents rapid config changes.

```bash
# Reset system
curl -X POST http://localhost:8000/reset

# Enable auto-load (creates natural oscillation)
curl -X POST http://localhost:8000/auto_load -d '{"enabled": true}'

# Run for 5 minutes
sleep 300
```

**Expected Outcome**:
- **v0.5-α**: Fewer config changes due to consensus requirement
- **v0.4**: More frequent batch adjustments
- **Key Metric**: Config changes per minute

**Metrics to Capture**:
```bash
# Config changes in 5 minutes
grep -v "NO_ACTION" decision_log_v2_predictive.jsonl | wc -l
grep -v "NO_ACTION" decision_log_v2_reactive.jsonl | wc -l

# Check for consensus blocks
grep "Consensus required but not met" decision_log_v2_predictive.jsonl | wc -l
```

---

### Test 5: Forecast Accuracy (Pre-emptive Batch Adjustment)

**Purpose**: Measure TTFT forecast accuracy and pre-emptive actions.

```bash
# Reset system
curl -X POST http://localhost:8000/reset
curl -X POST http://localhost:8000/auto_load -d '{"enabled": false}'

# Controlled load pattern (sawtooth)
for cycle in {1..5}; do
  # Ramp up
  for i in {1..50}; do curl -X POST http://localhost:8000/inference & sleep 0.2; done
  sleep 10
  # Cool down
  curl -X POST http://localhost:8000/reset
  sleep 10
done
```

**Expected Outcome**:
- Predictive controller adjusts batch *during* ramp-up
- Reactive controller adjusts *after* TTFT spike
- Fewer SLO violations in predictive mode

**Metrics to Capture**:
```bash
# SLO violations (TTFT > 600ms)
grep "ttft_ms" decision_log_v2_predictive.jsonl | jq 'select(.metrics.ttft_ms > 600)' | wc -l
grep "ttft_ms" decision_log_v2_reactive.jsonl | jq 'select(.metrics.ttft_ms > 600)' | wc -l

# Forecast vs actual TTFT correlation
grep "forecast=" decision_log_v2_predictive.jsonl | jq '.decision.reason'
```

---

## Comparison Metrics Summary

After running all tests, generate comparison report:

```bash
# Create comparison report
cat << 'EOF' > analyze_v2_results.py
import json
from collections import defaultdict

def analyze_logs(predictive_log, reactive_log):
    """Compare predictive vs reactive controller performance."""
    
    stats = {
        'predictive': defaultdict(int),
        'reactive': defaultdict(int)
    }
    
    # Parse predictive log
    with open(predictive_log) as f:
        for line in f:
            data = json.loads(line)
            action = data['decision']['action']
            stats['predictive'][action] += 1
            
            if data['metrics']['ttft_ms'] > 600:
                stats['predictive']['slo_violations'] += 1
    
    # Parse reactive log
    with open(reactive_log) as f:
        for line in f:
            data = json.loads(line)
            action = data['decision']['action']
            stats['reactive'][action] += 1
            
            if data['metrics']['ttft_ms'] > 600:
                stats['reactive']['slo_violations'] += 1
    
    # Print comparison
    print("=" * 60)
    print("v0.5-α PREDICTIVE vs v0.4 REACTIVE COMPARISON")
    print("=" * 60)
    print(f"\n{'Metric':<30} {'Predictive':<15} {'Reactive':<15}")
    print("-" * 60)
    
    for action in ['NO_ACTION', 'SCALE_OUT', 'REDUCE_BATCH', 'INCREASE_BATCH']:
        pred_count = stats['predictive'][action]
        react_count = stats['reactive'][action]
        print(f"{action:<30} {pred_count:<15} {react_count:<15}")
    
    print("-" * 60)
    pred_violations = stats['predictive']['slo_violations']
    react_violations = stats['reactive']['slo_violations']
    improvement = ((react_violations - pred_violations) / react_violations * 100) if react_violations > 0 else 0
    
    print(f"{'SLO Violations':<30} {pred_violations:<15} {react_violations:<15}")
    print(f"{'Improvement':<30} {improvement:.1f}%")
    print("=" * 60)

if __name__ == '__main__':
    analyze_logs('decision_log_v2_predictive.jsonl', 'decision_log_v2_reactive.jsonl')
EOF

python analyze_v2_results.py
```

---

## Success Criteria

For v0.5-α to be considered successful:

1. **Latency Improvement**: ≥ 20% reduction in SLO violations vs v0.4
2. **Pre-scaling**: SCALE_OUT occurs 5-10s earlier during load ramps
3. **Stability**: ≤ 50% of v0.4's config change frequency
4. **Emergency Handling**: PANIC modes trigger correctly under extreme load
5. **Forecast Accuracy**: Forecasted TTFT within 15% of actual at 10s horizon

---

## Troubleshooting

### Issue: Controllers always agree (no divergence)

**Cause**: Load insufficient to trigger predictive features.

**Solution**:
```bash
# Increase load intensity
for i in {1..500}; do curl -X POST http://localhost:8000/inference & done
```

### Issue: Emergency mode triggers too often

**Cause**: Thresholds too aggressive for current setup.

**Solution**: Adjust in `controller_v2.py`:
```python
self.PANIC_QUEUE_THRESHOLD = 500  # Increase to 1000
self.PANIC_TTFT_MULTIPLIER = 5    # Increase to 10
```

### Issue: Confidence always low

**Cause**: Insufficient history buffer.

**Solution**: Wait 20s (10 samples × 2s) for buffer to fill.

---

## Next Steps After Validation

1. **Archive Baseline Data**:
   ```bash
   mkdir -p validation_data/v05_alpha
   cp decision_log_v2_*.jsonl validation_data/v05_alpha/
   cp analyze_v2_results.py validation_data/v05_alpha/
   ```

2. **Proceed to v0.5-β**: Supervised RL training on archived logs

3. **Dashboard Integration**: Add v0.5-α metrics to dashboard for live comparison

4. **Hyperparameter Tuning**: Optimize thresholds based on test results
