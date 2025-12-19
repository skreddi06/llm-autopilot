# 🧪 v0.4 Control Feature Validation Guide

**Purpose**: Systematically validate all v0.4 features and capture baseline data for v0.5 predictive control.

**Duration**: ~10-15 minutes

**Prerequisites**:
- ✅ Terminal 1: `python run_autopilot.py` (running)
- ✅ Terminal 2: `streamlit run dashboard.py` (running at http://localhost:8501)
- ✅ Browser: Dashboard open and visible

---

## 📋 Test Sequence

### Test 1️⃣: Baseline System Health (2 min)

**Goal**: Verify system starts in healthy state

**Steps**:
1. Observe dashboard for 30 seconds
2. Record initial metrics:
   ```
   [ ] TTFT: _______ ms (should be < 600ms)
   [ ] GPU Utilization: _______ % (should be 50-80%)
   [ ] Queue Depth: _______ (should be < 20)
   [ ] Mode: _______ (should be "safe")
   [ ] Status: 🟢 HEALTHY
   ```

**Expected Behavior**:
- Status: 🟢 HEALTHY
- Mode: `safe`
- Decisions: Mostly `NO_ACTION`
- Auto-refresh working (watch timestamp update every 2s)

**✅ Pass Criteria**: System stable, no SLO breaches

---

### Test 2️⃣: Load Spike & Auto-Recovery (3 min)

**Goal**: Verify controller detects overload and takes corrective action

**Steps**:
1. **Generate Load Spike** (Terminal 3):
   ```bash
   for i in {1..50}; do curl -X POST http://localhost:8000/inference & done
   ```

2. **Observe Dashboard** (60-90 seconds):
   - Watch TTFT climb
   - Watch queue depth increase
   - Watch mode switch
   - Watch decision banner update

3. **Record Peak State**:
   ```
   [ ] Peak TTFT: _______ ms
   [ ] Peak Queue: _______
   [ ] Peak GPU: _______ %
   [ ] Mode Changed To: _______
   [ ] Action Taken: _______
   [ ] Batch Size Changed: _______ → _______
   ```

4. **Wait for Recovery** (60-90 seconds):
   - Don't generate more load
   - Watch metrics return to baseline
   - Note status color changes

5. **Record Recovery**:
   ```
   [ ] Recovery Time: _______ seconds
   [ ] Final TTFT: _______ ms
   [ ] Final Mode: _______
   [ ] Status: 🟢/🟡/🔴
   ```

**Expected Behavior**:
- 🔴 Status → CRITICAL when TTFT > 720ms
- Mode switches: `safe` → `latency_optimized` or `throughput_optimized`
- Actions: `REDUCE_BATCH` or `REBALANCE_LOAD`
- System self-heals within 60-120 seconds
- Status returns to 🟢 HEALTHY

**✅ Pass Criteria**: 
- Controller switches mode
- Takes at least one action
- System recovers to 🟢 within 2 minutes

---

### Test 3️⃣: Manual Mode Override (2 min)

**Goal**: Verify manual controls override autopilot

**Steps**:
1. **Enable Manual Mode**:
   - Sidebar → Check "🎮 Manual Mode"
   - Verify warning appears: "⚠️ Manual mode enabled"

2. **Adjust Configuration**:
   - Current batch size: _______
   - Set Batch Size to: **8**
   - Set GPU Count to: **2**
   - Click "Apply Configuration"

3. **Verify Changes**:
   ```
   [ ] Success message appears
   [ ] Server Configuration section updates
   [ ] Batch Size shows: 8
   [ ] GPU Count shows: 2
   ```

4. **Observe Impact** (30 seconds):
   - TTFT change: _______
   - GPU Utilization change: _______
   - Autopilot decisions: _______ (should be suspended)

5. **Disable Manual Mode**:
   - Uncheck "🎮 Manual Mode"
   - Verify: "🤖 Autopilot mode - System is autonomous"

**Expected Behavior**:
- Manual mode suspends autopilot decisions
- Configuration changes apply immediately
- System responds to new config
- Re-enabling autopilot resumes autonomous control

**✅ Pass Criteria**: 
- Manual config changes take effect
- No autopilot decisions while manual mode active
- Autopilot resumes after disabling manual mode

---

### Test 4️⃣: Traffic Control (2 min)

**Goal**: Verify auto-load toggle controls traffic generation

**Steps**:
1. **Current State**:
   - Queue Depth: _______
   - Queue Velocity: _______

2. **Disable Auto-Load**:
   - Uncheck "Enable Auto-Load"
   - Click "Apply Traffic Control"
   - Verify success message

3. **Observe for 60 seconds**:
   ```
   [ ] Queue Depth: _______ → _______ (should decrease)
   [ ] Queue Velocity: _______ → _______ (should go negative)
   [ ] TTFT: _______ → _______ (should decrease)
   [ ] Status: _______ (should stabilize to 🟢)
   ```

4. **Re-enable Auto-Load**:
   - Check "Enable Auto-Load"
   - Click "Apply Traffic Control"
   - Watch queue start building again

**Expected Behavior**:
- Disabling stops traffic bursts
- Queue drains naturally
- System stabilizes
- Re-enabling resumes traffic patterns

**✅ Pass Criteria**: 
- Auto-load toggle controls traffic
- Queue drains when disabled
- Traffic resumes when re-enabled

---

### Test 5️⃣: Dashboard Features (2 min)

**Goal**: Verify all dashboard components work

**Checklist**:
```
[ ] Real-time gauges update every 2s
[ ] TTFT gauge shows red threshold at 600ms
[ ] GPU gauge shows red threshold at 85%
[ ] Historical trend charts render (need 2+ data points)
[ ] TTFT chart shows SLO red line at 600ms
[ ] Recent Decisions table shows last 10 decisions
[ ] Server Configuration displays current batch/GPU
[ ] Advanced Metrics expander works
[ ] Mode indicator updates in real-time
[ ] Latest Decision banner appears for non-NO_ACTION
[ ] Status colors: 🟢 (< 600ms), 🟡 (600-720ms), 🔴 (> 720ms)
```

**✅ Pass Criteria**: All checkboxes ticked

---

### Test 6️⃣: Mode Transitions (3 min)

**Goal**: Verify controller switches modes appropriately

**Steps**:
1. **Start with Stable System** (mode: `safe`)

2. **Trigger Throughput Mode**:
   ```bash
   # Generate sustained moderate load
   for i in {1..100}; do curl -X POST http://localhost:8000/inference & sleep 0.1; done &
   ```
   - Wait 10-15 seconds
   - Record mode: _______
   - Expected: `throughput_optimized` (queue > 50, GPU < 85%)

3. **Trigger Latency Mode**:
   ```bash
   # Generate heavy burst
   for i in {1..80}; do curl -X POST http://localhost:8000/inference & done
   ```
   - Wait 10-15 seconds
   - Record mode: _______
   - Expected: `latency_optimized` (TTFT > SLO)

4. **Wait for Recovery to Safe**:
   - Stop generating load (kill background process if needed: `pkill -f "curl"`)
   - Wait 60-90 seconds
   - Record mode: _______
   - Expected: `safe`

**Expected Mode Transitions**:
```
safe → throughput_optimized → latency_optimized → safe
```

**✅ Pass Criteria**: Controller transitions through all 3 modes

---

## 📊 Data Collection

### Capture Baseline Data for v0.5

1. **Save Decision Log**:
   ```bash
   cp decision_log.jsonl v04_baseline_$(date +%Y%m%d_%H%M%S).jsonl
   ```

2. **Take Dashboard Screenshots**:
   - Healthy state (🟢)
   - Under load (🔴)
   - Recovery phase (🟡 → 🟢)

3. **Record Key Metrics**:
   ```
   Total Test Duration: _______ minutes
   Number of Load Spikes: _______
   Number of Mode Switches: _______
   Number of Actions Taken: _______
   Average Recovery Time: _______ seconds
   Max TTFT Observed: _______ ms
   Max Queue Depth: _______
   ```

4. **Analyze Decision Log**:
   ```bash
   # Count decisions by action type
   grep -o '"action":"[^"]*"' decision_log.jsonl | sort | uniq -c
   
   # Count mode occurrences
   grep -o '"mode":"[^"]*"' decision_log.jsonl | sort | uniq -c
   ```

---

## 🎯 Validation Summary

### ✅ All Tests Passed:
```
[ ] Test 1: Baseline Health
[ ] Test 2: Load Spike & Recovery
[ ] Test 3: Manual Mode Override
[ ] Test 4: Traffic Control
[ ] Test 5: Dashboard Features
[ ] Test 6: Mode Transitions
```

### 📈 Key Findings:
```
Strengths:
- _______________________________
- _______________________________
- _______________________________

Areas for Improvement (v0.5 targets):
- _______________________________
- _______________________________
- _______________________________
```

### 🧠 Insights for v0.5 Predictive Control:
```
1. Average time between load spike → action: _______ seconds
   → v0.5 target: Predict and act before spike
   
2. Queue velocity as early warning: _______
   → v0.5 target: Use velocity trends for forecasting
   
3. Most common action: _______
   → v0.5 target: Pre-scale before needing this action
   
4. Recovery time variance: _______ seconds
   → v0.5 target: Optimize recovery trajectory
```

---

## 🚀 Next Steps

Once validation complete:

1. ✅ Archive baseline data: `v04_baseline_*.jsonl`
2. ✅ Document findings in this file
3. ✅ Ready for v0.5 Architecture Blueprint
4. ✅ Proceed to Predictive Control Implementation

---

**Status**: [ ] Not Started | [ ] In Progress | [ ] Complete

**Date Completed**: _______

**Notes**:
```
_______________________________
_______________________________
_______________________________
```
