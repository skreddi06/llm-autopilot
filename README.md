# LLM Autopilot v1.0

An autonomous admission controller for LLM serving infrastructure, powered by Reinforcement Learning with Safety Shields.

## 🎯 What It Does

Controls LLM server capacity (batch size, GPU allocation, request admission) to:
- **Maximize throughput** under normal conditions
- **Maintain stability** during traffic surges  
- **Avoid the "Action Paradox"** (reducing capacity under load)

## 📊 Validated Results

| Scenario | Agent | Static | Reactive |
|----------|-------|--------|----------|
| Normal Load (Goodput) | **117** | 115 | 114 |
| Death Spiral (Stability σ) | **8.50** | 8.58 | 9.51 |

---

## 🏗️ Project Structure

```
llm-autopilot/
├── src/                        # Core Components
│   ├── ml_controller.py        # Hybrid RL + Shield controller (MAIN)
│   ├── vllm_client.py          # vLLM production adapter
│   ├── fake_llm_server.py      # Physics-based LLM simulator
│   ├── mock_vllm.py            # Mock vLLM for testing
│   ├── llm_env_v08.py          # Gym environment for RL training
│   ├── models.py               # Data models (Metrics, Action, Decision)
│   ├── actuator.py             # Action execution layer
│   ├── collector.py            # Metrics collection
│   ├── controller_v2.py        # Rule-based controller (baseline)
│   ├── predictor.py            # Surge prediction module
│   ├── dashboard.py            # Monitoring dashboard
│   └── decision_logger.py      # Decision audit logging
│
├── training/                   # Training Pipeline
│   ├── train_bc.py             # Behavioral Cloning (Shield → Agent)
│   ├── train_ppo.py            # PPO training (basic)
│   ├── train_ppo_v08.py        # PPO training (v08 with rewards)
│   ├── pretrain_ppo.py         # Pretraining on expert data
│   ├── finetune_ppo.py         # Fine-tuning existing models
│   └── generate_training_data.py # Generate expert demonstrations
│
├── benchmarks/                 # Validation Suite
│   ├── benchmark_showdown.py   # Agent vs Static vs Reactive (MAIN)
│   ├── benchmark_verification.py # Autonomy rate testing
│   ├── benchmark_hybrid.py     # Hybrid controller testing
│   └── benchmark_all.py        # Full benchmark suite
│
├── models/                     # Trained Models
│   └── ppo_cloned_v09.zip      # SHIPPED: BC-trained agent
│
├── tests/                      # Unit Tests
│   ├── test_run_autopilot.py   # Integration tests
│   ├── test_memory_cliff.py    # Memory limit testing
│   ├── test_mixed_load.py      # Mixed workload testing
│   ├── test_predictive_surge.py # Surge prediction tests
│   ├── test_roofline.py        # Performance bounds
│   ├── test_phase5_scheduling.py # Scheduling tests
│   └── test_student_driver.py  # Student policy tests
│
├── docs/                       # Documentation
│   ├── INVESTOR_NARRATIVE.md   # Business case
│   ├── CALIBRATED_SIMULATOR_SPEC.md # Simulator physics
│   └── V0*_*.md                # Version guides
│
├── run_bridge.py               # Production orchestrator (vLLM)
├── run_autopilot.py            # Local simulation runner
├── run_autopilot_v2.py         # Enhanced simulation runner
├── requirements.txt            # Dependencies
└── README.md                   # This file
```

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Simulation (No GPU Required)
```bash
# Start the fake LLM server
python run_autopilot.py
```

### 3. Run with Mock vLLM
```bash
# Terminal 1: Start mock server
python src/mock_vllm.py

# Terminal 2: Run controller
python run_bridge.py
```

### 4. Run Benchmarks
```bash
python benchmarks/benchmark_showdown.py
```

---

## 🔧 Training Pipeline

### Step 1: Generate Expert Data
```bash
python training/generate_training_data.py
```

### Step 2: Pretrain with PPO
```bash
python training/pretrain_ppo.py
```

### Step 3: Behavioral Cloning (Shield → Agent)
```bash
python training/train_bc.py
```

### Step 4: Verify
```bash
python benchmarks/benchmark_verification.py
```

---

## 📖 Architecture

The system uses a **Hybrid Shielded Controller**:

```
Metrics → [Shield Check] → [RL Agent] → Admission Control → vLLM
              ↓ (critical)      ↓ (normal)
         Override Action    Predicted Action
```

1. **Shield**: Deterministic safety rules (RESCUE, PANIC)
2. **RL Agent**: BC-trained PPO policy (ppo_cloned_v09)
3. **Admission Control**: Semaphore-based rate limiting

---

## 📄 License

MIT