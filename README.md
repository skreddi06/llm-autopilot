# LLM Autopilot v0.1

A self-healing autopilot system for LLM inference servers that maintains latency within SLO while maximizing GPU efficiency.

## Architecture

The system consists of 3 core components:

1. **Collector** (`collector.py`) - Continuously polls metrics from the LLM server
2. **Brain/Controller** (`controller.py`) - Makes intelligent decisions based on metrics
3. **Actuator** (`actuator.py`) - Applies configuration changes to the server

Plus a **Fake LLM Server** (`fake_llm_server.py`) for simulation and testing.

## Features

- Real-time latency control (TTFT, inter-token latency)
- GPU efficiency maximization through dynamic batch size adjustment
- Early overload detection via queue velocity monitoring
- Adaptive control with soft/hard breach handling
- Automatic scaling decisions

## Quick Start

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the autopilot:
```bash
python run_autopilot.py
```

The system will:
- Start a fake LLM server on `http://localhost:8000`
- Begin collecting metrics every 1 second
- Make decisions every 2 seconds
- Automatically adjust batch size and GPU count to maintain SLO

## Configuration

Default settings in `run_autopilot.py`:
- SLO latency target: 600ms
- Polling interval: 1 second
- Decision interval: 2 seconds
- Batch size range: 1-32
- GPU count range: 1-8

## Decision Logic

The controller implements:
- **Soft Breach**: Reduce batch size when latency increases
- **Hard Breach**: Scale out (add GPUs) when batch size is at minimum
- **Safe Mode**: Increase batch size when latency is well under SLO
- **Pre-scaling**: Scale out when queue velocity indicates imminent overload