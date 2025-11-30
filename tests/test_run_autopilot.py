import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
import asyncio
from run_autopilot import Autopilot
from models import Action, Metrics, Decision
import logging

@pytest.mark.asyncio
async def test_autopilot_starts_and_runs():
    """Test that the autopilot starts, initializes components, and runs the loop."""
    autopilot = Autopilot(decision_interval=1.0, poll_interval=0.5)
    try:
        # Start server and components
        await autopilot.start_server()
        await autopilot.initialize_components()

        # Start collector for a few cycles
        collector_task = asyncio.create_task(autopilot.collector.collect_loop())
        await asyncio.sleep(3)

        # Fetch metrics and make a decision
        metrics = autopilot.collector.get_current_metrics()
        assert metrics is not None, "Metrics should not be None"
        decision = autopilot.controller.make_decision(metrics)
        assert decision.action in [a for a in Action], "Invalid action from controller"

    finally:
        await autopilot.shutdown()

@pytest.mark.asyncio
async def test_controller_switches_modes_under_load():
    """Test that the controller switches modes under simulated load."""
    autopilot = Autopilot()
    try:
        await autopilot.start_server()
        await autopilot.initialize_components()

        # Simulate heavy load metrics
        fake_metrics = Metrics(
            ttft_ms=900,
            inter_token_latency_ms=80,
            prefill_latency_ms=540,
            decode_latency_ms=360,
            gpu_utilization=85.0,
            memory_efficiency=0.75,
            gpu_balance_index=0.95,
            comm_bubble_ratio=0.1,
            speculative_factor=0.4,
            queue_depth=60,
            timestamp=0.0
        )

        decision = autopilot.controller.make_decision(fake_metrics)
        assert decision.action in [Action.REDUCE_BATCH, Action.ENABLE_SPECULATIVE_DECODE]

    finally:
        await autopilot.shutdown()

import os

@pytest.mark.asyncio
async def test_decision_logger_output():
    """Test that the decision logger writes logs correctly."""
    log_file = "decision_log.txt"

    # Ensure the log file does not exist before the test
    if os.path.exists(log_file):
        try:
            os.remove(log_file)
        except PermissionError:
            pass  # Ignore if locked from a previous run

    autopilot = Autopilot(decision_interval=1.0, poll_interval=0.5)
    try:
        # Start server and components
        await autopilot.start_server()
        await autopilot.initialize_components()

        # Simulate a few decision cycles
        collector_task = asyncio.create_task(autopilot.collector.collect_loop())
        await asyncio.sleep(3)

        # Simulate logging
        fake_metrics = Metrics(
            ttft_ms=100,
            inter_token_latency_ms=50,
            prefill_latency_ms=30,
            decode_latency_ms=20,
            gpu_utilization=70.0,
            memory_efficiency=0.9,
            gpu_balance_index=0.8,
            comm_bubble_ratio=0.1,
            speculative_factor=0.5,
            queue_depth=10,
            timestamp=0.0
        )
        fake_decision = Decision(action=Action.NO_ACTION, reason="Test simulation")
        autopilot.logger.log_metrics_and_decision(fake_metrics, fake_decision)
        await asyncio.sleep(1)  # Allow time for the logger to write

        # Check if the log file is created
        assert os.path.exists(log_file), "Log file should be created."

        # Give the logger a moment to flush to disk
        for handler in logging.getLogger("decision_logger").handlers:
            handler.flush()

        # Read the log file and verify content
        with open(log_file, "r", encoding="utf-8") as f:
            logs = f.readlines()
            assert len(logs) > 0, "Log file should not be empty."

            # Verify that logs contain JSON entries for metrics and decisions
            found_metrics = any("metrics" in log for log in logs)
            found_decision = any("decision" in log for log in logs)
            assert found_metrics, "At least one log entry should include metrics."
            assert found_decision, "At least one log entry should include decision."

    finally:
        await autopilot.shutdown()

        # Close all file handlers cleanly before deleting
        for handler in logging.getLogger("decision_logger").handlers:
            handler.close()
            logging.getLogger("decision_logger").removeHandler(handler)

        # Now remove the file safely
        if os.path.exists(log_file):
            os.remove(log_file)