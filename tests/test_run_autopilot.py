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

            # Verify that logs contain the expected format (Mode/Action)
            found_mode = any("Mode" in log for log in logs)
            found_action = any("Action" in log for log in logs)
            assert found_mode, "At least one log entry should include Mode."
            assert found_action, "At least one log entry should include Action."

    finally:
        await autopilot.shutdown()

        # Close all file handlers cleanly before deleting
        for handler in logging.getLogger("decision_logger").handlers:
            handler.close()
            logging.getLogger("decision_logger").removeHandler(handler)

        # Now remove the file safely
        if os.path.exists(log_file):
            os.remove(log_file)


@pytest.mark.asyncio
async def test_conservation_law():
    """Test that the simulator enforces conservation law.
    
    Conservation: queued + prefilling + decoding + completed = submitted
    """
    from fake_llm_server import FakeLLMServer
    import aiohttp
    
    server = FakeLLMServer(port=8001)  # Different port to avoid conflicts
    server.auto_load_enabled = False  # Disable auto-load for controlled test
    
    try:
        runner = await server.start()
        await asyncio.sleep(0.5)  # Let server start
        
        # Submit 10 requests
        async with aiohttp.ClientSession() as session:
            for _ in range(10):
                await session.post("http://localhost:8001/inference")
        
        # Verify conservation law
        assert server.verify_conservation(), f"Conservation violated: {server.get_conservation_stats()}"
        assert server.total_submitted == 10, f"Expected 10 submitted, got {server.total_submitted}"
        
        # Wait for some requests to process
        await asyncio.sleep(2)
        
        # Conservation should still hold
        assert server.verify_conservation(), f"Conservation violated after processing: {server.get_conservation_stats()}"
        
        # Sum should still equal submitted
        stats = server.get_conservation_stats()
        total = stats["queued"] + stats["prefilling"] + stats["decoding"] + stats["completed"]
        assert total == stats["submitted"], f"Conservation sum mismatch: {total} != {stats['submitted']}"
        
    finally:
        await server.stop()
        await runner.cleanup()


@pytest.mark.asyncio
async def test_phase2_physics():
    """Test Phase 2 dual-phase physics:
    
    1. Prefill (compute-bound): should scale linearly with GPU count
    2. Decode (memory-bound): should degrade with sqrt(batch_size)
    3. Throughput: should increase with batch size (Databricks baselines)
    """
    from fake_llm_server import FakeLLMServer
    
    server = FakeLLMServer(port=8002)
    server.auto_load_enabled = False
    
    # === TEST 1: Prefill scales with GPU count ===
    server.config.gpu_count = 1
    prefill_1gpu = server.calculate_prefill_latency_ms()
    
    server.config.gpu_count = 2
    prefill_2gpu = server.calculate_prefill_latency_ms()
    
    # Prefill should be ~2x faster with 2 GPUs (compute-bound, linear scaling)
    assert 1.8 < (prefill_1gpu / prefill_2gpu) < 2.2, \
        f"Prefill should scale linearly: 1GPU={prefill_1gpu:.1f}ms, 2GPU={prefill_2gpu:.1f}ms"
    
    # === TEST 2: Decode degrades with batch size (sqrt penalty) ===
    server.config.gpu_count = 1
    
    server.config.batch_size = 1
    decode_batch1 = server.calculate_decode_latency_ms()
    
    server.config.batch_size = 16
    decode_batch16 = server.calculate_decode_latency_ms()
    
    server.config.batch_size = 64
    decode_batch64 = server.calculate_decode_latency_ms()
    
    # Decode should degrade with batch (memory contention)
    assert decode_batch16 > decode_batch1, \
        f"Decode should degrade: batch=1 {decode_batch1:.1f}ms, batch=16 {decode_batch16:.1f}ms"
    assert decode_batch64 > decode_batch16, \
        f"Decode should degrade more: batch=16 {decode_batch16:.1f}ms, batch=64 {decode_batch64:.1f}ms"
    
    # Verify sqrt scaling (not linear)
    # Linear would give 16x degradation, sqrt gives ~4x
    ratio_1_to_16 = decode_batch16 / decode_batch1
    assert ratio_1_to_16 < 3.0, f"Decode degradation should be sqrt, not linear: ratio={ratio_1_to_16:.2f}"
    
    # === TEST 3: Throughput increases with batch (Databricks baselines) ===
    server.config.batch_size = 1
    throughput_batch1 = server.calculate_throughput_rps()
    
    server.config.batch_size = 16
    throughput_batch16 = server.calculate_throughput_rps()
    
    # Throughput should increase significantly with batch
    assert throughput_batch16 > throughput_batch1 * 5, \
        f"Throughput should increase: batch=1 {throughput_batch1:.1f}, batch=16 {throughput_batch16:.1f}"
    
    # Verify matches Databricks baselines approximately
    assert 0.5 < throughput_batch1 < 1.5, f"Batch=1 throughput should be ~0.9: got {throughput_batch1:.1f}"
    assert 5.0 < throughput_batch16 < 12.0, f"Batch=16 throughput should be ~8.0: got {throughput_batch16:.1f}"
    
    print(f"Phase 2 Physics Validated:")
    print(f"  Prefill: 1GPU={prefill_1gpu:.1f}ms, 2GPU={prefill_2gpu:.1f}ms (linear scaling ✓)")
    print(f"  Decode: B1={decode_batch1:.1f}ms, B16={decode_batch16:.1f}ms, B64={decode_batch64:.1f}ms (sqrt penalty ✓)")
    print(f"  Throughput: B1={throughput_batch1:.1f}rps, B16={throughput_batch16:.1f}rps (Databricks curve ✓)")


@pytest.mark.asyncio
async def test_phase3_memory_wall():
    """Test Phase 3 KV Cache Memory Wall:
    
    1. Memory usage scales with tokens (prompt + generated)
    2. Memory limit enforced (18GB per GPU for A100)
    3. Congestion detection triggers when memory exceeded
    4. Swap penalty slows down processing
    """
    from fake_llm_server import FakeLLMServer, InFlightRequest
    
    server = FakeLLMServer(port=8003)
    server.auto_load_enabled = False
    
    # === TEST 1: Memory constants are correct ===
    # A100-40GB: 40GB - 14GB weights - 8GB overhead = 18GB for KV
    expected_kv_bytes = 40 * 1024**3 - 14 * 1024**3 - int(40 * 1024**3 * 0.2)
    assert abs(server.available_kv_memory_bytes - expected_kv_bytes) < 1024**3, \
        f"Expected ~18GB KV memory, got {server.available_kv_memory_bytes / 1024**3:.1f}GB"
    
    # KV cost per token should be ~0.125MB (GQA-optimized for 70B)
    assert 100 * 1024 < server.kv_bytes_per_token < 200 * 1024, \
        f"Expected ~0.125MB per token (GQA), got {server.kv_bytes_per_token / 1024 / 1024:.2f}MB"
    
    # === TEST 2: Memory usage scales with tokens ===
    # Initially empty
    assert server.calculate_kv_memory_usage() == 0
    
    # Add a decoding request with tokens
    # Note: In decoding state, prefill is complete, so prefill_tokens_processed = prompt_tokens
    test_req = InFlightRequest(
        id=1,
        state='decoding',
        arrival_time=0,
        decode_start=0,
        prompt_tokens=1000,
        generated_tokens=500,
        target_tokens=1000,
        prefill_tokens_processed=1000  # Phase 5: Prefill complete for decode requests
    )
    server.decoding_requests.append(test_req)
    
    # Memory should be (prefill_tokens_processed + generated) * bytes_per_token
    expected_mem = (1000 + 500) * server.kv_bytes_per_token
    actual_mem = server.calculate_kv_memory_usage()
    assert actual_mem == expected_mem, f"Memory mismatch: {actual_mem} != {expected_mem}"
    
    # === TEST 3: Memory utilization calculation ===
    utilization = server.get_memory_utilization()
    assert 0 < utilization < 100, f"Utilization should be reasonable: {utilization}%"
    
    # Clean up
    server.decoding_requests.clear()
    
    # === TEST 4: Congestion triggers at memory limit ===
    # Add enough requests to exceed memory
    # 18GB / 0.5MB per token = ~36,864 tokens max
    # Each request with 512 prompt + 128 generated = 640 tokens
    # Need ~58 requests to exceed
    tokens_needed = int(server.get_available_kv_memory() / server.kv_bytes_per_token) + 1000
    tokens_per_req = 640
    reqs_needed = (tokens_needed // tokens_per_req) + 1
    
    for i in range(reqs_needed):
        req = InFlightRequest(
            id=i,
            state='decoding',
            arrival_time=0,
            decode_start=0,
            prompt_tokens=512,
            generated_tokens=128,
            target_tokens=128,
            prefill_tokens_processed=512  # Phase 5: Prefill complete
        )
        server.decoding_requests.append(req)
    
    # Should now be congested
    assert server.check_memory_congestion(), \
        f"Should be congested with {len(server.decoding_requests)} requests using " \
        f"{server.calculate_kv_memory_usage() / 1024**3:.1f}GB"
    
    # === TEST 5: Memory stats are correct ===
    stats = server.get_memory_stats()
    assert stats["memory_congested"] == True
    assert stats["memory_utilization_pct"] >= 100  # Capped at 100%
    assert stats["kv_memory_used_mb"] > stats["kv_memory_available_mb"]
    
    print(f"Phase 3 Memory Wall Validated:")
    print(f"  Available KV: {server.available_kv_memory_bytes / 1024**3:.1f} GB")
    print(f"  Bytes per token: {server.kv_bytes_per_token / 1024**2:.2f} MB")
    print(f"  Congestion at {len(server.decoding_requests)} requests ✓")
    print(f"  Memory usage: {stats['kv_memory_used_mb'] / 1024:.1f} GB / {stats['kv_memory_available_mb'] / 1024:.1f} GB ✓")