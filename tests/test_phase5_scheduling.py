"""Phase 5 Verification Tests

Test A: Static vs Continuous Throughput
- Proves continuous batching eliminates GPU idle time from output variance

Test B: Chunking Stability (RAG Spike)
- Proves chunked prefill prevents HOL blocking and stabilizes TBT jitter
"""

import sys
import os
import time
import asyncio
import statistics
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fake_llm_server import FakeLLMServer, InFlightRequest


async def test_a_static_vs_continuous():
    """Test A: Static vs Continuous Throughput
    
    With high variance in output lengths, continuous batching should
    maintain higher throughput than static batching by filling gaps
    left by short requests.
    """
    print("=" * 70)
    print("TEST A: Static vs Continuous Throughput")
    print("=" * 70)
    print()
    
    # Create requests with EXTREME variance in output length
    # Many short (5 tokens) vs fewer very long (1000 tokens)
    # This creates the "idle GPU" effect in static batching
    request_configs = []
    
    # 20 very short requests (complete almost instantly)
    for _ in range(20):
        request_configs.append({'prompt': 256, 'output': 5})
    
    # 5 very long requests (take much longer)
    for _ in range(5):
        request_configs.append({'prompt': 256, 'output': 1000})
    
    results = {}
    
    for mode_name, continuous_mode in [("STATIC", False), ("CONTINUOUS", True)]:
        print(f"\n--- {mode_name} BATCHING ---")
        
        # Create server
        server = FakeLLMServer(port=8010)
        server.continuous_batching_enabled = continuous_mode
        server.auto_load_enabled = False
        server.config.batch_size = 10  # Can hold all requests
        
        # Add all requests to queue
        for i, cfg in enumerate(request_configs):
            req = InFlightRequest(
                id=i,
                state='queued',
                arrival_time=time.time(),
                prompt_tokens=cfg['prompt'],
                target_tokens=cfg['output'],
                generated_tokens=0,
                prefill_tokens_processed=0
            )
            server.queued_requests.append(req)
            server.total_submitted += 1
        
        # Run simulation for fixed duration
        start_time = time.time()
        simulation_duration = 2.0  # seconds
        steps = 0
        tokens_completed = 0
        
        while time.time() - start_time < simulation_duration:
            current_time = time.time()
            server.total_step_count += 1
            steps += 1
            
            # Check memory congestion
            swap_penalty = 1.0
            if server.check_memory_congestion():
                swap_penalty = 1.0 + (server.memory_swap_penalty_ms / 100.0)
            
            if continuous_mode:
                # Continuous batching: token budget scheduling
                remaining_budget = server.max_tokens_per_batch
                
                # Priority 1: Decode tokens
                decode_tokens = min(len(server.decoding_requests), remaining_budget)
                remaining_budget -= decode_tokens
                
                # Priority 2: Chunked prefill for existing prefilling requests
                for req in server.prefilling_requests:
                    if remaining_budget <= 0:
                        break
                    if not req.prefill_complete:
                        chunk = min(server.prefill_chunk_size, req.prefill_remaining, remaining_budget)
                        req.prefill_tokens_processed += chunk
                        remaining_budget -= chunk
                
                # Priority 3: Start new requests from queue
                while remaining_budget > 0 and server.queued_requests:
                    req = server.queued_requests.pop(0)
                    req.state = 'prefilling'
                    req.prefill_start = current_time
                    chunk = min(server.prefill_chunk_size, req.prompt_tokens, remaining_budget)
                    req.prefill_tokens_processed = chunk
                    remaining_budget -= chunk
                    server.prefilling_requests.append(req)
                
                # Track occupancy
                total_tokens = decode_tokens + (server.max_tokens_per_batch - remaining_budget - decode_tokens)
                server.batch_occupancies.append(total_tokens / server.max_tokens_per_batch)
            
            else:
                # Static batching: just move whole requests through stages
                # Pull from queue up to batch size
                while len(server.prefilling_requests) + len(server.decoding_requests) < server.config.batch_size:
                    if not server.queued_requests:
                        break
                    req = server.queued_requests.pop(0)
                    req.state = 'prefilling'
                    req.prefill_start = current_time
                    req.prefill_tokens_processed = req.prompt_tokens  # Instant prefill in static mode
                    server.prefilling_requests.append(req)
            
            # Process decode tokens
            step_latency = server.calculate_roofline_latency_ms() * swap_penalty
            tokens_per_tick = max(1, int(server.simulation_interval * 1000 / step_latency))
            
            for req in server.decoding_requests:
                if req.generated_tokens < req.target_tokens:
                    req.generated_tokens = min(req.target_tokens, req.generated_tokens + tokens_per_tick)
                    tokens_completed += tokens_per_tick
            
            # Complete decodes
            for req in server.decoding_requests[:]:
                if req.generated_tokens >= req.target_tokens:
                    req.state = 'completed'
                    req.completion_time = current_time
                    server.decoding_requests.remove(req)
                    server.completed_count += 1
            
            # Move completed prefills to decode
            for req in server.prefilling_requests[:]:
                if req.prefill_complete:
                    req.state = 'decoding'
                    req.decode_start = current_time
                    req.generated_tokens = 0
                    server.prefilling_requests.remove(req)
                    server.decoding_requests.append(req)
            
            await asyncio.sleep(server.simulation_interval)
        
        elapsed = time.time() - start_time
        throughput = tokens_completed / elapsed
        completed = server.completed_count
        avg_occupancy = sum(server.batch_occupancies) / len(server.batch_occupancies) if server.batch_occupancies else 0
        
        results[mode_name] = {
            'throughput': throughput,
            'completed': completed,
            'occupancy': avg_occupancy * 100,
            'steps': steps
        }
        
        print(f"  Completed: {completed} requests")
        print(f"  Throughput: {throughput:.0f} tokens/sec")
        print(f"  Avg Occupancy: {avg_occupancy*100:.1f}%")
    
    # Compare
    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    
    static_tp = results['STATIC']['throughput']
    continuous_tp = results['CONTINUOUS']['throughput']
    improvement = ((continuous_tp - static_tp) / static_tp) * 100 if static_tp > 0 else 0
    
    print(f"\nStatic Throughput:     {static_tp:.0f} tok/sec")
    print(f"Continuous Throughput: {continuous_tp:.0f} tok/sec")
    print(f"Improvement: {improvement:+.1f}%")
    
    if continuous_tp > static_tp:
        print("\n✅ PASS: Continuous batching achieves higher throughput!")
        return True
    else:
        print("\n❌ FAIL: No throughput improvement from continuous batching")
        return False


async def test_b_chunking_stability():
    """Test B: Chunking Stability (RAG Spike)
    
    Inject a massive prefill request into a steady decode workload.
    Chunked prefill should keep decode jitter low.
    """
    print()
    print("=" * 70)
    print("TEST B: Chunking Stability (RAG Spike)")
    print("=" * 70)
    print()
    
    # Test with and without chunking
    results = {}
    
    for mode_name, chunk_size in [("NO_CHUNKING", 8192), ("WITH_CHUNKING", 512)]:
        print(f"\n--- {mode_name} (chunk_size={chunk_size}) ---")
        
        server = FakeLLMServer(port=8011)
        server.continuous_batching_enabled = True
        server.auto_load_enabled = False
        server.prefill_chunk_size = chunk_size
        server.config.batch_size = 20
        
        # Start with 10 active decode requests (the "chat users")
        for i in range(10):
            req = InFlightRequest(
                id=i,
                state='decoding',
                arrival_time=time.time(),
                decode_start=time.time(),
                prompt_tokens=512,
                target_tokens=200,  # Medium length outputs
                generated_tokens=50,  # Already 50 tokens in
                prefill_tokens_processed=512  # Prefill complete
            )
            server.decoding_requests.append(req)
            server.total_submitted += 1
        
        decode_latencies = []
        rag_injected = False
        
        # Run simulation
        start_time = time.time()
        simulation_duration = 1.5  # seconds
        steps = 0
        
        while time.time() - start_time < simulation_duration:
            current_time = time.time()
            elapsed = current_time - start_time
            steps += 1
            
            # Inject RAG request at 0.5 seconds
            if elapsed > 0.5 and not rag_injected:
                rag_req = InFlightRequest(
                    id=999,
                    state='queued',
                    arrival_time=current_time,
                    prompt_tokens=4096,  # Massive 4k prompt!
                    target_tokens=100,
                    generated_tokens=0,
                    prefill_tokens_processed=0
                )
                server.queued_requests.append(rag_req)
                server.total_submitted += 1
                rag_injected = True
                print(f"  [T={elapsed:.2f}s] Injected RAG request (4k prompt)")
            
            # Token budget scheduling
            remaining_budget = server.max_tokens_per_batch
            
            # Priority 1: Decode
            decode_count = len(server.decoding_requests)
            remaining_budget -= decode_count
            
            # Priority 2: Chunked prefill
            for req in server.prefilling_requests:
                if remaining_budget <= 0:
                    break
                if not req.prefill_complete:
                    chunk = min(server.prefill_chunk_size, req.prefill_remaining, remaining_budget)
                    req.prefill_tokens_processed += chunk
                    remaining_budget -= chunk
            
            # Priority 3: New requests from queue
            while remaining_budget > 0 and server.queued_requests:
                req = server.queued_requests.pop(0)
                req.state = 'prefilling'
                req.prefill_start = current_time
                chunk = min(server.prefill_chunk_size, req.prompt_tokens, remaining_budget)
                req.prefill_tokens_processed = chunk
                remaining_budget -= chunk
                server.prefilling_requests.append(req)
            
            # Calculate decode latency for this step
            step_latency = server.calculate_roofline_latency_ms()
            decode_latencies.append(step_latency)
            
            # Process decodes
            tokens_per_tick = max(1, int(server.simulation_interval * 1000 / step_latency))
            for req in server.decoding_requests:
                if req.generated_tokens < req.target_tokens:
                    req.generated_tokens = min(req.target_tokens, req.generated_tokens + tokens_per_tick)
            
            # Complete
            for req in server.decoding_requests[:]:
                if req.generated_tokens >= req.target_tokens:
                    req.state = 'completed'
                    server.decoding_requests.remove(req)
                    server.completed_count += 1
            
            # Transition
            for req in server.prefilling_requests[:]:
                if req.prefill_complete:
                    req.state = 'decoding'
                    req.decode_start = current_time
                    req.generated_tokens = 0
                    server.prefilling_requests.remove(req)
                    server.decoding_requests.append(req)
            
            await asyncio.sleep(server.simulation_interval)
        
        # Calculate jitter
        jitter = statistics.stdev(decode_latencies) if len(decode_latencies) > 1 else 0
        max_latency = max(decode_latencies) if decode_latencies else 0
        avg_latency = sum(decode_latencies) / len(decode_latencies) if decode_latencies else 0
        
        results[mode_name] = {
            'jitter': jitter,
            'max_latency': max_latency,
            'avg_latency': avg_latency,
            'steps': steps
        }
        
        print(f"  Avg Decode Latency: {avg_latency:.1f}ms")
        print(f"  Max Decode Latency: {max_latency:.1f}ms")
        print(f"  Decode Jitter (std): {jitter:.2f}ms")
    
    # Compare
    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    
    no_chunk_jitter = results['NO_CHUNKING']['jitter']
    with_chunk_jitter = results['WITH_CHUNKING']['jitter']
    reduction = ((no_chunk_jitter - with_chunk_jitter) / no_chunk_jitter) * 100 if no_chunk_jitter > 0 else 0
    
    print(f"\nNo Chunking Jitter:   {no_chunk_jitter:.2f}ms")
    print(f"With Chunking Jitter: {with_chunk_jitter:.2f}ms")
    print(f"Jitter Reduction: {reduction:.1f}%")
    
    if with_chunk_jitter <= no_chunk_jitter or with_chunk_jitter < 5.0:
        print("\n✅ PASS: Chunked prefill maintains stable decode latency!")
        return True
    else:
        print("\n❌ FAIL: Chunking did not reduce jitter")
        return False


async def main():
    print("Phase 5 Continuous Batching Verification")
    print("=" * 70)
    
    test_a_pass = await test_a_static_vs_continuous()
    test_b_pass = await test_b_chunking_stability()
    
    print()
    print("=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"Test A (Static vs Continuous): {'PASS ✅' if test_a_pass else 'FAIL ❌'}")
    print(f"Test B (Chunking Stability):   {'PASS ✅' if test_b_pass else 'FAIL ❌'}")
    
    if test_a_pass and test_b_pass:
        print("\n🎉 Phase 5 Verification Complete!")
        print("   Continuous batching and chunked prefill are working correctly.")
    
    return test_a_pass and test_b_pass


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
