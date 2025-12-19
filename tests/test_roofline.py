"""Roofline Verification Test

Tests that the simulator exhibits the complete 3-zone performance curve:
1. Linear Growth: Low batch, throughput rises (memory-bound)
2. Plateau: Mid batch, throughput flattens (compute-bound)
3. Cliff: High batch, throughput crashes (memory wall)

This validates Phase 4 (Compute Saturation) implementation.
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fake_llm_server import FakeLLMServer, InFlightRequest


def test_roofline_3_zones():
    """Test that the roofline model produces 3 distinct performance zones."""
    
    print("=" * 70)
    print("ROOFLINE VERIFICATION TEST (3-Zone Curve)")
    print("=" * 70)
    print()
    
    server = FakeLLMServer()
    server.auto_load_enabled = False
    
    # Use moderate sequence length
    PROMPT_TOKENS = 512
    GENERATED_TOKENS = 128
    TOTAL_TOKENS = PROMPT_TOKENS + GENERATED_TOKENS
    
    print(f"Configuration:")
    print(f"  Effective TFLOPS: {server.effective_tflops:.0f}")
    print(f"  FLOPs per token: {server.flops_per_token / 1e9:.0f} GFLOPs")
    print(f"  Saturation threshold: {server.compute_saturation_threshold * 100:.0f}%")
    print(f"  KV Memory: {server.available_kv_memory_bytes / 1024**3:.1f} GB")
    print()
    
    batch_sizes = [1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256]
    
    header = f"{'Batch':<8} {'Latency':<12} {'Compute%':<12} {'Memory%':<12} {'Throughput':<12} {'Zone':<15}"
    print(header)
    print("-" * 71)
    
    results = []
    
    for batch_size in batch_sizes:
        # Reset server state
        server.decoding_requests.clear()
        server.prefilling_requests.clear()
        server.config.batch_size = batch_size
        
        # Fill batch with requests
        for i in range(batch_size):
            req = InFlightRequest(
                id=i,
                state='decoding',
                arrival_time=0,
                decode_start=0,
                prompt_tokens=PROMPT_TOKENS,
                generated_tokens=GENERATED_TOKENS,
                target_tokens=GENERATED_TOKENS
            )
            server.decoding_requests.append(req)
        
        # Calculate metrics
        latency_ms = server.calculate_roofline_latency_ms(batch_size)
        compute_util = server.get_compute_utilization() * 100
        memory_util = server.get_memory_utilization()
        is_congested = server.check_memory_congestion()
        
        # Throughput = batch_size / latency (tokens per second)
        throughput = (batch_size / latency_ms) * 1000 if latency_ms > 0 else 0
        
        # Apply swap penalty to throughput if congested
        if is_congested:
            throughput /= 4  # 4x slowdown
        
        results.append({
            'batch': batch_size,
            'latency': latency_ms,
            'compute': compute_util,
            'memory': memory_util,
            'throughput': throughput,
            'congested': is_congested
        })
    
    # Determine zones based on throughput behavior
    # Zone detection by comparing throughput growth
    for i, r in enumerate(results):
        if r['congested']:
            r['zone'] = "CLIFF 📉"
        elif i > 0:
            # Check if throughput is growing significantly
            prev_throughput = results[i-1]['throughput']
            growth_rate = (r['throughput'] - prev_throughput) / prev_throughput if prev_throughput > 0 else 1
            
            # Plateau: throughput growth < 10% despite batch doubling
            if growth_rate < 0.1 and not results[i-1].get('congested', False):
                r['zone'] = "PLATEAU 🔒"
            else:
                r['zone'] = "LINEAR 📈"
        else:
            r['zone'] = "LINEAR 📈"
    
    # Print results
    print(header)
    print("-" * 71)
    for r in results:
        print(f"{r['batch']:<8} {r['latency']:<12.2f} {r['compute']:<12.1f} {r['memory']:<12.1f} {r['throughput']:<12.1f} {r['zone']:<15}")
    
    print()
    print("=" * 70)
    print("ZONE ANALYSIS")
    print("=" * 70)
    
    # Analyze zones
    linear_zone = [r for r in results if 'LINEAR' in r['zone']]
    plateau_zone = [r for r in results if 'PLATEAU' in r['zone']]
    cliff_zone = [r for r in results if 'CLIFF' in r['zone']]
    
    print(f"\n📈 LINEAR Zone: {len(linear_zone)} batch sizes")
    if linear_zone:
        throughputs = [r['throughput'] for r in linear_zone]
        print(f"   Throughput range: {min(throughputs):.1f} - {max(throughputs):.1f} tokens/sec")
        print(f"   Throughput increases linearly with batch size")
    
    print(f"\n🔒 PLATEAU Zone: {len(plateau_zone)} batch sizes")
    if plateau_zone:
        throughputs = [r['throughput'] for r in plateau_zone]
        print(f"   Throughput range: {min(throughputs):.1f} - {max(throughputs):.1f} tokens/sec")
        print(f"   Throughput flattens (compute saturated)")
    
    print(f"\n📉 CLIFF Zone: {len(cliff_zone)} batch sizes")
    if cliff_zone:
        throughputs = [r['throughput'] for r in cliff_zone]
        print(f"   Throughput range: {min(throughputs):.1f} - {max(throughputs):.1f} tokens/sec")
        print(f"   Throughput crashes (memory swap penalty)")
    
    # Verify all 3 zones exist
    print()
    print("=" * 70)
    print("VERIFICATION RESULTS")
    print("=" * 70)
    
    all_zones_present = len(linear_zone) > 0 and len(plateau_zone) > 0 and len(cliff_zone) > 0
    
    if all_zones_present:
        print("\n✅ ALL 3 ZONES DETECTED!")
        print("   The simulator correctly models the roofline performance curve.")
        print("   - Linear growth at low batch (memory-limited)")
        print("   - Plateau at mid batch (compute-limited)")
        print("   - Cliff at high batch (memory wall)")
        return True
    else:
        missing = []
        if len(linear_zone) == 0:
            missing.append("LINEAR")
        if len(plateau_zone) == 0:
            missing.append("PLATEAU")
        if len(cliff_zone) == 0:
            missing.append("CLIFF")
        
        print(f"\n❌ MISSING ZONES: {', '.join(missing)}")
        return False


if __name__ == "__main__":
    success = test_roofline_3_zones()
    sys.exit(0 if success else 1)
