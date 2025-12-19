import asyncio
import time
import random
from fake_llm_server import FakeLLMServer, InFlightRequest
from ml_controller import MLController
from actuator import Actuator
from models import Metrics

async def run_hybrid_benchmark():
    # Setup
    server = FakeLLMServer()
    actuator = Actuator("http://localhost:8000")
    
    # Initialize Hybrid Controller (PPO v08 + Shield)
    controller = MLController(model_path="ppo_finetuned_v08")
    
    print("="*60)
    print("🚀 BENCHMARKING HYBRID SHIELDED CONTROLLER")
    print("   Model: ppo_finetuned_v08 (Agent)")
    print("   Shield: Rescue (Queue>10 Mem<0.6 -> INC) + Panic (Mem>0.85 -> DEFER)")
    print("="*60)

    # Scenarios to test
    scenarios = [
        ("Surge", 1000)
    ]
    
    total_survived = 0
    
    for name, duration in scenarios:
        print(f"\nRunning {name} Scenario ({duration} steps)...")
        # Start server background tasks
        runner = await server.start()
        # Disable auto load to have manual control
        server.auto_load_enabled = False
        
        metrics_history = []
        decisions = []

        # Inject Traffic Pattern (Surge)
        if name == "Surge":
            print("   Injecting Surge (1000 requests)...")
            for i in range(1000):
                 req = InFlightRequest(
                     id=i,
                     state='queued',
                     arrival_time=time.time(),
                     priority=0
                 )
                 # Mock properties usually set by API
                 req.prompt_tokens = 128
                 req.max_tokens = 128
                 req.prefill_tokens_processed = 0
                 req.generated_tokens = 0
                 
                 server.queued_requests.append(req)
            
            # Wait a bit for surge to register in metrics
            await asyncio.sleep(0.5)
        
        for step in range(duration):
             # MANUAL METRIC CONSTRUCTION
             # Using internal state of FakeLLMServer
             
             q_depth = len(server.queued_requests)
             q_iw = len([r for r in server.queued_requests if r.priority==0])
             q_niw = len([r for r in server.queued_requests if r.priority>0])
             
             # Calculate Utilization manually
             active_reqs = len(server.prefilling_requests) + len(server.decoding_requests)
             capacity = server.config.batch_size * server.config.gpu_count
             gpu_util = min(100.0, (active_reqs / max(1, capacity)) * 100)
             
             # Calculate Memory Efficiency (KV Cache)
             # Approx: (active_reqs * 128 tokens) / (available_memory)
             # But let's just use the MLController's approximation logic style
             # or try to read server.memory_headroom if it exists.
             # Easier:
             kv_util = min(100.0, (active_reqs / (server.config.gpu_count * 16)) * 100) # Rough heuristic
             
             niw_flight = active_reqs # Simplified
             
             current_metrics = Metrics(
                 ttft_ms=getattr(server, 'current_ttft', 0.0),
                 inter_token_latency_ms=getattr(server, 'current_itl', 0.0),
                 prefill_latency_ms=0.0,
                 decode_latency_ms=0.0,
                 gpu_utilization=gpu_util,
                 memory_efficiency=kv_util,
                 gpu_balance_index=1.0,
                 comm_bubble_ratio=0.0,
                 speculative_factor=0.0,
                 queue_depth=q_depth,
                 timestamp=time.time(),
                 queue_velocity=0.0, 
                 queue_depth_iw=q_iw,
                 queue_depth_niw=q_niw,
                 niw_in_flight=niw_flight
             )
            
             # Update controller state
             try:
                 controller.update_config(batch_size=server.batch_size, gpu_count=server.gpu_count)
             except AttributeError:
                  pass

             decision = controller.make_decision(current_metrics)
            
             await actuator.apply_decision(decision)
             metrics_history.append(current_metrics)
             decisions.append(decision)
            
             await asyncio.sleep(0.01) # Short tick

        # Cleanup
        await server.stop()
        await runner.cleanup()

        # Analysis
        stats = controller.get_stats()
        final_q = metrics_history[-1].queue_depth
        survived = final_q < 500 # Should drain substantially
        
        print(f"   Final Queue: {final_q}")
        print(f"   Decisions: {stats['decision_count']}")
        print(f"   Overrides: {stats['safety_overrides']} ({stats['override_rate']*100:.1f}%)")
        print(f"   Survived: {survived}")
        
        if survived:
            total_survived += 1
            
        # Check action distribution
        actions = {}
        for d in decisions:
            val = d.action.value if hasattr(d.action, 'value') else str(d.action)
            actions[val] = actions.get(val, 0) + 1
        print("   Actions:", actions)
        
    print("\n" + "="*60)
    if total_survived == len(scenarios):
        print("✅ HYBRID CONTROLLER PASSED! (Survived all scenarios)")
    else:
        print("❌ HYBRID CONTROLLER FAILED!")

if __name__ == "__main__":
    asyncio.run(run_hybrid_benchmark())
