"""
v0.5-α Autopilot Runner with Predictive Controller
Runs both v0.4 reactive and v0.5-α predictive controllers side-by-side for comparison.
"""

import asyncio
import signal
import sys
from typing import Optional

from collector import MetricsCollector
from controller import Controller as ReactiveController
from controller_v2 import PredictiveController
from actuator import Actuator
from fake_llm_server import FakeLLMServer
from decision_logger import DecisionLogger
from models import Decision, Action


class AutopilotV2:
    """
    Dual-controller autopilot for v0.5-α validation.
    
    Runs both reactive (v0.4) and predictive (v0.5-α) controllers simultaneously:
    - Predictive controller makes actual decisions (applied to server)
    - Reactive controller runs in shadow mode (logged but not applied)
    - Comparison metrics logged for analysis
    """
    
    def __init__(
        self,
        slo_latency_ms: int = 600,
        decision_interval: float = 2.0,
        comparison_mode: bool = True
    ):
        self.slo_latency_ms = slo_latency_ms
        self.decision_interval = decision_interval
        self.comparison_mode = comparison_mode
        
        # Components
        self.fake_server = FakeLLMServer()
        self.collector = MetricsCollector(
            metrics_url="http://localhost:8000/metrics",
            poll_interval=1.0
        )
        self.predictive_controller = PredictiveController(slo_latency_ms=slo_latency_ms)
        self.reactive_controller = ReactiveController(slo_latency_ms=slo_latency_ms) if comparison_mode else None
        self.actuator = Actuator(configure_url="http://localhost:8000/configure")
        
        # Loggers
        self.predictive_logger = DecisionLogger(
            log_file="decision_log_v2_predictive.jsonl",
            txt_log_file="decision_log_v2_predictive.txt"
        )
        self.reactive_logger = DecisionLogger(
            log_file="decision_log_v2_reactive.jsonl",
            txt_log_file="decision_log_v2_reactive.txt"
        ) if comparison_mode else None
        self.comparison_logger = DecisionLogger(
            log_file="decision_log_v2_comparison.jsonl",
            txt_log_file="decision_log_v2_comparison.txt"
        ) if comparison_mode else None
        
        self._running = False
        self._tasks = []
    
    async def decision_loop(self):
        """
        Main control loop with dual-controller comparison.
        """
        print(f"🧠 Decision loop started (interval={self.decision_interval}s)")
        print(f"🔬 Comparison mode: {'ENABLED' if self.comparison_mode else 'DISABLED'}")
        
        while self._running:
            try:
                # Get current metrics (with moving average)
                metrics = self.collector.get_moving_average(window=3)
                
                if metrics:
                    # Predictive controller decision (APPLIED)
                    predictive_decision = self.predictive_controller.make_decision(metrics)
                    
                    # Log predictive decision
                    self.predictive_logger.log_decision(
                        predictive_decision,
                        metrics,
                        mode=self.predictive_controller.mode
                    )
                    
                    # Update collector with predictive decision
                    self.collector.update_last_decision(
                        predictive_decision,
                        self.predictive_controller.mode
                    )
                    
                    # Apply predictive decision
                    if predictive_decision.action != Action.NO_ACTION:
                        await self.actuator.apply_decision(predictive_decision)
                        print(f"✅ Applied predictive: {predictive_decision.action.value} - {predictive_decision.reason}")
                    
                    # Reactive controller decision (SHADOW MODE)
                    if self.comparison_mode and self.reactive_controller:
                        reactive_decision = self.reactive_controller.make_decision(metrics)
                        
                        # Log reactive decision
                        self.reactive_logger.log_decision(
                            reactive_decision,
                            metrics,
                            mode=self.reactive_controller.mode
                        )
                        
                        # Log comparison
                        self._log_comparison(predictive_decision, reactive_decision, metrics)
                
                await asyncio.sleep(self.decision_interval)
                
            except Exception as e:
                print(f"❌ Error in decision loop: {e}")
                await asyncio.sleep(self.decision_interval)
    
    def _log_comparison(self, predictive: Decision, reactive: Decision, metrics):
        """
        Log comparison between predictive and reactive decisions.
        """
        comparison_data = {
            "predictive_action": predictive.action.value,
            "reactive_action": reactive.action.value,
            "agreement": predictive.action == reactive.action,
            "predictive_reason": predictive.reason,
            "reactive_reason": reactive.reason,
            "ttft_ms": metrics.ttft_ms,
            "queue_depth": metrics.queue_depth,
            "queue_velocity": metrics.queue_velocity
        }
        
        if predictive.action != reactive.action:
            print(f"🔀 DIVERGENCE: Predictive={predictive.action.value} vs Reactive={reactive.action.value}")
        
        # Log as special decision for comparison tracking
        comparison_decision = Decision(
            action=Action.NO_ACTION,
            reason=f"Comparison: P={predictive.action.value} R={reactive.action.value}"
        )
        self.comparison_logger.log_decision(comparison_decision, metrics, mode="comparison")
    
    async def start(self):
        """
        Start all components and control loop.
        """
        print("🚀 Starting Autopilot v0.5-α...")
        self._running = True
        
        # Start fake server
        print("🖥️  Starting fake LLM server on http://localhost:8000")
        server_task = asyncio.create_task(self.fake_server.start())
        self._tasks.append(server_task)
        await asyncio.sleep(0.5)  # Let server initialize
        
        # Start collector
        print("📊 Starting metrics collector on http://localhost:8080")
        await self.collector.start()
        collector_task = asyncio.create_task(self.collector.collect_loop())
        self._tasks.append(collector_task)
        
        # Start decision loop
        decision_task = asyncio.create_task(self.decision_loop())
        self._tasks.append(decision_task)
        
        print("✅ Autopilot v0.5-α running")
        print(f"📈 Dashboard: streamlit run dashboard.py")
        print(f"📊 Live metrics: http://localhost:8080/live_metrics")
        print(f"🧠 Last decision: http://localhost:8080/last_decision")
        print(f"🔬 Predictive log: decision_log_v2_predictive.jsonl")
        if self.comparison_mode:
            print(f"🔬 Reactive log: decision_log_v2_reactive.jsonl")
            print(f"🔬 Comparison log: decision_log_v2_comparison.jsonl")
        print("\n⏸️  Press Ctrl+C to stop\n")
    
    async def stop(self):
        """
        Graceful shutdown of all components.
        """
        print("\n🛑 Stopping Autopilot v0.5-α...")
        self._running = False
        
        # Stop collector
        await self.collector.stop()
        
        # Cancel all tasks
        for task in self._tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        
        # Stop fake server
        await self.fake_server.stop()
        
        # Close loggers
        self.predictive_logger.close()
        if self.reactive_logger:
            self.reactive_logger.close()
        if self.comparison_logger:
            self.comparison_logger.close()
        
        # Cleanup actuator
        await self.actuator.cleanup()
        
        print("✅ Autopilot v0.5-α stopped")


async def main():
    """
    Main entry point with signal handling.
    """
    autopilot = AutopilotV2(
        slo_latency_ms=600,
        decision_interval=2.0,
        comparison_mode=True  # Enable side-by-side comparison
    )
    
    # Signal handler for graceful shutdown
    def signal_handler(sig, frame):
        print("\n⚠️  Received interrupt signal")
        asyncio.create_task(autopilot.stop())
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        await autopilot.start()
        
        # Keep running until stopped
        while autopilot._running:
            await asyncio.sleep(1)
            
    except KeyboardInterrupt:
        print("\n⚠️  Keyboard interrupt")
    finally:
        await autopilot.stop()


if __name__ == "__main__":
    asyncio.run(main())
