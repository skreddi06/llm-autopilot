"""Main orchestrator that runs the LLM Autopilot system."""

import asyncio
import logging
import signal
import sys
from fake_llm_server import FakeLLMServer
from collector import Collector
from controller import Controller
from actuator import Actuator
from models import Metrics, ServerConfig
from decision_logger import DecisionLogger

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class Autopilot:
    """Main autopilot orchestrator."""
    
    def __init__(
        self,
        server_host: str = "localhost",
        server_port: int = 8000,
        slo_latency_ms: float = 600.0,
        poll_interval: float = 1.0,
        decision_interval: float = 2.0
    ):
        self.server_host = server_host
        self.server_port = server_port
        self.server_url = f"http://{server_host}:{server_port}"
        self.poll_interval = poll_interval
        self.decision_interval = decision_interval
        
        # Initialize components
        self.server = FakeLLMServer(host=server_host, port=server_port)
        self.collector = Collector(server_url=self.server_url, poll_interval=poll_interval)
        self.controller = Controller(slo_latency_ms=slo_latency_ms)
        self.actuator = Actuator(server_url=self.server_url)
        self.logger = DecisionLogger()
        
        # Control flags
        self.running = False
        self.server_runner = None
    
    async def start_server(self):
        """Start the fake LLM server."""
        logger.info("Starting fake LLM server...")
        self.server_runner = await self.server.start()
        # Give server a moment to start
        await asyncio.sleep(0.5)
    
    async def initialize_components(self):
        """Initialize all components."""
        logger.info("Initializing autopilot components...")
        await self.actuator.initialize()
        
        # Fetch initial config and update controller
        initial_config = self.actuator.get_current_config()
        if initial_config:
            self.controller.update_config(initial_config)
            logger.info(f"Initial server config: batch_size={initial_config.batch_size}, gpu_count={initial_config.gpu_count}")
    
    async def decision_loop(self):
        """Main decision loop."""
        logger.info("Starting autopilot decision loop...")
        logger.info(f"SLO target: {self.controller.slo_latency_ms}ms")
        logger.info(f"Decision interval: {self.decision_interval}s")
        
        try:
            while self.running:
                # Get current metrics
                metrics = self.collector.get_current_metrics()
                
                if metrics:
                    # Update controller with current config
                    current_config = self.actuator.get_current_config()
                    if current_config:
                        self.controller.update_config(current_config)
                    
                    # Calculate moving average of TTFT (last 3 metrics)
                    recent = self.collector.get_metrics_history(3)
                    if recent:
                        avg_ttft = sum(m.ttft_ms for m in recent) / len(recent)
                        metrics.ttft_ms = avg_ttft
                    
                    # Make decision
                    decision = self.controller.make_decision(metrics)
                    
                    # Log metrics and decision
                    self.logger.log_metrics_and_decision(metrics, decision)
                    print("[Autopilot] Metrics and decision logged.")
                    
                    logger.info("=" * 80)
                    logger.info(f"Metrics: TTFT={metrics.ttft_ms:.1f}ms, "
                              f"Inter-token={metrics.inter_token_latency_ms:.1f}ms, "
                              f"GPU={metrics.gpu_utilization:.1f}%, "
                              f"Queue={metrics.queue_depth}, "
                              f"Velocity={metrics.queue_velocity:.2f}/s")
                    logger.info(f"Decision: {decision.action.value} - {decision.reason}")
                    
                    # Apply decision if not NO_ACTION
                    if decision.action.value != "no_action":
                        success = await self.actuator.apply_decision(decision)
                        if success:
                            # Update controller with new config
                            new_config = self.actuator.get_current_config()
                            if new_config:
                                self.controller.update_config(new_config)
                    else:
                        logger.info("No action taken")
                    logger.info("=" * 80)
                    
                    await asyncio.sleep(self.decision_interval)
                else:
                    await asyncio.sleep(self.poll_interval)
        except asyncio.CancelledError:
            logger.info("Decision loop cancelled")
    
    async def run(self):
        """Run the autopilot system."""
        self.running = True

        try:
            # Start server
            await self.start_server()

            # Initialize components
            await self.initialize_components()

            # Start collector in background
            collector_task = asyncio.create_task(self.collector.collect_loop())

            # Start live metrics HTTP server
            await self.collector.start_http_server()

            # Wait a bit for initial metrics
            await asyncio.sleep(self.poll_interval * 2)

            # Start decision loop
            decision_task = asyncio.create_task(self.decision_loop())

            # Wait for both tasks
            await asyncio.gather(collector_task, decision_task)
        except KeyboardInterrupt:
            logger.info("Received shutdown signal")
        finally:
            await self.shutdown()

    async def shutdown(self):
        """Shutdown all components gracefully."""
        logger.info("Shutting down autopilot...")
        self.running = False

        # Stop collector
        self.collector.stop()

        # Stop live metrics HTTP server
        await self.collector.stop_http_server()

        # Cleanup actuator
        await self.actuator.cleanup()

        # Stop server
        if self.server_runner:
            await self.server_runner.cleanup()

        # Shutdown the decision logger
        self.logger.shutdown()

        logger.info("Autopilot shutdown complete")


def setup_signal_handlers(autopilot):
    """Setup signal handlers for graceful shutdown."""
    def signal_handler(sig, frame):
        logger.info("Received interrupt signal")
        asyncio.create_task(autopilot.shutdown())
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


async def main():
    """Main entry point."""
    logger.info("=" * 80)
    logger.info("LLM Autopilot v0.1")
    logger.info("=" * 80)
    
    autopilot = Autopilot(
        server_host="localhost",
        server_port=8000,
        slo_latency_ms=600.0,
        poll_interval=1.0,
        decision_interval=2.0
    )
    
    setup_signal_handlers(autopilot)
    
    try:
        await autopilot.run()
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
    finally:
        await autopilot.shutdown()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Exiting...")


