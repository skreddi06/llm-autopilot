"""Actuator that applies decisions to the LLM server."""

import asyncio
import logging
from typing import Optional
from aiohttp import ClientSession, ClientError
from models import Decision, ServerConfig

logger = logging.getLogger(__name__)


class Actuator:
    """Applies controller decisions to the server."""
    
    def __init__(self, server_url: str = "http://localhost:8000"):
        self.server_url = server_url
        self.session: Optional[ClientSession] = None
        self.current_config: Optional[ServerConfig] = None
    
    async def initialize(self):
        """Initialize the actuator (create HTTP session)."""
        self.session = ClientSession()
        # Fetch current config
        await self._fetch_current_config()
    
    async def _fetch_current_config(self):
        """Fetch current server configuration."""
        try:
            async with self.session.get(f"{self.server_url}/config") as response:
                if response.status == 200:
                    data = await response.json()
                    self.current_config = ServerConfig(
                        batch_size=int(data["batch_size"]),
                        gpu_count=int(data["gpu_count"])
                    )
                    logger.info(f"Fetched current config: batch_size={self.current_config.batch_size}, gpu_count={self.current_config.gpu_count}")
                else:
                    logger.warning(f"Failed to fetch config: HTTP {response.status}")
        except Exception as e:
            logger.error(f"Error fetching config: {e}")
    
    async def apply_decision(self, decision: Decision) -> bool:
        """Apply a decision to the server."""
        if not self.session:
            await self.initialize()

        if decision.action.value == "no_action":
            logger.debug(f"No action needed: {decision.reason}")
            return True

        try:
            config_update = {}

            if decision.action.value == "reduce_batch":
                new_batch_size = max(self.current_config.batch_size - 1, 1)
                config_update["batch_size"] = new_batch_size
                logger.info(f"Reducing batch size to {new_batch_size}")

            elif decision.action.value == "increase_batch":
                new_batch_size = min(self.current_config.batch_size + 1, 32)
                config_update["batch_size"] = new_batch_size
                logger.info(f"Increasing batch size to {new_batch_size}")

            elif decision.action.value == "scale_out":
                new_gpu_count = min(self.current_config.gpu_count + 1, 8)
                config_update["gpu_count"] = new_gpu_count
                logger.info(f"Scaling out to {new_gpu_count} GPUs")

            elif decision.action.value == "rebalance_load":
                config_update["rebalance"] = True
                logger.info("Rebalancing load across GPUs")

            elif decision.action.value == "enable_speculative_decode":
                config_update["speculative_mode"] = True
                logger.info("Enabling speculative decoding")

            elif decision.action.value == "enable_overlap_mode":
                config_update["overlap_mode"] = True
                logger.info("Enabling overlap mode")

            # Send the configuration update
            await self._update_config(**config_update)
            return True

        except Exception as e:
            logger.error(f"Failed to apply decision: {e}")
            return False

    async def _update_config(self, batch_size: Optional[int] = None, gpu_count: Optional[int] = None, **kwargs):
        """Update the server configuration."""
        config_update = {}
        if batch_size is not None:
            config_update["batch_size"] = batch_size
        if gpu_count is not None:
            config_update["gpu_count"] = gpu_count

        # Include additional parameters
        config_update.update(kwargs)

        try:
            async with self.session.post(f"{self.server_url}/configure", json=config_update) as response:
                if response.status == 200:
                    logger.info("Configuration updated successfully")
                else:
                    logger.warning(f"Failed to update config: HTTP {response.status}")
        except Exception as e:
            logger.error(f"Error updating config: {e}")
    
    def get_current_config(self) -> Optional[ServerConfig]:
        """Get the current server configuration."""
        return self.current_config
    
    async def cleanup(self):
        """Clean up resources."""
        if self.session:
            await self.session.close()
            self.session = None

