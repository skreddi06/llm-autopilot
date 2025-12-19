"""Decision Logger for recording metrics and decisions."""

import logging
import json
import time
from typing import Optional
from models import Metrics, Decision, Action
from logging.handlers import RotatingFileHandler
from datetime import datetime

logger = logging.getLogger("decision_logger")

class DecisionLogger:
    """Logs metrics and decisions for debugging and visualization."""

    def __init__(self, log_file: str = "decision_log.txt", jsonl_file: str = "decision_log.jsonl"):
        self.log_file = log_file
        self.jsonl_file = jsonl_file
        self._setup_logger()

    def _setup_logger(self):
        """Set up the logger to write to a file and console."""
        # Rotating file handler for human-readable logs
        file_handler = RotatingFileHandler(self.log_file, maxBytes=5_000_000, backupCount=5)
        file_formatter = logging.Formatter("%(asctime)s - %(message)s")
        file_handler.setFormatter(file_formatter)

        # Console handler
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter("%(asctime)s - %(message)s")
        console_handler.setFormatter(console_formatter)

        # Add handlers to logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        logger.setLevel(logging.INFO)

    def log_metrics(self, metrics: Metrics):
        """Log the current metrics as JSON."""
        logger.info(json.dumps({
            "timestamp": time.time(),
            "metrics": metrics.__dict__
        }))

    def log_decision(self, decision: Decision):
        """Log the decision made by the controller as JSON."""
        logger.info(json.dumps({
            "timestamp": time.time(),
            "decision": decision.__dict__
        }))

    def log_metrics_and_decision(self, metrics: Metrics, decision: Decision):
        """Log both metrics and the decision together."""
        def custom_serializer(obj):
            if isinstance(obj, Action):
                return obj.value  # Serialize enums as their value
            raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

        # Create structured log entry
        entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "mode": decision.reason.split(",")[0],  # Extract mode from reason
            "action": decision.action.value,
            "metrics": metrics.__dict__,
            "reason": decision.reason
        }

        # Write to human-readable log
        logger.info(f"Mode: {entry['mode']}, Action: {entry['action']}, Reason: {entry['reason']}")

        # Write to JSONL log
        with open(self.jsonl_file, "a", encoding="utf-8") as jsonl:
            jsonl.write(json.dumps(entry, default=custom_serializer) + "\n")

        # Debug confirmation
        print(f"[DecisionLogger] Logged metrics + decision at {entry['timestamp']}")

    def shutdown(self):
        """Clean up and close all file handlers."""
        for handler in logger.handlers:
            handler.close()
            logger.removeHandler(handler)