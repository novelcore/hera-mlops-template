"""Manager for the config-validation step.

Wires the step config (env-driven timeouts) to the ConfigValidationService and
exposes a single run() entry the CLI/entry point calls.
"""

import logging
import os

from app.logger import setup_logging
from app.services.config_validation import ConfigValidationService


class Manager:
    def __init__(self) -> None:
        setup_logging(level=os.environ.get("LOG_LEVEL", "INFO"))
        self._log = logging.getLogger(__name__)
        self._service = ConfigValidationService(
            timeout=int(os.environ.get("TIMEOUT", "15")),
            max_retries=int(os.environ.get("MAX_RETRIES", "3")),
        )

    def run(self, resolved: dict) -> None:
        self._log.info("Starting config-validation")
        self._service.run(resolved)
        self._log.info("Config validation completed successfully")
