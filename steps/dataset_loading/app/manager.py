"""Manager for the Dataset Loading pipeline step.

Wires together configuration, the boto3 S3 client, and
``DatasetLoadingService``.  The ``run`` method is the single orchestration
entry point called by the CLI.
"""

import logging
from typing import Optional

import boto3
from botocore.config import Config as BotoConfig

from app.logger import setup_logging
from app.models.config import Config
from app.models.dataset import YoloDatasetParams
from app.services.dataset_loading import DatasetLoadingService


class Manager:
    """Orchestrates the dataset loading step.

    Parameters
    ----------
    config:
        Optional pre-built ``Config`` instance.  When *None* a default
        ``Config()`` is constructed (which reads from env vars / .env).
    """

    def __init__(self, config: Optional[Config] = None) -> None:
        self._config = config or Config()
        setup_logging(level=self._config.log_level)
        self._logger = logging.getLogger(__name__)

        self._s3_client = self._build_s3_client()
        self._service = DatasetLoadingService(
            s3_client=self._s3_client,
            max_retries=self._config.max_retries,
            lakefs_endpoint=self._config.lakefs_endpoint,
            lakefs_access_key=self._config.lakefs_access_key,
            lakefs_secret_key=self._config.lakefs_secret_key,
        )

    # ------------------------------------------------------------------
    # S3 / LakeFS client factory
    # ------------------------------------------------------------------

    def _build_s3_client(self) -> object:
        """Construct a boto3 S3 client from the step config.

        For the ``s3`` source the standard AWS environment variables are used.
        For ``lakefs`` the LakeFS S3-compatible endpoint is configured via
        ``LAKEFS_ENDPOINT``, ``LAKEFS_ACCESS_KEY``, and ``LAKEFS_SECRET_KEY``.

        The correct client is selected at *run time* based on the ``--source``
        CLI flag; this factory always builds a client that covers both cases by
        preferring LakeFS credentials when they are present.
        """
        boto_cfg = BotoConfig(
            retries={"max_attempts": self._config.max_retries, "mode": "adaptive"},
            connect_timeout=self._config.timeout,
            read_timeout=self._config.timeout,
        )

        # LakeFS takes precedence when its endpoint is configured
        if self._config.lakefs_endpoint:
            self._logger.debug(
                "Configuring boto3 for LakeFS at %s", self._config.lakefs_endpoint
            )
            return boto3.client(
                "s3",
                endpoint_url=self._config.lakefs_endpoint,
                aws_access_key_id=self._config.lakefs_access_key,
                aws_secret_access_key=self._config.lakefs_secret_key,
                config=boto_cfg,
            )

        # Plain S3 (AWS or custom endpoint such as MinIO)
        return boto3.client(
            "s3",
            endpoint_url=self._config.s3_endpoint_url,
            aws_access_key_id=self._config.aws_access_key_id,
            aws_secret_access_key=self._config.aws_secret_access_key,
            region_name=self._config.aws_default_region,
            config=boto_cfg,
        )

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def run(
        self,
        version: str,
        source: str,
        output_dir: str,
        lakefs_repo: Optional[str] = None,
        lakefs_branch: Optional[str] = None,
        path_override: Optional[str] = None,
        labels_only: bool = False,
        manifest_only: bool = False,
        sample_size: Optional[int] = None,
        seed: int = 42,
    ) -> None:
        """Execute the dataset loading step end-to-end.

        Parameters
        ----------
        version:
            Dataset version tag (e.g. ``"v1"``).
        source:
            Storage backend: ``"s3"`` or ``"lakefs"``.
        output_dir:
            Local directory where the YOLO dataset tree will be written.
        lakefs_repo:
            LakeFS repository name (used when source is ``"lakefs"``).
        lakefs_branch:
            LakeFS branch name (used when source is ``"lakefs"``).
        path_override:
            Optional full S3 URI that overrides the default location.
        sample_size:
            If provided, keep only this many image+label pairs per split.
        seed:
            Random seed used for reproducible sampling.
        """
        self._logger.info(
            "Starting %s | version=%s source=%s", self._config.app_name, version, source
        )

        params = YoloDatasetParams(
            version=version,
            source=source,
            output_dir=output_dir,
            lakefs_repo=lakefs_repo,
            lakefs_branch=lakefs_branch,
            path_override=path_override or None,
            labels_only=labels_only,
            manifest_only=manifest_only,
            sample_size=sample_size,
            seed=seed,
        )

        try:
            stats = self._service.run(params=params)
        except Exception as e:
            self._logger.error("Dataset loading failed: %s", e)
            raise

        self._logger.info(
            "Dataset loading finished | train=%d val=%d test=%d",
            stats.train_images,
            stats.val_images,
            stats.test_images,
        )
