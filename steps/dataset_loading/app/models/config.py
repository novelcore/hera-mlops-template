"""Application configuration for the dataset loading step.

All values are read from environment variables (or a .env file).
See env.example for the full list of supported variables.
"""

from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    """Step-level runtime settings loaded from environment variables."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # -------------------------------------------------------------------------
    # Application settings
    # -------------------------------------------------------------------------
    app_name: str = Field(default="io-dataset-loading", description="Application name.")
    log_level: str = Field(
        default="INFO", description="Logging level (DEBUG/INFO/WARNING/ERROR)."
    )
    max_retries: int = Field(
        default=3, description="Maximum number of S3/LakeFS request retries."
    )
    timeout: int = Field(default=120, description="Request timeout in seconds.")

    # -------------------------------------------------------------------------
    # AWS / S3 connection settings
    # -------------------------------------------------------------------------
    aws_default_region: Optional[str] = Field(
        default=None,
        description="AWS region (e.g. 'eu-central-1'). Omit to use the instance role region.",
    )
    aws_access_key_id: Optional[str] = Field(
        default=None,
        description="AWS access key ID. Omit to use IAM role credentials.",
    )
    aws_secret_access_key: Optional[str] = Field(
        default=None,
        description="AWS secret access key. Omit to use IAM role credentials.",
    )
    s3_endpoint_url: Optional[str] = Field(
        default=None,
        description="Custom S3 endpoint URL (e.g. for MinIO). Leave unset for AWS.",
    )

    # -------------------------------------------------------------------------
    # LakeFS connection settings (S3-compatible API)
    # -------------------------------------------------------------------------
    lakefs_endpoint: Optional[str] = Field(
        default=None,
        description="LakeFS server base URL (e.g. 'https://lakefs.example.com').",
    )
    lakefs_access_key: Optional[str] = Field(
        default=None,
        description="LakeFS access key ID.",
    )
    lakefs_secret_key: Optional[str] = Field(
        default=None,
        description="LakeFS secret access key.",
    )
