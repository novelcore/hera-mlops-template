"""Config-validation service: pre-GPU liveness checks over the resolved config.

This is the pipeline's first *developer* step. Its job is to fail fast — before
any expensive GPU step starts — if the run's external dependencies are missing.

What it does NOT do: re-validate the config schema. In this pipeline the config
tree is composed and TYPE/ENUM-validated upstream by compose-and-validate (the
Hydra structured-config layer). Re-implementing a parallel schema here would
just be a second source of truth to drift. So this step reads the already-
resolved params.yaml and checks that the world it references actually exists:

  - dataset path      : the dataset prefix exists and is non-empty in lakeFS/S3;
  - MLflow tracking   : the tracking server answers /health;
  - pretrained weights: an s3:// weights override actually exists;
  - checkpoint resume : when the config resumes, the checkpoint is present.

Endpoints + credentials come from the ENV the platform injects into this step
(LAKEFS_ENDPOINT / LAKEFS_ACCESS_KEY / LAKEFS_SECRET_KEY, or the AWS_* chain for
source=s3, and MLFLOW_TRACKING_URI). The developer wires none of that. A check
whose endpoint/credentials were not injected is SKIPPED with a clear log (local
render, or a step that opted out) rather than failing — the checks are a fail-
fast convenience, and compose-time schema validation is the hard contract.
"""

import logging
import os
from urllib.parse import urlparse

# Same canonical S3 layout the dataset-loading service uses when no explicit
# path is given — kept in sync so the check probes exactly where the loader
# will read.
_DEFAULT_BUCKET = "io-audio-text-data"


class ConfigValidationError(Exception):
    """A required external dependency is missing or unreachable."""


class ConfigValidationService:
    def __init__(self, timeout: int = 15, max_retries: int = 3) -> None:
        self._timeout = timeout
        self._max_retries = max_retries
        self._log = logging.getLogger(__name__)

    # ------------------------------------------------------------------ public

    def run(self, resolved: dict) -> None:
        """Run all liveness checks over the resolved params.yaml dict.

        Raises ConfigValidationError on the first missing/unreachable
        dependency. Network checks self-skip when their endpoint/credentials
        were not injected, so this is always safe to call.
        """
        data = resolved.get("data") or {}
        model = resolved.get("model") or {}
        platform = resolved.get("platform") or {}

        self._check_mlflow()

        s3 = self._s3_client(str(data.get("source") or "lakefs"))
        if s3 is None:
            self._log.info(
                "object-store checks skipped (no injected credentials) — "
                "schema validation already ran at compose time"
            )
            return

        self._check_dataset_path(s3, data, platform)
        self._check_pretrained_weights(s3, model)
        self._check_checkpoint_resume(s3, data, platform)
        self._log.info("config-validation: all liveness checks passed")

    # ------------------------------------------------------------------ s3/lakefs

    def _s3_client(self, source: str):
        """Build a boto3 S3 client from the platform-injected env, or return
        None when boto3 / credentials are unavailable (→ caller skips)."""
        try:
            import boto3
            from botocore.config import Config as BotoConfig
        except ImportError:
            self._log.info("boto3 not installed — skipping object-store checks")
            return None

        cfg = BotoConfig(
            retries={"max_attempts": self._max_retries, "mode": "adaptive"},
            connect_timeout=self._timeout, read_timeout=self._timeout,
        )

        if source == "lakefs":
            endpoint = os.environ.get("LAKEFS_ENDPOINT")
            if not endpoint:
                self._log.info("LAKEFS_ENDPOINT not injected — skipping lakeFS checks")
                return None
            return boto3.client(
                "s3", endpoint_url=endpoint,
                aws_access_key_id=os.environ.get("LAKEFS_ACCESS_KEY"),
                aws_secret_access_key=os.environ.get("LAKEFS_SECRET_KEY"),
                config=cfg,
            )

        # source == s3 (AWS or a custom endpoint such as MinIO)
        if not (os.environ.get("AWS_ACCESS_KEY_ID") or os.environ.get("S3_ENDPOINT_URL")):
            self._log.info("no AWS/S3 credentials injected — skipping S3 checks")
            return None
        return boto3.client(
            "s3", endpoint_url=os.environ.get("S3_ENDPOINT_URL"),
            aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
            region_name=os.environ.get("AWS_DEFAULT_REGION"),
            config=cfg,
        )

    def _dataset_prefix(self, data: dict, platform: dict) -> str:
        """The s3:// URI the dataset-loading step will read (mirrors that
        step's _resolve_source), so the check probes exactly what it uses."""
        override = str(data.get("path_override") or "").strip()
        if override:
            return override

        source = str(data.get("source") or "lakefs")
        ref = str(data.get("ref") or "main")
        version = str(data.get("version") or "") or ref

        if source == "lakefs":
            repo = str((platform.get("lakefs") or {}).get("repository") or "")
            return f"s3://{repo}/{ref}/dataset/{version}/"
        return f"s3://{_DEFAULT_BUCKET}/upload-initial/dataset/{version}/"

    def _check_dataset_path(self, s3, data: dict, platform: dict) -> None:
        uri = self._dataset_prefix(data, platform)
        parsed = urlparse(uri)
        bucket, prefix = parsed.netloc, parsed.path.lstrip("/")
        self._log.info("checking dataset path %s", uri)
        try:
            resp = s3.list_objects_v2(Bucket=bucket, Prefix=prefix, MaxKeys=1)
        except Exception as exc:  # NoSuchBucket etc. — a repo/branch typo
            raise ConfigValidationError(
                f"dataset path unreachable: {uri} — the repository/branch may not "
                f"exist ({exc})"
            ) from exc
        if resp.get("KeyCount", 0) == 0:
            raise ConfigValidationError(
                f"dataset path not found or empty: {uri} — check the ref/version "
                f"and that the dataset was uploaded."
            )
        # A non-empty prefix isn't enough: require data.yaml at the root so a run
        # doesn't pass config-validation and then fail in dataset-loading on a
        # missing/misplaced data.yaml. Cheap (one HEAD), and closes the biggest
        # false-confidence gap between the two steps.
        try:
            s3.head_object(Bucket=bucket, Key=prefix + "data.yaml")
        except Exception as exc:
            raise ConfigValidationError(
                f"dataset at {uri} has no data.yaml at its root — a valid "
                f"Ultralytics YOLO dataset must include data.yaml ({exc})"
            ) from exc
        self._log.info("dataset path OK (%s, data.yaml present)", uri)

    def _check_pretrained_weights(self, s3, model: dict) -> None:
        weights = model.get("pretrained_weights")
        if not weights:
            return  # standard Ultralytics names ship in the training image
        if not str(weights).startswith("s3://"):
            raise ConfigValidationError(
                f"model.pretrained_weights must be null or an s3:// path, got: {weights!r}"
            )
        parsed = urlparse(weights)
        self._log.info("checking pretrained weights %s", weights)
        try:
            s3.head_object(Bucket=parsed.netloc, Key=parsed.path.lstrip("/"))
        except Exception as exc:  # botocore ClientError 404 etc.
            raise ConfigValidationError(
                f"pretrained weights not found: {weights} ({exc})"
            ) from exc
        self._log.info("pretrained weights OK")

    def _check_checkpoint_resume(self, s3, data: dict, platform: dict) -> None:
        """When the config resumes training, prove the checkpoint exists.

        The checkpoint bucket/prefix are platform facts
        (platform.checkpoints, injected from the project lakeFS repo).
        """
        resume_from = str(data.get("resume_from") or "").strip()
        if not resume_from:
            return

        checkpoints = platform.get("checkpoints") or {}
        if resume_from == "auto":
            bucket = str(checkpoints.get("bucket") or "")
            prefix = str(checkpoints.get("prefix") or "").rstrip("/") + "/"
            if not bucket:
                raise ConfigValidationError(
                    "resume_from='auto' but platform.checkpoints.bucket is unset"
                )
            self._log.info("scanning for checkpoints under s3://%s/%s", bucket, prefix)
            resp = s3.list_objects_v2(Bucket=bucket, Prefix=prefix, MaxKeys=100)
            pt = [o for o in resp.get("Contents", []) if o["Key"].endswith(".pt")]
            if not pt:
                raise ConfigValidationError(
                    f"resume_from='auto' but no .pt checkpoint under s3://{bucket}/{prefix}"
                )
            self._log.info("found %d checkpoint(s) for auto-resume", len(pt))
        else:
            parsed = urlparse(resume_from)
            self._log.info("checking checkpoint %s", resume_from)
            try:
                s3.head_object(Bucket=parsed.netloc, Key=parsed.path.lstrip("/"))
            except Exception as exc:
                raise ConfigValidationError(
                    f"checkpoint not found: {resume_from} ({exc})"
                ) from exc
            self._log.info("checkpoint OK")

    # ------------------------------------------------------------------ mlflow

    def _check_mlflow(self) -> None:
        # Gate on the INJECTED env, not any params fallback: the enhancer sets
        # MLFLOW_TRACKING_URI in-cluster, so its presence signals a real
        # environment with a reachable server. Locally the env is absent, so we
        # skip — never probe a placeholder endpoint.
        uri = os.environ.get("MLFLOW_TRACKING_URI")
        if not uri:
            self._log.info("MLFLOW_TRACKING_URI not injected — skipping MLflow check")
            return
        try:
            import requests
        except ImportError:
            self._log.info("requests not installed — skipping MLflow check")
            return
        health = uri.rstrip("/") + "/health"
        self._log.info("checking MLflow %s", health)
        try:
            resp = requests.get(health, timeout=self._timeout)
        except requests.RequestException as exc:
            raise ConfigValidationError(
                f"MLflow unreachable at {health}: {type(exc).__name__}"
            ) from exc
        if not resp.ok:
            raise ConfigValidationError(
                f"MLflow health check failed at {health}: HTTP {resp.status_code}"
            )
        self._log.info("MLflow OK")
