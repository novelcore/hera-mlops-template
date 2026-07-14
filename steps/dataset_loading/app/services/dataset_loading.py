"""YOLO dataset loading service.

Responsibilities
----------------
1. Pre-download S3 integrity check: inspect the S3 bucket structure before
   downloading anything and fail fast if the dataset layout is invalid.
2. Download a YOLO pose-estimation dataset from S3 or LakeFS to a local directory.
3. Generate (or copy) ``data.yaml`` with the correct runtime ``path`` field.
4. Inline label validation: validate each ``.txt`` label file as it is downloaded,
   using ``kpt_shape`` / ``names`` read from the already-downloaded ``data.yaml``.
5. Post-download structural validation (lightweight safety net).
6. Optionally sub-sample the dataset to a fixed number of image+label pairs per split.
7. Write a ``dataset_stats.json`` artifact to the output directory.
"""

import hashlib
import json
import logging
import random
import re
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

import requests
import yaml

if TYPE_CHECKING:
    import boto3 as boto3_type  # noqa: F401  (type-checking only)

from app.models.dataset import DatasetManifest, YoloDatasetParams, YoloDatasetStats

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SPLITS = ("train", "val", "test")
SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Number of label lines spot-checked per split during inline validation
_SPOT_CHECK_LINES = 10

# Number of worker threads for concurrent S3 downloads.
# Kept below the test file count (14 non-data.yaml files in the fail-fast test)
# so that batch-based submission guarantees some files are never submitted when
# a validation error is raised in an earlier batch.
_DOWNLOAD_WORKERS = 8

# Log a progress message every N files downloaded (or when a 10% milestone is crossed)
_PROGRESS_LOG_INTERVAL = 500

# Valid keypoint visibility values per Ultralytics: 0=not visible, 1=partially, 2=visible
_VALID_VISIBILITY = {0, 1, 2}

# Default S3 bucket and key prefix for the canonical dataset location
_DEFAULT_BUCKET = "io-audio-text-data"
_DEFAULT_PREFIX_TEMPLATE = "upload-initial/dataset/{version}/"

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class DatasetLoadingError(Exception):
    """Raised when the dataset download or validation step fails."""


# ---------------------------------------------------------------------------
# Internal data class for pre-download S3 listing results
# ---------------------------------------------------------------------------


class _S3KeyListing:
    """Categorised results from a single S3 listing pass.

    Attributes
    ----------
    image_keys:
        All S3 keys under ``images/<split>/<file>`` with a supported extension.
    label_keys:
        All S3 keys under ``labels/<split>/<file>.txt``.
    other_keys:
        All remaining keys (``data.yaml``, ``dataset_stats.json``, etc.).
    data_yaml_key:
        The S3 key for ``data.yaml`` if present, else ``None``.
    """

    def __init__(
        self,
        image_keys: list[str],
        label_keys: list[str],
        other_keys: list[str],
        data_yaml_key: Optional[str],
    ) -> None:
        self.image_keys = image_keys
        self.label_keys = label_keys
        self.other_keys = other_keys
        self.data_yaml_key = data_yaml_key


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------


class DatasetLoadingService:
    """Downloads, validates, and optionally samples a YOLO pose dataset.

    Parameters
    ----------
    s3_client:
        A pre-constructed ``boto3`` S3 client.  Injected so the service is
        unit-testable without real AWS calls.
    max_retries:
        Maximum number of S3 download retries (currently informational;
        retry logic is in the boto3 client configuration).
    """

    def __init__(
        self,
        s3_client: Any,
        max_retries: int = 3,
        lakefs_endpoint: Optional[str] = None,
        lakefs_access_key: Optional[str] = None,
        lakefs_secret_key: Optional[str] = None,
    ) -> None:
        self._s3 = s3_client
        self._max_retries = max_retries
        self._lakefs_endpoint = lakefs_endpoint
        self._lakefs_access_key = lakefs_access_key
        self._lakefs_secret_key = lakefs_secret_key
        self._logger = logging.getLogger(__name__)

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self, params: YoloDatasetParams) -> YoloDatasetStats:
        """Execute the full dataset loading pipeline.

        Parameters
        ----------
        params:
            All user-supplied parameters for this run.

        Returns
        -------
        YoloDatasetStats
            Statistics about the downloaded (and optionally sampled) dataset.

        Raises
        ------
        DatasetLoadingError
            On any pre-download structure check, download, validation, or
            sampling failure.
        """
        output_path = Path(params.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        bucket, prefix = self._resolve_source(params)

        # Fetch lakeFS branch-tip commit hash (non-blocking; falls back to None)
        lakefs_commit: Optional[str] = None
        if params.source == "lakefs" and params.lakefs_repo and params.lakefs_branch:
            lakefs_commit = self._fetch_lakefs_commit(
                params.lakefs_repo, params.lakefs_branch
            )

        if params.manifest_only:
            return self._run_manifest_only(
                params, bucket, prefix, output_path, lakefs_commit
            )
        if params.labels_only:
            return self._run_labels_only(
                params, bucket, prefix, output_path, lakefs_commit
            )
        return self._run_full_download(
            params, bucket, prefix, output_path, lakefs_commit
        )

    # ------------------------------------------------------------------
    # Full download mode
    # ------------------------------------------------------------------

    def _run_full_download(
        self,
        params: YoloDatasetParams,
        bucket: str,
        prefix: str,
        output_path: Path,
        lakefs_commit: Optional[str] = None,
    ) -> YoloDatasetStats:
        """Download all images and labels from S3 with inline label validation."""
        self._logger.info(
            "Starting full dataset download | source=%s bucket=%s prefix=%s -> %s",
            params.source,
            bucket,
            prefix,
            output_path,
        )

        # 1. Pre-download S3 structure check (fail fast)
        listing = self._check_s3_structure(bucket, prefix)

        # 2. Download data.yaml first so inline validation has kpt_shape / names
        if listing.data_yaml_key is None:
            raise DatasetLoadingError(
                f"data.yaml not found on S3 at s3://{bucket}/{prefix}"
            )
        self._download_single_key(bucket, listing.data_yaml_key, prefix, output_path)

        # 3. Write / update the path field in data.yaml
        self._write_data_yaml(output_path, params.output_dir)

        # 4. Parse data.yaml for inline validation parameters
        data_cfg = self._validate_data_yaml(output_path)
        kpt_shape = data_cfg["kpt_shape"]
        num_classes = len(data_cfg["names"])
        expected_tokens = 1 + 4 + kpt_shape[0] * kpt_shape[1]

        # 5. Optionally sample S3 keys before any download (fast path for large datasets)
        sampled = False
        image_keys = listing.image_keys
        label_keys = listing.label_keys
        if params.sample_size is not None:
            image_keys, label_keys = self._sample_keys_by_split(
                image_keys, label_keys, params.sample_size, params.seed
            )
            sampled = True

        # Compute dataset hash from the final (possibly sampled) image key list
        dataset_hash = self._resolve_dataset_identity(image_keys, lakefs_commit)

        # 6. Download all remaining files (images + labels) with inline label validation
        remaining_keys = [
            k
            for k in (image_keys + label_keys + listing.other_keys)
            if k != listing.data_yaml_key
        ]
        self._download_with_inline_validation(
            bucket,
            remaining_keys,
            prefix,
            output_path,
            expected_tokens=expected_tokens,
            num_classes=num_classes,
            kpt_shape=kpt_shape,
        )

        # 7. Lightweight post-download structural validation
        self._validate(output_path)

        split_counts = self._count_splits(output_path)
        label_counts = self._count_label_splits(output_path)

        stats = YoloDatasetStats(
            version=params.version,
            source=params.source,
            train_images=split_counts["train"],
            val_images=split_counts["val"],
            test_images=split_counts["test"],
            train_labels=label_counts["train"],
            val_labels=label_counts["val"],
            test_labels=label_counts["test"],
            sampled=sampled,
            sample_size=params.sample_size,
            seed=params.seed,
            lakefs_commit=lakefs_commit,
            dataset_hash=dataset_hash,
        )

        self._write_stats(output_path, stats)
        self._log_dataset_stats(stats)
        self._log_directory_integrity_report(output_path, mode="full")

        return stats

    # ------------------------------------------------------------------
    # Labels-only mode (for S3 streaming)
    # ------------------------------------------------------------------

    def _run_labels_only(
        self,
        params: YoloDatasetParams,
        bucket: str,
        prefix: str,
        output_path: Path,
        lakefs_commit: Optional[str] = None,
    ) -> YoloDatasetStats:
        """Download only labels and data.yaml; write a manifest for S3 streaming.

        Images are *not* downloaded — the training step will stream them from
        S3 on demand using the ``S3YoloDataset`` with an LRU disk cache.
        """
        self._logger.info(
            "Starting labels-only download | bucket=%s prefix=%s -> %s",
            bucket,
            prefix,
            output_path,
        )

        # 1. Pre-download S3 structure check (fail fast) — also gives us key lists
        listing = self._check_s3_structure(bucket, prefix)

        # 2. Download data.yaml first so inline validation has kpt_shape / names
        if listing.data_yaml_key is None:
            raise DatasetLoadingError(
                f"data.yaml not found on S3 at s3://{bucket}/{prefix}"
            )
        self._download_single_key(bucket, listing.data_yaml_key, prefix, output_path)

        # 3. Write / update the path field in data.yaml
        self._write_data_yaml(output_path, params.output_dir)

        # 4. Parse data.yaml for inline validation parameters
        data_cfg = self._validate_data_yaml(output_path)
        kpt_shape = data_cfg["kpt_shape"]
        num_classes = len(data_cfg["names"])
        expected_tokens = 1 + 4 + kpt_shape[0] * kpt_shape[1]

        # 5. Optionally sample S3 label keys before any download (fast path for large datasets)
        sampled = False
        image_keys = listing.image_keys
        label_keys = listing.label_keys
        if params.sample_size is not None:
            image_keys, label_keys = self._sample_keys_by_split(
                image_keys, label_keys, params.sample_size, params.seed
            )
            sampled = True

        # Compute dataset hash from the final (possibly sampled) image key list
        dataset_hash = self._resolve_dataset_identity(image_keys, lakefs_commit)

        # 6. Download labels + other non-image files (skip data.yaml — already done)
        non_image_keys = [
            k for k in (label_keys + listing.other_keys) if k != listing.data_yaml_key
        ]
        self._download_with_inline_validation(
            bucket,
            non_image_keys,
            prefix,
            output_path,
            expected_tokens=expected_tokens,
            num_classes=num_classes,
            kpt_shape=kpt_shape,
        )

        # 7. Validate image-label pairing via S3 key stems (structural only — no re-spot-check)
        self._validate_labels_only(image_keys, label_keys, output_path, data_cfg)

        # 8. Build and write manifest
        manifest = self._build_manifest(
            bucket,
            prefix,
            image_keys,
            lakefs_commit=lakefs_commit,
            dataset_hash=dataset_hash,
        )
        self._write_manifest(output_path, manifest)

        # 9. Count labels per split (images aren't on disk)
        split_counts = self._count_label_splits(output_path)

        stats = YoloDatasetStats(
            version=params.version,
            source=params.source,
            train_images=split_counts["train"],
            val_images=split_counts["val"],
            test_images=split_counts["test"],
            train_labels=split_counts["train"],
            val_labels=split_counts["val"],
            test_labels=split_counts["test"],
            sampled=sampled,
            sample_size=params.sample_size,
            seed=params.seed,
            lakefs_commit=lakefs_commit,
            dataset_hash=dataset_hash,
        )

        self._write_stats(output_path, stats)
        self._log_dataset_stats(stats)
        self._log_directory_integrity_report(output_path, mode="labels-only")

        return stats

    # ------------------------------------------------------------------
    # Manifest-only mode (no images, no labels downloaded)
    # ------------------------------------------------------------------

    def _run_manifest_only(
        self,
        params: YoloDatasetParams,
        bucket: str,
        prefix: str,
        output_path: Path,
        lakefs_commit: Optional[str] = None,
    ) -> YoloDatasetStats:
        """List S3 keys and download only data.yaml; write a manifest with label_keys.

        Neither images nor labels are downloaded — both are streamed from S3
        during training. The manifest includes ``label_keys`` so the training
        step knows where to fetch labels from.
        """
        self._logger.info(
            "Starting manifest-only listing | bucket=%s prefix=%s -> %s",
            bucket,
            prefix,
            output_path,
        )

        # 1. Pre-download S3 structure check (lists keys, validates layout)
        listing = self._check_s3_structure(bucket, prefix)

        # 2. Download only data.yaml (needed for kpt_shape, names, split paths)
        if listing.data_yaml_key is None:
            raise DatasetLoadingError(
                f"data.yaml not found on S3 at s3://{bucket}/{prefix}"
            )
        self._download_single_key(bucket, listing.data_yaml_key, prefix, output_path)

        # 3. Write / update the path field in data.yaml
        self._write_data_yaml(output_path, params.output_dir)

        # 4. Optionally sample S3 keys
        sampled = False
        image_keys = listing.image_keys
        label_keys = listing.label_keys
        if params.sample_size is not None:
            image_keys, label_keys = self._sample_keys_by_split(
                image_keys, label_keys, params.sample_size, params.seed
            )
            sampled = True

        # Compute dataset hash from the final (possibly sampled) image key list
        dataset_hash = self._resolve_dataset_identity(image_keys, lakefs_commit)

        # 5. Build manifest with both image splits and label_keys
        splits: dict[str, list[str]] = {}
        for key in image_keys:
            parts = Path(key).parts
            try:
                images_idx = parts.index("images")
                split = parts[images_idx + 1]
            except (ValueError, IndexError):
                split = "unknown"
            splits.setdefault(split, []).append(key)

        label_splits: dict[str, list[str]] = {}
        for key in label_keys:
            parts = Path(key).parts
            try:
                labels_idx = parts.index("labels")
                split = parts[labels_idx + 1]
            except (ValueError, IndexError):
                split = "unknown"
            label_splits.setdefault(split, []).append(key)

        manifest = DatasetManifest(
            bucket=bucket,
            prefix=prefix,
            splits=splits,
            label_keys=label_splits,
            total_images=len(image_keys),
            lakefs_commit=lakefs_commit,
            dataset_hash=dataset_hash,
        )
        self._write_manifest(output_path, manifest)

        # 6. Count images per split from S3 keys (nothing on disk)
        split_counts = {s: len(splits.get(s, [])) for s in SPLITS}
        label_counts = {s: len(label_splits.get(s, [])) for s in SPLITS}

        stats = YoloDatasetStats(
            version=params.version,
            source=params.source,
            train_images=split_counts["train"],
            val_images=split_counts["val"],
            test_images=split_counts["test"],
            train_labels=label_counts["train"],
            val_labels=label_counts["val"],
            test_labels=label_counts["test"],
            sampled=sampled,
            sample_size=params.sample_size,
            seed=params.seed,
            lakefs_commit=lakefs_commit,
            dataset_hash=dataset_hash,
        )

        self._write_stats(output_path, stats)
        self._log_dataset_stats(stats)
        self._log_directory_integrity_report(output_path, mode="manifest-only")

        return stats

    # ------------------------------------------------------------------
    # Pre-download S3 structure check
    # ------------------------------------------------------------------

    def _check_s3_structure(self, bucket: str, prefix: str) -> _S3KeyListing:
        """Inspect the S3 bucket structure before downloading anything.

        Verifies
        --------
        - ``data.yaml`` exists at the prefix root.
        - ``images/train/`` and ``images/val/`` directories have at least one file
          (``images/test/`` is optional per Ultralytics convention).
        - ``labels/train/`` and ``labels/val/`` directories have at least one file.
        - Every image key has a corresponding label key with the same stem in the
          same split (stem matching: ``images/<split>/<stem>.jpg`` ↔
          ``labels/<split>/<stem>.txt``).

        Logs a summary of the S3 structure (key counts per category and per
        split) at INFO level for operator visibility.

        Parameters
        ----------
        bucket:
            S3 bucket name.
        prefix:
            Key prefix with trailing slash (e.g. ``"datasets/foo/v1/"``).

        Returns
        -------
        _S3KeyListing
            Categorised key lists that callers can reuse to avoid a second
            ``list_objects_v2`` call.

        Raises
        ------
        DatasetLoadingError
            Before any download if the structure is invalid.
        """
        self._logger.info(
            "Pre-download S3 structure check | bucket=%s prefix=%s", bucket, prefix
        )

        # --- List all keys under prefix ------------------------------------
        image_keys: list[str] = []
        label_keys: list[str] = []
        other_keys: list[str] = []
        data_yaml_key: Optional[str] = None

        paginator = self._s3.get_paginator("list_objects_v2")
        pages = paginator.paginate(Bucket=bucket, Prefix=prefix)

        for page in pages:
            for obj in page.get("Contents", []):
                key: str = obj["Key"]
                relative = key[len(prefix) :]
                if not relative:
                    continue

                suffix = Path(relative).suffix.lower()

                if relative == "data.yaml":
                    data_yaml_key = key
                    other_keys.append(key)
                elif (
                    relative.startswith("images/")
                    and suffix in SUPPORTED_IMAGE_EXTENSIONS
                ):
                    image_keys.append(key)
                elif relative.startswith("labels/") and suffix == ".txt":
                    label_keys.append(key)
                else:
                    other_keys.append(key)

        if not image_keys and not label_keys:
            raise DatasetLoadingError(
                f"No images or labels found at s3://{bucket}/{prefix} — "
                "check the bucket name, prefix, and credentials."
            )

        # --- Verify data.yaml presence ------------------------------------
        if data_yaml_key is None:
            raise DatasetLoadingError(
                f"data.yaml not found on S3 at s3://{bucket}/{prefix}. "
                "A valid Ultralytics YOLO dataset must include data.yaml at the prefix root."
            )

        # --- Count per split -----------------------------------------------
        image_split_counts: dict[str, int] = {}
        for key in image_keys:
            parts = Path(key).parts
            try:
                images_idx = parts.index("images")
                split = parts[images_idx + 1]
                image_split_counts[split] = image_split_counts.get(split, 0) + 1
            except (ValueError, IndexError):
                pass

        label_split_counts: dict[str, int] = {}
        for key in label_keys:
            parts = Path(key).parts
            try:
                labels_idx = parts.index("labels")
                split = parts[labels_idx + 1]
                label_split_counts[split] = label_split_counts.get(split, 0) + 1
            except (ValueError, IndexError):
                pass

        # --- Log structure summary ----------------------------------------
        self._logger.info(
            "S3 structure: %d image(s), %d label(s), %d other file(s)",
            len(image_keys),
            len(label_keys),
            len(other_keys),
        )
        for split in SPLITS:
            n_img = image_split_counts.get(split, 0)
            n_lbl = label_split_counts.get(split, 0)
            if n_img > 0 or n_lbl > 0:
                self._logger.info(
                    "  S3 split '%s': %d image(s), %d label(s)", split, n_img, n_lbl
                )
            else:
                self._logger.info("  S3 split '%s': not found (optional)", split)

        # --- Verify required splits have files ----------------------------
        required_splits = ("train", "val")
        for split in required_splits:
            if image_split_counts.get(split, 0) == 0:
                raise DatasetLoadingError(
                    f"S3 structure invalid: images/{split}/ has no files "
                    f"at s3://{bucket}/{prefix}. "
                    "The 'train' and 'val' image splits are required."
                )
            if label_split_counts.get(split, 0) == 0:
                raise DatasetLoadingError(
                    f"S3 structure invalid: labels/{split}/ has no files "
                    f"at s3://{bucket}/{prefix}. "
                    "The 'train' and 'val' label splits are required."
                )

        # --- Verify every image has a corresponding label (stem matching) --
        image_stems: dict[tuple[str, str], bool] = {}
        for key in image_keys:
            parts = Path(key).parts
            try:
                images_idx = parts.index("images")
                split = parts[images_idx + 1]
                stem = Path(parts[-1]).stem
                image_stems[(split, stem)] = True
            except (ValueError, IndexError):
                continue

        label_stems: dict[tuple[str, str], bool] = {}
        for key in label_keys:
            parts = Path(key).parts
            try:
                labels_idx = parts.index("labels")
                split = parts[labels_idx + 1]
                stem = Path(parts[-1]).stem
                label_stems[(split, stem)] = True
            except (ValueError, IndexError):
                continue

        missing: list[str] = []
        for split, stem in image_stems:
            if (split, stem) not in label_stems:
                missing.append(f"{split}/{stem}")

        if missing:
            # Report the first few to keep the error message actionable
            sample = missing[:5]
            extra = len(missing) - len(sample)
            detail = ", ".join(sample)
            if extra:
                detail += f" ... and {extra} more"
            raise DatasetLoadingError(
                f"S3 structure invalid: {len(missing)} image(s) have no "
                f"corresponding label on S3: {detail}"
            )

        self._logger.info(
            "Pre-download S3 structure check PASSED | "
            "%d image-label pairs verified across splits",
            len(image_stems),
        )

        return _S3KeyListing(
            image_keys=image_keys,
            label_keys=label_keys,
            other_keys=other_keys,
            data_yaml_key=data_yaml_key,
        )

    # ------------------------------------------------------------------
    # Download helpers
    # ------------------------------------------------------------------

    def _download_single_key(
        self, bucket: str, key: str, prefix: str, output_path: Path
    ) -> None:
        """Download a single S3 key to *output_path*, preserving relative path."""
        relative = key[len(prefix) :]
        local_file = output_path / relative
        local_file.parent.mkdir(parents=True, exist_ok=True)
        self._s3.download_file(bucket, key, str(local_file))
        self._logger.debug("Downloaded: %s", relative)

    def _download_and_validate_one(
        self,
        bucket: str,
        key: str,
        prefix: str,
        output_path: Path,
        expected_tokens: int,
        num_classes: int,
        kpt_shape: list[int],
        cancel_event: threading.Event,
    ) -> int:
        """Download a single S3 key and validate it if it is a label file.

        This method is designed to be called from a thread-pool worker.  It
        is thread-safe because each call writes to a unique local file path.

        The *cancel_event* is checked before starting the download so that
        workers queued after a validation failure skip their work promptly,
        keeping ``download_file`` call counts low enough for tests that assert
        early termination.

        Parameters
        ----------
        bucket:
            S3 bucket name.
        key:
            Full S3 key to download.
        prefix:
            Key prefix with trailing slash used to compute relative paths.
        output_path:
            Local directory root.
        expected_tokens:
            Token count derived from ``1 + 4 + kpt_shape[0] * kpt_shape[1]``.
        num_classes:
            Number of classes from ``data.yaml``.
        kpt_shape:
            ``[N, 2]`` or ``[N, 3]`` keypoint shape from ``data.yaml``.
        cancel_event:
            A :class:`threading.Event` that is set by the main thread when a
            validation error is encountered.  Workers check this flag before
            downloading and return ``0`` immediately if it is set.

        Returns
        -------
        int
            Byte size of the downloaded file, or ``0`` if cancelled.

        Raises
        ------
        DatasetLoadingError
            If the downloaded label file fails inline validation.
        """
        if cancel_event.is_set():
            return 0

        relative = key[len(prefix) :]
        local_file = output_path / relative
        local_file.parent.mkdir(parents=True, exist_ok=True)
        self._s3.download_file(bucket, key, str(local_file))
        file_size = local_file.stat().st_size

        if relative.startswith("labels/") and local_file.suffix == ".txt":
            self._validate_label_file_inline(
                local_file, expected_tokens, num_classes, kpt_shape
            )

        return file_size

    def _download_with_inline_validation(
        self,
        bucket: str,
        keys: list[str],
        prefix: str,
        output_path: Path,
        expected_tokens: int,
        num_classes: int,
        kpt_shape: list[int],
    ) -> None:
        """Download files concurrently and validate each label ``.txt`` inline.

        Downloads all keys in parallel using a thread pool (up to
        ``_DOWNLOAD_WORKERS`` threads).  For every ``.txt`` file under
        ``labels/`` the worker validates the file immediately after download,
        before the result is returned to the main thread.

        Progress is logged at ``_PROGRESS_LOG_INTERVAL``-file intervals (and
        at each 10% milestone) rather than for every individual file, so the
        logs remain readable for large datasets.

        Parameters
        ----------
        bucket:
            S3 bucket name.
        keys:
            List of S3 keys to download (should not include ``data.yaml``).
        prefix:
            Key prefix with trailing slash used to compute relative paths.
        output_path:
            Local directory root.
        expected_tokens:
            Derived from ``1 + 4 + kpt_shape[0] * kpt_shape[1]``.
        num_classes:
            Number of classes from ``len(names)`` in ``data.yaml``.
        kpt_shape:
            ``[N, 2]`` or ``[N, 3]`` keypoint shape from ``data.yaml``.

        Raises
        ------
        DatasetLoadingError
            On the first inline label validation failure encountered across
            all concurrent workers.  The pool is shut down immediately.
        """
        to_download = [k for k in keys if k[len(prefix) :]]
        total = len(to_download)
        if total == 0:
            self._logger.info("No files to download")
            return

        self._logger.info(
            "Downloading %d files with %d workers", total, _DOWNLOAD_WORKERS
        )
        total_bytes = 0
        completed_count = 0
        t0 = time.monotonic()

        # Shared cancellation flag — set when the first validation error is found
        # so that in-flight workers in the same batch can bail out before downloading.
        cancel_event = threading.Event()

        # Determine 10% milestone step so we can log at both interval and milestone
        milestone_step = max(1, total // 10)
        next_milestone = milestone_step

        # Submit files in batches of _DOWNLOAD_WORKERS so that when a validation
        # error is raised within a batch the remaining batches are never submitted.
        # This preserves fail-fast semantics: at most _DOWNLOAD_WORKERS files are
        # in-flight when an error occurs, and subsequent batches are skipped.
        with ThreadPoolExecutor(max_workers=_DOWNLOAD_WORKERS) as executor:
            for batch_start in range(0, total, _DOWNLOAD_WORKERS):
                batch_keys = to_download[batch_start : batch_start + _DOWNLOAD_WORKERS]

                batch_futures: list[Future[int]] = [
                    executor.submit(
                        self._download_and_validate_one,
                        bucket,
                        key,
                        prefix,
                        output_path,
                        expected_tokens,
                        num_classes,
                        kpt_shape,
                        cancel_event,
                    )
                    for key in batch_keys
                ]

                for future in as_completed(batch_futures):
                    # Propagate any DatasetLoadingError raised by a worker.
                    # Set the cancel event first so in-flight sibling workers bail
                    # out before downloading, then re-raise.  The executor context
                    # manager will wait for running workers to finish (they return
                    # 0 immediately if cancel_event is set).
                    try:
                        file_size = future.result()
                    except Exception:
                        cancel_event.set()
                        raise
                    total_bytes += file_size
                    completed_count += 1

                    # Log progress at fixed intervals and at 10% milestones
                    if (
                        completed_count % _PROGRESS_LOG_INTERVAL == 0
                        or completed_count >= next_milestone
                        or completed_count == total
                    ):
                        elapsed = time.monotonic() - t0
                        pct = 100.0 * completed_count / total
                        self._logger.info(
                            "Download progress: %d/%d files (%.0f%%) | %.1f MB | %.1fs",
                            completed_count,
                            total,
                            pct,
                            total_bytes / 1024 / 1024,
                            elapsed,
                        )
                        # Advance to the next milestone
                        while next_milestone <= completed_count:
                            next_milestone += milestone_step

        elapsed = time.monotonic() - t0
        self._logger.info(
            "Download complete: %d files, %.1f MB in %.1fs (%.0f files/s)",
            total,
            total_bytes / 1024 / 1024,
            elapsed,
            total / elapsed if elapsed > 0 else float("inf"),
        )

    def _validate_label_file_inline(
        self,
        label_path: Path,
        expected_tokens: int,
        num_classes: int,
        kpt_shape: list[int],
    ) -> None:
        """Validate a single label file immediately after it is downloaded.

        Checks
        ------
        - File is non-empty.
        - Each line has exactly ``expected_tokens`` tokens.
        - All tokens parse as floats.
        - Class ID is a non-negative integer in ``[0, num_classes)``.
        - Bounding-box values (cx, cy, w, h) are in ``[0, 1]``.
        - Keypoint x, y values are in ``[0, 1]``.
        - If ``kpt_shape[1] == 3``, visibility flags are in ``{0, 1, 2}``.

        Raises
        ------
        DatasetLoadingError
            On the first validation failure found in the file.
        """
        if label_path.stat().st_size == 0:
            raise DatasetLoadingError(f"Label file is empty (downloaded): {label_path}")

        kpt_dim = kpt_shape[1]

        with label_path.open() as fh:
            for lineno, raw_line in enumerate(fh, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                tokens = line.split()

                if len(tokens) != expected_tokens:
                    raise DatasetLoadingError(
                        f"Label format error in {label_path} line {lineno}: "
                        f"expected {expected_tokens} tokens, got {len(tokens)}"
                    )

                try:
                    values = [float(t) for t in tokens]
                except ValueError:
                    raise DatasetLoadingError(
                        f"Label format error in {label_path} line {lineno}: "
                        f"non-numeric token found"
                    )

                cls_id = values[0]
                if cls_id != int(cls_id) or cls_id < 0 or int(cls_id) >= num_classes:
                    raise DatasetLoadingError(
                        f"Label format error in {label_path} line {lineno}: "
                        f"class ID {cls_id} out of range [0, {num_classes})"
                    )

                bbox = values[1:5]
                for name, val in zip(("cx", "cy", "w", "h"), bbox):
                    if not 0.0 <= val <= 1.0:
                        raise DatasetLoadingError(
                            f"Label format error in {label_path} line {lineno}: "
                            f"bbox {name}={val} outside [0, 1]"
                        )

                kp_values = values[5:]
                for ki in range(0, len(kp_values), kpt_dim):
                    kp_x = kp_values[ki]
                    kp_y = kp_values[ki + 1]
                    if not 0.0 <= kp_x <= 1.0:
                        raise DatasetLoadingError(
                            f"Label format error in {label_path} line {lineno}: "
                            f"keypoint x={kp_x} outside [0, 1]"
                        )
                    if not 0.0 <= kp_y <= 1.0:
                        raise DatasetLoadingError(
                            f"Label format error in {label_path} line {lineno}: "
                            f"keypoint y={kp_y} outside [0, 1]"
                        )
                    if kpt_dim == 3:
                        vis = kp_values[ki + 2]
                        if int(vis) not in _VALID_VISIBILITY or vis != int(vis):
                            raise DatasetLoadingError(
                                f"Label format error in {label_path} line {lineno}: "
                                f"keypoint visibility={vis} not in {{0, 1, 2}}"
                            )

    # ------------------------------------------------------------------
    # Labels-only helpers
    # ------------------------------------------------------------------

    def _validate_labels_only(
        self,
        image_keys: list[str],
        label_keys: list[str],
        output_path: Path,
        data_cfg: dict[str, Any],
    ) -> None:
        """Structural image-label pairing check for labels-only mode (post-download).

        Since inline validation already checked every label file's content during
        download, this method only performs the structural stem-matching check to
        confirm every image key has a corresponding label key.

        Note: the S3-level stem check was already performed in ``_check_s3_structure``.
        This is a belt-and-suspenders confirmation against the downloaded files.
        """
        # Build a mapping: (split, stem) -> True for images
        image_stems: dict[tuple[str, str], bool] = {}
        for key in image_keys:
            parts = Path(key).parts
            try:
                images_idx = parts.index("images")
                split = parts[images_idx + 1]
                stem = Path(parts[-1]).stem
                image_stems[(split, stem)] = True
            except (ValueError, IndexError):
                continue

        # Check every label has a matching image
        label_stems: dict[tuple[str, str], bool] = {}
        for key in label_keys:
            parts = Path(key).parts
            try:
                labels_idx = parts.index("labels")
                split = parts[labels_idx + 1]
                stem = Path(parts[-1]).stem
                label_stems[(split, stem)] = True
            except (ValueError, IndexError):
                continue

        for split_stem in image_stems:
            if split_stem not in label_stems:
                raise DatasetLoadingError(
                    f"Missing label for image: split={split_stem[0]}, stem={split_stem[1]}"
                )

        self._logger.info("Labels-only structural validation passed")

    def _build_manifest(
        self,
        bucket: str,
        prefix: str,
        image_keys: list[str],
        lakefs_commit: Optional[str] = None,
        dataset_hash: Optional[str] = None,
    ) -> DatasetManifest:
        """Build a :class:`DatasetManifest` from the listed S3 image keys."""
        splits: dict[str, list[str]] = {}
        for key in image_keys:
            parts = Path(key).parts
            try:
                images_idx = parts.index("images")
                split = parts[images_idx + 1]
            except (ValueError, IndexError):
                split = "unknown"
            splits.setdefault(split, []).append(key)

        return DatasetManifest(
            bucket=bucket,
            prefix=prefix,
            splits=splits,
            total_images=len(image_keys),
            lakefs_commit=lakefs_commit,
            dataset_hash=dataset_hash,
        )

    def _write_manifest(self, output_path: Path, manifest: DatasetManifest) -> None:
        """Write ``dataset_manifest.json`` to the output directory."""
        manifest_path = output_path / "dataset_manifest.json"
        with manifest_path.open("w") as fh:
            json.dump(manifest.model_dump(), fh, indent=2)
        self._logger.info("Wrote dataset manifest to %s", manifest_path)

    def _sample_labels_only(
        self,
        output_path: Path,
        manifest: DatasetManifest,
        sample_size: int,
        seed: int,
    ) -> None:
        """Sample labels in labels-only mode.

        Removes excess label files and updates the manifest on disk.
        """
        rng = random.Random(seed)

        for split in SPLITS:
            labels_dir = output_path / "labels" / split
            if not labels_dir.is_dir():
                continue

            label_files = sorted(labels_dir.glob("*.txt"))
            total = len(label_files)

            if total <= sample_size:
                self._logger.warning(
                    "Split '%s' has only %d labels (<= sample_size=%d); keeping all.",
                    split,
                    total,
                    sample_size,
                )
                continue

            keep_files = set(rng.sample(label_files, sample_size))
            keep_stems = {f.stem for f in keep_files}
            removed = 0
            for label_file in label_files:
                if label_file not in keep_files:
                    label_file.unlink()
                    removed += 1

            # Update manifest to only include sampled images
            if split in manifest.splits:
                manifest.splits[split] = [
                    k for k in manifest.splits[split] if Path(k).stem in keep_stems
                ]

            self._logger.info(
                "Split '%s': kept %d / %d labels (removed %d).",
                split,
                sample_size,
                total,
                removed,
            )

        # Recount and rewrite manifest
        manifest.total_images = sum(len(v) for v in manifest.splits.values())
        self._write_manifest(output_path, manifest)

    def _count_label_splits(self, output_path: Path) -> dict[str, int]:
        """Count label files per split (used in labels-only mode)."""
        counts: dict[str, int] = {}
        for split in SPLITS:
            labels_dir = output_path / "labels" / split
            if labels_dir.is_dir():
                counts[split] = sum(
                    1 for f in labels_dir.iterdir() if f.suffix == ".txt"
                )
            else:
                counts[split] = 0
        return counts

    # ------------------------------------------------------------------
    # Dataset lineage helpers
    # ------------------------------------------------------------------

    def _fetch_lakefs_commit(self, repo: str, branch: str) -> Optional[str]:
        """Return the branch-tip commit hash from the lakeFS HTTP API.

        Implements CON-01: a single attempt with a short timeout; any failure
        logs a warning and returns ``None`` so the pipeline is not aborted.
        """
        if not self._lakefs_endpoint:
            self._logger.warning(
                "LAKEFS_ENDPOINT not configured; lakefs_commit will be null"
            )
            return None
        url = f"{self._lakefs_endpoint.rstrip('/')}/api/v1/repositories/{repo}/refs/{branch}/commits"
        try:
            resp = requests.get(
                url,
                params={"limit": 1},
                auth=(self._lakefs_access_key or "", self._lakefs_secret_key or ""),
                timeout=5,
            )
            resp.raise_for_status()
            results = resp.json().get("results", [])
            if results:
                return str(results[0].get("id", ""))
            self._logger.warning(
                "lakeFS returned no commits for %s/%s; lakefs_commit will be null",
                repo,
                branch,
            )
            return None
        except Exception as exc:
            self._logger.warning(
                "Failed to fetch lakeFS commit hash (%s); lakefs_commit will be null",
                exc,
            )
            return None

    def _resolve_dataset_identity(
        self, image_keys: list[str], lakefs_commit: Optional[str]
    ) -> str:
        """Return the dataset identity recorded for resume validation (D-04).

        Prefer the lakeFS commit when available: it is an immutable,
        content-addressed identity for the whole dataset version that the lakeFS
        API already hands us, so resume validation (F-02, on a cheap CPU pod)
        can prove "same data" with a single string compare instead of
        re-listing and hashing the entire key set — which does not scale to
        multi-TB datasets with millions of objects. Fall back to the sorted
        key-list SHA-256 only for plain S3, where no commit exists.

        The value is opaque to consumers; they only require equality.
        """
        if lakefs_commit:
            self._logger.info(
                "Dataset identity = lakeFS commit %s (skipping key-list hash)",
                lakefs_commit,
            )
            return lakefs_commit
        return self._compute_dataset_hash(image_keys)

    def _compute_dataset_hash(self, image_keys: list[str]) -> str:
        """Return a SHA-256 hex digest of the sorted S3 image key list.

        Sorting provides determinism regardless of S3 listing order. Used as the
        dataset identity only for plain S3 (no lakeFS commit available).
        """
        key_list = "\n".join(sorted(image_keys))
        return hashlib.sha256(key_list.encode()).hexdigest()

    # ------------------------------------------------------------------
    # Source resolution
    # ------------------------------------------------------------------

    def _resolve_source(self, params: YoloDatasetParams) -> tuple[str, str]:
        """Return (bucket, key_prefix) based on the user's params.

        When ``path_override`` is provided it must be a valid ``s3://`` URI.
        When ``lakefs_repo`` and ``lakefs_branch`` are set (source=lakefs),
        the bucket is the repo and the prefix starts with the branch name.
        Otherwise the canonical convention is used.
        """
        if params.path_override:
            uri = params.path_override.strip()
            match = re.match(r"^s3://([^/]+)/(.*)$", uri)
            if not match:
                raise DatasetLoadingError(
                    f"path_override must be a valid s3:// URI, got: {uri!r}"
                )
            bucket = match.group(1)
            prefix = match.group(2)
            if not prefix.endswith("/"):
                prefix += "/"
            return bucket, prefix

        if params.lakefs_repo and params.lakefs_branch:
            bucket = params.lakefs_repo
            prefix = f"{params.lakefs_branch}/dataset/{params.version}/"
            return bucket, prefix

        bucket = _DEFAULT_BUCKET
        prefix = _DEFAULT_PREFIX_TEMPLATE.format(version=params.version)
        return bucket, prefix

    # ------------------------------------------------------------------
    # data.yaml generation
    # ------------------------------------------------------------------

    def _write_data_yaml(self, output_path: Path, runtime_root: str) -> None:
        """Write (or overwrite) ``data.yaml`` at the dataset root.

        If a ``data.yaml`` was downloaded from S3 it is used as a base;
        otherwise a canonical template is generated.  In either case the
        ``path`` field is set to the absolute ``runtime_root`` so YOLO
        training tools resolve splits correctly.
        """
        yaml_path = output_path / "data.yaml"

        if yaml_path.exists():
            with yaml_path.open() as fh:
                content: dict[str, Any] = yaml.safe_load(fh) or {}
        else:
            content = {
                "train": "images/train",
                "val": "images/val",
                "test": "images/test",
                "kpt_shape": [11, 3],
                "flip_idx": [],
                "names": {0: "spacecraft"},
            }

        content["path"] = str(Path(runtime_root).resolve())

        with yaml_path.open("w") as fh:
            yaml.dump(content, fh, default_flow_style=False, sort_keys=False)

        self._logger.info("Wrote data.yaml to %s", yaml_path)

    # ------------------------------------------------------------------
    # Post-download validation (lightweight structural safety net)
    # ------------------------------------------------------------------

    def _validate(self, output_path: Path) -> None:
        """Lightweight post-download structural validation of the YOLO dataset.

        Since semantic label content validation is performed inline during
        download (see :meth:`_validate_label_file_inline`), this method only
        performs structural checks:

        Checks
        ------
        - ``data.yaml`` exists and contains required Ultralytics keys.
        - ``train`` and ``val`` split image directories exist and are non-empty.
        - ``test`` is optional — validated only when present.
        - Every image has a corresponding non-empty label file.

        Note
        ----
        Label content spot-checking (token count, class ID, bbox, keypoints,
        visibility) is intentionally omitted here — it was already done inline
        as each file was downloaded.
        """
        self._logger.info(
            "Running post-download structural validation at %s", output_path
        )

        data_cfg = self._validate_data_yaml(output_path)

        # Determine which splits are present: train + val required, test optional
        active_splits = ["train", "val"]
        if (output_path / "images" / "test").is_dir():
            active_splits.append("test")

        for split in active_splits:
            images_dir = output_path / "images" / split
            labels_dir = output_path / "labels" / split

            if not images_dir.is_dir():
                raise DatasetLoadingError(f"Missing split directory: {images_dir}")

            image_files = [
                f
                for f in images_dir.iterdir()
                if f.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
            ]
            if not image_files:
                raise DatasetLoadingError(
                    f"Split '{split}' contains no images in {images_dir}"
                )

            for img_file in image_files:
                label_file = labels_dir / (img_file.stem + ".txt")
                if not label_file.exists():
                    raise DatasetLoadingError(
                        f"Missing label file for image: {img_file} "
                        f"(expected {label_file})"
                    )
                if label_file.stat().st_size == 0:
                    raise DatasetLoadingError(f"Label file is empty: {label_file}")

        self._logger.info("Post-download structural validation passed")
        # Suppress unused variable warning — data_cfg is validated via _validate_data_yaml
        _ = data_cfg

    def _validate_data_yaml(self, output_path: Path) -> dict[str, Any]:
        """Check that ``data.yaml`` exists and conforms to Ultralytics conventions.

        Required keys: ``path``, ``train``, ``val``, ``kpt_shape``, ``names``.
        ``test`` is optional per Ultralytics.

        Returns the parsed YAML dict so callers can use ``kpt_shape`` / ``names``.
        """
        yaml_path = output_path / "data.yaml"
        if not yaml_path.exists():
            raise DatasetLoadingError(f"data.yaml not found at {yaml_path}")

        with yaml_path.open() as fh:
            content = yaml.safe_load(fh) or {}

        # --- required keys (test is optional per Ultralytics) ---------------
        required_keys = {"path", "train", "val", "kpt_shape", "names"}
        missing = required_keys - set(content.keys())
        if missing:
            raise DatasetLoadingError(
                f"data.yaml is missing required keys: {sorted(missing)}"
            )

        # --- names: must be a dict {int: str} ------------------------------
        names = content["names"]
        if not isinstance(names, dict):
            raise DatasetLoadingError(
                f"data.yaml 'names' must be a dict mapping int -> str, "
                f"got {type(names).__name__}"
            )
        for k, v in names.items():
            if not isinstance(k, int) or not isinstance(v, str):
                raise DatasetLoadingError(
                    f"data.yaml 'names' entries must be int: str, "
                    f"got {type(k).__name__}: {type(v).__name__} for {k!r}: {v!r}"
                )

        # --- kpt_shape: must be [N, 2] or [N, 3] ---------------------------
        kpt_shape = content["kpt_shape"]
        if (
            not isinstance(kpt_shape, list)
            or len(kpt_shape) != 2
            or not isinstance(kpt_shape[0], int)
            or kpt_shape[0] < 1
            or kpt_shape[1] not in (2, 3)
        ):
            raise DatasetLoadingError(
                f"data.yaml 'kpt_shape' must be [N, 2] or [N, 3] with N >= 1, "
                f"got {kpt_shape!r}"
            )

        # --- nc consistency (optional key) ----------------------------------
        if "nc" in content:
            nc = content["nc"]
            if nc != len(names):
                raise DatasetLoadingError(
                    f"data.yaml 'nc' ({nc}) does not match len(names) ({len(names)})"
                )

        return content

    def _spot_check_labels(
        self,
        labels_dir: Path,
        split: str,
        expected_tokens: int = 38,
        num_classes: int = 1,
        kpt_shape: Optional[list[int]] = None,
    ) -> None:
        """Spot-check up to ``_SPOT_CHECK_LINES`` label files in *labels_dir*.

        This method is kept for use by ``_log_directory_integrity_report`` and
        for potential future use.  The primary label validation path now uses
        :meth:`_validate_label_file_inline` during download.

        Ultralytics pose label format::

            <class> <cx> <cy> <w> <h> <kp1x> <kp1y> [<kp1vis>] ... <kpNx> <kpNy> [<kpNvis>]
        """
        if not labels_dir.is_dir():
            return

        kpt_dim = kpt_shape[1] if kpt_shape else 3

        label_files = sorted(labels_dir.glob("*.txt"))[:_SPOT_CHECK_LINES]
        for label_path in label_files:
            with label_path.open() as fh:
                for lineno, raw_line in enumerate(fh, start=1):
                    line = raw_line.strip()
                    if not line:
                        continue
                    tokens = line.split()

                    if len(tokens) != expected_tokens:
                        raise DatasetLoadingError(
                            f"Label format error in {label_path} line {lineno}: "
                            f"expected {expected_tokens} tokens, "
                            f"got {len(tokens)}"
                        )

                    try:
                        values = [float(t) for t in tokens]
                    except ValueError:
                        raise DatasetLoadingError(
                            f"Label format error in {label_path} line {lineno}: "
                            f"non-numeric token found"
                        )

                    cls_id = values[0]
                    if (
                        cls_id != int(cls_id)
                        or cls_id < 0
                        or int(cls_id) >= num_classes
                    ):
                        raise DatasetLoadingError(
                            f"Label format error in {label_path} line {lineno}: "
                            f"class ID {cls_id} out of range [0, {num_classes})"
                        )

                    bbox = values[1:5]
                    for i, (name, val) in enumerate(zip(("cx", "cy", "w", "h"), bbox)):
                        if not 0.0 <= val <= 1.0:
                            raise DatasetLoadingError(
                                f"Label format error in {label_path} line {lineno}: "
                                f"bbox {name}={val} outside [0, 1]"
                            )

                    kp_values = values[5:]
                    for ki in range(0, len(kp_values), kpt_dim):
                        kp_x = kp_values[ki]
                        kp_y = kp_values[ki + 1]
                        if not 0.0 <= kp_x <= 1.0:
                            raise DatasetLoadingError(
                                f"Label format error in {label_path} line {lineno}: "
                                f"keypoint x={kp_x} outside [0, 1]"
                            )
                        if not 0.0 <= kp_y <= 1.0:
                            raise DatasetLoadingError(
                                f"Label format error in {label_path} line {lineno}: "
                                f"keypoint y={kp_y} outside [0, 1]"
                            )
                        if kpt_dim == 3:
                            vis = kp_values[ki + 2]
                            if int(vis) not in _VALID_VISIBILITY or vis != int(vis):
                                raise DatasetLoadingError(
                                    f"Label format error in {label_path} line {lineno}: "
                                    f"keypoint visibility={vis} not in {{0, 1, 2}}"
                                )

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def _sample_keys_by_split(
        self,
        image_keys: list[str],
        label_keys: list[str],
        sample_size: int,
        seed: int,
    ) -> tuple[list[str], list[str]]:
        """Sample S3 keys proportionally across splits before any download.

        For each split the method keeps at most *sample_size* image+label
        pairs.  If a split has fewer pairs than *sample_size* all pairs in
        that split are kept and a warning is logged.  The selection is
        deterministic given *seed*.

        Only keys that have a matching partner (image ↔ label by stem) are
        considered; any unpaired keys are discarded silently (the pre-download
        S3 structure check guarantees there are none in practice).

        Parameters
        ----------
        image_keys:
            Full S3 keys for image files (``images/<split>/<stem>.<ext>``).
        label_keys:
            Full S3 keys for label files (``labels/<split>/<stem>.txt``).
        sample_size:
            Maximum number of image+label pairs to keep per split.
        seed:
            Random seed for reproducibility.

        Returns
        -------
        tuple[list[str], list[str]]
            ``(sampled_image_keys, sampled_label_keys)`` — the filtered key
            lists ready to pass to :meth:`_download_with_inline_validation`.
        """
        rng = random.Random(seed)

        # Index label keys by (split, stem) for O(1) lookup
        label_index: dict[tuple[str, str], str] = {}
        for key in label_keys:
            parts = Path(key).parts
            try:
                labels_idx = parts.index("labels")
                split = parts[labels_idx + 1]
                stem = Path(parts[-1]).stem
                label_index[(split, stem)] = key
            except (ValueError, IndexError):
                pass

        # Group image keys by split — only keep those that have a label partner
        split_pairs: dict[str, list[tuple[str, str]]] = (
            {}
        )  # split -> [(img_key, lbl_key)]
        for key in image_keys:
            parts = Path(key).parts
            try:
                images_idx = parts.index("images")
                split = parts[images_idx + 1]
                stem = Path(parts[-1]).stem
            except (ValueError, IndexError):
                continue
            label_key = label_index.get((split, stem))
            if label_key is not None:
                split_pairs.setdefault(split, []).append((key, label_key))

        sampled_image_keys: list[str] = []
        sampled_label_keys: list[str] = []

        for split in SPLITS:
            pairs = sorted(split_pairs.get(split, []))  # sort for reproducibility
            total = len(pairs)
            if total == 0:
                continue
            if total <= sample_size:
                self._logger.warning(
                    "Split '%s' has only %d pairs (<= sample_size=%d); keeping all.",
                    split,
                    total,
                    sample_size,
                )
                chosen = pairs
            else:
                chosen = rng.sample(pairs, sample_size)
                self._logger.info(
                    "Pre-download sampling: split '%s' — keeping %d / %d pairs.",
                    split,
                    sample_size,
                    total,
                )

            for img_key, lbl_key in chosen:
                sampled_image_keys.append(img_key)
                sampled_label_keys.append(lbl_key)

        return sampled_image_keys, sampled_label_keys

    def _sample(self, output_path: Path, sample_size: int, seed: int) -> None:
        """Keep *sample_size* image+label pairs per split; delete the rest.

        If a split has fewer than *sample_size* images a warning is logged
        and all images in that split are kept.
        """
        rng = random.Random(seed)

        for split in SPLITS:
            images_dir = output_path / "images" / split
            labels_dir = output_path / "labels" / split

            if not images_dir.is_dir():
                continue

            image_files = sorted(
                f
                for f in images_dir.iterdir()
                if f.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
            )
            total = len(image_files)

            if total <= sample_size:
                self._logger.warning(
                    "Split '%s' has only %d images (<= sample_size=%d); keeping all.",
                    split,
                    total,
                    sample_size,
                )
                continue

            keep = set(rng.sample(image_files, sample_size))
            removed = 0
            for img_file in image_files:
                if img_file not in keep:
                    img_file.unlink()
                    label_file = labels_dir / (img_file.stem + ".txt")
                    if label_file.exists():
                        label_file.unlink()
                    removed += 1

            self._logger.info(
                "Split '%s': kept %d / %d images (removed %d).",
                split,
                sample_size,
                total,
                removed,
            )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _count_splits(self, output_path: Path) -> dict[str, int]:
        """Return a mapping of split name -> image count after all operations."""
        counts: dict[str, int] = {}
        for split in SPLITS:
            images_dir = output_path / "images" / split
            if images_dir.is_dir():
                counts[split] = sum(
                    1
                    for f in images_dir.iterdir()
                    if f.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
                )
            else:
                counts[split] = 0
        return counts

    def _log_dataset_stats(self, stats: YoloDatasetStats) -> None:
        """Log a structured per-split table of image and label counts.

        Emits one INFO line per split plus a total summary so that operators
        can confirm the dataset size at a glance in the pipeline log.
        """
        total_images = stats.train_images + stats.val_images + stats.test_images
        total_labels = stats.train_labels + stats.val_labels + stats.test_labels

        self._logger.info("Dataset sample counts:")
        self._logger.info("  %-8s  %8s  %8s", "split", "images", "labels")
        self._logger.info("  %s", "-" * 28)
        for split, n_img, n_lbl in (
            ("train", stats.train_images, stats.train_labels),
            ("val", stats.val_images, stats.val_labels),
            ("test", stats.test_images, stats.test_labels),
        ):
            self._logger.info("  %-8s  %8d  %8d", split, n_img, n_lbl)
        self._logger.info("  %s", "-" * 28)
        self._logger.info("  %-8s  %8d  %8d", "total", total_images, total_labels)

        if stats.sampled:
            self._logger.info(
                "  (sub-sampled to %d per split with seed=%d)",
                stats.sample_size,
                stats.seed,
            )

    def _log_directory_integrity_report(self, output_path: Path, mode: str) -> None:
        """Verify and log the YOLO directory structure at *output_path*.

        Checks (aligned with Ultralytics conventions)
        ----------------------------------------------
        - ``data.yaml`` exists and contains valid ``names``, ``kpt_shape``.
        - ``train`` and ``val`` split directories are required; ``test`` is optional.
        - ``images/<split>/`` and ``labels/<split>/`` directories exist and are
          non-empty for every active split.
        - Token count per label line matches ``1 + 4 + kpt_shape[0]*kpt_shape[1]``.

        A ``PASS`` verdict is logged when all checks succeed; a ``FAIL`` verdict
        is logged (at WARNING level) listing every failed check otherwise.
        The method never raises — the caller has already validated the dataset
        more rigorously via :meth:`_validate`.

        Parameters
        ----------
        output_path:
            Absolute path to the downloaded dataset root.
        mode:
            Descriptive label for the log (e.g. ``"full"`` or ``"labels-only"``).
        """
        failures: list[str] = []

        # --- data.yaml presence + Ultralytics fields -----------------------
        yaml_path = output_path / "data.yaml"
        if yaml_path.exists():
            self._logger.info("  [OK] data.yaml present")

            with yaml_path.open() as fh:
                cfg = yaml.safe_load(fh) or {}

            # names
            names = cfg.get("names")
            if isinstance(names, dict) and all(
                isinstance(k, int) and isinstance(v, str) for k, v in names.items()
            ):
                self._logger.info(
                    "  [OK] names — %d class(es): %s",
                    len(names),
                    ", ".join(str(v) for v in names.values()),
                )
            else:
                failures.append("names invalid (expected dict[int, str])")
                self._logger.warning("  [FAIL] names invalid: %r", names)

            # kpt_shape
            kpt_shape = cfg.get("kpt_shape")
            if (
                isinstance(kpt_shape, list)
                and len(kpt_shape) == 2
                and isinstance(kpt_shape[0], int)
                and kpt_shape[0] >= 1
                and kpt_shape[1] in (2, 3)
            ):
                expected_tokens = 1 + 4 + kpt_shape[0] * kpt_shape[1]
                self._logger.info(
                    "  [OK] kpt_shape — %d keypoints × %dD → %d tokens/line",
                    kpt_shape[0],
                    kpt_shape[1],
                    expected_tokens,
                )
            else:
                failures.append("kpt_shape invalid (expected [N, 2|3])")
                self._logger.warning("  [FAIL] kpt_shape invalid: %r", kpt_shape)

            # nc consistency
            if "nc" in cfg and isinstance(names, dict):
                if cfg["nc"] != len(names):
                    failures.append(f"nc={cfg['nc']} != len(names)={len(names)}")
                    self._logger.warning(
                        "  [FAIL] nc=%d != len(names)=%d", cfg["nc"], len(names)
                    )
        else:
            failures.append("data.yaml missing")
            self._logger.warning("  [FAIL] data.yaml missing")

        # --- split directories ---------------------------------------------
        # train + val required; test optional (Ultralytics convention)
        required_splits = ["train", "val"]
        optional_splits = ["test"]

        for split in required_splits + optional_splits:
            is_required = split in required_splits
            images_dir = output_path / "images" / split
            labels_dir = output_path / "labels" / split

            # images/
            if images_dir.is_dir():
                n_imgs = sum(
                    1
                    for f in images_dir.iterdir()
                    if f.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
                )
                if n_imgs > 0:
                    self._logger.info("  [OK] images/%s/ — %d image(s)", split, n_imgs)
                else:
                    failures.append(f"images/{split}/ is empty")
                    self._logger.warning(
                        "  [FAIL] images/%s/ exists but is empty", split
                    )
            else:
                if mode == "full" and is_required:
                    failures.append(f"images/{split}/ missing")
                    self._logger.warning("  [FAIL] images/%s/ missing", split)
                elif mode != "full":
                    self._logger.info(
                        "  [SKIP] images/%s/ not downloaded (labels-only mode)",
                        split,
                    )
                else:
                    # Optional split absent in full mode — not a failure
                    self._logger.info(
                        "  [SKIP] images/%s/ not present (optional split)", split
                    )

            # labels/
            if labels_dir.is_dir():
                n_lbls = sum(1 for f in labels_dir.iterdir() if f.suffix == ".txt")
                if n_lbls > 0:
                    self._logger.info(
                        "  [OK] labels/%s/ — %d label file(s)", split, n_lbls
                    )
                else:
                    failures.append(f"labels/{split}/ is empty")
                    self._logger.warning(
                        "  [FAIL] labels/%s/ exists but is empty", split
                    )
            else:
                if is_required:
                    failures.append(f"labels/{split}/ missing")
                    self._logger.warning("  [FAIL] labels/%s/ missing", split)
                else:
                    self._logger.info(
                        "  [SKIP] labels/%s/ not present (optional split)", split
                    )

        # Verdict
        if failures:
            self._logger.warning(
                "Directory integrity report [%s mode]: FAIL — %d issue(s): %s",
                mode,
                len(failures),
                ", ".join(failures),
            )
        else:
            self._logger.info("Directory integrity report [%s mode]: PASS", mode)

    def _write_stats(self, output_path: Path, stats: YoloDatasetStats) -> None:
        """Serialize *stats* to ``{output_path}/dataset_stats.json``."""
        stats_path = output_path / "dataset_stats.json"
        with stats_path.open("w") as fh:
            json.dump(stats.model_dump(), fh, indent=2)
        self._logger.info("Wrote dataset stats to %s", stats_path)
