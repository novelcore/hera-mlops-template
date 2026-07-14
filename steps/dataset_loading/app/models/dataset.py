"""Domain models for the YOLO dataset loading step."""

from typing import Optional

from pydantic import BaseModel, Field


class YoloDatasetParams(BaseModel):
    """Parameters passed into the dataset loading step from the CLI / orchestrator."""

    version: str = Field(description="Dataset version tag (e.g. 'v1').")
    source: str = Field(
        pattern=r"^(s3|lakefs)$",
        description="Storage backend: 's3' or 'lakefs'.",
    )
    lakefs_repo: Optional[str] = Field(
        default=None,
        description="LakeFS repository name. Used to construct the S3-compatible path.",
    )
    lakefs_branch: Optional[str] = Field(
        default=None,
        description="LakeFS branch name. Used to construct the S3-compatible path.",
    )
    path_override: Optional[str] = Field(
        default=None,
        description=(
            "Full S3 URI override (e.g. 's3://my-bucket/path/'). "
            "When set, the conventional bucket/prefix is ignored."
        ),
    )
    labels_only: bool = Field(
        default=False,
        description=(
            "When True, download only label files and data.yaml (no images). "
            "Used with S3 streaming mode where images are fetched on-demand."
        ),
    )
    manifest_only: bool = Field(
        default=False,
        description=(
            "When True, download only data.yaml and list S3 keys (no images, no labels). "
            "Both images and labels are streamed from S3 during training."
        ),
    )
    sample_size: Optional[int] = Field(
        default=None,
        ge=1,
        description="If set, keep only this many image+label pairs per split.",
    )
    seed: int = Field(
        default=42,
        description="Random seed used for reproducible sampling.",
    )
    output_dir: str = Field(description="Local directory where the dataset is written.")


class DatasetManifest(BaseModel):
    """Manifest describing the S3 dataset layout for downstream S3 streaming.

    Written to ``{output_dir}/dataset_manifest.json`` in labels-only mode so
    that the training step knows which bucket/prefix to stream from and how
    many images to expect per split.
    """

    bucket: str = Field(description="S3 bucket name.")
    prefix: str = Field(description="S3 key prefix (with trailing slash).")
    splits: dict[str, list[str]] = Field(
        description="Mapping of split name to list of S3 image keys."
    )
    label_keys: Optional[dict[str, list[str]]] = Field(
        default=None,
        description=(
            "Mapping of split name to list of S3 label keys. "
            "Present only in manifest-only mode; signals that labels "
            "should be streamed from S3 during training."
        ),
    )
    total_images: int = Field(
        ge=0, description="Total number of images across all splits."
    )
    lakefs_commit: Optional[str] = Field(
        default=None,
        description=(
            "LakeFS branch-tip commit hash at fetch time. "
            "Null when source=s3 or the lakeFS API was unreachable."
        ),
    )
    dataset_hash: Optional[str] = Field(
        default=None,
        description=(
            "Dataset identity enforced at resume (D-04): the lakeFS commit when "
            "available, otherwise the SHA-256 of the sorted S3 image key list. "
            "Consumers only require equality."
        ),
    )


class YoloDatasetStats(BaseModel):
    """Statistics artifact written to {output_dir}/dataset_stats.json after a run."""

    version: str = Field(description="Dataset version tag.")
    source: str = Field(description="Storage backend used: 's3' or 'lakefs'.")
    train_images: int = Field(ge=0, description="Number of images in the train split.")
    val_images: int = Field(ge=0, description="Number of images in the val split.")
    test_images: int = Field(ge=0, description="Number of images in the test split.")
    train_labels: int = Field(
        ge=0, description="Number of label files in the train split."
    )
    val_labels: int = Field(ge=0, description="Number of label files in the val split.")
    test_labels: int = Field(
        ge=0, description="Number of label files in the test split."
    )
    sampled: bool = Field(description="True when the dataset was sub-sampled.")
    sample_size: Optional[int] = Field(
        default=None,
        description="The requested sample size (None when not sampled).",
    )
    seed: int = Field(description="Random seed used during sampling.")
    lakefs_commit: Optional[str] = Field(
        default=None,
        description=(
            "LakeFS branch-tip commit hash at fetch time. "
            "Null when source=s3 or the lakeFS API was unreachable."
        ),
    )
    dataset_hash: Optional[str] = Field(
        default=None,
        description=(
            "Dataset identity enforced at resume (D-04): the lakeFS commit when "
            "available, otherwise the SHA-256 of the sorted S3 image key list."
        ),
    )
