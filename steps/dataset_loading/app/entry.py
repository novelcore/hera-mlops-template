"""dataset-loading — the Hera-way entry point.

This is the thin adapter between the platform's step contract and the real
dataset-loading service (app/services/dataset_loading.py, ported verbatim from
the production YOLO pipeline). It:

  1. reads its config slice + platform endpoints from the single resolved
     params.yaml that compose-and-validate produced (--params) — the same file
     every step receives; this step reads cfg["data"] and cfg["platform"];
  2. maps that config onto the service's parameters and runs it in MANIFEST-ONLY
     mode — the only mode that works on the cluster, because each step runs in
     its own pod with no shared filesystem, so images/labels are streamed from
     object storage during training rather than handed between pods on disk;
  3. emits two step outputs that model-training consumes (declared in
     pipeline.py via outputs=["data-yaml", "manifest-summary"]):
       - data-yaml         : the dataset's data.yaml (kpt_shape, names, splits)
       - manifest-summary  : {bucket, prefix, label_keys sentinel} — the compact
                             pointer to the object-store layout (the full key
                             list is far too large for an Argo parameter).

Credentials + endpoints are NOT read from config or any file: the platform
injects them as env (LAKEFS_ENDPOINT, LAKEFS_ACCESS_KEY, LAKEFS_SECRET_KEY, or
the AWS_* chain for source=s3), which the service's env-var Config picks up.
The developer writes none of that wiring.
"""

import argparse
import json
import sys
from pathlib import Path

import yaml

from app.manager import Manager

# The config sections this step reads from params.yaml. The compose-and-validate
# release gate fails the render if the config tree has no `data` section, so a
# renamed/deleted section is caught before any run.
READS = ["data"]

# Where the service writes the dataset tree (this pod's ephemeral volume) and
# where the platform expects this step's declared outputs.
DATASET_DIR = "/work/dataset"
OUTPUT_DIR = Path("/work/output")


def _int_field(value, name: str, default: int) -> int:
    """Coerce a form value to int with a clear error (not a raw ValueError)."""
    raw = str(value).strip() if value is not None else ""
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        raise ValueError(f"{name} must be an integer, got {raw!r}")


def _resolve_dataset_params(cfg: dict) -> dict:
    """Map the experiment `data` config slice + platform lakeFS repo onto the
    dataset-loading service's run parameters.

    The Hydra `data` slice is the developer-facing shape (ref/source/version/…);
    the lakeFS repository is a platform fact, so it comes from cfg["platform"],
    never from the developer's config.
    """
    data = cfg.get("data") or {}
    platform = cfg.get("platform") or {}
    lakefs = platform.get("lakefs") or {}

    source = str(data.get("source") or "lakefs")

    # The lakeFS ref (branch/tag/commit) the developer picked is the branch the
    # service lists under; the repository is platform-owned. `version` is a
    # provenance tag — default it to the ref so the S3 prefix is well-formed.
    ref = str(data.get("ref") or "main")
    version = str(data.get("version") or "") or ref

    # sample_size is a string in the form ("" = full dataset); coerce to a
    # positive int, or None when empty. A bad value is a form mistake, so fail
    # with a clear message here rather than a raw int() ValueError or a verbose
    # Pydantic error deep in the service.
    raw_sample = str(data.get("sample_size") or "").strip()
    if raw_sample:
        try:
            sample_size = int(raw_sample)
        except ValueError:
            raise ValueError(
                f"data.sample_size must be a positive integer or empty, got {raw_sample!r}"
            )
        if sample_size < 1:
            raise ValueError(
                f"data.sample_size must be >= 1 (or empty for the full dataset), got {sample_size}"
            )
    else:
        sample_size = None

    params = {
        "version": version,
        "source": source,
        "output_dir": DATASET_DIR,
        # MANIFEST-ONLY is mandatory on the cluster (no shared FS between step
        # pods). Full-download / labels-only assume a shared volume this
        # platform does not provide; the service supports them for local use.
        "manifest_only": True,
        "sample_size": sample_size,
        "seed": _int_field(data.get("seed", 42), "data.seed", default=42),
    }

    path_override = str(data.get("path_override") or "").strip()
    if path_override:
        params["path_override"] = path_override
    elif source == "lakefs":
        params["lakefs_repo"] = str(lakefs.get("repository") or "")
        params["lakefs_branch"] = ref

    return params


def _write_outputs(cfg: dict) -> None:
    """Extract the two step outputs from the service's on-disk artifacts.

    model-training reconstructs /work/dataset from these:
      - data-yaml        : the full data.yaml content (verbatim)
      - manifest-summary : a compact {bucket, prefix, label_keys} pointer; the
                           label_keys sentinel tells training to stream labels
                           from object storage too (manifest-only mode).
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    dataset_dir = Path(DATASET_DIR)

    data_yaml_text = (dataset_dir / "data.yaml").read_text()
    (OUTPUT_DIR / "data-yaml.json").write_text(data_yaml_text)

    manifest = json.loads((dataset_dir / "dataset_manifest.json").read_text())
    bucket, prefix = manifest.get("bucket"), manifest.get("prefix")
    if not bucket or not prefix:
        print(f"[dataset-loading] ERROR: manifest missing bucket/prefix: {manifest}",
              file=sys.stderr)
        sys.exit(1)

    # Non-empty sentinel = labels are also streamed from object storage (we never
    # embed the actual key lists — too large for an Argo parameter).
    label_keys_sentinel = {"_present": True} if manifest.get("label_keys") else {}
    summary = {"bucket": bucket, "prefix": prefix, "label_keys": label_keys_sentinel}
    (OUTPUT_DIR / "manifest-summary.json").write_text(
        json.dumps(summary, separators=(",", ":"))
    )
    print(f"[dataset-loading] outputs written: data-yaml + manifest-summary "
          f"(bucket={bucket} prefix={prefix})")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", required=True,
                        help="Resolved params.yaml content (from compose-and-validate).")
    args, _ = parser.parse_known_args()
    cfg = yaml.safe_load(args.params)

    params = _resolve_dataset_params(cfg)
    print(f"[dataset-loading] source={params['source']} version={params['version']} "
          f"manifest-only -> {DATASET_DIR}")

    # Manager reads endpoints + credentials from the platform-injected env
    # (LAKEFS_ENDPOINT / LAKEFS_ACCESS_KEY / LAKEFS_SECRET_KEY, or the AWS_*
    # chain for source=s3). The developer wires none of this.
    Manager().run(**params)

    _write_outputs(cfg)


if __name__ == "__main__":
    main()
