"""Generate (and optionally upload) a toy Ultralytics-pose dataset.

A brand-new project's lakeFS repository is EMPTY, so the very first pipeline
run fails at ``config-validation`` with ``dataset path not found or empty``.
This tool produces a small, fully valid pose dataset so the whole pipeline
(config-validation → dataset-loading → model-training → …) can be exercised
end to end before real data exists:

* stdlib only — runs inside ``./run.sh``'s venv, a CI pod, or a laptop;
* real, decodable PNG images: one filled ellipse per image on a flat
  background, with its 4 extreme points (top/right/bottom/left) drawn as dots;
* labels follow the contract ``steps/dataset_loading`` validates: one line per
  object, ``class cx cy w h`` + ``K*3`` keypoint tokens (x, y, visibility), all
  normalised to [0, 1], visibility in {0, 1, 2};
* ``data.yaml`` with ``path/train/val/test/kpt_shape/flip_idx/names``.

Layout (what ``config-validation`` resolves for ``data.ref``)::

    s3://{repo}/{ref}/dataset/{ref}/data.yaml
                                    images/{train,val,test}/toy_*.png
                                    labels/{train,val,test}/toy_*.txt

Usage::

    python -m tools.toy_dataset --out ./toy-dataset            # local only
    python -m tools.toy_dataset --out ./toy-dataset --upload \\
        --repo <project> --branch main                          # + lakeFS

Upload reads ``LAKEFS_ENDPOINT`` and either ``LAKEFS_ACCESS_KEY_ID`` +
``LAKEFS_SECRET_ACCESS_KEY`` or ``LAKEFS_BEARER_TOKEN`` from the environment.
It is a SYNC of ``dataset/{branch}/``: files are uploaded, remote-only objects
under that prefix are deleted, and one commit is made — so re-running replaces
a previous toy set instead of mixing with it.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import random
import struct
import sys
import urllib.parse
import urllib.request
import zlib
from pathlib import Path

KPT_NAMES = ("top", "right", "bottom", "left")
KPT_SHAPE = (len(KPT_NAMES), 3)
FLIP_IDX = [0, 3, 2, 1]  # a horizontal flip swaps right <-> left
SPLITS = {"train": 16, "val": 8, "test": 4}
SIZE = 96
BG, FILL, DOT = (24, 28, 36), (230, 120, 40), (40, 220, 90)


def _png(pixels: list[list[tuple[int, int, int]]]) -> bytes:
    """Encode an RGB pixel grid as a PNG (stdlib only)."""
    raw = b"".join(b"\x00" + bytes(c for px in row for c in px) for row in pixels)

    def chunk(tag: bytes, body: bytes) -> bytes:
        return (struct.pack(">I", len(body)) + tag + body
                + struct.pack(">I", zlib.crc32(tag + body) & 0xFFFFFFFF))

    ihdr = struct.pack(">IIBBBBB", len(pixels[0]), len(pixels), 8, 2, 0, 0, 0)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr)
            + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b""))


def _sample(rng: random.Random) -> tuple[bytes, str]:
    """One image + its YOLO-pose label line."""
    rx, ry = rng.uniform(0.12, 0.3), rng.uniform(0.12, 0.3)
    cx, cy = rng.uniform(rx + 0.05, 0.95 - rx), rng.uniform(ry + 0.05, 0.95 - ry)
    kpts = [(cx, cy - ry), (cx + rx, cy), (cx, cy + ry), (cx - rx, cy)]
    # Occasionally mark one keypoint "partially visible" so the flag varies.
    vis = [2] * len(kpts)
    if rng.random() < 0.25:
        vis[rng.randrange(len(kpts))] = 1

    def px(x: float, y: float) -> tuple[int, int, int]:
        if any(abs(x - kx) < 0.03 and abs(y - ky) < 0.03 for kx, ky in kpts):
            return DOT
        return FILL if ((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2 <= 1 else BG

    grid = [[px((i + 0.5) / SIZE, (j + 0.5) / SIZE) for i in range(SIZE)]
            for j in range(SIZE)]
    tokens = [0, cx, cy, 2 * rx, 2 * ry]
    for (kx, ky), v in zip(kpts, vis):
        tokens += [kx, ky, v]
    line = " ".join(str(t) if isinstance(t, int) else f"{t:.6f}" for t in tokens)
    return _png(grid), line + "\n"


def generate(out: Path, seed: int = 7) -> dict[str, bytes]:
    """Write the dataset under ``out`` and return {relative path: bytes}."""
    rng = random.Random(seed)
    files: dict[str, bytes] = {
        "data.yaml": (
            "path: .\ntrain: images/train\nval: images/val\ntest: images/test\n"
            f"kpt_shape: [{KPT_SHAPE[0]}, {KPT_SHAPE[1]}]\n"
            f"flip_idx: {FLIP_IDX}\nnames:\n  0: blob\n"
        ).encode()
    }
    for split, n in SPLITS.items():
        for i in range(n):
            img, label = _sample(rng)
            files[f"images/{split}/toy_{split}_{i:03d}.png"] = img
            files[f"labels/{split}/toy_{split}_{i:03d}.txt"] = label.encode()
    for rel, data in files.items():
        (out / rel).parent.mkdir(parents=True, exist_ok=True)
        (out / rel).write_bytes(data)
    return files


class LakeFS:
    """Minimal lakeFS REST client (basic-auth keys or bearer token)."""

    def __init__(self, endpoint: str, repo: str, branch: str, opener=None):
        self.base = endpoint.rstrip("/") + "/api/v1"
        self.repo, self.branch = repo, branch
        self.open = opener or urllib.request.urlopen
        if os.environ.get("LAKEFS_BEARER_TOKEN"):
            self.auth = "Bearer " + os.environ["LAKEFS_BEARER_TOKEN"]
        else:
            keys = f'{os.environ["LAKEFS_ACCESS_KEY_ID"]}:{os.environ["LAKEFS_SECRET_ACCESS_KEY"]}'
            self.auth = "Basic " + base64.b64encode(keys.encode()).decode()

    def _req(self, method: str, path: str, body: bytes | None = None,
             ctype: str | None = None):
        req = urllib.request.Request(self.base + path, data=body, method=method,
                                     headers={"Authorization": self.auth})
        if ctype:
            req.add_header("Content-Type", ctype)
        with self.open(req, timeout=120) as resp:
            return resp.read()

    def list(self, prefix: str) -> list[str]:
        paths, after = [], ""
        while True:
            q = urllib.parse.urlencode({"prefix": prefix, "amount": 1000, "after": after})
            page = json.loads(self._req(
                "GET", f"/repositories/{self.repo}/refs/{self.branch}/objects/ls?{q}"))
            paths += [o["path"] for o in page["results"]]
            if not page["pagination"]["has_more"]:
                return paths
            after = page["pagination"]["next_offset"]

    def put(self, path: str, data: bytes) -> None:
        body = (b"--B\r\nContent-Disposition: form-data; name=\"content\"; filename=\"f\"\r\n"
                b"Content-Type: application/octet-stream\r\n\r\n" + data + b"\r\n--B--\r\n")
        q = urllib.parse.urlencode({"path": path})
        self._req("POST", f"/repositories/{self.repo}/branches/{self.branch}/objects?{q}",
                  body, "multipart/form-data; boundary=B")

    def delete(self, path: str) -> None:
        q = urllib.parse.urlencode({"path": path})
        self._req("DELETE", f"/repositories/{self.repo}/branches/{self.branch}/objects?{q}")

    def commit(self, message: str) -> str:
        out = self._req("POST", f"/repositories/{self.repo}/branches/{self.branch}/commits",
                        json.dumps({"message": message}).encode(), "application/json")
        return json.loads(out)["id"]


def sync(client: LakeFS, files: dict[str, bytes], prefix: str) -> tuple[int, int, str]:
    """Upload ``files`` under ``prefix``, delete remote-only objects, commit."""
    wanted = {prefix + rel: data for rel, data in files.items()}
    stale = [p for p in client.list(prefix) if p not in wanted]
    for path, data in wanted.items():
        client.put(path, data)
    for path in stale:
        client.delete(path)
    commit = client.commit(
        f"toy dataset: {len(wanted)} objects under {prefix} (tools/toy_dataset.py)")
    return len(wanted), len(stale), commit


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, required=True, help="local output directory")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--upload", action="store_true", help="sync to lakeFS after generating")
    ap.add_argument("--repo", help="lakeFS repository (usually the project name)")
    ap.add_argument("--branch", default="main", help="lakeFS branch = the data.ref you train on")
    ap.add_argument("--endpoint", default=os.environ.get("LAKEFS_ENDPOINT"),
                    help="lakeFS API endpoint (default: $LAKEFS_ENDPOINT)")
    args = ap.parse_args(argv)

    files = generate(args.out, args.seed)
    print(f"generated {len(files)} objects under {args.out} "
          f"({', '.join(f'{s}={n}' for s, n in SPLITS.items())})")
    if not args.upload:
        return 0
    if not (args.repo and args.endpoint):
        print("--upload needs --repo and --endpoint/$LAKEFS_ENDPOINT", file=sys.stderr)
        return 2
    prefix = f"dataset/{args.branch}/"
    n, deleted, commit = sync(LakeFS(args.endpoint, args.repo, args.branch), files, prefix)
    print(f"synced {n} objects to lakefs://{args.repo}/{args.branch}/{prefix} "
          f"(deleted {deleted} stale) commit {commit[:12]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
