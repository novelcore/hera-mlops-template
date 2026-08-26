"""tools/toy_dataset.py — the generated set must satisfy the dataset-loading
contract (mirrors steps/dataset_loading `_check_s3_structure` +
`_validate_label_file_inline`) and the lakeFS sync must replace, not mix."""
import io
import json
import struct
import sys
import zlib
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import toy_dataset  # noqa: E402


def test_generated_set_matches_loader_contract(tmp_path):
    files = toy_dataset.generate(tmp_path)
    data = yaml.safe_load((tmp_path / "data.yaml").read_text())
    for key in ("path", "train", "val", "kpt_shape", "names"):
        assert key in data, key
    k, dim = data["kpt_shape"]
    expected_tokens = 1 + 4 + k * dim
    num_classes = len(data["names"])

    images = {p for p in files if p.startswith("images/")}
    labels = {p for p in files if p.startswith("labels/")}
    assert images and labels
    for split in ("train", "val"):
        assert any(p.startswith(f"images/{split}/") for p in images), split
    # every image has a label with the same stem in the same split, and vice versa
    stems = lambda paths: {(Path(p).parts[1], Path(p).stem) for p in paths}  # noqa: E731
    assert stems(images) == stems(labels)

    for rel in labels:
        lines = files[rel].decode().splitlines()
        assert lines, rel
        for line in lines:
            toks = line.split()
            assert len(toks) == expected_tokens, (rel, line)
            cls = int(toks[0])
            assert 0 <= cls < num_classes
            box = [float(t) for t in toks[1:5]]
            assert all(0.0 <= v <= 1.0 for v in box), box
            for i in range(k):
                x, y, v = toks[5 + i * dim:5 + (i + 1) * dim]
                assert 0.0 <= float(x) <= 1.0 and 0.0 <= float(y) <= 1.0
                assert int(v) in {0, 1, 2}


def test_images_are_decodable_png(tmp_path):
    files = toy_dataset.generate(tmp_path)
    png = next(v for p, v in files.items() if p.endswith(".png"))
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    width, height, depth, ctype = struct.unpack(">IIBB", png[16:26])
    assert (width, height, depth, ctype) == (toy_dataset.SIZE, toy_dataset.SIZE, 8, 2)
    idat_len = struct.unpack(">I", png[33:37])[0]
    assert png[37:41] == b"IDAT"
    raw = zlib.decompress(png[41:41 + idat_len])
    assert len(raw) == height * (1 + width * 3)


def test_generation_is_deterministic(tmp_path):
    a = toy_dataset.generate(tmp_path / "a")
    b = toy_dataset.generate(tmp_path / "b")
    assert a == b
    assert a != toy_dataset.generate(tmp_path / "c", seed=8)


class _FakeLakeFS:
    """Records requests; pre-seeded with one stale object under the prefix."""

    def __init__(self):
        self.calls = []

    def __call__(self, req, timeout=0):
        self.calls.append((req.get_method(), req.full_url))
        body = b"{}"
        if "/objects/ls" in req.full_url:
            body = json.dumps({"results": [{"path": "dataset/main/old.txt"},
                                           {"path": "dataset/main/data.yaml"}],
                               "pagination": {"has_more": False, "next_offset": ""}}).encode()
        elif req.full_url.endswith("/commits"):
            body = json.dumps({"id": "abc123def456"}).encode()
        return io.BytesIO(body)


def test_sync_uploads_deletes_stale_and_commits(monkeypatch):
    monkeypatch.setenv("LAKEFS_ACCESS_KEY_ID", "k")
    monkeypatch.setenv("LAKEFS_SECRET_ACCESS_KEY", "s")
    fake = _FakeLakeFS()
    client = toy_dataset.LakeFS("http://lakefs", "proj", "main", opener=fake)
    n, deleted, commit = toy_dataset.sync(
        client, {"data.yaml": b"x", "images/train/a.png": b"y"}, "dataset/main/")
    assert (n, deleted, commit) == (2, 1, "abc123def456")
    methods = [m for m, _ in fake.calls]
    assert methods == ["GET", "POST", "POST", "DELETE", "POST"]
    assert "path=dataset%2Fmain%2Fold.txt" in fake.calls[3][1]
    assert fake.calls[1][1].endswith("branches/main/objects?path=dataset%2Fmain%2Fdata.yaml")
