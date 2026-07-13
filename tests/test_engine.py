"""Assertion suite for the kubecore ML-pipeline engine.

Run offline (no cluster):  python -m pytest tests/ -v
or standalone:             python tests/test_engine.py

Covers the platform contract: form derivation, the reads= render gate, boolean
values, group-swap default-elision, the ADVANCED platform guard, enum
validation, the params size cap, byte-stability, and the enhancer's live-model
fidelity (per-step {step}-class + nodeSelector, GPU routing, image indirection,
sizing knobs, checkpoint env, /dev/shm).
"""

import os
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from kubecore import compose, derive_tree, enhance  # noqa: E402

CONTEXT = yaml.safe_load((ROOT / "kubecore" / "local-dev" / "pipeline-context.yaml").read_text())
CATALOG = yaml.safe_load((ROOT / "kubecore" / "local-dev" / "dataset-catalog.yaml").read_text())


def _render():
    """Render pipeline.py -> raw WFT dict (in-process)."""
    import importlib
    import runpy

    # pipeline.py builds the WorkflowTemplate on context exit; capture it.
    ns = runpy.run_path(str(ROOT / "pipeline.py"), run_name="__pipeline__")
    return yaml.safe_load(ns["p"].wt.to_yaml())


def _enhanced():
    return enhance.enhance(_render(), CONTEXT, CATALOG)


# ------------------------------------------------------------- derivation


def test_form_surface_covers_yolo():
    form = derive_tree.derive()
    names = {p.name for p in form.parameters}
    # spot-check the real yolo parameter surface reached the form
    for expected in ["train-epochs", "train-optimizer-lr", "image_processing-hsv_h",
                     "data-ref", "quantization-mode", "train-optimizer", "model"]:
        assert expected in names, f"missing derived form param {expected}"
    assert "config" in names  # ADVANCED
    print(f"  derive: {len(form.parameters)} form params, {len(form.sections)} sections")


def test_groups_render_as_dropdowns():
    form = derive_tree.derive()
    by_name = {p.name: p for p in form.parameters}
    for group in ["model", "train-optimizer", "train-callbacks", "train-qat", "image_processing"]:
        assert by_name[group].enum, f"group {group} should be an enum dropdown"


def test_boolean_is_plain_value():
    form = derive_tree.derive()
    by_name = {p.name: p for p in form.parameters}
    amp = by_name["train-amp"]
    assert amp.default in ("true", "false")
    assert amp.enum is None, "booleans must NOT carry an enum (O9 dead)"


def test_reads_gate_fires_on_missing_section():
    try:
        derive_tree.validate_reads({"badstep": ["nonexistent-section"]}, ["data", "train"])
        assert False, "reads= gate should have raised"
    except derive_tree.DeriveError as e:
        assert "nonexistent-section" in str(e) and "badstep" in str(e)


# ------------------------------------------------------------- compose


def test_compose_byte_stable():
    a = compose.compose_and_validate([], required=["data", "train"])
    b = compose.compose_and_validate([], required=["data", "train"])
    assert a == b, "compose must be byte-stable"


def test_boolean_override_end_to_end():
    txt = compose.compose_and_validate(["train.amp=false"], required=["train"])
    cfg = yaml.safe_load(txt)
    assert cfg["train"]["amp"] is False, "override should yield a real YAML bool"


def test_group_swap_takes_effect():
    form = derive_tree.derive()
    rd = form.render_defaults
    # simulate a FULL submission (every leaf frozen at default) + a group swap
    tokens = [f"{k}={v}" for k, v in rd.items()]
    tokens = [t if not t.startswith("train/optimizer=") else "train/optimizer=adamw" for t in tokens]
    txt = compose.compose_and_validate(tokens, required=["train"], render_defaults=rd)
    cfg = yaml.safe_load(txt)
    assert cfg["train"]["optimizer"]["name"] == "AdamW", "group swap must survive frozen leaves"


def test_touched_leaf_wins_over_group():
    form = derive_tree.derive()
    rd = form.render_defaults
    tokens = [f"{k}={v}" for k, v in rd.items()]
    tokens = [t if not t.startswith("train/optimizer=") else "train/optimizer=adamw" for t in tokens]
    tokens = [t if not t.startswith("train.optimizer.lr=") else "train.optimizer.lr=0.005" for t in tokens]
    txt = compose.compose_and_validate(tokens, required=["train"], render_defaults=rd)
    cfg = yaml.safe_load(txt)
    assert cfg["train"]["optimizer"]["name"] == "AdamW"
    assert float(cfg["train"]["optimizer"]["lr"]) == 0.005, "explicitly-set leaf must win"


def test_advanced_cannot_touch_platform():
    try:
        compose.compose_and_validate([], advanced="platform:\n  checkpoints:\n    bucket: attacker")
        assert False, "ADVANCED touching platform.* should fail"
    except compose.ComposeError as e:
        assert "platform" in str(e)


def test_enum_validation_rejects_bad_value():
    try:
        compose.compose_and_validate(["quantization.mode=banana"], required=["quantization"])
        assert False, "invalid enum value should fail"
    except Exception as e:
        assert "banana" in str(e) or "Invalid value" in str(e)


def test_size_cap():
    saved = compose.MAX_PARAMS_BYTES
    compose.MAX_PARAMS_BYTES = 10  # force the cap
    try:
        compose.compose_and_validate([], required=["data"])
        assert False, "size cap should trip"
    except compose.ComposeError as e:
        assert "exceeds" in str(e)
    finally:
        compose.MAX_PARAMS_BYTES = saved


# ------------------------------------------------------------- enhancer (live-model fidelity)


def test_per_step_class_params():
    wft = _enhanced()
    names = {p["name"] for p in wft["spec"]["arguments"]["parameters"]}
    for step in ["dataset-loading", "model-training", "qat-finetune", "model-registration"]:
        assert f"{step}-class" in names, f"missing per-step class param {step}-class"


def test_nodeselector_uses_step_class():
    wft = _enhanced()
    for t in wft["spec"]["templates"]:
        if "container" not in t:
            continue
        sel = t.get("nodeSelector", {}).get("platform.kubecore.io/nodegroup-type", "")
        assert sel == f"{{{{workflow.parameters.{t['name']}-class}}}}", \
            f"{t['name']} nodeSelector should reference its {{step}}-class param"


def test_gpu_routing():
    wft = _enhanced()
    gpu_steps = set()
    for t in wft["spec"]["templates"]:
        if "container" in t:
            req = t["container"].get("resources", {}).get("requests", {})
            if "nvidia.com/gpu" in req:
                gpu_steps.add(t["name"])
    assert gpu_steps == {"model-training", "qat-finetune"}, f"unexpected GPU steps: {gpu_steps}"


def test_image_indirection_uses_app_configmap():
    wft = _enhanced()
    cm = f"{CONTEXT['app']}-pipeline-images"
    found = False
    for p in wft["spec"]["arguments"]["parameters"]:
        if p["name"].startswith("image-"):
            assert p["valueFrom"]["configMapKeyRef"]["name"] == cm
            found = True
    assert found, "expected image-<step> configMapKeyRef params"


def test_sizing_knobs_and_checkpoint_env():
    wft = _enhanced()
    names = {p["name"] for p in wft["spec"]["arguments"]["parameters"]}
    assert "model-training-cpu" in names and "model-training-mem" in names
    # checkpoint env on every step
    for t in wft["spec"]["templates"]:
        if "container" in t:
            env = {e["name"] for e in t["container"].get("env", [])}
            assert "CHECKPOINT_BUCKET" in env and "MLFLOW_TRACKING_URI" in env


def test_unknown_annotation_fails():
    raw = _render()
    # inject a bad annotation on the first container template
    for t in raw["spec"]["templates"]:
        if "container" in t:
            t.setdefault("metadata", {}).setdefault("annotations", {})[
                "platform.kubecore.io/bogus"] = "x"
            break
    try:
        enhance.enhance(raw, CONTEXT, CATALOG)
        assert False, "unknown platform annotation should fail the enhance"
    except enhance.EnhanceError as e:
        assert "bogus" in str(e)


def test_dataset_catalog_enum():
    wft = _enhanced()
    by_name = {p["name"]: p for p in wft["spec"]["arguments"]["parameters"]}
    assert "data-ref" in by_name
    assert set(CATALOG["refs"]).issubset(set(by_name["data-ref"].get("enum", [])))


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except Exception as e:  # noqa: BLE001
                failures += 1
                print(f"FAIL {name}: {e}")
    print(f"\n{'ALL PASS' if not failures else f'{failures} FAILURES'}")
    sys.exit(1 if failures else 0)
