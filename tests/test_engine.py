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
    # arguments.parameters carry `value` (not `default`) so Argo accepts the WFT
    assert amp.value in ("true", "false")
    assert amp.enum is None, "booleans must NOT carry an enum (O9 dead)"


def test_reads_gate_fires_on_missing_section():
    try:
        derive_tree.validate_reads({"badstep": ["nonexistent-section"]}, ["data", "train"])
        assert False, "reads= gate should have raised"
    except derive_tree.DeriveError as e:
        assert "nonexistent-section" in str(e) and "badstep" in str(e)


def test_reads_gate_catches_schema_backed_key_rename():
    """A schema-backed section (data/quantization) whose YAML key is renamed
    still appears in the composed tree (schema backfill) — the gate must catch
    that via the on-disk source check, else the rename silently drops values."""
    # 'data' is schema-backed and present in the tree, but simulate its config
    # source being absent (renamed key) by asserting the source-check logic.
    assert "data" in derive_tree.SCHEMA_BACKED
    assert derive_tree._has_config_source("data"), "data.yaml should be a real source normally"
    # If a step reads 'data' but the section has no real source, gate must fire.
    import unittest.mock as mock
    with mock.patch.object(derive_tree, "_has_config_source", lambda s: s != "data"):
        try:
            derive_tree.validate_reads({"dataset-loading": ["data"]},
                                       ["data", "train", "quantization"])
            assert False, "gate should fire for schema-backed section with no source"
        except derive_tree.DeriveError as e:
            assert "structured-config schema but no config/data.yaml" in str(e)


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


def test_every_arguments_param_is_submittable():
    """Argo REJECTS a workflow whose spec.arguments.parameters has an entry
    lacking `value`/`valueFrom` (a `default` alone is invalid there). Every
    derived + enhancer-injected form param must carry value/valueFrom, else the
    rendered WFT can't be submitted at all. (Regression: caught live when a run
    failed 'spec.arguments.image_processing.value ... is required'.)"""
    wft = _enhanced()
    bad = [p["name"] for p in wft["spec"]["arguments"]["parameters"]
           if "value" not in p and "valueFrom" not in p]
    assert not bad, f"arguments params missing value/valueFrom (unsubmittable): {bad}"


def test_wft_name_forced_app_scoped_no_cross_app_collision():
    """The enhancer MUST force metadata.name to {app}-pipeline, overriding
    whatever pipeline.py declared — so a template's hardcoded pipeline name
    (copied into every seeded app) can't make one app's render overwrite
    another app's WFT in the shared ml-{project} namespace."""
    raw = _render()
    hardcoded = raw["metadata"]["name"]  # whatever pipeline.py wrote
    # enhance with a DIFFERENT app identity than the hardcoded name implies
    ctx = dict(CONTEXT)
    ctx["app"] = "some-other-app"
    out = enhance.enhance(_render(), ctx, CATALOG)
    assert out["metadata"]["name"] == "some-other-app-pipeline", \
        f"expected forced name some-other-app-pipeline, got {out['metadata']['name']}"
    assert out["metadata"]["name"] != hardcoded or hardcoded == "some-other-app-pipeline", \
        "enhancer must not trust the developer's pipeline name for a different app"
    # image ConfigMap must also be that app's, never another app's
    cms = {p.get("valueFrom", {}).get("configMapKeyRef", {}).get("name")
           for p in out["spec"]["arguments"]["parameters"]}
    cms.discard(None)
    assert cms == {"some-other-app-pipeline-images"}, f"image CM not app-scoped: {cms}"


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


def _dag_tasks(wft):
    dag = next(t for t in wft["spec"]["templates"] if t["name"] == "main")
    return {t["name"]: t for t in dag["dag"]["tasks"]}


def test_first_two_steps_present_and_ordered():
    """The two real first steps are wired: compose-and-validate (platform) ->
    config-validation -> dataset-loading -> model-training."""
    wft = _enhanced()
    tasks = _dag_tasks(wft)
    for name in ("compose-and-validate", "config-validation", "dataset-loading",
                 "model-training"):
        assert name in tasks, f"missing DAG task {name}"
    # config-validation gates on compose; dataset-loading gates on both;
    # model-training gates on dataset-loading (all pre-GPU work done first).
    assert "compose-and-validate" in tasks["config-validation"]["depends"]
    assert "config-validation" in tasks["dataset-loading"]["depends"]
    assert "dataset-loading" in tasks["model-training"]["depends"]


def test_dataset_loading_emits_step_outputs():
    """dataset-loading exposes the data-yaml + manifest-summary outputs
    model-training consumes (the object-store handoff, not shared disk)."""
    wft = _enhanced()
    tmpl = next(t for t in wft["spec"]["templates"] if t["name"] == "dataset-loading")
    outs = {p["name"] for p in tmpl.get("outputs", {}).get("parameters", [])}
    assert {"data-yaml", "manifest-summary"}.issubset(outs), outs


def test_every_declared_step_has_a_buildable_dockerfile():
    """Robustness gate: every step the pipeline declares must have a buildable
    steps/<dir>/Dockerfile (or the image escape-hatch), so a forgotten
    Dockerfile is caught at render time — not as a run-time ImagePullBackOff."""
    wft = _enhanced()
    steps = [t for t in wft["spec"]["templates"] if "container" in t]
    missing = []
    for t in steps:
        annotations = t.get("metadata", {}).get("annotations", {})
        if "platform.kubecore.io/image" in annotations:
            continue
        dockerfile = ROOT / "steps" / t["name"].replace("-", "_") / "Dockerfile"
        if not dockerfile.is_file():
            missing.append(t["name"])
    assert not missing, f"steps with no buildable Dockerfile: {missing}"


def test_first_two_steps_get_platform_env():
    """The platform auto-injects data-source env into the first two steps —
    a developer wires no endpoints/credentials."""
    wft = _enhanced()
    for step_name in ("config-validation", "dataset-loading"):
        tmpl = next(t for t in wft["spec"]["templates"] if t["name"] == step_name)
        env = {e["name"] for e in tmpl["container"].get("env", [])}
        assert {"MLFLOW_TRACKING_URI", "LAKEFS_ENDPOINT", "LAKEFS_ACCESS_KEY"}.issubset(env), \
            f"{step_name} missing injected platform env: {env}"


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
