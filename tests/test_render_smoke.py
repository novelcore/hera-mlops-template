"""Smoke test: the template renders + enhances to a valid WorkflowTemplate.

Sanity check that ships with the template — pipeline.py builds, the config tree
composes, and the enhancer produces a runnable WFT with the example hello-world
step. Run:  python -m pytest tests/ -v
"""

import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from kubecore import enhance  # noqa: E402

CONTEXT = yaml.safe_load((ROOT / "kubecore" / "local-dev" / "pipeline-context.yaml").read_text())
CATALOG = yaml.safe_load((ROOT / "kubecore" / "local-dev" / "dataset-catalog.yaml").read_text())


def _render():
    import runpy
    ns = runpy.run_path(str(ROOT / "pipeline.py"), run_name="__pipeline__")
    return yaml.safe_load(ns["p"].wt.to_yaml())


def _enhanced():
    return enhance.enhance(_render(), CONTEXT, CATALOG)


def test_pipeline_renders():
    raw = _render()
    assert raw["kind"] == "WorkflowTemplate"
    assert raw["spec"]["templates"]


def test_enhance_produces_hello_world():
    steps = [t["name"] for t in _enhanced()["spec"]["templates"] if "container" in t]
    assert "hello-world" in steps
    assert "compose-and-validate" in steps


def test_config_tree_becomes_form():
    params = {p["name"] for p in _enhanced()["spec"].get("arguments", {}).get("parameters", [])}
    assert "experiment-name" in params
