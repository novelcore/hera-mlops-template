"""PRD-1016: the enhance_hpc pass — MeluXina twin-routing contract."""
import copy
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kubecore.enhance import enhance  # noqa: E402

CTX_PATH = Path(__file__).resolve().parents[1] / "kubecore/local-dev/pipeline-context.yaml"


def _raw():
    return {
        "apiVersion": "argoproj.io/v1alpha1", "kind": "WorkflowTemplate",
        "metadata": {"name": "x"},
        "spec": {"entrypoint": "p", "arguments": {"parameters": []}, "templates": [
            {"name": "p", "dag": {"tasks": [
                {"name": "dataset-loading", "template": "dataset-loading"},
                {"name": "model-training", "template": "model-training",
                 "depends": "dataset-loading"},
                {"name": "qat-finetune", "template": "qat-finetune",
                 "depends": "model-training",
                 "when": "{{=workflow.parameters.quantization-mode == 'qat'}}"},
                {"name": "model-registration", "template": "model-registration",
                 "depends": "model-training && (qat-finetune.Succeeded || qat-finetune.Skipped)"},
            ]}},
            {"name": "dataset-loading", "container": {"image": "i", "command": ["python", "-m", "load"], "resources": {"requests": {"cpu": "1"}}}},
            {"name": "model-training", "container": {"image": "i", "command": ["python", "-m", "train"], "resources": {"requests": {"nvidia.com/gpu": 1}}}},
            {"name": "qat-finetune", "container": {"image": "i", "command": ["python", "-m", "qat"], "resources": {"requests": {"nvidia.com/gpu": 1}}}},
            {"name": "model-registration", "container": {"image": "i", "command": ["python", "-m", "reg"], "resources": {"requests": {"cpu": "1"}}}},
        ]},
    }


def _ctx(hpc=True):
    ctx = yaml.safe_load(CTX_PATH.read_text())
    if not hpc:
        ctx.pop("hpc", None)
    return ctx


def test_hpc_routes_gpu_step_behind_target_param():
    out = enhance(copy.deepcopy(_raw()), _ctx())
    spec = out["spec"]
    params = {p["name"]: p for p in spec["arguments"]["parameters"]}
    tpls = {t["name"]: t for t in spec["templates"]}
    tasks = {t["name"]: t for t in tpls["p"]["dag"]["tasks"]}

    assert params["target"]["enum"] == ["gcp", "meluxina"]
    assert "!= 'meluxina'" in tasks["model-training"]["when"]
    mel = tasks["model-training-meluxina"]
    assert "== 'meluxina'" in mel["when"]
    assert mel["template"] == "meluxina-run"
    # quantization-gated GPU step keeps in-cluster-only behaviour this slice
    assert "qat-finetune-meluxina" not in tasks
    # downstream depends gate on the Succeeded||Skipped twin pair
    reg = tasks["model-registration"]["depends"]
    assert "(model-training.Succeeded || model-training.Skipped)" in reg
    assert "model-training-meluxina.Succeeded" in reg
    # release gate: container template, never script; no duplicate params
    mr = tpls["meluxina-run"]
    assert "container" in mr and "script" not in mr
    names = [p["name"] for p in spec["arguments"]["parameters"]]
    assert len(names) == len(set(names))
    # idempotent submit + operational lessons encoded in the program
    src = mr["container"]["command"][2]
    assert "adopting existing job" in src and "sif-cache" in src and "bash -l" in src
    assert "{{" not in src


def test_twin_arguments_never_carry_step_scoped_tags():
    """A templated step arg ({{inputs.parameters.params}}) copied into a task
    argument fails Argo spec validation for the WHOLE WorkflowTemplate —
    every submission, gcp target included (live incident 2026-08-25)."""
    raw = _raw()
    tpl = next(t for t in raw["spec"]["templates"] if t["name"] == "model-training")
    tpl["container"]["args"] = ["{{inputs.parameters.params}}", "--epochs", "1"]
    out = enhance(copy.deepcopy(raw), _ctx())
    dag = next(t for t in out["spec"]["templates"] if t["name"] == "p")
    mel = next(t for t in dag["dag"]["tasks"] if t["name"] == "model-training-meluxina")
    cmd = next(p for p in mel["arguments"]["parameters"] if p["name"] == "step-command")
    assert "{{" not in cmd["value"]
    assert json.loads(cmd["value"]) == ["python", "-m", "train", "--epochs", "1"]


def test_submit_code_normalizes_tag_digest_refs(monkeypatch):
    """Apptainer rejects name:tag@digest (live job 5140397); the submit code
    must normalize to digest-only before docker:// pull."""
    from kubecore.meluxina import MELUXINA_SUBMIT_CODE
    for k, v in {"SLURM_TOKEN": "t", "WF_UID": "u", "STEP_NAME": "s",
                 "IMAGE_REF": "reg.example.com/unknown:v1-abc@sha256:deadbeef",
                 "STEP_COMMAND": "[]"}.items():
        monkeypatch.setenv(k, v)
    # execute only the prologue up to the normalization (stop before network)
    prologue = MELUXINA_SUBMIT_CODE.split("jid = None")[0]
    g = {}
    exec(prologue, g)
    assert g["img"] == "reg.example.com/unknown@sha256:deadbeef"


def test_registry_token_only_for_gar_hosts():
    """A GCP token presented to Zot turns anonymous-OK pulls into 401
    (live job 5140432); the batch must gate credentials on *-docker.pkg.dev."""
    from kubecore.meluxina import MELUXINA_SUBMIT_CODE
    assert '-docker.pkg.dev/*)' in MELUXINA_SUBMIT_CODE
    # the export must live INSIDE the case arm, never unconditional
    import re
    line = next(l for l in MELUXINA_SUBMIT_CODE.split(chr(92)+"n")
                if 'APPTAINER_DOCKER_USERNAME' in l)
    assert 'docker.pkg.dev' in line


def test_submit_code_shell_quotes_command_tokens(monkeypatch):
    """argv tokens with spaces/quotes must survive the env->sh -c ride
    (live job 5140493: a python -c one-liner arrived as bare words)."""
    from kubecore.meluxina import MELUXINA_SUBMIT_CODE
    import json as _json
    for k, v in {"SLURM_TOKEN": "t", "WF_UID": "u", "STEP_NAME": "s",
                 "IMAGE_REF": "r/x:t",
                 "STEP_COMMAND": _json.dumps(["python", "-c", "import torch; print(1)"])}.items():
        monkeypatch.setenv(k, v)
    prologue = MELUXINA_SUBMIT_CODE.split("jid = None")[0]
    g = {}
    exec(prologue, g)
    assert g["cmd"] == "python -c 'import torch; print(1)'"


def test_hpc_absent_leaves_output_untouched():
    out = enhance(copy.deepcopy(_raw()), _ctx(hpc=False))
    s = json.dumps(out)
    assert "meluxina" not in s and '"target"' not in s
