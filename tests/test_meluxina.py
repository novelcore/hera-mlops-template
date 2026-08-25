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
    # step-scoped tags must never survive; expression tags ({{=...}}) are the
    # ONLY templating allowed (they are task-context-valid and self-escaping)
    assert "{{inputs." not in cmd["value"]
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


def test_twin_command_substitutes_task_arguments():
    """F-04: the twin resolves {{inputs.parameters.X}} from the DAG task's
    own arguments (literals or task-context-valid workflow refs), so the
    real invocation reaches MeluXina instead of the nvidia-smi fallback."""
    raw = _raw()
    tpl = next(t for t in raw["spec"]["templates"] if t["name"] == "model-training")
    tpl["container"]["args"] = ["--config", "{{inputs.parameters.config}}",
                                "--epochs", "{{inputs.parameters.epochs}}"]
    dag = next(t for t in raw["spec"]["templates"] if t["name"] == "p")
    task = next(t for t in dag["dag"]["tasks"] if t["name"] == "model-training")
    task["arguments"] = {"parameters": [
        {"name": "config", "value": "{{workflow.parameters.config}}"},
        {"name": "epochs", "value": "5"},
    ]}
    out = enhance(copy.deepcopy(raw), _ctx())
    odag = next(t for t in out["spec"]["templates"] if t["name"] == "p")
    mel = next(t for t in odag["dag"]["tasks"] if t["name"] == "model-training-meluxina")
    val = next(p for p in mel["arguments"]["parameters"]
               if p["name"] == "step-command")["value"]
    # Param-carrying tokens ride as {{=toJson(...)}} expressions so Argo
    # JSON-escapes the substituted value (live wf mgznz 2026-08-25: a plain
    # {{workflow.parameters.config}} inside json.dumps output put raw
    # newlines in the JSON string -> submit pod json.loads died on
    # "invalid control character"). Literals stay plain JSON.
    assert val == ('["python", "-m", "train", "--config", '
                   "{{=toJson(workflow.parameters['config'])}}"
                   ', "--epochs", "5"]')
    assert "{{inputs." not in val
    # simulate Argo substituting a multi-line, quote-carrying value: the
    # result must parse as JSON and preserve the value byte-for-byte
    nasty = 'line1\nline2 "quoted" \\backslash'
    substituted = val.replace(
        "{{=toJson(workflow.parameters['config'])}}", json.dumps(nasty))
    assert json.loads(substituted) == [
        "python", "-m", "train", "--config", nasty, "--epochs", "5"]


def test_meluxina_run_carries_wallet_plumbing():
    """F-08: the submit pod gets the machine-key mount (optional — non-OIDC
    deployments still submit), the Zitadel coordinates, the PUBLIC
    endpoints, and dataset coordinates — never the key into Slurm env."""
    raw = _raw()
    raw["spec"]["arguments"]["parameters"] = [
        {"name": "lakefs-repo", "value": "r"}, {"name": "data-ref", "value": "dev"}]
    out = enhance(copy.deepcopy(raw), _ctx())
    mr = next(t for t in out["spec"]["templates"] if t["name"] == "meluxina-run")
    vols = {v["name"]: v for v in mr["volumes"]}
    assert vols["mlflow-svc"]["secret"]["optional"] is True
    assert vols["mlflow-svc"]["secret"]["secretName"] == "PLACEHOLDER-mlflow-svc"
    mounts = {m["name"]: m for m in mr["container"]["volumeMounts"]}
    assert mounts["mlflow-svc"]["mountPath"] == "/etc/mlflow-svc"
    env = {e["name"]: e.get("value") for e in mr["container"]["env"]}
    assert env["ZITADEL_MACHINE_KEY_FILE"] == "/etc/mlflow-svc/ZITADEL_MACHINE_KEY"
    assert env["ZITADEL_DOMAIN"] == "oidc.internal.invalid"
    assert env["MLFLOW_EXTERNAL_URL"] == "https://mlflow.internal.invalid"
    assert env["LAKEFS_EXTERNAL_URL"] == "https://lakefs.internal.invalid"
    # dataset coordinates prefer the pipeline's own workflow params
    assert env["DATASET_REPO"] == "{{workflow.parameters.lakefs-repo}}"
    assert env["DATASET_REF"] == "{{workflow.parameters.data-ref}}"


def test_dataset_coordinates_fall_back_to_context():
    """enhance() itself injects the lakefs-repo workflow param (earlier
    pass), so repo always resolves via the param; data-ref comes only from
    the app's own authoring — absent, ref stays empty (stage-in disabled,
    not broken)."""
    out = enhance(copy.deepcopy(_raw()), _ctx())
    mr = next(t for t in out["spec"]["templates"] if t["name"] == "meluxina-run")
    env = {e["name"]: e.get("value") for e in mr["container"]["env"]}
    assert env["DATASET_REPO"] == "{{workflow.parameters.lakefs-repo}}"
    assert env["DATASET_REF"] == ""


def test_submit_code_stagein_and_wallet_invariants():
    """T-03/D-04 invariants pinned at source level: the machine key never
    enters the Slurm environment; stage-in fails loudly; the bearer rides
    both MLflow and lakeFS; the batch bind-mounts the staged version."""
    from kubecore.meluxina import MELUXINA_SUBMIT_CODE as src
    # wallet: minted in-cluster, key file read locally, never exported
    assert "mint_wallet" in src and "ZITADEL_MACHINE_KEY_FILE" in src
    assert "ZITADEL_MACHINE_KEY=" not in src  # no key material into env
    assert "MLFLOW_TRACKING_TOKEN=" in src and "LAKEFS_BEARER_TOKEN=" in src
    # stage-in: commit-keyed Lustre cache, loud failure, read-only bind
    assert "data-cache" in src and "|| exit 232" in src
    assert "/kubecore/dataset:ro" in src
    assert "APPTAINERENV_KUBECORE_DATASET_DIR" in src
    # the stage-in payload rides base64 in env and runs on system python3
    assert "STAGEIN_B64" in src and "base64 -d" in src
    # no Argo tags anywhere in the program (survives templating verbatim)
    assert "{{" not in src


def test_stagein_code_is_valid_python():
    """The embedded stage-in payload must parse standalone — it executes on
    the compute node's system python3 with no packaging step in between."""
    import ast
    from kubecore.meluxina import MELUXINA_SUBMIT_CODE
    g = {}
    # executing the module-level defs is network-free; STAGEIN is a constant
    prologue = MELUXINA_SUBMIT_CODE.split("API = ")[0]
    exec(prologue, g)
    ast.parse(g["STAGEIN"])


def test_cmd_json_mixed_and_literal_tokens():
    """_cmd_json escaping table: literal -> plain JSON; pure-tag ->
    toJson(param); mixed -> toJson of single-quoted concatenation (quotes
    in the literal part escaped for the expr string)."""
    from kubecore.meluxina import _cmd_json
    out = _cmd_json(["run", "--epochs={{workflow.parameters.epochs}}",
                     "it's", "{{workflow.parameters.cfg}}"])
    assert out == ('["run", '
                   "{{=toJson('--epochs=' + workflow.parameters['epochs'])}}"
                   ', "it\'s", '
                   "{{=toJson(workflow.parameters['cfg'])}}]")
