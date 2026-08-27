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
                   "{{=toJson(workflow.parameters.config)}}"
                   ', "--epochs", "5"]')
    assert "{{inputs." not in val
    # simulate Argo substituting a multi-line, quote-carrying value: the
    # result must parse as JSON and preserve the value byte-for-byte
    nasty = 'line1\nline2 "quoted" \\backslash'
    substituted = val.replace(
        "{{=toJson(workflow.parameters.config)}}", json.dumps(nasty))
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
    assert "data-cache" in src and "|| fail 232" in src
    # failures self-diagnose into the job comment (live 5143859 was a black
    # box); the waiter cancels a still-PENDING job on SIGTERM and resubmits
    # once with fresh credentials on a pull failure
    assert "fail(){ scontrol update" in src
    assert "signal.signal(signal.SIGTERM" in src
    assert "rc == 231 and not resubmitted" in src
    assert "/kubecore/dataset:ro" in src
    assert "APPTAINERENV_KUBECORE_DATASET_DIR" in src
    # the step runs from the image WORKDIR (Apptainer ignores it otherwise)
    assert 'PWD_OPT="--pwd $STEP_WORKDIR"' in src and "STEP_WORKDIR=" in src
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
                   "{{=toJson('--epochs=' + workflow.parameters.epochs)}}"
                   ', "it\'s", '
                   "{{=toJson(workflow.parameters.cfg)}}]")
    # tasks-output tags (the live mgznz payload: compose-and-validate's
    # multi-line params.yaml output) get the same treatment — hyphenated
    # segments via bracket access, identifier segments via dots
    out2 = _cmd_json(["--params",
                      "{{tasks.compose-and-validate.outputs.parameters.params}}"])
    assert out2 == ('["--params", '
                    "{{=toJson(tasks['compose-and-validate']"
                    ".outputs.parameters.params)}}]")


def test_fetch_workdir_reads_oci_config_via_index(monkeypatch):
    """Apptainer ignores the image WORKDIR (live job 5148071: python -m
    app.entry ran from the host cwd -> 'No module named app'). The submit
    code resolves WorkingDir from the registry: index -> amd64 manifest ->
    config blob; GAR gets the bearer, and any failure degrades to ''."""
    import io, json as _json, urllib.request
    from kubecore.meluxina import MELUXINA_SUBMIT_CODE
    g = {}
    exec(MELUXINA_SUBMIT_CODE.split("API = ")[0], g)
    calls = []
    def fake_urlopen(req, timeout=0):
        calls.append((req.full_url, req.headers.get("Authorization")))
        u = req.full_url
        if u.endswith("/manifests/sha256:img"):
            body = {"manifests": [
                {"digest": "sha256:arm", "platform": {"architecture": "arm64", "os": "linux"}},
                {"digest": "sha256:amd", "platform": {"architecture": "amd64", "os": "linux"}}]}
        elif u.endswith("/manifests/sha256:amd"):
            body = {"config": {"digest": "sha256:cfg"}}
        elif u.endswith("/blobs/sha256:cfg"):
            body = {"config": {"WorkingDir": "/app"}}
        else:
            raise AssertionError(u)
        return io.BytesIO(_json.dumps(body).encode())
    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    wd = g["fetch_workdir"]("europe-central2-docker.pkg.dev/proj/repo/unknown@sha256:img", "tok")
    assert wd == "/app"
    assert calls[0][0] == "https://europe-central2-docker.pkg.dev/v2/proj/repo/unknown/manifests/sha256:img"
    assert all(a == "Bearer tok" for _, a in calls)  # GAR host -> bearer on every call
    # tag refs and failures
    def boom(req, timeout=0):
        raise OSError("registry down")
    monkeypatch.setattr(urllib.request, "urlopen", boom)
    assert g["fetch_workdir"]("zot.internal:5000/unknown:v1", "") == ""


# ---------------------------------------------------------------- time limit


def test_parse_time_limit_units():
    from kubecore.meluxina import parse_time_limit
    assert parse_time_limit("90") == 90
    assert parse_time_limit("90m") == 90
    assert parse_time_limit("12h") == 720
    import pytest
    for bad in ("", "0", "1d", "abc", "12 h"):
        with pytest.raises(ValueError):
            parse_time_limit(bad)


def _rendered_twin(annotation=None):
    from kubecore.meluxina import HPC_ANNOTATION
    raw = _raw()
    if annotation is not None:
        for t in raw["spec"]["templates"]:
            if t["name"] == "model-training":
                t.setdefault("metadata", {}).setdefault("annotations", {})[HPC_ANNOTATION] = annotation
    out = enhance(copy.deepcopy(raw), yaml.safe_load(CTX_PATH.read_text()))
    dag = next(t for t in out["spec"]["templates"] if t["name"] == "p")
    twin = next(t for t in dag["dag"]["tasks"] if t["name"] == "model-training-meluxina")
    run = next(t for t in out["spec"]["templates"] if t["name"] == "meluxina-run")
    return {p["name"]: p["value"] for p in twin["arguments"]["parameters"]}, run


def test_twin_carries_default_time_limit_and_deadline():
    from kubecore.meluxina import DEFAULT_TIME_LIMIT_MINUTES, QUEUE_ALLOWANCE_MINUTES
    args, run = _rendered_twin()
    assert args["time-limit"] == str(DEFAULT_TIME_LIMIT_MINUTES)
    assert args["deadline-seconds"] == str((DEFAULT_TIME_LIMIT_MINUTES + QUEUE_ALLOWANCE_MINUTES) * 60)
    # the template resolves the per-twin deadline and hands the limit to Slurm
    assert run["activeDeadlineSeconds"] == "{{inputs.parameters.deadline-seconds}}"
    names = {p["name"] for p in run["inputs"]["parameters"]}
    assert {"time-limit", "deadline-seconds"} <= names
    env = {e["name"]: e.get("value") for e in run["container"]["env"]}
    assert env["SLURM_TIME_LIMIT"] == "{{inputs.parameters.time-limit}}"
    assert "SLURM_TIME_LIMIT" in run["container"]["command"][2]


def test_twin_honours_step_time_limit_annotation():
    from kubecore.meluxina import QUEUE_ALLOWANCE_MINUTES
    args, _ = _rendered_twin("12h")
    assert args["time-limit"] == "720"
    assert args["deadline-seconds"] == str((720 + QUEUE_ALLOWANCE_MINUTES) * 60)


def test_bad_time_limit_annotation_fails_the_render():
    import pytest
    with pytest.raises(ValueError):
        _rendered_twin("1d")


def test_authoring_hpc_time_limit_validation():
    import pytest
    from kubecore.authoring import AuthoringError, pipeline, step
    for bad in ("1d", 720, "12 h"):
        with pytest.raises(AuthoringError):
            with pipeline("t"):
                step("train", gpu=True, hpc_time_limit=bad)
    with pytest.raises(AuthoringError):
        with pipeline("t"):
            step("prep", hpc_time_limit="1h")  # cpu steps never route to HPC


# ---------------------------------------------------------------- F-05 stage-out


def _raw_with_outputs():
    raw = _raw()
    tpl = next(t for t in raw["spec"]["templates"] if t["name"] == "model-training")
    tpl["outputs"] = {"parameters": [
        {"name": "training-result", "valueFrom": {"path": "/work/output/training-result.json", "default": "{}"}}]}
    dag = next(t for t in raw["spec"]["templates"] if t["name"] == "p")
    reg = next(t for t in dag["dag"]["tasks"] if t["name"] == "model-registration")
    reg["arguments"] = {"parameters": [
        {"name": "training-result", "value": "{{tasks.model-training.outputs.parameters.training-result}}"},
        {"name": "params", "value": "{{tasks.dataset-loading.outputs.parameters.params}}"}]}
    return raw


def test_stage_out_declares_step_outputs_on_the_runner_and_twin():
    out = enhance(_raw_with_outputs(), _ctx())
    run = next(t for t in out["spec"]["templates"] if t["name"] == "meluxina-run")
    outs = {o["name"]: o for o in run["outputs"]["parameters"]}
    assert "slurm-job-id" in outs
    assert outs["training-result"]["valueFrom"] == {"path": "/tmp/outputs/training-result.json", "default": "{}"}
    dag = next(t for t in out["spec"]["templates"] if t["name"] == "p")
    twin = next(t for t in dag["dag"]["tasks"] if t["name"] == "model-training-meluxina")
    args = {p["name"]: p["value"] for p in twin["arguments"]["parameters"]}
    assert args["step-outputs"] == "training-result"
    # qat-finetune carries its own `when` in the fixture, so it is not routed
    # (existing contract) — no twin, nothing to assert about its outputs.
    assert not any(t["name"] == "qat-finetune-meluxina" for t in dag["dag"]["tasks"])
    env = {e["name"]: e.get("value") for e in run["container"]["env"]}
    assert env["STEP_OUTPUTS"] == "{{inputs.parameters.step-outputs}}"


def test_stage_out_rewrites_downstream_references_to_the_twin_that_ran():
    out = enhance(_raw_with_outputs(), _ctx())
    dag = next(t for t in out["spec"]["templates"] if t["name"] == "p")
    reg = next(t for t in dag["dag"]["tasks"] if t["name"] == "model-registration")
    args = {p["name"]: p["value"] for p in reg["arguments"]["parameters"]}
    assert args["training-result"] == (
        "{{=tasks['model-training'].status == 'Skipped' ? "
        "tasks['model-training-meluxina'].outputs.parameters['training-result'] : "
        "tasks['model-training'].outputs.parameters['training-result']}}")
    # references to un-routed steps are left alone
    assert args["params"] == "{{tasks.dataset-loading.outputs.parameters.params}}"


def test_stage_out_batch_and_waiter_invariants():
    from kubecore.meluxina import MELUXINA_SUBMIT_CODE as code
    import ast, re
    assert "-B $KAOS_WORK:/work" in code                      # writable /work for the step
    assert 'rm -rf "$KAOS_WORK"' in code and "[ $rc -eq 0 ] && rm -rf" in code  # scratch kept on failure
    assert "STAGEOUT_B64" in code and "STEP_OUTPUTS" in code
    assert "fetch_outputs()" in code and "/tmp/outputs/" in code
    # Execute the payload DEFINITIONS exactly as python3 -c will on the node
    # (the submit code is raw; nested payloads must survive escape processing),
    # then parse what the batch script will actually run.
    ns = {}
    exec(code.split("\nWALLET = None", 1)[0], ns)
    for payload in ("STAGEIN", "STAGEOUT"):
        ast.parse(ns[payload])
    stageout = ns["STAGEOUT"]
    assert "\r\n" not in stageout.replace("\\r\\n", "")  # escapes intact, no raw CR/LF in literals
    assert "hpc-outputs/" in stageout and "branches/%s/objects" in stageout


def test_slurm_token_is_read_fresh_per_request():
    """The MeluXina JWT rotates every 25 min (60 min lifetime); a queue wait
    longer than that must not blind the waiter (live run tvsm8)."""
    import re
    from kubecore.meluxina import MELUXINA_SUBMIT_CODE as code
    out = enhance(_raw(), _ctx())
    run = next(t for t in out["spec"]["templates"] if t["name"] == "meluxina-run")
    vols = {v["name"]: v for v in run["volumes"]}
    assert vols["meluxina-jwt"]["secret"]["secretName"] == "meluxina-jwt"
    mounts = {m["name"]: m["mountPath"] for m in run["container"]["volumeMounts"]}
    assert mounts["meluxina-jwt"] == "/etc/meluxina-jwt"
    assert "def tok():" in code and "/etc/meluxina-jwt/token" in code
    # every Slurm request builds its headers at call time — no frozen header
    # dict (the stage-in/out payloads keep their own lakeFS bearer `H`)
    slurm = code.split("TOKEN_FILE = ", 1)[1]
    assert not re.search(r"headers=H\b", slurm)
    assert slurm.count("headers=hdrs()") >= 3  # get, submit, cancel
