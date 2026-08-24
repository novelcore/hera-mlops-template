"""PRD-1016 — the MeluXina HPC leg the `platform.kubecore.io/hpc` annotation reserved.

`enhance_hpc` runs as the last enhancement pass. Active only when the pipeline
context carries `hpc.enabled` (Derive idiom: the operator sets it iff the
parent KubePool is HPC-ready, so non-HPC pools render byte-identically).

What it does (F-02/F-03/F-06):
  * re-emits the `target` submit-form parameter (gcp | <provider>);
  * every GPU step WITHOUT an existing task `when` (quantization-gated steps
    keep in-cluster-only behaviour this slice) gains a `-meluxina` twin DAG
    task with complementary `when:` clauses on `target`;
  * downstream `depends` tokens that reference a routed step are rewritten to
    gate on the Succeeded||Skipped twin pair (exactly one twin runs);
  * one `meluxina-run` CONTAINER template is appended (the release gate
    rejects script templates): an idempotent Slurm REST submit keyed on
    workflow.uid — a retried pod ADOPTS the queued/running job instead of
    double-submitting a full-node run — then a poll to terminal state,
    emitting `slurm-job-id` as an output.

Operational notes baked in from the live F-01/F-02 shakedown (2026-08):
  * the rotating Slurm JWT arrives via the `meluxina-jwt` Secret (PRD-1016
    F-07 ExternalSecret in the ml namespace, 5m refresh);
  * the batch script restores the Lmod init (`bash -l` + explicit source) —
    Slurm's mandatory `environment` field wipes it;
  * images pull through a digest-keyed Lustre SIF cache (D-02: cold ~8 min,
    warm ~7 s, measured on mel2107); registry auth is a best-effort
    metadata-server access token (GAR);
  * data-plane env (lakeFS/MLflow endpoints + wallet) lands with F-04/F-08 —
    until then a meluxina run proves scheduling+image+GPU end-to-end.
"""

import json
import re

# The submit/poll program the meluxina-run container executes via
# `python3 -c`. Plain string — no Argo tags inside (all run-time inputs
# arrive via env), so it survives every templating layer verbatim.
MELUXINA_SUBMIT_CODE = r'''
import json, os, sys, time, urllib.request
API = 'https://slurm-api.lxp.lu/slurm/v0.0.44'
SDB = 'https://slurm-api.lxp.lu/slurmdb/v0.0.44'
TOK = os.environ['SLURM_TOKEN'].strip()
H = {'X-SLURM-USER-NAME': 'u104378', 'X-SLURM-USER-TOKEN': TOK,
     'Content-Type': 'application/json'}

def get(url):
    return json.load(urllib.request.urlopen(
        urllib.request.Request(url, headers=H), timeout=30))

jobname = 'kaos-' + os.environ['WF_UID'] + '-' + os.environ['STEP_NAME']
img = os.environ['IMAGE_REF']
# Apptainer rejects tag@digest refs ('Docker references with both a tag and
# digest are currently not supported', live job 5140397). The platform pins
# images as name:tag@sha256:... (Zot-retention lesson) — normalize to the
# digest-only form, which Apptainer accepts and which is the stronger pin.
if '@' in img:
    name, digest = img.split('@', 1)
    if ':' in name.rsplit('/', 1)[-1]:
        name = name.rsplit(':', 1)[0]
    img = name + '@' + digest
cmd = ' '.join(json.loads(os.environ.get('STEP_COMMAND') or '[]'))

jid = None
for j in (get(API + '/jobs').get('jobs') or []):
    st = j.get('job_state') or []
    if j.get('name') == jobname and any(
            x in ('PENDING', 'RUNNING', 'SUSPENDED') for x in st):
        jid = j.get('job_id')
        print('adopting existing job', jid, flush=True)
        break

if jid is None:
    reg = ''
    try:
        r = urllib.request.Request(
            'http://metadata.google.internal/computeMetadata/v1/instance/'
            'service-accounts/default/token',
            headers={'Metadata-Flavor': 'Google'})
        reg = json.load(urllib.request.urlopen(r, timeout=5)).get(
            'access_token', '')
    except Exception as e:
        print('no registry token from metadata server (anonymous pull):', e,
              flush=True)
    batch = '\n'.join([
        '#!/bin/bash -l',
        'set +e',
        'for f in /usr/share/lmod/lmod/init/bash /etc/profile.d/lmod.sh; do'
        ' [ -r "$f" ] && source "$f" && break; done',
        'module load Apptainer 2>/dev/null || module load apptainer 2>/dev/null',
        'command -v apptainer >/dev/null || exit 210',
        'SCR=/project/scratch/p201342',
        'export APPTAINER_CACHEDIR=$SCR/kaos-apptainer-cache'
        ' APPTAINER_TMPDIR=$SCR/kaos-tmp',
        'mkdir -p $APPTAINER_CACHEDIR $APPTAINER_TMPDIR $SCR/sif-cache',
        'KEY=$(printf %s "$IMAGE_REF" | sha256sum | cut -c1-16)',
        'SIF=$SCR/sif-cache/$KEY.sif',
        'if [ ! -f "$SIF" ]; then',
        # GCP token ONLY for GAR hosts: presenting it to Zot turns an
        # anonymous-OK pull into 401 authentication required (live job
        # 5140432 vs the anonymous F-01 pull that worked on the same repo).
        '  case "$IMAGE_REF" in *-docker.pkg.dev/*)'
        ' [ -n "$REG_TOKEN" ] && export'
        ' APPTAINER_DOCKER_USERNAME=oauth2accesstoken'
        ' APPTAINER_DOCKER_PASSWORD=$REG_TOKEN;; esac',
        '  apptainer pull "$SIF" docker://$IMAGE_REF || exit 231',
        'fi',
        'if [ -n "$STEP_CMD" ]; then apptainer exec --nv "$SIF" /bin/sh -c'
        ' "$STEP_CMD"; else apptainer exec --nv "$SIF" nvidia-smi -L; fi',
    ])
    env = ['PATH=/usr/bin:/bin:/usr/local/bin', 'HOME=/home/users/u104378',
           'USER=u104378', 'IMAGE_REF=' + img, 'REG_TOKEN=' + reg,
           'STEP_CMD=' + cmd]
    body = {'job': {'name': jobname, 'partition': 'gpu',
                    'account': 'p201342', 'qos': 'default', 'time_limit': 240,
                    'current_working_directory': '/home/users/u104378',
                    'environment': env, 'tasks': 1, 'nodes': '1'},
            'script': batch}
    req = urllib.request.Request(API + '/job/submit',
                                 data=json.dumps(body).encode(),
                                 headers=H, method='POST')
    resp = json.load(urllib.request.urlopen(req, timeout=30))
    jid = resp.get('job_id')
    print('submitted job', jid, 'errors:', resp.get('errors'), flush=True)
    if not jid:
        sys.exit(1)

open('/tmp/slurm-job-id', 'w').write(str(jid))
TERMINAL = ('COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT', 'NODE_FAIL',
            'PREEMPTED', 'OUT_OF_MEMORY')
while True:
    time.sleep(30)
    try:
        job = (get(SDB + '/job/' + str(jid)).get('jobs') or [{}])[0]
        st = (job.get('state') or {}).get('current') or []
        print('state:', st, flush=True)
        if any(x in TERMINAL for x in st):
            rc = ((job.get('exit_code') or {}).get('return_code')
                  or {}).get('number')
            print('terminal:', st, 'exit:', rc, flush=True)
            sys.exit(0 if ('COMPLETED' in st and rc in (0, None)) else 1)
    except SystemExit:
        raise
    except Exception as e:
        print('poll error (transient):', e, flush=True)
'''


def enhance_hpc(spec: dict, ctx: dict, steps: list, gpu_step_names: set) -> None:
    """Route GPU steps to MeluXina behind the `target` param (module doc)."""
    hpc = ctx.get("hpc") or {}
    if not hpc.get("enabled"):
        return
    provider = hpc.get("provider", "meluxina")

    parameters = spec["arguments"]["parameters"]
    if not any(p.get("name") == "target" for p in parameters):
        parameters.append({
            "name": "target", "value": "gcp", "enum": ["gcp", provider],
            "description": ("Computation target for this run. gcp = "
                            "in-cluster pools; %s = HPC burst (GPU training "
                            "runs on %s via Slurm)." % (provider, provider)),
        })

    entry = spec.get("entrypoint")
    dag_tpl = next((t for t in spec.get("templates", [])
                    if t.get("name") == entry and "dag" in t), None)
    if dag_tpl is None:
        return
    tasks = dag_tpl["dag"]["tasks"]
    by_tpl = {s["name"]: s for s in steps}

    routed = []
    for task in list(tasks):
        step = by_tpl.get(task.get("template"))
        if step is None or step["name"] not in gpu_step_names or task.get("when"):
            continue
        task["when"] = "{{=workflow.parameters.target != '%s'}}" % provider
        container = step.get("container") or {}
        # step-command: only tag-free tokens survive. Hera-authored args carry
        # Argo tags ({{inputs.parameters.params}}) that resolve against the
        # STEP template's inputs — copied into a task argument they fail spec
        # validation for the entire WorkflowTemplate (live-caught 2026-08-25:
        # one templated token bricked every submission, gcp runs included).
        # Templated tokens are dropped; an empty result falls back to the
        # in-template nvidia-smi probe until F-04 wires the real invocation.
        cmd = [tok for tok in ((container.get("command") or [])
                               + (container.get("args") or []))
               if "{{" not in tok]
        twin = {
            "name": task["name"] + "-" + provider,
            "template": "meluxina-run",
            "when": "{{=workflow.parameters.target == '%s'}}" % provider,
            "arguments": {"parameters": [
                {"name": "step-name", "value": task["name"]},
                {"name": "image", "value": container.get("image", "")},
                {"name": "step-command", "value": json.dumps(cmd)},
            ]},
        }
        if task.get("depends"):
            twin["depends"] = task["depends"]
        tasks.append(twin)
        routed.append(task["name"])

    if not routed:
        return

    # A bare task token in Argo depends grammar means Succeeded; a routed dep
    # is now a twin pair where exactly one twin runs and the other is Skipped.
    for task in tasks:
        dep = task.get("depends")
        if not dep or task["name"].endswith("-" + provider):
            continue
        for r in routed:
            if task["name"] == r:
                continue
            pair = ("((%s.Succeeded || %s.Skipped) && "
                    "(%s-%s.Succeeded || %s-%s.Skipped))"
                    % (r, r, r, provider, r, provider))
            dep = re.sub(r"(?<![\w.-])%s(?![\w.-])" % re.escape(r), pair, dep)
        task["depends"] = dep

    spec["templates"].append({
        "name": "meluxina-run",
        "inputs": {"parameters": [
            {"name": "step-name"}, {"name": "image"}, {"name": "step-command"}]},
        "activeDeadlineSeconds": 14400,
        "metadata": {"labels": {"platform.kubecore.io/compute-type": "hpc"}},
        "container": {
            "image": "python:3.12-alpine",
            "command": ["python3", "-c", MELUXINA_SUBMIT_CODE],
            "env": [
                {"name": "SLURM_TOKEN", "valueFrom": {"secretKeyRef": {
                    "name": "meluxina-jwt", "key": "token"}}},
                {"name": "WF_UID", "value": "{{workflow.uid}}"},
                {"name": "STEP_NAME", "value": "{{inputs.parameters.step-name}}"},
                {"name": "IMAGE_REF", "value": "{{inputs.parameters.image}}"},
                {"name": "STEP_COMMAND", "value": "{{inputs.parameters.step-command}}"},
            ],
        },
        "outputs": {"parameters": [{"name": "slurm-job-id",
                                    "valueFrom": {"path": "/tmp/slurm-job-id"}}]},
    })
