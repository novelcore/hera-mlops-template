# Pipeline Developer Guide

The operating manual for authoring and running your pipeline. You need
Python and YAML. You do not need Kubernetes — anything you don't see in
this guide is handled for you.

---

## 1. Mental model

Your repo is:

```
your-repo/
├── pipeline.py              # the DAG: steps, what they read, how they connect
├── config/                  # THE experiment config tree (all your parameters)
│   ├── config.yaml          #   defaults list (which group options are default)
│   ├── experiment.yaml      #   a section of scalar leaves (add your own)
│   └── train/optimizer/{sgd,adam,adamw}.yaml     # a config GROUP (example)
├── kubecore/                # platform-owned helpers (seeded — do not edit)
└── steps/
    ├── hello_world/         # Dockerfile + entry point (your code)
    └── ...
```

**One rule to remember: the config tree IS the submit form.** Every
scalar value in `config/` becomes a form field automatically. Every
group directory becomes a dropdown. Add a leaf → a field appears.
There is no parameter wiring anywhere else.

**What happens on push:** push to `dev` → the platform builds your step
images and releases a new pipeline version (~minutes).

**What happens on submit:** the values you set on the Argo form are
composed with your tree into ONE resolved `params.yaml` — by the
pipeline's first step, before any real compute — and every step receives
it. That file *is* the experiment: it's also archived to MLflow so any
run can be reproduced exactly.

```
   you edit                 platform                you run
┌─────────────┐   push   ┌──────────────┐  ~min  ┌──────────────────────┐
│ config/     │ ───────► │ build+release│ ─────► │ Argo UI submit form  │
│ pipeline.py │          └──────────────┘        │ = your config tree   │
│ steps/*/    │                                  └──────────┬───────────┘
└─────────────┘                                             ▼
                                    [compose] → params.yaml → all steps
```

---

## 2. Anatomy of the config tree

```yaml
# config/train.yaml (excerpt — a SECTION with scalar leaves)
train:
  epochs: 100        # total training epochs
  batch_size: 16     # -1 for auto-batch
  cos_lr: true       # plain booleans are fine
  loss:
    weight: 12.0     # nested scalars work too
```

```yaml
# config/train/optimizer/sgd.yaml (one OPTION of a config GROUP)
name: "SGD"
lr: 0.01
momentum: 0.937
weight_decay: 0.0005
```

```yaml
# config/config.yaml — the defaults list picks each group's default
defaults:
  - _self_
  - experiment
  - train
  - train/optimizer: sgd
  # ...
```

What each thing becomes on the submit form:

| you write in config/ | what happens |
|---|---|
| a scalar leaf (`train.epochs: 100`) | a form field named by its path, dots→dashes: **`train-epochs`**, default `100` |
| a nested scalar (`train.loss.weight`) | same rule: **`train-loss-weight`** |
| a **boolean** (`train.amp: true`) | a plain form field (`true`/`false`) — no special treatment |
| a **group directory** (`train/optimizer/*.yaml`) | ONE dropdown (**`train-optimizer`**) whose options are the file names; picking one swaps the whole subtree |
| a **list** or other complex structure | NOT a form field — change it by swapping a group option or via the ADVANCED override |

Some fields are dropdowns with a fixed set of valid values because the
platform declares them (e.g. `data-source`: lakefs/s3) — submitting
anything else fails the run immediately, with the allowed values listed
in the error.

**The boolean non-rule:** booleans are just values now. Write
`enabled: true`, submit `false`, your step reads a real `False`. (If you
used the previous Typer-based version of this platform: the
true/false-Enum workaround is gone.)

---

## 3. Add a parameter

Add a leaf to the tree. That's the whole task.

```yaml
# config/train.yaml
train:
  ...
  gradient_clip: 10.0   # NEW
```

Push. The form now has `train-gradient_clip` with default `10.0`, and
every step that declares `reads=["train"]` sees
`cfg["train"]["gradient_clip"]` in its params — no pipeline.py change,
no step-code change (unless you want to act on it), no wiring.

To add a *behavior variant* instead of a single knob, add a new option
file to a group:

```yaml
# config/train/optimizer/lion.yaml  (NEW file = new dropdown option)
name: "Lion"
lr: 0.0003
momentum: 0.95
weight_decay: 0.01
```

Push. The `train-optimizer` dropdown now offers `lion`.

---

## 4. Anatomy of pipeline.py (the whole file)

`pipeline.py` declares structure and nothing else. This is the real,
complete file from this repo — one example step:

```python
from kubecore.authoring import pipeline, step  # platform-owned, do not edit

with pipeline("ml-pipeline") as p:  # platform renames to {app}-pipeline at release
    hello = step("hello-world", reads=["experiment"])
```

As your pipeline grows you add steps and wire them with `needs=`. An
illustrative multi-step DAG using every argument:

```python
with pipeline("ml-pipeline") as p:  # platform renames to {app}-pipeline at release
    prep = step("prepare", reads=["experiment", "data"],
                outputs=["prepared-manifest"])
    run = step("run", gpu=True, needs=[prep],
               reads=["experiment", "train"],
               when="{{workflow.parameters.train-enabled}} == true",
               outputs=["run-result"])
    report = step("report", needs=[prep, run],
                  reads=["experiment"])
```

Per step:

- **`reads=[...]`** — which top-level config sections this step
  consumes. This is a declared contract: if a section you read is ever
  renamed or deleted, the *release fails* with a clear error instead of
  your step crashing at 2am.
- **`gpu=True`** — the GPU switch. One flag; node routing, GPU
  reservation, everything else is automatic.
- **`needs=[...]`** — ordering + data: every output of a needed step is
  fed to yours automatically.
- **`outputs=[...]`** — small result files your step writes to
  `/work/output/<name>.json`.
- **`when=...`** — conditional execution keyed on any form parameter
  (parameter name = config path with dashes: `train.enabled` →
  `train-enabled`).

---

## 5. Add a step

> **New here? Read [ADD-A-STEP.md](ADD-A-STEP.md)** — a complete, copy-paste,
> zero-assumptions walkthrough (every command, every gotcha, how to verify).
> The version below is the quick reference once you know the shape.

Three small edits — say we add `model-evaluation` after training:

1. **config**: add a section for it (if it needs its own knobs):

   ```yaml
   # config/evaluation.yaml
   evaluation:
     split: "val"
     iou_threshold: 0.5
   ```

   and register it in `config/config.yaml`'s defaults list: `- evaluation`.

2. **code**: `mkdir steps/model_evaluation` with a Dockerfile and an
   entry point that reads its slice (this is the real pattern from
   `steps/hello_world/app/entry.py`):

   ```python
   import argparse
   import yaml

   READS = ["evaluation"]

   def main() -> None:
       parser = argparse.ArgumentParser()
       parser.add_argument("--params", required=True)
       args, _ = parser.parse_known_args()
       cfg = yaml.safe_load(args.params)
       config = {section: cfg[section] for section in READS}
       ...

   if __name__ == "__main__":
       main()
   ```

3. **pipeline.py**: one line —

   ```python
   evaluate = step("model-evaluation", reads=["evaluation"], needs=[train])
   ```

Push. Done. The form gains `evaluation-split` and
`evaluation-iou_threshold` (plus automatic per-run sizing knobs for the
new step), and it runs after training with `training-result` fed in.

---

## 6. Swap behavior wholesale (groups)

Groups exist so you can swap an entire parameter subtree in one move —
on the form (pick `adamw` in the `train-optimizer` dropdown) or as the
new default in the repo (edit one line of `config/config.yaml`):

```yaml
defaults:
  - train/optimizer: adamw     # was: sgd
```

Everything inside the option file (`lr`, `momentum`, `weight_decay`, …)
changes together — no field-by-field editing, no forgotten stragglers.
Use groups for anything with internally-consistent presets: optimizers,
augmentation recipes, callbacks, export settings.

**How a group interacts with its individual fields on the form:**
fields you leave untouched follow the group you selected; fields you
explicitly change win over the group. Picking `adamw` gives you the
full AdamW preset; picking `adamw` *and* setting
`train-optimizer-lr` to `0.005` gives you AdamW with your lr.

---

## 7. Delete a step

1. Remove its `step(...)` line; remove it from any other step's
   `needs=[...]`.
2. Delete `steps/<name>/`; delete (or keep) its config section — if you
   delete the section, also remove it from `config/config.yaml`'s
   defaults list and from any remaining step's `reads=`.

Miss the `reads=` cleanup and the release fails immediately with the
exact section and step named (see §10) — nothing silently breaks.

---

## 8. GPU & resources

- `step("run", gpu=True, ...)` is the entire GPU
  configuration.
- Per-run sizing is automatic: every step gets `{step}-cpu` and
  `{step}-mem` fields on the submit form (defaults = a whole node).
  Dial a smoke-run down at submit time; you never configure this.

---

### 8.1 Running steps on MeluXina (HPC classes)

When the project's pool is HPC-enabled, every step's `{step}-class` dropdown
also offers the MeluXina classes — `meluxina-gpu` (4× A100-40GB), `meluxina-cpu`
(2× EPYC 7452, 128 cores, 512 GB), `meluxina-largemem` (4 TB). Pick one for a
step and that step runs as a Slurm job on MeluXina; every other step keeps its
in-cluster class. There is no global switch: placement is per step, per run
(a GPU step on `meluxina-gpu`, a lighter step on `gpu-small`, another step
in-cluster is a normal combination).

What the platform does for an HPC-placed step: waits in the queue on your
behalf, stages the dataset ref to Lustre (cached), pulls the step image into an
Apptainer SIF (cached), runs the same command the pod would run, uploads the
step's declared outputs to lakeFS and hands them to the next step exactly as
if it had run in the cluster. MLflow and lakeFS are reached through the
public endpoints with a short-lived token minted per job.

Give long steps a wall-clock limit (default 4 h):

```python
run = step("run", gpu=True, needs=[prep], hpc_time_limit="12h",
           reads=["experiment", "train"])
```

`hpc_time_limit` takes minutes or hours (`"90m"`, `"12h"`, `"720"`); Slurm kills
the job when it elapses. The platform adds a queue allowance on top for its own
wait budget; a run that outlives that budget keeps running on MeluXina but is
no longer observed by the pipeline. Queue time is the one variable you don't
control — expect minutes to hours depending on the partition's load.

## 9. Data flow between steps

**Parameters (small)**: declare `outputs=["training-result"]`, write
`/work/output/training-result.json`, and every step that `needs=` yours
receives it automatically (`--training-result <content>`). Keep these
small (KBs) — manifests, metrics, references.

**Your experiment config**: arrives resolved in `--params`; read only
your declared sections.

**THE DATA RULE — steps do not share a filesystem.** Each step runs on
its own machine with its own disk. Move data like this:

| data | mechanism |
|---|---|
| small manifests / results | `outputs=` parameters (above) |
| datasets | lakeFS — resolve a ref, stream; endpoints in `cfg["platform"]["lakefs"]` and env |
| checkpoints | the checkpoint bucket — `cfg["platform"]["checkpoints"]["bucket"]/["prefix"]` |
| metrics, artifacts, models | MLflow (below) |

---

## 10. Local iteration loop (authoring preview only — not a way to run the pipeline)

This is a **render preview**, not a run. It lets you check your DAG and see the
resolved `params.yaml` on a laptop before you push. **Pipelines only ever run in
the cluster, submitted from Argo Workflows** (§12) — there is no local execution
and no local submit. No cluster needed for the preview:

```bash
./run.sh                                  # or the commands below
python pipeline.py                        # -> out/raw-workflow-template.yaml (your DAG)
python -m kubecore.compose --output out/params.yaml         # composed defaults
python -m kubecore.compose experiment.name=demo train/optimizer=adamw   # try overrides
```

`out/params.yaml` is exactly what your steps will receive at run time. Errors you
can hit, verbatim:

**A step reads a section that doesn't exist** (typo, or you renamed a
tree section) — fails at render, names both sides:

```
platform.derive_tree.DeriveError: step 'report' reads config section 'reporting' which does not exist in the config tree (top-level sections: experiment, data, train)
```

**Unknown/typo'd override key** — fails in the compose step, in
seconds, before any compute:

```
compose-and-validate FAILED: Could not override 'train.epochz'.
To append to your config use +train.epochz=5
```

**Invalid value for a platform-declared dropdown**:

```
compose-and-validate FAILED: Error merging override data.source=banana
  cause: Invalid value 'banana', expected one of [lakefs, s3]
```

**Unknown key inside the ADVANCED override YAML**:

```
compose-and-validate FAILED: Key 'epochz' is not in struct
```

**ADVANCED override touching the platform section** (not allowed):

```
compose-and-validate FAILED: ADVANCED override may not modify platform.*
```

The same checks run in the cluster: a bad submission kills the run at
the first step, pre-GPU.

---

## 11. Using platform tools from step code

Two ways in, both zero-config:

**The `platform` section of your params** — every resolved params.yaml
contains it (you don't declare or read= it, it's just there):

```python
cfg = yaml.safe_load(args.params)
bucket = cfg["platform"]["checkpoints"]["bucket"]     # e.g. "my-project"
prefix = cfg["platform"]["checkpoints"]["prefix"]     # "main/checkpoints"
repo   = cfg["platform"]["lakefs"]["repository"]
```

**Ambient environment variables** — set on every step container:

| env var | what it is |
|---|---|
| `MLFLOW_TRACKING_URI` | your project's MLflow tracking server |
| `LAKEFS_ENDPOINT` | your project's lakeFS endpoint |
| `LAKEFS_ACCESS_KEY` / `LAKEFS_SECRET_KEY` | lakeFS credentials (managed secret) |

MLflow just works:

```python
import mlflow
mlflow.set_experiment(cfg["experiment"]["name"])
with mlflow.start_run():
    mlflow.log_metric("accuracy", 0.87)
```

lakeFS speaks the S3 API (repo = bucket, `ref/path` = key):

```python
import os, boto3
s3 = boto3.client("s3",
    endpoint_url=os.environ["LAKEFS_ENDPOINT"],
    aws_access_key_id=os.environ["LAKEFS_ACCESS_KEY"],
    aws_secret_access_key=os.environ["LAKEFS_SECRET_KEY"])
s3.download_file("my-project", "main/dataset.yaml", "/work/dataset.yaml")
```

When you run locally, the same `platform` section is filled from a
local fallback file, so `python -m app.entry --params
"$(cat out/params.yaml)"` (run from `steps/hello_world/`) behaves like
the real thing.

---

## 12. Running the pipeline (from Argo Workflows only)

Open the workflow template in the Argo UI, press **Submit**:

- **`pipeline-info`** (first field) — read-only cheat-sheet; leave it.
- **Group dropdowns** (any config group directory you define, e.g.
  `train-optimizer`) — swap whole presets. Fields you
  leave untouched follow the group you pick; fields you explicitly
  change win over the group.
- **Your scalar fields** — every leaf of your tree
  (`experiment-name`, `train-epochs`, `data-ref`, …), plus dropdowns the
  platform fills live (`data-ref` lists the datasets that actually
  exist; `cpu-class`/`gpu-class` list this cluster's node classes) and
  fixed-choice fields it validates (e.g. `data-source`).
- **Per-step sizing** — `{step}-cpu` / `{step}-mem`.
- **`config` (ADVANCED)** — normally empty. When non-empty it is an
  override YAML merged LAST over everything above — for scripted
  submissions:

  ```bash
  argo submit --from workflowtemplate/{app}-pipeline \
      -p config="$(cat my-sweep-point.yaml)"
  ```

  Keys must exist in the tree — a typo in the YAML fails the run
  immediately (§10). The `platform:` section is off-limits: an
  ADVANCED override that touches it is rejected.

Defaults are runnable as-is: Submit with no changes = a standard full
training run.

---

## 13. What CI does for you (PR checks + per-step builds)

You never run the render or the image builds by hand — the platform's
**in-cluster CI (Argo Workflows)** does it and reports back on the commit.
There are **no GitHub Actions** in this repo, and none are needed: every
check you see on GitHub is posted by a workflow running on the cluster, and
its link opens that run.

**On a PR to `dev`** (`wft-render` commit status):

- The platform renders your pipeline exactly the way it will after merge
  (`pipeline.py` → enhance with the *real* cluster context → the
  release-hygiene gate → the step ↔ folder check).
- The **`wft-render` status check** gates the PR: green = your DAG builds,
  every `reads=` section exists, every step is declared, every step has a
  buildable `steps/<dir>/Dockerfile` (or a verbatim-image annotation), no
  duplicate form params. The check's description carries the step / form-param
  counts; its link opens the Argo run — if the render fails, the error is in
  that run's `hera-enhance-gate` (or `hera-render`) log.
- Nothing is written anywhere on a PR. The gate only checks.

**On merge to `dev`** (`kubecore-ml-ci` commit status + `ml-ci` deployment):

- The same render runs again and this time **commits** the rendered
  WorkflowTemplate to the project's GitOps repo.
- Every `steps/*/` whose contents changed is built (kaniko → the platform
  registry) and wired into the WorkflowTemplate as `image-<step>`. The matrix
  is discovered from the folders, so a step you add gets built automatically,
  with no CI edits.
- One **`kubecore-ml-ci`** status on the commit (pending → success/failure,
  link = the Argo build run) and one GitHub **Deployment** (`ml-ci`) that
  resolves to the same run. A missed webhook is caught within ~3 minutes by
  the platform's reconciler — no build is ever lost.

So the whole loop for adding a step (§5) is: add `steps/<name>/` + a config
section + one `pipeline.py` line → **open a PR** → `wft-render` renders your
WFT → merge → `kubecore-ml-ci` builds your step image and updates the
WorkflowTemplate → it is ready to run in the Argo UI.

---

## 14. Pitfalls FAQ

**"My new parameter didn't show up on the form."**
Is it inside a *list* or other complex structure? Only scalar leaves
and groups reach the form. Move it to a scalar leaf, or change it via
group options / the ADVANCED override.

**"I renamed a config section and now the release fails."**
That's the `reads=` gate doing its job — the error names the step still
reading the old name. Update that step's `reads=` (and its `READS` in
entry code).

**"Two leaves ended up with the same form name."**
Possible only if your key names contain dashes that collide with the
path mapping (e.g. `train.a-b` vs `train.a.b`). The release fails with
`form-name collision: ...`; rename one key.

**"How do I make something a dropdown?"**
Make it a group: a directory of option files. Scalars are free-text
fields; groups are dropdowns.

**"I picked a group option but my run used the old values."**
It didn't — untouched fields follow the group (§6). If a value really
didn't change, you (or a script) explicitly set that field, and
explicitly-set fields win over the group by design.

**"My run failed with `Invalid value ... expected one of [...]`."**
That field has a fixed set of valid values (e.g. `data-source`:
lakefs/s3). Pick one of the listed values — the check exists to
stop a typo from wasting a training run.

**"My step needs a value from a section it doesn't read."**
Add the section to its `reads=` (and entry `READS`). Reading
undeclared sections works at runtime but is unprotected — the render
gate only defends what you declare.

**"Booleans?"**
Plain values. `true`/`false` in YAML, on the form, everywhere.

**"Step didn't rebuild after my push."**
A push only rebuilds steps whose `steps/<dir>/` changed. Config-tree or
pipeline.py changes re-release the pipeline without rebuilding images —
that's normal; parameters live in the template, not the images.

**"Can I put big things (file contents, long lists) in config?"**
The resolved params.yaml is capped (200KB, checked with a clear error).
Config is for knobs; bulk data goes through lakeFS/checkpoints (§9).
