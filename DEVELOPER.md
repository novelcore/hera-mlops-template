# Pipeline Developer Guide

The operating manual for authoring and running your ML pipeline. You need
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
│   ├── data.yaml  train.yaml  quantization.yaml  ...
│   ├── train/optimizer/{sgd,adam,adamw}.yaml     # a config GROUP
│   └── image_processing/{mosaic_default,...}.yaml
├── kubecore/                # platform-owned helpers (seeded — do not edit)
└── steps/
    ├── model_training/      # Dockerfile + entry point (your code)
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
  batch_size: 16     # -1 for Ultralytics auto-batch
  cos_lr: true       # plain booleans are fine
  loss:
    pose: 12.0       # nested scalars work too
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
  - train
  - train/optimizer: sgd
  - image_processing: mosaic_default
  # ...
```

What each thing becomes on the submit form:

| you write in config/ | what happens |
|---|---|
| a scalar leaf (`train.epochs: 100`) | a form field named by its path, dots→dashes: **`train-epochs`**, default `100` |
| a nested scalar (`train.loss.pose`) | same rule: **`train-loss-pose`** |
| a **boolean** (`train.amp: true`) | a plain form field (`true`/`false`) — no special treatment |
| a **group directory** (`train/optimizer/*.yaml`) | ONE dropdown (**`train-optimizer`**) whose options are the file names; picking one swaps the whole subtree |
| a **list** or other complex structure | NOT a form field — change it by swapping a group option or via the ADVANCED override |

Some fields are dropdowns with a fixed set of valid values because the
platform declares them (e.g. `data-source`: lakefs/s3,
`quantization-mode`: none/ptq/qat) — submitting anything else fails the
run immediately, with the allowed values listed in the error.

**The boolean non-rule:** booleans are just values now. Write
`amp: true`, submit `false`, your step reads a real `False`. (If you
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

`pipeline.py` declares structure and nothing else — this is the real,
complete step list from this repo:

```python
from kubecore.authoring import pipeline, step  # platform-owned, do not edit

with pipeline("ml-pipeline") as p:  # platform renames to {app}-pipeline at release
    load = step("dataset-loading", reads=["data"],
                outputs=["data-yaml", "manifest-summary"])
    train = step("model-training", gpu=True, needs=[load],
                 reads=["experiment", "data", "model", "train", "image_processing", "logging"],
                 outputs=["training-result"])
    qat = step("qat-finetune", gpu=True, needs=[train],
               reads=["experiment", "train", "quantization"],
               when="{{workflow.parameters.quantization-mode}} == qat",
               outputs=["qat-result"])
    quant = step("model-quantization", needs=[train, qat],
                 reads=["experiment", "quantization"],
                 when="{{workflow.parameters.quantization-mode}} != none",
                 outputs=["quantization-result"])
    register = step("model-registration", needs=[train, quant],
                    reads=["data", "model", "registration"])
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
  (parameter name = config path with dashes: `quantization.mode` →
  `quantization-mode`).

---

## 5. Add a step

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
   `steps/model_training/entry.py`):

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
augmentation recipes, callbacks, QAT settings.

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

- `step("model-training", gpu=True, ...)` is the entire GPU
  configuration.
- Per-run sizing is automatic: every step gets `{step}-cpu` and
  `{step}-mem` fields on the submit form (defaults = a whole node).
  Dial a smoke-run down at submit time; you never configure this.

---

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

## 10. Local iteration loop

No cluster needed:

```bash
./run.sh                                  # or the commands below
python pipeline.py                        # -> out/raw-workflow-template.yaml (your DAG)
python -m kubecore.compose --output out/params.yaml         # composed defaults
python -m kubecore.compose train.epochs=5 train/optimizer=adamw   # try overrides
```

`out/params.yaml` is exactly what your steps will receive. Errors you
can hit, verbatim:

**A step reads a section that doesn't exist** (typo, or you renamed a
tree section) — fails at render, names both sides:

```
platform.derive_tree.DeriveError: step 'model-registration' reads config section 'registration' which does not exist in the config tree (top-level sections: data, quantization, experiment, model, train, image_processing, registry, logging)
```

**Unknown/typo'd override key** — fails in the compose step, in
seconds, before any compute:

```
compose-and-validate FAILED: Could not override 'train.epochz'.
To append to your config use +train.epochz=5
```

**Invalid value for a platform-declared dropdown**:

```
compose-and-validate FAILED: Error merging override quantization.mode=banana
  cause: Invalid value 'banana', expected one of [none, ptq, qat]
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
bucket = cfg["platform"]["checkpoints"]["bucket"]     # "yolo"
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
    mlflow.log_metric("mAP50", 0.87)
```

lakeFS speaks the S3 API (repo = bucket, `ref/path` = key):

```python
import os, boto3
s3 = boto3.client("s3",
    endpoint_url=os.environ["LAKEFS_ENDPOINT"],
    aws_access_key_id=os.environ["LAKEFS_ACCESS_KEY"],
    aws_secret_access_key=os.environ["LAKEFS_SECRET_KEY"])
s3.download_file("yolo", "main/data.yaml", "/work/data.yaml")
```

When you run locally, the same `platform` section is filled from a
local fallback file, so `python steps/model_training/entry.py --params
"$(cat out/params.yaml)"` behaves like the real thing.

---

## 12. Running the pipeline

Open the workflow template in the Argo UI, press **Submit**:

- **`pipeline-info`** (first field) — read-only cheat-sheet; leave it.
- **Group dropdowns** (`model`, `train-optimizer`, `train-callbacks`,
  `train-qat`, `image_processing`) — swap whole presets. Fields you
  leave untouched follow the group you pick; fields you explicitly
  change win over the group.
- **Your scalar fields** — every leaf of your tree
  (`train-epochs`, `data-ref`, `train-loss-pose`, …), plus dropdowns the
  platform fills live (`data-ref` lists the datasets that actually
  exist; `cpu-class`/`gpu-class` list this cluster's node classes) and
  fixed-choice fields it validates (`data-source`, `quantization-mode`).
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

## 13. Pitfalls FAQ

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
That field has a fixed set of valid values (e.g. `quantization-mode`:
none/ptq/qat). Pick one of the listed values — the check exists to
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
