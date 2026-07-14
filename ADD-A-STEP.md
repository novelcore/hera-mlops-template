# Add a step — the complete walkthrough (for dummies)

This is the copy-paste, zero-assumptions guide to adding a new step to your
pipeline. Follow it top to bottom. If you can `mkdir` and edit a text file, you
can do this. No platform knowledge needed — the platform wires up the image,
the form fields, the compute node, the credentials, and the CI checks for you.

We'll add a step called **`model-evaluation`** that runs after training. Swap in
your own name wherever you see it.

> **The golden rule of naming.** Pick a step name in `lowercase-with-hyphens`
> (e.g. `model-evaluation`). The **folder** for it uses `underscores`
> (`steps/model_evaluation/`). The platform maps one to the other for you. So:
> - step name (in `pipeline.py`): `model-evaluation`
> - folder (on disk): `steps/model_evaluation/`
>
> Get this mapping right and everything else just works.

---

## What you will touch (that's it)

| # | File | Why |
|---|------|-----|
| 1 | `steps/model_evaluation/app/entry.py` | the code your step runs |
| 2 | `steps/model_evaluation/app/__init__.py` | empty file (makes it a Python package) |
| 3 | `steps/model_evaluation/Dockerfile` | how your step is packaged into an image |
| 4 | `config/evaluation.yaml` | *(optional)* the knobs your step exposes on the form |
| 5 | `config/config.yaml` | *(optional)* register the new config section |
| 6 | `pipeline.py` | ONE line: add the step to the DAG |

Everything else — building the image, publishing it, putting it on the submit
form, picking the compute node, injecting lakeFS/MLflow credentials, wiring the
CI status checks — **the platform does automatically.** You never edit anything
under `kubecore/`. That directory is the platform engine; leave it alone.

---

## Step 1 — make the folder

From the repo root:

```bash
mkdir -p steps/model_evaluation/app
touch steps/model_evaluation/app/__init__.py
```

That `__init__.py` is an empty file. It just tells Python "this is a package."
Don't put anything in it.

---

## Step 2 — write the code (`steps/model_evaluation/app/entry.py`)

Every step is a small Python program that:
1. receives the resolved config as a single `--params` string, and
2. reads the slice(s) of config it cares about.

Copy this exactly, then change `READS` and the body:

```python
"""model-evaluation step: what it does in one sentence."""

import argparse
import yaml

# The config sections this step reads. List every top-level section you use.
# The platform checks at render time that these sections exist — so a typo
# here fails your PR early, not at 3am on a GPU node.
READS = ["evaluation", "model"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", required=True,
                        help="Resolved params.yaml content (from compose-and-validate).")
    args, _ = parser.parse_known_args()
    cfg = yaml.safe_load(args.params)

    # Your config is here. Read only what you declared in READS.
    evaluation = cfg["evaluation"]
    model = cfg["model"]

    # Platform endpoints (if you need them) live under cfg["platform"]:
    #   cfg["platform"]["mlflow"]["tracking_uri"]
    #   cfg["platform"]["lakefs"]["endpoint"] / ["repository"]
    # You do NOT need to configure credentials — the platform injects them as
    # env vars (LAKEFS_ACCESS_KEY, LAKEFS_SECRET_KEY, MLFLOW_TRACKING_URI).

    print(f"[model-evaluation] split={evaluation['split']} "
          f"iou={evaluation['iou_threshold']} model={model['variant']}")

    # ... do your real work here ...


if __name__ == "__main__":
    main()
```

**Key facts:**
- The command that runs in the cluster is `python -m app.entry` — the same as
  running it locally. So keep your entry point at `app/entry.py`.
- `--params` is a YAML string containing **every** config section, already
  resolved from the submit form. You just `cfg["your-section"]`.
- If your step needs a previous step's output (e.g. training results), see
  "Passing data between steps" below.

---

## Step 3 — write the Dockerfile (`steps/model_evaluation/Dockerfile`)

Copy this. Add any Python packages your step needs to the `pip install` line.

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Add whatever your step imports. PyYAML is always needed (to read --params).
RUN pip install --no-cache-dir PyYAML

COPY app/ ./app/

# Argo runs: python -m app.entry --params <resolved params.yaml content>
ENTRYPOINT ["python", "-m", "app.entry"]
```

That's the whole Dockerfile. The platform builds it (kaniko → the registry) and
wires the resulting image into the workflow as `image-model-evaluation`
automatically. **You never write an image name or a registry URL.**

> **The #1 mistake:** declaring a step in `pipeline.py` but forgetting this
> Dockerfile. The PR render check catches it with a clear message
> ("step 'model-evaluation': expected steps/model_evaluation/Dockerfile") — so
> you find out at PR time, not when it `ImagePullBackOff`s in the cluster.

---

## Step 4 — add your config knobs (optional)

Only if your step has parameters a user should be able to change on the submit
form. Create `config/evaluation.yaml`:

```yaml
evaluation:
  split: "val"              # which split to evaluate on
  iou_threshold: 0.5        # IoU threshold for matching
```

**Every scalar leaf here becomes a field on the Argo submit form**, named
`evaluation-split`, `evaluation-iou_threshold` (section + key, dots→hyphens).
Users can change them at submit time; you read them via `cfg["evaluation"]`.

Then register the section in `config/config.yaml`'s `defaults:` list so it gets
composed in:

```yaml
defaults:
  - evaluation      # <- add this line
  # ... the others ...
```

If your step needs no new knobs, skip Step 4 entirely — just read existing
sections (e.g. `READS = ["model", "data"]`).

---

## Step 5 — add the step to the DAG (`pipeline.py`) — ONE line

Open `pipeline.py`. You'll see the pipeline as a list of `step(...)` calls. Add
your step where it belongs in the order, with `needs=` pointing at whatever must
run first:

```python
with pipeline("ml-pipeline") as p:
    validate = step("config-validation", reads=["data", "model"])
    load     = step("dataset-loading", reads=["data"], needs=[validate],
                    outputs=["data-yaml", "manifest-summary"])
    train    = step("model-training", gpu=True, needs=[load],
                    reads=["experiment", "data", "model", "train"],
                    outputs=["training-result"])
    # 👇 YOUR NEW STEP — one line
    evaluate = step("model-evaluation", reads=["evaluation", "model"], needs=[train])
    register = step("model-registration", needs=[train], reads=["data", "model", "registration"])
```

**What each argument means:**
- **first argument** (`"model-evaluation"`) — the step name (`lowercase-hyphens`).
- **`reads=[...]`** — the config sections your `entry.py` READS. Must match.
- **`needs=[...]`** — the steps that must finish before yours starts. This is
  the whole DAG: just wiring `needs`. `needs=[train]` means "after training."
- **`gpu=True`** — add this only if your step needs a GPU.
- **`outputs=[...]`** — small result files your step writes for a later step to
  consume (see below). Omit if your step produces nothing downstream needs.
- **`when="..."`** — an optional condition (advanced; see DEVELOPER.md).

> **Python ordering gotcha:** `needs=[train]` refers to the `train` variable, so
> `train` must be defined *above* your step in the file. If you need a step
> that's defined later, move your step down. (If you get it wrong, you'll see a
> clear `NameError: name 'x' is not defined` — not a silent failure.)

---

## Step 6 — check it locally (optional but nice)

You don't need a cluster. From the repo root:

```bash
./run.sh
```

This renders your pipeline exactly the way the platform will (it does NOT run
it — running only happens in Argo). It writes `out/workflow-template.yaml`. If
your step shows up there with an `image-model-evaluation` reference and its
config fields are in the form, you're good. If you made a mistake (missing
config section, bad name), it fails here with a clear message.

To just build your step's image locally and confirm it's valid:

```bash
docker build steps/model_evaluation
```

---

## Step 7 — commit and push. That's it.

```bash
git add steps/model_evaluation config/evaluation.yaml config/config.yaml pipeline.py
git commit -m "add model-evaluation step"
git push
```

Now the platform takes over:
1. **On your PR** — a `wft-render` check renders the pipeline and comments the
   result (steps, form parameters, and a link to where it'll run). If anything's
   wrong, the comment tells you exactly what.
2. **On merge** — your step's image builds and gets its own `build/model-evaluation`
   status check. The workflow template updates. Your step is now on the Argo
   submit form, ready to run.

You did not touch the platform. You did not write a registry URL, an image name,
a credential, a node selector, or a resource request. You wrote a small Python
program, a Dockerfile, maybe a config file, and one line in `pipeline.py`.

---

## Passing data between steps

If your step needs a previous step's output, the producing step declares it and
your step gets it as an input automatically.

**Producer** declares what it writes (as `outputs=`) and writes small JSON files
to `/work/output/<name>.json`:

```python
# in the producing step (e.g. model-training in pipeline.py):
train = step("model-training", ..., outputs=["training-result"])
```

```python
# in the producing step's entry.py, write the file:
import json, os
os.makedirs("/work/output", exist_ok=True)
with open("/work/output/training-result.json", "w") as f:
    json.dump({"best_map": 0.99, "weights_uri": "s3://..."}, f)
```

**Consumer** (your step) just `needs=` the producer — the platform feeds every
declared output in as a `--<name>` argument:

```python
evaluate = step("model-evaluation", reads=["evaluation"], needs=[train])
```

```python
# in your entry.py, read the input:
parser.add_argument("--training-result", default="{}")
args, _ = parser.parse_known_args()
result = yaml.safe_load(args.training_result)   # {"best_map": 0.99, ...}
```

Keep these small (they pass as workflow parameters). For bulk data (datasets,
weights), pass a pointer (an `s3://` URI) and stream it — don't inline it.

---

## Cheat sheet

```
add a knob        -> 1 line in config/<section>.yaml            (becomes a form field)
change the DAG    -> edit needs=[...] in pipeline.py            (that's the whole DAG)
add a step        -> mkdir steps/<name>/app + Dockerfile + 1 pipeline.py line
                     (platform auto-does: image, form fields, node, creds, CI check)
run it            -> only from the Argo UI (Submit). ./run.sh is a local preview only.
never edit        -> anything under kubecore/  (that's the platform engine)
```

If you remember one thing: **you describe *what* your step is (name, what it
reads, what it needs); the platform handles *how* it runs.**
