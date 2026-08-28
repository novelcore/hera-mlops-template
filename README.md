# hera-mlops-template

A KubeCore ML pipeline you author in **Python** ([Hera](https://github.com/argoproj-labs/hera))
with parameters in a **[Hydra](https://github.com/facebookresearch/hydra) config tree**.
Clone it, add your steps and parameters, push — the platform builds your images
and releases a runnable pipeline. You never touch Kubernetes.

This template is the source new ML KubeApps are seeded from (`spec.type:
ml-pipeline`). It replaces the old `kubeline.yaml` DSL.

## Why this exists

The whole DAG is **one 38-line `pipeline.py`** and a tree of small YAML files —
the same YOLO training pipeline that used to be a **1551-line `kubeline.yaml`**.

```python
# pipeline.py — the ENTIRE pipeline definition
from kubecore.authoring import pipeline, step

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

**One rule: the `config/` tree IS the submit form.** Every scalar in `config/`
becomes a form field; every group directory becomes a dropdown. Add a leaf → a
field appears. No parameter wiring anywhere.

## Layout

```
pipeline.py            the DAG (steps, reads=, gpu=, needs=, when=)
config/                THE experiment config tree — all your parameters
  config.yaml            the defaults list (which group options are default)
  data.yaml train.yaml … sections of scalar leaves
  train/optimizer/*.yaml a config GROUP → a dropdown
steps/<name>/          your step code + Dockerfile (one dir per step)
kubecore/              platform-owned helpers — SEEDED, DO NOT EDIT
pyproject.toml         PEP 621, pinned render deps
```

## Two core moves (full guide: [DEVELOPER.md](DEVELOPER.md))

- **Add a parameter** → add a leaf to `config/…`. Push. The form has it.
- **Add a step** → `mkdir steps/<name>` + a Dockerfile + an entry that reads its
  config slice, then one `step("<name>", reads=[…], needs=[…])` line in
  `pipeline.py`. Push.

## Local iteration (no cluster)

```bash
./run.sh                                              # venv + render + enhance + compose
python -m kubecore.compose train.epochs=5 train/optimizer=adamw   # try overrides locally
```

`out/params.yaml` is exactly what your steps receive at run time; a typo or a
bad `reads=` fails locally with the same message the cluster gives you.

## What the platform does for you (you never write this)

Image supply-chain (Zot registry), MLflow/lakeFS/checkpoint env + secrets,
per-step compute-class selection + node scheduling from your KubePool's classes,
per-run sizing knobs, `/dev/shm`, GitOps release, and the Argo submit form —
all injected at release time. Your `pipeline.py` stays pure structure.

**Adding your first step?** Start with **[ADD-A-STEP.md](ADD-A-STEP.md)** — a
complete copy-paste walkthrough for newcomers (every command, every gotcha).

See **[DEVELOPER.md](DEVELOPER.md)** for the complete operating manual, and
**[MECHANISMS.md](MECHANISMS.md)** for how the platform works internally
(add/remove steps, enhancement, runtime config flow, multi-tenant safety).
## Running steps on MeluXina (HPC)

When the project's pool is HPC-enabled, every step's `{step}-class` dropdown on the
Argo submit form also offers the MeluXina classes:

| class | what it is | typical use |
|---|---|---|
| `meluxina-gpu` | 4× A100-40GB, 2× EPYC 7452, 512 GB | training, QAT fine-tune |
| `meluxina-cpu` | 2× EPYC 7452 (128 cores), 512 GB | heavy preprocessing, PTQ export |
| `meluxina-largemem` | 4 TB RAM | datasets that do not fit a node |

Pick one for a step and only that step runs as a Slurm job on MeluXina; every other
step keeps its in-cluster class — mixing (training on `meluxina-gpu`, quantization on
`gpu-small`, registration in-cluster) is the normal case. Nothing in this repository
changes: the platform stages the dataset ref to Lustre, pulls the step image into an
Apptainer SIF, runs the same command the pod would run, and brings the step's declared
outputs back so the next step consumes them as usual. MLflow and lakeFS are reached
through the public endpoints with a short-lived token minted per job.

- Queue time is real (minutes to hours) — the pipeline waits for you; the
  `pipeline-info` field on the form lists the classes and the account.
- Give long steps a wall-clock limit in `pipeline.py`: `step(..., hpc_time_limit="12h")`
  (default 4 h). See [DEVELOPER.md §8.1](DEVELOPER.md).
- Dataset layout is unchanged: `s3://{repo}/{ref}/dataset/{version}/`.
- Steps pinned with a `compute-class` annotation stay in-cluster.

