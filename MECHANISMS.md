# How it all works — the full mechanism

The complete internal machinery of the Hera + Hydra ML pipeline platform: what
happens when you add or remove a step, how enhancement transforms your DAG into a
runnable WorkflowTemplate, how runtime config flows from your `config/` tree into
a running container, and how every piece connects. `DEVELOPER.md` tells you *what
to do*; this tells you *what the platform does* — for anyone extending or
debugging it.

---

## 0. The three files that are the whole system

```
pipeline.py   your DAG:   step("model-training", reads=[...], gpu=True, needs=[...])
config/       your knobs:  one Hydra tree — scalar leaf → form field, group dir → dropdown
steps/<n>/    your code:   a Dockerfile + app/entry.py per step (name ↔ directory)
```

Everything else — images, secrets, scheduling, sizing, the submit form, the run
— is derived by the platform from these three. You never write YAML for a
WorkflowTemplate, a Pod, a nodeSelector, or a secret.

---

## 1. The pipeline of pipelines (four transforms)

Your three files become a running training job through four transforms, each in a
different place and time:

```
  ┌─ RENDER (CI, tokenless) ───────────────────────────────────────────────┐
  │  pipeline.py  →  authoring.step()/pipeline()  →  raw WorkflowTemplate    │
  │                  (+ derive_tree: config tree → submit-form parameters)   │
  └────────────────────────────────────────────────────────────────────────┘
                                   │  raw WFT (structure + form params, no wiring)
                                   ▼
  ┌─ ENHANCE (CI, platform step, no client code) ──────────────────────────┐
  │  enhance()  →  injects name, images, env, secrets, scheduling, sizing,   │
  │                per-step class dropdowns, dynamic enums, pipeline-info     │
  └────────────────────────────────────────────────────────────────────────┘
                                   │  enhanced WFT (a complete, runnable release)
                                   ▼  git commit → gitops → ArgoCD → live WFT
  ┌─ SUBMIT (Argo UI / agent) ─────────────────────────────────────────────┐
  │  the enhanced WFT's arguments ARE the submit form (= your config tree)   │
  └────────────────────────────────────────────────────────────────────────┘
                                   │  a Workflow (one run) with your chosen values
                                   ▼
  ┌─ COMPOSE (run time, in the pipeline's first container) ─────────────────┐
  │  compose.py (`python -m kubecore.compose`)  →  resolves the config tree   │
  │  + your form values → ONE params.yaml → every downstream step reads it    │
  └────────────────────────────────────────────────────────────────────────┘
```

RENDER + ENHANCE run once per release (on push). SUBMIT + COMPOSE run once per
training run. The render/enhance split is the security boundary: RENDER runs your
arbitrary `pipeline.py` with **no credentials**; ENHANCE runs only platform code.

---

## 2. RENDER — pipeline.py → raw WorkflowTemplate

`kubecore/authoring.py` gives you `step()` and `pipeline()`. When the `with
pipeline(...)` block exits, `_build()` runs:

1. **Derive the submit form from the config tree** (`derive_tree.derive()`):
   - composes the Hydra `config/` tree once;
   - every **scalar leaf** → one form `Parameter` (dotted path, dots→dashes:
     `train.optimizer.lr` → `train-optimizer-lr`), value = the leaf's value;
   - every **config group directory** → one dropdown `Parameter` (options = the
     `.yaml` files in the dir, default = the defaults-list choice);
   - **lists / complex nodes are not flattened** (change them via a group or the
     ADVANCED override);
   - each entry also emits its **Hydra override token**
     (`train.epochs={{workflow.parameters.train-epochs}}`) and a
     **render-defaults** entry (used later for group-swap elision);
   - a **form-name collision** (two leaves mapping to the same dashed name) is a
     hard error.
2. **Run the `reads=` gate** (`derive_tree.validate_reads()`): every section a
   step declares in `reads=` must exist in the composed tree. For schema-backed
   sections (`data`, `quantization`) it also requires a real on-disk source, so a
   renamed YAML key can't be silently backfilled by the ConfigStore schema.
3. **Build the DAG** (plain Hera):
   - task 1 is always **`compose-and-validate`** — its args are ALL the override
     tokens + the render-defaults manifest + the ADVANCED param + `--require`
     for each read section; its output parameter is the resolved `params.yaml`;
   - each of your steps becomes a `Container` running `python -m app.entry`, with
     an input parameter `params` fed from compose's output, plus one input per
     output of each `needs=` step;
   - `gpu=True` → `Resources(gpus=1)` (emits `nvidia.com/gpu` into requests +
     limits — the enhancer keys GPU scheduling off exactly this);
   - `when=` → the task's `when`; a `needs=` step that is itself conditional gets
     a **skip-tolerant depends** (`(dep.Succeeded || dep.Skipped || dep.Omitted)`)
     so the DAG doesn't wedge when an optional upstream skips.

Output: a valid Argo `WorkflowTemplate` with your DAG structure and the full
submit form — but no images, no secrets, no scheduling yet. That's ENHANCE's job.

**The image is a sentinel** (`platform-managed`) at this point — you never name an
image; the enhancer rewrites it.

---

## 3. ENHANCE — raw WFT → runnable release

`kubecore/enhance.py` runs in CI (a platform step, no client code) on the raw WFT
+ the `pipeline-context.yaml` the operator wrote (the platform API). It operates
on plain dicts (no Hera dependency). The exact ordered pipeline (`enhance()`):

| Step | What it injects | Rule |
|---|---|---|
| `enhance_metadata` | **Forces** `metadata.name = {app}-pipeline`; namespace, labels, provenance annotations | name is a multi-tenant identity — the platform owns it (see §6) |
| `enhance_spec_top_level` | `serviceAccountName` | fill-absent |
| `enhance_arguments` | endpoints (`mlflow-endpoint`, `lakefs-endpoint`, `lakefs-repo`), one `image-{step}` param per step (a `configMapKeyRef` into `{app}-pipeline-images`) | image is a supply-chain concern |
| `enhance_image` | rewrites each step's image → `{{workflow.parameters.image-{step}}}` | **always** overwritten (the one non-fill-absent rule), unless a `platform.kubecore.io/image` annotation gives a verbatim utility image |
| `enhance_env` | `MLFLOW_TRACKING_URI`, `LAKEFS_ENDPOINT`, lakeFS creds (`secretKeyRef`), `CHECKPOINT_BUCKET/PREFIX` | default-on per step; opt out via `platform.kubecore.io/inject` |
| `enhance_class_param` | a per-step `{step}-class` dropdown (options = the pool's classes of that tier) | so a submitter routes each step onto any allowed class |
| `enhance_sizing_knobs` | `{step}-cpu` / `{step}-mem` form params + a `podSpecPatch` wiring them | per-run sizing; runs BEFORE scheduling so dev-set values are distinguishable |
| `enhance_scheduling` | `nodeSelector` (`nodegroup-type: {{workflow.parameters.{step}-class}}`), tolerations, whole-node `requests==limits` from the class `allocatable`, `nvidia.com/gpu` for GPU steps | fill-absent; developer-set resources win |
| `enhance_volumes` | `/dev/shm` emptyDir (size via `shm` annotation), optional `/workspace` PVC (via `workspace` annotation) | fill-absent; raw Hera volumes win |
| `enhance_workspace_pvc` | one workflow-level `volumeClaimTemplate` if any step requested a workspace | opt-in only |
| `enhance_dynamic_enums` | live `enum` values the developer can't know (e.g. `data-ref` from the dataset-catalog probe) | fill-absent enum |
| `enhance_pipeline_info` | a read-only `pipeline-info` doc parameter (first on the form) | — |

**Golden rule: only fill absent keys.** If you set a nodeSelector, an env var, or
a memory request in `pipeline.py`, the enhancer leaves it. The two deliberate
exceptions — because they're platform-identity/supply-chain concerns, not
developer choices — are the **WFT name** and the **image**.

**Unknown `platform.kubecore.io/*` annotations hard-fail the enhance** (typo
protection). The known vocabulary: `compute-class, inject, source, image, shm,
workspace, hpc`.

Output: a complete, runnable `WorkflowTemplate` — CI commits it to the gitops
repo, ArgoCD syncs it, and it becomes the live release. That live WFT's
`arguments.parameters` block **is** the Argo submit form, and it **is** the schema
agents read.

---

## 4. Live runtime config flow — from tree to running container

At **submit** time the form values arrive as Hydra override tokens to the
**compose-and-validate** step only. At **run** time that step
(`python -m kubecore.compose`) does:

1. **order + elide overrides** (`order_overrides`): group tokens first; a scalar
   token whose value equals its render-time default is **dropped** (so an
   untouched field follows whatever group you selected); explicitly-changed
   fields apply after the groups and win. Without this, a full form submission
   would freeze every leaf at its default and silently revert a group swap.
2. **compose** the tree + overrides (Hydra Compose API), including the platform
   config group (see below).
3. **validate**: unknown/typo'd override → Hydra strict mode rejects it;
   type/enum-invalid value → structured-config (dataclass) validation; a declared
   `--require` section missing → hard error; ADVANCED override touching
   `platform.*` → hard error; resolved params > 200 KB → hard error.
4. emit the single resolved **`params.yaml`** as its Argo output parameter.

Every downstream step receives that `params.yaml` as its `--params` input and
reads only its declared sections. **params.yaml IS the experiment** — it's also
archived to MLflow so any run reproduces exactly. All failures are concentrated
here, in seconds, **before any GPU time is spent.**

**Two channels reach step code, both zero-config:**
- **`cfg["platform"]`** — a `platform` config group the enhancer/CI materializes
  from `pipeline-context` (checkpoints bucket/prefix, lakeFS repo/endpoint, MLflow
  URI). It's in every `params.yaml`; you don't declare or `reads=` it.
- **ambient env vars** — `MLFLOW_TRACKING_URI`, `LAKEFS_ENDPOINT`,
  `LAKEFS_ACCESS_KEY/SECRET_KEY`, `CHECKPOINT_BUCKET/PREFIX` — set on every step
  container by the enhancer. `import mlflow` just works.

**The data plane** (verified live): no shared volumes between steps. Small results
cross as Argo parameters carrying file *content*; bulk data (datasets,
checkpoints, models) goes through lakeFS / the checkpoint bucket / MLflow, with
references passed as parameters. Two concurrent runs are fully isolated (no shared
PVC, per-run output params, per-pod `/work`).

---

## 5. Adding / removing / altering a step — the full mechanism

### Add a parameter
You add a leaf to `config/…`. On push, RENDER re-derives the form and the new leaf
becomes a form field (dotted-path name); every step that `reads=` its section sees
it in `params.yaml`. **No image rebuild** — parameters live in the WFT, not the
images. Nothing else changes.

### Add a step
You `mkdir steps/<name>` (+ Dockerfile + `app/entry.py`), add one
`step("<name>", reads=[...], needs=[...])` line, and (if it has its own knobs) a
config section. On push:
1. **CI `detect-steps`** walks `steps/*/` and finds a directory with a Dockerfile
   that has no built image yet → it's in the build set (**step name ↔ directory**;
   `dataset_loading/` → `dataset-loading`). It builds the image (kaniko → Zot).
2. **RENDER** adds the step to the DAG + wires its `needs=`/`reads=`/`when=`.
3. **ENHANCE** gives it an `image-<name>` param, a `<name>-class` dropdown,
   `<name>-cpu/mem` sizing knobs, env/secrets, scheduling — everything.
4. **COMMIT** patches `{app}-pipeline-images` to add the new `image-<name>` key so
   it resolves on the first submit.

One `step()` line + a directory → a fully-wired, scheduled, imaged pipeline step.

### Remove a step
Delete its `step()` line, its entry from any other step's `needs=`, and
`steps/<name>/`. On push, RENDER simply doesn't emit it. Miss a `needs=`/`reads=`
cleanup and the reads= gate fails the release loudly, naming both sides.

### Alter behavior
- change a knob → edit the leaf (or set it on the form per-run);
- swap a whole preset → pick a different **group** option (optimizer, augmentation
  recipe, callbacks) — one dropdown swaps the entire subtree;
- GPU on/off → `gpu=True` / remove it;
- conditional → `when="{{workflow.parameters.<param>}} == <value>"`.

---

## 6. Multi-tenant safety — why one app can't break another

- **WFT name forced to `{app}-pipeline`** (§3): two apps in the same
  `ml-{project}` namespace get distinct WFTs — one app's render can never
  overwrite another's, even if both copied the same `pipeline("...")` literal.
- **image ConfigMap is `{app}-pipeline-images`** — no cross-app image bleed.
- **gitops discovery is slash-anchored** (`*/{app}/{branch}/…`) — `yolo` ≠
  `yolo-training`.
- **credential split** (§7): untrusted `pipeline.py` runs with no token and a
  locked-down executor SA (workflowtaskresults only) — it can't read another
  tenant's secrets via the K8s API.

---

## 7. The credential split (CI security boundary)

`pipeline.py` is arbitrary developer Python. It runs in the **`hera-render`**
step: `automountServiceAccountToken: false`, no GitHub token, a locked-down
`hera-render-exec` SA (workflowtaskresults only, **no** secret access), and no
egress beyond package installs. Its only input is the already-cloned repo; its
only output is a raw WFT on the workspace PVC. The **`hera-enhance-commit`** step
— which holds the GitHub App token and writes to the gitops repo — runs **only
platform code** (the enhancer + gate + git), never client code. So untrusted
Python can never execute in a credentialed context. (See `ci-integration/`.)

---

## 8. No Crossplane fights (why MRs don't revert CI's work)

The gitops `workflow-template.yaml` is a Crossplane `RepositoryFile` MR with
`managementPolicies: ["Observe","Create","Delete"]` + `overwriteOnCreate: false`.
Crossplane seeds a one-task placeholder **once** and never issues an Update, so
CI's git commit of the real rendered WFT is never reverted. ArgoCD `selfHeal` then
keeps the live cluster WFT equal to what CI committed — so the platform's stated
guarantee ("cluster state == last released pipeline") holds, and manual `kubectl`
edits to the live WFT are reverted (everything goes through git, by design).

---

## 9. Compute sizing — whole-node, per-tier, GPU-aware

Each step runs sole-tenant on its class's node: `requests == limits` from the
class **allocatable** (Guaranteed QoS). The operator computes allocatable from the
machine type via GKE's tiered kube-reservation curve, and — for **GPU** classes —
subtracts an extra headroom (nvidia device-plugin + driver + larger system
reservation) so a training step actually fits a fresh accelerator node. Per-run,
the `{step}-cpu`/`{step}-mem` form knobs override via `podSpecPatch`, and
`{step}-class` picks which of the pool's classes the step lands on.

---

## 10. Where each piece lives (quick map)

| Concern | File / place |
|---|---|
| Your DAG | `pipeline.py` |
| Your knobs | `config/**` (Hydra tree) |
| Your code | `steps/<name>/app/entry.py` + `Dockerfile` |
| step()/pipeline() → Hera DAG | `kubecore/authoring.py` |
| config tree → form + gate + schemas | `kubecore/derive_tree.py` |
| raw WFT → runnable release | `kubecore/enhance.py` |
| run-time compose+validate | `kubecore/compose.py` (`python -m kubecore.compose`) |
| CI render/enhance/gate/commit + credential split | `ci-integration/` (`ml-ci-build` chart) |
| platform API (endpoints, classes, secrets names) | `{app}-pipeline-context` ConfigMap (operator-written) |
| image indirection | `{app}-pipeline-images` ConfigMap (CI-patched) |
| the released pipeline | `WorkflowTemplate/{app}-pipeline` in `ml-{project}` |

See `DEVELOPER.md` (how to author), `kubecore/README.md` (engine internals),
`PLATFORM.md` (repos + operator + onboarding), `ci-integration/README.md` (the CI
render path).
