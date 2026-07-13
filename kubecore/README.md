# kubecore/ — the platform engine (PLATFORM-OWNED, do not edit in an app repo)

This directory is seeded into every ML app repo and is **replaced wholesale by
the platform on upgrade** — edits here are lost. It is the render/compose/enhance
engine that turns a developer's `pipeline.py` + `config/` tree into a released
Argo `WorkflowTemplate`. Developers never touch it; this README is for the
**platform team** that maintains and extends it.

> Named `kubecore` (not `platform`) on purpose: a top-level `platform` package
> shadows Python's stdlib `platform` module and breaks any dependency that calls
> `platform.system()`.

## The four modules

| Module | When it runs | Holds credentials? | Job |
|---|---|---|---|
| `authoring.py` | render time (CI, tokenless) | no | `step()`/`pipeline()` → plain Hera `Container`s + Argo DAG; injects the compose-and-validate step as task 1; `gpu=True` → `Resources(gpus=1)` |
| `derive_tree.py` | render time (CI, tokenless) | no | composes the Hydra `config/` tree, derives the submit-form parameters (group→dropdown, scalar-leaf→field, dots→dashes), runs the `reads=` render gate, registers structured-config schemas |
| `enhance.py` | render time (CI, platform step) | no client code | post-processes the raw WFT: forces the app-scoped name + image indirection, injects env/secrets/scheduling/sizing/per-step class dropdowns/dynamic enums. Plain dicts, **no Hera dependency** |
| `compose.py` | **run time** (inside the pipeline's first step container) | no | `python -m kubecore.compose` — composes tree+overrides → the single resolved `params.yaml`, validates, fails fast pre-GPU |

`authoring` + `derive_tree` are **imported** by the developer's `pipeline.py`.
`compose` runs as a program in the compose step's image. `enhance` runs in the
CI enhance step and is **never** import-reachable from client code (the D4 trust
boundary — untrusted `pipeline.py` runs tokenless; the enhancer runs separately).

## The invariants the engine enforces (and why)

- **WFT name is forced to `{app}-pipeline`** (`enhance.pipeline_wft_name`),
  overriding whatever `pipeline.py` wrote. The name is a multi-tenant identity
  concern — two apps in one `ml-{project}` namespace must not collide. This is
  the one authoritative name source (same rationale as force-overriding `image`).
- **`image` is always rewritten** to the `image-{step}` ConfigMap indirection
  (`{app}-pipeline-images`), unless the `platform.kubecore.io/image` annotation
  gives a verbatim utility image. Supply-chain concern (Zot, scanned, versioned).
- **GPU iff the step requests `nvidia.com/gpu`** (`gpu=True` → `Resources(gpus=1)`).
  The enhancer routes GPU steps to the pool's gpu class + tolerations. No annotation.
- **`reads=` render gate** fails the release if a step declares a section that
  isn't in the composed tree. For **schema-backed** sections (`data`,
  `quantization`) it additionally requires a real on-disk source contributing the
  key — because the ConfigStore schema backfills defaults and would otherwise mask
  a rename (see `_has_config_source`).
- **Only-fill-absent everywhere else**: a developer-set nodeSelector/env/resource
  wins; the platform fills only what's missing.

## Extending it (platform team)

- **Add a structured-config schema** for a section (types + enum validation):
  add a dataclass + a `SCHEMAS` entry in `derive_tree.py`, and add the schema name
  to `config/config.yaml`'s defaults list. It's then in `SCHEMA_BACKED` and the
  gate protects it.
- **Add an annotation** to the platform vocabulary: add the key to
  `KNOWN_ANNOTATIONS` in `enhance.py` and handle it. Unknown `platform.kubecore.io/*`
  keys hard-fail the enhance (typo protection).
- **Change the compute model**: the per-step `{step}-class` dropdowns + nodeSelector
  wiring live in `enhance_class_param` / `enhance_scheduling`; they read
  `computeClasses.{cpu,gpu,all}` from the pipeline-context the operator writes.

## Contract with the platform (inputs)

`enhance.py` consumes `pipeline-context.yaml` (the `{app}-pipeline-context`
ConfigMap the k8smlapp composition writes): `app`, `project`, `namespace`,
`serviceAccountName`, `mlflow.*`, `lakefs.*`, `computeClasses.*`, `checkpoints.*`.
`compose.py`/`derive_tree.py` mount the `platform` config group from that same
context (H6). Local-dev fallbacks live in `kubecore/local-dev/` and `kubecore/config/`.

## Tests

`tests/test_engine.py` — 21 assertions, run offline: `python -m pytest tests/` or
`python tests/test_engine.py`. Covers derivation, both gates, group-swap elision,
the ADVANCED platform guard, enum validation, the size cap, byte-stability, the
app-scoped-name anti-collision enforcement, and the schema-rename gate.
