# Hera + Hydra pipeline authoring — platform integration

How the pieces fit, how a client onboards, and how the platform team operates it.
Developers author pipelines in **Python (Hera) + a Hydra config tree**; the
platform renders, wires, and delivers the Argo `WorkflowTemplate` through the
existing GitOps path.

## The pieces (3 repos + the operator)

| Artifact | Repo | Role |
|---|---|---|
| App repo | `novelcore/{project}-{app}` (seeded from the template) | developer-owned: `pipeline.py`, `config/`, `steps/*`, vendored `kubecore/` |
| Template repo | `novelcore/hera-mlops-template` | the source new ML apps are seeded from (GitHub template) |
| CI chart | `novelcore/charts` → `kubecore-ci-workflows` (≥0.12.31) | `ml-ci-build` renders/enhances/gates/commits the WFT |
| Operator + composition | `novelcore/kubecore-operator` | routes `spec.type: ml-pipeline` → seeds the app repo + gitops scaffold + CI chain; pins the CI chart version |

## The flow (author → run)

```
developer edits pipeline.py / config/ / steps/*   →  push to `dev`
     │
     ▼  GitHub webhook → WorkflowEventBinding → ml-ci-build (child cluster)
clone (token) → detect-steps (frontend=hera; build set = steps/*) → version
     ├─ build-push  (kaniko → Zot, per changed step; compose step = repo-root context)
     └─ render-hera:
          hera-render        TOKENLESS  →  python pipeline.py → raw WFT
          hera-enhance-commit  TOKEN    →  enhance (force {app}-pipeline name,
                                            wire images/env/scheduling/classes) →
                                            gate (dup-param + @script hard errors) →
                                            patch missing image-<step> keys →
                                            commit WFT to gitops repo
     ▼
gitops repo kubeapps/{app}/main/workflow-template.yaml
     ▼  ApplicationSet auto-discovery → ArgoCD selfHeal
WorkflowTemplate/{app}-pipeline in ml-{project}   ← operational state
     ▼
runs: Argo UI / agents (workflowTemplateRef); the submit form IS the config tree
```

## How a client onboards (create your own workflow KubeApp)

1. Create a `KubeApp` (`schema.kubecore.io/v1beta1`) referencing the template:
   ```yaml
   apiVersion: schema.kubecore.io/v1beta1
   kind: KubeApp
   metadata: {name: my-training, namespace: <org>}
   spec:
     kubeAppTemplateRef: hera-mlops-template   # <-- the Hera+Hydra template
     kubeProjectRef: <project>
     profile: medium
     visibility: private
   ```
2. The operator seeds `novelcore/{project}-my-training` from the template, writes
   the gitops scaffold + pipeline-context, and sets up the CI chain. **Zero operator
   or composition changes were needed to add this template — routing keys only on
   `spec.type: ml-pipeline`.**
3. The developer clones the app repo and works in Python (see the app repo's
   `README.md` / `DEVELOPER.md`). Add a parameter = add a config leaf; add a step =
   `mkdir steps/<name>` + a Dockerfile + one `step()` line.

## Multi-tenant / scale properties

- **No cross-app WFT collision**: the enhancer forces the WFT name to
  `{app}-pipeline`, so two apps in the same namespace never overwrite each other
  (a template's hardcoded pipeline name can't leak across apps).
- **App-scoped image ConfigMap**: `{app}-pipeline-images` — no cross-app image bleed.
- **Slash-anchored gitops discovery**: `*/{app}/{branch}/…` — `app` ≠ `app-training`.
- **Per-step compute**: each step gets a `{step}-class` dropdown (the KubePool's
  allowed classes) + `{step}-cpu/mem` sizing knobs; GPU steps route to the gpu class.
- **Credential split**: untrusted `pipeline.py` runs tokenless (no GitHub token, a
  locked-down executor SA); only the platform-owned commit step holds the token.

## No Crossplane MR reverts

The WFT is a Crossplane `RepositoryFile` MR with
`managementPolicies: [Observe, Create, Delete]` + `overwriteOnCreate: false`:
Crossplane seeds a one-task placeholder **once** and never Updates, so CI's git
commit of the real WFT is never reverted. ArgoCD `selfHeal` then keeps the live
WFT equal to what CI committed (direct `kubectl` edits are reverted — everything
goes through git, by design).

## Platform operations

- **Deploy the Hera CI path**: merge `novelcore/charts` PR (chart ≥0.12.31), which
  auto-publishes via `release-charts.yml`; the operator pins it
  (`internal/operators/kubepool/phases/syncing_tools.go`, `ciWorkflows.version`) and
  installs it on every `features.ml=true` KubePool reconcile.
- **Extend the engine** (schemas, annotations, compute model): see
  `kubecore/README.md`.
- **Coexistence**: `ml-ci-build` detects each app's frontend and routes it to the
  matching render path, so apps using other authoring frontends are unaffected.

## Verification

- Offline: `./run.sh` (render→enhance→compose) and `python tests/test_engine.py`
  (23 assertions) in any app repo.
- In-cluster (validated end-to-end): create KubeApp → operator
  seeds the app repo → push → CI clone (token) → detect Hera frontend → build the
  step images → **render-hera tokenless** → enhance (forced `{app}-pipeline`
  name) → gate → commit to gitops → **no Crossplane revert** (WFT/images MRs stay
  Synced) → ArgoCD sync → **submittable** WFT → real run: `compose-and-validate`
  composed+validated the Hydra config into a real `params.yaml`, and the steps
  ran, with a GPU step scheduled with the correct GPU request + the CI-built
  image. A GPU container is gated only on GPU node capacity in the zone (an
  accelerator stockout during validation is not a platform issue).
- Whole-node GPU sizing: GPU compute-class allocatable subtracts an extra GPU
  node headroom (nvidia device-plugin + driver + larger system reservation) so a
  GPU step actually fits a fresh accelerator node — verified in-cluster (an
  oversized request never scheduled a GPU node; the corrected request triggered
  the scale-up).
