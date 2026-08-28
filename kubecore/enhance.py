"""Platform-owned enhancer. The successor of render_wft.py's platform half.

Takes the RAW WorkflowTemplate a developer's vanilla-Hera pipeline.py rendered
and a platform context (pipeline-context.yaml, sourced from the
`{app}-pipeline-context` ConfigMap the operator writes into the app namespace)
and injects everything the developer never had to think about: namespace,
service account, secrets, compute-class routing, node scheduling, image
supply-chain wiring, per-run sizing knobs ({step}-cpu/{step}-mem/{step}-disk via
podSpecPatch), per-step compute-class dropdowns ({step}-class, matching the
live render-wft), dynamic enum dropdowns (compute classes from the pool,
dataset refs from the dataset-catalog probe), and the read-only pipeline-info
form parameter.

Golden rule: only fill ABSENT keys. If a developer already set a field (a
nodeSelector, an env var, a memory request) we leave it alone. The one
deliberate exception is `image`: the platform always overwrites it, because the
image running in a pipeline step is a supply-chain concern (Zot registry,
scanned, versioned) that a developer's local placeholder must never leak into
(see CON-18 / Zot registry). The `platform.kubecore.io/image` annotation is the
sanctioned escape hatch for unbuilt utility images.

Compute model (matches the LIVE WorkflowTemplate on gke-dev): each step gets a
`{step}-class` submit-form parameter defaulting to the pool's cpu/gpu tier
class; the step's nodeSelector is
`platform.kubecore.io/nodegroup-type: {{workflow.parameters.{step}-class}}`, so
the submitter routes each step onto any of the KubePool's allowed classes. The
allowed classes come from the pool (computeClasses.all).

GPU convention (matches live render-wft): a template is a GPU step iff its
rendered resources request `nvidia.com/gpu`. No annotation.

Platform annotation vocabulary (all optional, validated — an unknown
`platform.kubecore.io/*` key FAILS the enhance, typo protection; this is the
release-gate rule):

  platform.kubecore.io/compute-class  pin a named class from the pool
                                      catalog (computeClasses.all); disables
                                      the {step}-class dropdown for this step
  platform.kubecore.io/inject         "none" or comma list like
                                      "mlflow=false" — opt out of
                                      data-source env/secret wiring
  platform.kubecore.io/source         images-ConfigMap key differs
                                      from step name (shared image)
  platform.kubecore.io/image          use this image verbatim, skip
                                      the ConfigMap indirection
  platform.kubecore.io/shm            /dev/shm emptyDir sizeLimit (e.g. "4Gi")
  platform.kubecore.io/workspace      per-run RWO PVC at /workspace (e.g. "50Gi")
  platform.kubecore.io/hpc            future MeluXina leg (accepted,
                                      not acted on)

This file deals in plain dicts (parsed with pyyaml), not Hera objects. The
enhancer runs at CI render time, after Hera has already produced Argo-native
YAML, so it has no reason to depend on Hera at all — this is what keeps the
platform layer decoupled from whatever authoring library or version a
developer's repo pins.
"""

import argparse
from pathlib import Path

import yaml

from kubecore.meluxina import enhance_hpc, hpc_classes

ANNOTATION_PREFIX = "platform.kubecore.io/"
KNOWN_ANNOTATIONS = {"compute-class", "inject", "source", "image", "shm", "workspace", "hpc"}
WORKSPACE_MOUNT = "/workspace"
# Mount path for the Zitadel machine-key secret backing MLflow bearer auth
# (#868) — matches kubeline's working reference (/etc/mlflow-svc).
MLFLOW_SVC_MOUNT = "/etc/mlflow-svc"


class EnhanceError(Exception):
    pass


def _param_names(parameters: list) -> set:
    return {p["name"] for p in parameters}


def images_configmap(ctx: dict) -> str:
    """The pipeline-images ConfigMap name — {app}-pipeline-images, from
    context (NOT a hardcoded literal)."""
    app = ctx.get("app") or ctx.get("pipelineName")
    if not app:
        raise EnhanceError("context is missing 'app' — cannot derive the pipeline-images ConfigMap name")
    return f"{app}-pipeline-images"


# ---------------------------------------------------------------- annotations


def platform_annotations(step: dict) -> dict:
    """Extract + validate this step's platform.kubecore.io/* annotations.

    Unknown platform.kubecore.io/* keys fail the enhance (typo
    protection — the release-gate rule).
    """
    annotations = step.get("metadata", {}).get("annotations", {})
    result = {}
    for key, value in annotations.items():
        if not key.startswith(ANNOTATION_PREFIX):
            continue
        short = key[len(ANNOTATION_PREFIX):]
        if short not in KNOWN_ANNOTATIONS:
            known = ", ".join(sorted(ANNOTATION_PREFIX + k for k in KNOWN_ANNOTATIONS))
            raise EnhanceError(
                f"step '{step['name']}': unknown platform annotation '{key}' (known: {known})"
            )
        result[short] = value
    return result


def inject_disabled(annots: dict, source_name: str) -> bool:
    """True if `platform.kubecore.io/inject` opts this step out of
    the named data source ("mlflow", "lakefs")."""
    spec = annots.get("inject")
    if spec is None:
        return False
    if spec.strip() == "none":
        return True
    return any(item.strip() == f"{source_name}=false" for item in spec.split(","))


# ------------------------------------------------------------------ metadata


def pipeline_wft_name(ctx: dict) -> str:
    """The authoritative WorkflowTemplate name: {app}-pipeline. This is the
    convention the k8smlapp composition seeds the placeholder with
    ({{ $appName }}-pipeline), so the CI-rendered WFT must match it — otherwise
    it lands as a DIFFERENT object in the shared ml-{project} namespace."""
    app = ctx.get("app")
    if not app:
        raise EnhanceError("context is missing 'app' — cannot derive the WorkflowTemplate name")
    return f"{app}-pipeline"


def enhance_metadata(wft: dict, ctx: dict) -> None:
    meta = wft["metadata"]

    # FORCE the WFT name to the app-authoritative {app}-pipeline, OVERRIDING
    # whatever pipeline.py declared. The name is an IDENTITY / multi-tenant
    # isolation concern the platform must own — not a developer choice. A
    # template's hardcoded pipeline("yolo-training-pipeline") copied into every
    # seeded app would otherwise make app B's render OVERWRITE app A's live WFT
    # in the shared ml-{project} namespace (the cross-app contamination class).
    # This is the one-and-only name authority, same rationale as force-
    # overriding `image` (a supply-chain concern) — see enhance_image.
    authoritative = pipeline_wft_name(ctx)
    if meta.get("name") != authoritative:
        meta["name"] = authoritative
    meta.setdefault("namespace", ctx["namespace"])

    labels = meta.setdefault("labels", {})
    labels.setdefault("platform.kubecore.io/project", ctx["project"])
    labels.setdefault("platform.kubecore.io/app", ctx["app"])
    labels.setdefault("platform.kubecore.io/managed-by", "kubecore-ci")

    annotations = meta.setdefault("annotations", {})
    annotations.setdefault("platform.kubecore.io/generated-by", "ml-ci-build/hera-render")
    annotations.setdefault("platform.kubecore.io/cost-tracking", "enabled")


def enhance_spec_top_level(spec: dict, ctx: dict) -> None:
    spec.setdefault("serviceAccountName", ctx["serviceAccountName"])


def step_templates(spec: dict) -> list:
    """Container-backed templates only (skip the DAG template itself)."""
    return [t for t in spec["templates"] if "container" in t]


# ----------------------------------------------------------------- arguments


def enhance_arguments(spec: dict, ctx: dict, steps: list) -> None:
    args = spec.setdefault("arguments", {})
    parameters = args.setdefault("parameters", [])
    existing = _param_names(parameters)
    cm = images_configmap(ctx)

    def add(param: dict) -> None:
        if param["name"] not in existing:
            parameters.append(param)
            existing.add(param["name"])

    add({"name": "mlflow-endpoint", "value": ctx["mlflow"]["trackingUri"]})
    add({"name": "lakefs-endpoint", "value": ctx["lakefs"]["endpoint"]})
    add({"name": "lakefs-repo", "value": ctx["lakefs"]["repository"]})

    for step in steps:
        annots = platform_annotations(step)
        if "image" in annots:
            continue  # verbatim utility image: no ConfigMap indirection, no parameter
        step_name = step["name"]
        configmap_key = annots.get("source", step_name)
        add(
            {
                "name": f"image-{step_name}",
                "valueFrom": {
                    "configMapKeyRef": {
                        "name": cm,
                        "key": configmap_key,
                    }
                },
            }
        )


# ------------------------------------------------------------------- compute


def is_gpu_step(step: dict) -> bool:
    """Live render-wft convention: GPU iff resources request nvidia.com/gpu."""
    resources = step["container"].get("resources", {})
    return "nvidia.com/gpu" in resources.get("requests", {}) or "nvidia.com/gpu" in resources.get(
        "limits", {}
    )


def step_tier(step: dict) -> str:
    return "gpu" if is_gpu_step(step) else "cpu"


def tier_default(ctx: dict, step: dict) -> dict:
    """The class a step defaults to: its own tier's default, else the OTHER
    tier's (an ml environment may declare only cpu or only gpu flavours —
    PRD-1124 D-02 — and a gpu-declaring step submitted with a cpu class
    legitimately runs CPU-only, see enhance_gpu_routing). Neither tier
    present is an authoring error: the environment declares no compute."""
    classes = ctx["computeClasses"]
    tier = step_tier(step)
    for candidate in (tier, "cpu" if tier == "gpu" else "gpu"):
        if classes.get(candidate):
            return classes[candidate]
    raise EnhanceError(
        f"step '{step['name']}': the target ml environment declares no compute flavours "
        f"(environments[].ml.cpu / .gpu are both empty) — add at least one"
    )


def default_class_name(step: dict, ctx: dict, annots: dict) -> str:
    """The step's default compute class name: a pinned compute-class
    annotation wins, else the pool's gpu/cpu tier default."""
    pinned = annots.get("compute-class")
    if pinned is not None:
        if pinned not in {c["name"] for c in ctx["computeClasses"]["all"]}:
            known = ", ".join(c["name"] for c in ctx["computeClasses"]["all"])
            raise EnhanceError(
                f"step '{step['name']}': compute-class '{pinned}' not in pool catalog ({known})"
            )
        return pinned
    return tier_default(ctx, step)["name"]


def resolve_compute_class(step: dict, ctx: dict, annots: dict) -> dict:
    """The class config used for whole-node sizing/tolerations. Pinned class
    merges over its tier default (catalog entries omit toleration config)."""
    name = default_class_name(step, ctx, annots)
    for cls in ctx["computeClasses"]["all"]:
        if cls["name"] == name:
            base = ctx["computeClasses"].get(cls.get("tier", step_tier(step))) or tier_default(ctx, step)
            return {**base, **cls}
    # tier default itself (name came straight from computeClasses.{cpu,gpu})
    return tier_default(ctx, step)


def catalog_toleration_pairs(ctx: dict, step: dict, annots: dict) -> list:
    """Unique (tolerationKey, tolerationValue) pairs a step must tolerate.

    A pinned (compute-class annotation) step tolerates only its pinned
    class's taint — it can never be routed elsewhere. Every other step
    tolerates the FULL catalog (#865): the {step}-class dropdown may route
    it onto any class pool at submit time, and tolerations render statically.
    """
    classes = list(ctx["computeClasses"]["all"])
    if "compute-class" in annots:
        classes = [c for c in classes if c["name"] == annots["compute-class"]] or classes
    pairs = []
    for cls in classes:
        tier_default = ctx["computeClasses"].get(cls.get("tier", "cpu"), {})
        merged = {**tier_default, **cls}
        pair = (merged.get("tolerationKey"), merged.get("tolerationValue"))
        if pair[0] and pair not in pairs:
            pairs.append(pair)
    return pairs


def gpu_class_names(ctx: dict) -> list:
    """Names of every gpu-tier class in the pool catalog (dedup, ordered)."""
    names = [c["name"] for c in ctx["computeClasses"]["all"] if c.get("tier") == "gpu"]
    tier_default = ctx["computeClasses"].get("gpu", {}).get("name")
    if tier_default and tier_default not in names:
        names.insert(0, tier_default)
    return names


def enhance_gpu_routing(spec: dict, step: dict, ctx: dict, annots: dict) -> None:
    """Make a gpu-declaring step class-routable at submit time (#865 fix 2).

    A gpu=True step used to carry a STATIC nvidia.com/gpu request +
    GPU-only scheduling, so submitting {step}-class=<cpu class> produced a
    pod that targets a CPU node while requesting a GPU — permanently
    Unschedulable. CPU smoke tests and PTQ-on-CPU (routine on the kubeline
    template, which had separate cpu/gpu step variants) were impossible.

    Mechanics — static render, runtime routing, using only PROVEN eval
    sites (the platform's podSpecPatch sizing model + {{=}} in DAG task
    arguments, the one expression site Argo reliably evaluates):
      1. strip the developer's typed nvidia.com/gpu request/limit (it stays
         the gpu-capability DECLARATION, detected before this runs);
      2. give the template a `gpus` input (default "1" — direct invocation
         keeps GPU behavior);
      3. extend the sizing podSpecPatch with nvidia.com/gpu:
         "{{inputs.parameters.gpus}}" under requests+limits — a "0" request
         for an extended resource is ignored by the scheduler, so the pod
         lands on CPU nodes;
      4. every DAG task invoking this step passes gpus via an {{=}}
         expression mapping the {step}-class param to '1' iff it names a
         gpu-tier class.

    Pinned (compute-class annotation) steps are untouched: their class is a
    render-time constant, and enhance_scheduling keeps the static request.
    """
    if "compute-class" in annots or not is_gpu_step(step):
        return
    step_name = step["name"]

    # 1. The typed request was the capability declaration; routing owns it now.
    resources = step["container"].get("resources", {})
    resources.get("requests", {}).pop("nvidia.com/gpu", None)
    resources.get("limits", {}).pop("nvidia.com/gpu", None)

    # 2. Template input with a GPU-preserving default.
    inputs = step.setdefault("inputs", {})
    input_params = inputs.setdefault("parameters", [])
    if not any(p.get("name") == "gpus" for p in input_params):
        input_params.append({"name": "gpus", "default": "1"})

    # 3. Extend the sizing patch (enhance_sizing_knobs ran first and always
    #    sets it) with the parameterized GPU quantity — YAML-level merge into
    #    the same `main` container entry, never a second patch document.
    patch_obj = yaml.safe_load(step.get("podSpecPatch") or "") or {}
    containers = patch_obj.setdefault("containers", [])
    main = next((c for c in containers if c.get("name") == "main"), None)
    if main is None:
        main = {"name": "main"}
        containers.append(main)
    res = main.setdefault("resources", {})
    res.setdefault("requests", {})["nvidia.com/gpu"] = "{{inputs.parameters.gpus}}"
    res.setdefault("limits", {})["nvidia.com/gpu"] = "{{inputs.parameters.gpus}}"
    step["podSpecPatch"] = yaml.safe_dump(patch_obj, default_flow_style=False)

    # 4. DAG tasks pass gpus from the class param.
    gpu_names = gpu_class_names(ctx)
    expr_list = ", ".join(f"'{n}'" for n in gpu_names)
    expr = (
        f"{{{{=workflow.parameters['{step_name}-class'] in [{expr_list}] ? '1' : '0'}}}}"
    )
    for template in spec["templates"]:
        for task in (template.get("dag", {}) or {}).get("tasks", []):
            if task.get("template") != step_name:
                continue
            args = task.setdefault("arguments", {})
            params = args.setdefault("parameters", [])
            if not any(p.get("name") == "gpus" for p in params):
                params.append({"name": "gpus", "value": expr})


def enhance_class_param(spec: dict, step: dict, ctx: dict, annots: dict) -> None:
    """Per-step compute-class dropdown ({step}-class), matching the live WFT.

    Default = the step's tier class; enum = every class of that tier from the
    pool catalog. A `compute-class` annotation PINS the class (no dropdown —
    the nodeSelector uses the literal name)."""
    if "compute-class" in annots:
        return  # pinned: nodeSelector uses the literal name (enhance_scheduling)
    parameters = spec["arguments"]["parameters"]
    existing = _param_names(parameters)
    step_name = step["name"]
    param_name = f"{step_name}-class"
    if param_name in existing:
        return
    tier = step_tier(step)
    default = tier_default(ctx, step)["name"]
    # Full pool catalog, ALL tiers, ALWAYS attached (#865): the old
    # tier-filtered list collapsed to length 1 on a one-class-per-tier pool,
    # and enum-only-when->1 then rendered a FREE-TEXT box (typo -> forever-
    # Pending run). Cross-tier entries are legitimate choices: a gpu-capable
    # step submitted with a cpu class runs CPU-only (see the gpus routing in
    # enhance_gpu_routing).
    options = [c["name"] for c in ctx["computeClasses"]["all"]]
    if default not in options:
        options.insert(0, default)
    # HPC classes (kubecore-operator#1191): every step can be placed on the
    # HPC provider per run — the class is provider-prefixed (meluxina-gpu),
    # and kubecore/meluxina.py gates the in-cluster/Slurm twin pair on it.
    hpc_names = [c["name"] for c in hpc_classes(ctx)]
    options += [n for n in hpc_names if n not in options]
    param = {
        "name": param_name,
        "value": default,
        "enum": options,
        "description": (
            f"Compute class for the '{step_name}' step "
            f"(the KubePool's class catalog; default = its {tier} tier"
            + ("; meluxina-* classes run the step as a Slurm job on MeluXina — "
               "queue time applies, see pipeline-info" if hpc_names else "")
            + ")."
        ),
    }
    parameters.append(param)


def enhance_image(step: dict, ctx: dict, annots: dict) -> None:
    # Always overwritten: image is a supply-chain concern, not a developer
    # choice (see module docstring). The `image` annotation is the sanctioned
    # escape hatch (verbatim utility image).
    if "image" in annots:
        step["container"]["image"] = annots["image"]
        return
    step_name = step["name"]
    step["container"]["image"] = f"{{{{workflow.parameters.image-{step_name}}}}}"


def enhance_env(step: dict, ctx: dict, annots: dict) -> None:
    container = step["container"]
    env = container.setdefault("env", [])
    existing = {e["name"] for e in env}  # developer-set keys win

    def add(entry: dict) -> None:
        if entry["name"] not in existing:
            env.append(entry)
            existing.add(entry["name"])

    if not inject_disabled(annots, "mlflow"):
        add({"name": "MLFLOW_TRACKING_URI", "value": ctx["mlflow"]["trackingUri"]})
        # MLflow AUTH (#868): the tracking server sits behind mlflow-oidc-auth,
        # so a bare MLFLOW_TRACKING_URI 401s on any real logging/registry call.
        # Mirror kubeline's working reference: the step mints a Zitadel machine
        # JWT from the key file (mounted by enhance_volumes from
        # mlflow.svcSecret) and sends it as the bearer. Injected only when the
        # context carries the OIDC coordinates — a context without them (OIDC
        # off) keeps today's env exactly.
        mlflow_ctx = ctx.get("mlflow") or {}
        oidc_domain = (mlflow_ctx.get("oidcDomain") or "").strip()
        oidc_project = (mlflow_ctx.get("oidcProjectId") or "").strip()
        if oidc_domain and oidc_project and (mlflow_ctx.get("svcSecret") or "").strip():
            add({"name": "MLFLOW_TRACKING_AUTH", "value": "zitadel"})
            add({"name": "ZITADEL_DOMAIN", "value": oidc_domain})
            add({"name": "ZITADEL_MLFLOW_PROJECT_ID", "value": oidc_project})
            add({"name": "ZITADEL_MACHINE_KEY_FILE",
                 "value": f"{MLFLOW_SVC_MOUNT}/ZITADEL_MACHINE_KEY"})
    if not inject_disabled(annots, "lakefs"):
        add({"name": "LAKEFS_ENDPOINT", "value": ctx["lakefs"]["endpoint"]})
        keys = ctx["lakefs"]["adminSecretKeys"]
        secret_name = ctx["lakefs"]["adminSecret"]
        add(
            {
                "name": "LAKEFS_ACCESS_KEY",
                "valueFrom": {"secretKeyRef": {"name": secret_name, "key": keys["accessKey"]}},
            }
        )
        add(
            {
                "name": "LAKEFS_SECRET_KEY",
                "valueFrom": {"secretKeyRef": {"name": secret_name, "key": keys["secretKey"]}},
            }
        )
    # checkpoint bucket/prefix (the live WFT injects these; step code derives
    # checkpoint locations from them). Only when the context declares them.
    checkpoints = ctx.get("checkpoints")
    if checkpoints:
        add({"name": "CHECKPOINT_BUCKET", "value": str(checkpoints.get("bucket", ""))})
        add({"name": "CHECKPOINT_PREFIX", "value": str(checkpoints.get("prefix", ""))})


# How long a step may sit Pending before the platform calls it unschedulable.
# A step whose node cannot be created (cloud capacity stockout, exhausted quota,
# a class whose pool is at max) otherwise waits FOREVER with no signal: Argo does
# not time out a Pending pod, and the developer sees a run that is neither
# succeeding nor failing. Alexandra's run sat Pending 17 HOURS this way
# (2026-07-15), and a GPU stockout reproduced it exactly (2026-07-16: 105
# FailedScaleUp over 2d5h, still Pending).
#
# GPU steps get longer: accelerator pools legitimately take minutes to scale from
# zero, and a transient stockout often clears. The point is not to fail fast —
# it is to fail *visibly* instead of hanging silently until someone notices.
PENDING_DEADLINE_SECONDS = 1800  # CPU steps: 30 min
GPU_PENDING_DEADLINE_SECONDS = 5400  # GPU steps: 90 min (scale-from-zero + retries)


def enhance_scheduling(step: dict, ctx: dict, annots: dict) -> None:
    compute_class = resolve_compute_class(step, ctx, annots)
    gpu = is_gpu_step(step)

    # nodeSelector: pinned class -> literal name; else the {step}-class param.
    if "compute-class" in annots:
        selector_value = annots["compute-class"]
    else:
        selector_value = f"{{{{workflow.parameters.{step['name']}-class}}}}"
    step.setdefault("nodeSelector", {}).setdefault(
        "platform.kubecore.io/nodegroup-type", selector_value
    )

    # PRD-858: an ml-environment-targeted app pins EVERY step onto that
    # environment's own scale-to-zero class pools — the nodeSelector gains the
    # environment label and the tolerations gain the kubenv env-exclusivity
    # taint, mirroring exactly what kubenv-gcp stamps on those pools. Absent
    # mlEnvironment (legacy project-wide apps) leaves output byte-for-byte
    # unchanged. Same fill-absent discipline as everything else here.
    ml_env = ctx.get("mlEnvironment") or {}
    ml_env_name = (ml_env.get("name") or "").strip()
    if ml_env_name:
        # The operator's selector is the authority (PRD-1124): project +
        # environment, plus app for an APP-dedicated ml environment (D-07,
        # spec.dedicatedEnvironment.ml → app-tainted pools). Copy it verbatim
        # so the keys never drift from what kubenv-gcp stamps on the pools;
        # fall back to the legacy project/environment pair for a context
        # that predates the map (pipeline-context rendered by an older
        # operator). Project scoping is load-bearing either way: environment
        # names repeat across projects (live cross-tenant leak, 2026-08-27).
        selector = ml_env.get("nodeSelector") or {}
        if not selector:
            selector = {"platform.kubecore.io/environment": ml_env_name}
            project = (ctx.get("project") or "").strip()
            if project:
                selector["platform.kubecore.io/project"] = project
        for key, value in selector.items():
            step["nodeSelector"].setdefault(key, value)

    # Bound how long this step may sit unschedulable (see the constants above).
    # fill-absent: a developer who set their own deadline keeps it.
    step.setdefault(
        "activeDeadlineSeconds",
        GPU_PENDING_DEADLINE_SECONDS if gpu else PENDING_DEADLINE_SECONDS,
    )

    tolerations = step.setdefault("tolerations", [])
    # Tolerate EVERY catalog class's taint, not just the default class's
    # (#865): the {step}-class dropdown routes at SUBMIT time via the
    # nodeSelector param, but tolerations are rendered statically — a run
    # submitted onto any non-default class (including cross-tier) must
    # already tolerate that pool's taint or it Pends forever. Tolerations
    # are permissive; the nodeSelector still pins the exact pool.
    for pair in catalog_toleration_pairs(ctx, step, annots):
        key, value = pair
        if not any(
            t.get("key") == key and t.get("value") == value for t in tolerations
        ):
            tolerations.append(
                {"key": key, "operator": "Equal", "value": value, "effect": "NoSchedule"}
            )
    if gpu and not any(t.get("key") == "nvidia.com/gpu" for t in tolerations):
        # GKE auto-taints accelerator nodes; tolerate it so the pod schedules.
        # Harmless on CPU nodes (tolerations never attract, only permit).
        tolerations.append({"key": "nvidia.com/gpu", "operator": "Exists", "effect": "NoSchedule"})
    if ml_env_name:
        # The pools carry the env-exclusivity taint (kubenv={env}) — or the
        # app-exclusivity taint (app={app}) for a dedicated ml environment
        # (D-07); the operator's mlEnvironment.tolerations says which. Without
        # it every step Pends forever on its own pools (PRD-858).
        env_tolerations = ml_env.get("tolerations") or [
            {"key": "platform.kubecore.io/kubenv", "operator": "Equal",
             "value": ml_env_name, "effect": "NoSchedule"}
        ]
        for tol in env_tolerations:
            if not any(t.get("key") == tol.get("key") and t.get("value") == tol.get("value") for t in tolerations):
                tolerations.append({
                    "key": tol.get("key"), "operator": tol.get("operator", "Equal"),
                    "value": tol.get("value"), "effect": tol.get("effect", "NoSchedule"),
                })

    # Sizing. Developer-set requests always win; a missing limit mirrors the
    # (possibly developer-set) request. Only fields the developer left entirely
    # unset get whole-node allocatable sizing (requests==limits).
    resources = step["container"].setdefault("resources", {})
    requests = resources.setdefault("requests", {})
    limits = resources.setdefault("limits", {})
    allocatable = compute_class["allocatable"]
    defaults = {"cpu": str(allocatable["cpu"]), "memory": f"{allocatable['memoryGiB']}Gi"}
    for field_name, default in defaults.items():
        requests.setdefault(field_name, default)
        limits.setdefault(field_name, requests[field_name])  # limits follow requests
    # Disk request floor (#892) — request only, deliberately NO limit (see
    # DISK_REQUEST_DEFAULT): kubeline-parity eviction shield for every step.
    requests.setdefault("ephemeral-storage", DISK_REQUEST_DEFAULT)
    # NOTE (#865): the GPU request is deliberately NOT set here for
    # class-routable steps — enhance_gpu_routing moves it into the
    # podSpecPatch keyed to the runtime class choice, so a gpu-capable step
    # submitted with a cpu class carries nvidia.com/gpu: "0" (which the
    # scheduler ignores) instead of an unsatisfiable GPU request on a CPU
    # node. Pinned (compute-class annotation) gpu steps keep the static
    # request: their class never changes at submit time.
    if gpu and "compute-class" in annots:
        requests.setdefault("nvidia.com/gpu", "1")
        limits.setdefault("nvidia.com/gpu", "1")


def _parse_cpu(value) -> float:
    """Kubernetes CPU quantity -> cores ("500m" -> 0.5, "2" -> 2.0)."""
    text = str(value).strip()
    if text.endswith("m"):
        return float(text[:-1]) / 1000.0
    return float(text)


_MEM_UNITS = {
    "Ki": 1 / (1024 * 1024), "Mi": 1 / 1024, "Gi": 1.0, "Ti": 1024.0,
    "K": 1000 / 1024**3, "M": 1000**2 / 1024**3, "G": 1000**3 / 1024**3,
}


def _parse_mem_gib(value) -> float:
    """Kubernetes memory quantity -> GiB ("512Mi" -> 0.5, "27Gi" -> 27.0)."""
    text = str(value).strip()
    for suffix, factor in _MEM_UNITS.items():
        if text.endswith(suffix):
            return float(text[: -len(suffix)]) * factor
    return float(text) / 1024**3  # bare bytes


def validate_scheduling(step: dict, compute_class: dict, gpu: bool = None, env_has_gpu: bool = True) -> None:
    """Fail the render if this step can NEVER schedule on its compute class.

    The step's requests are compared against the class's whole-node budget
    (capacity minus the node's kube-reservation and DaemonSet footprint, computed
    by the platform). A request above that budget produces a pod that sits
    Pending on "Insufficient cpu/memory" forever — a run that burns hours before
    anyone notices. Catching it at render time turns a silent hang into a build
    error naming the step, the class, and the number that does not fit.

    Only what is knowable at render time is checked: a per-run {step}-cpu/-mem
    override can still overshoot, so the same budget is what those knobs default
    to.
    """
    allocatable = compute_class.get("allocatable")
    if not allocatable:
        return  # unknown machine type: the platform omits allocatable, nothing to check
    requests = step["container"].get("resources", {}).get("requests", {})
    step_name = step["name"]
    class_name = compute_class.get("name", "?")
    machine = compute_class.get("machineType", "?")

    if "cpu" in requests:
        want = _parse_cpu(requests["cpu"])
        if want > allocatable["cpu"]:
            raise EnhanceError(
                f"step '{step_name}' requests {requests['cpu']} cpu but compute class "
                f"'{class_name}' ({machine}) can only give a pod {allocatable['cpu']} cpu — "
                f"the pod would never schedule. Lower the request or pick a bigger class."
            )
    if "memory" in requests:
        want = _parse_mem_gib(requests["memory"])
        if want > allocatable["memoryGiB"]:
            raise EnhanceError(
                f"step '{step_name}' requests {requests['memory']} memory but compute class "
                f"'{class_name}' ({machine}) can only give a pod {allocatable['memoryGiB']}Gi — "
                f"the pod would never schedule. Lower the request or pick a bigger class."
            )
    disk_size_gb = compute_class.get("diskSizeGb")
    if "ephemeral-storage" in requests and disk_size_gb:
        want = _parse_mem_gib(requests["ephemeral-storage"])
        if want > disk_size_gb:
            raise EnhanceError(
                f"step '{step_name}' requests {requests['ephemeral-storage']} ephemeral-storage "
                f"but compute class '{class_name}' ({machine}) has only a {disk_size_gb}GB boot "
                f"disk — the pod would never schedule. Lower disk= or pick a bigger class."
            )
    # A gpu-declaring step on an accelerator-less class can never schedule —
    # unless the target ml environment declares NO gpu flavours at all
    # (PRD-1124 D-02: cpu-only environment). Then the step defaults to the cpu
    # class and the gpus routing (enhance_gpu_routing) sets its nvidia.com/gpu
    # request to 0, so it schedules and runs CPU-only by design.
    if (is_gpu_step(step) if gpu is None else gpu) and env_has_gpu and not compute_class.get("guestAccelerator"):
        raise EnhanceError(
            f"step '{step_name}' requests a GPU but compute class '{class_name}' "
            f"({machine}) has no accelerator — the pod would never schedule."
        )


# Ephemeral-storage baseline for every step (#892). Unlike cpu/mem there is
# no whole-node "allocatable disk" in the class catalog, so the platform sets
# a flat floor: kubeline requested 10Gi on its heavy steps (model-training,
# model-quantization) and that proved sufficient; qat-class steps that need
# more declare disk="20Gi" in pipeline.py. Request-only, never a limit: the
# request is the eviction shield (a pod under its request is evicted LAST and
# the scheduler won't place it on a nearly-full node); a limit would instead
# kill legitimate spiky export scratch (onnx2tf temp trees).
DISK_REQUEST_DEFAULT = "10Gi"


def enhance_sizing_knobs(spec: dict, step: dict, compute_class: dict) -> None:
    """Per-run sizing knobs, mirroring the live WFT's podSpecPatch model.

    Emits {step}-cpu / {step}-mem / {step}-disk workflow parameters
    (defaults = the developer-set requests when present, else the class's
    whole-node allocatable for cpu/mem and DISK_REQUEST_DEFAULT for disk)
    and wires them into the template via podSpecPatch so a submitter can
    dial a single run down/up without re-rendering. The developer writes
    nothing for this.

    MUST run before enhance_scheduling so "developer-set" can still be
    told apart from platform whole-node fill.
    """
    parameters = spec["arguments"]["parameters"]
    existing = _param_names(parameters)
    step_name = step["name"]
    dev_requests = step["container"].get("resources", {}).get("requests", {})
    allocatable = compute_class["allocatable"]

    knobs = {
        f"{step_name}-cpu": str(dev_requests.get("cpu", allocatable["cpu"])),
        f"{step_name}-mem": str(dev_requests.get("memory", f"{allocatable['memoryGiB']}Gi")),
        f"{step_name}-disk": str(dev_requests.get("ephemeral-storage", DISK_REQUEST_DEFAULT)),
    }
    for name, default in knobs.items():
        if name not in existing:
            parameters.append(
                {
                    # arguments.parameters need `value` (not `default`) — Argo
                    # rejects a workflow whose arguments param lacks value/valueFrom
                    "name": name,
                    "value": default,
                    "description": f"Per-run resource override for step '{step_name}'.",
                }
            )
            existing.add(name)

    # ephemeral-storage under requests only — see DISK_REQUEST_DEFAULT.
    patch = (
        "containers:\n"
        "- name: main\n"
        "  resources:\n"
        "    requests:\n"
        f'      cpu: "{{{{workflow.parameters.{step_name}-cpu}}}}"\n'
        f'      memory: "{{{{workflow.parameters.{step_name}-mem}}}}"\n'
        f'      ephemeral-storage: "{{{{workflow.parameters.{step_name}-disk}}}}"\n'
        "    limits:\n"
        f'      cpu: "{{{{workflow.parameters.{step_name}-cpu}}}}"\n'
        f'      memory: "{{{{workflow.parameters.{step_name}-mem}}}}"\n'
    )
    step.setdefault("podSpecPatch", patch)


# ------------------------------------------------------------------- volumes


def enhance_volumes(step: dict, annots: dict, ctx: dict = None) -> None:
    """Inject /dev/shm (sizable via shm annotation), opt-in per-run RWO
    workspace PVC (workspace annotation), and — when the context carries the
    MLflow OIDC coordinates (#868) — the Zitadel machine-key secret mount the
    MLflow auth env from enhance_env points at. Raw developer volumes win
    (only-fill-absent)."""
    container = step["container"]
    volume_mounts = container.setdefault("volumeMounts", [])
    mounted = {m.get("mountPath") for m in volume_mounts}
    volumes = step.setdefault("volumes", [])
    vol_names = {v.get("name") for v in volumes}

    # MLflow Zitadel machine key (#868) — same gating as the env injection in
    # enhance_env: only when the context has domain + project id + secret, and
    # the developer hasn't opted the step out of mlflow injection.
    mlflow_ctx = (ctx or {}).get("mlflow") or {}
    if (
        not inject_disabled(annots, "mlflow")
        and (mlflow_ctx.get("oidcDomain") or "").strip()
        and (mlflow_ctx.get("oidcProjectId") or "").strip()
        and (mlflow_ctx.get("svcSecret") or "").strip()
        and MLFLOW_SVC_MOUNT not in mounted
    ):
        if "mlflow-svc" not in vol_names:
            volumes.append({
                "name": "mlflow-svc",
                "secret": {"secretName": mlflow_ctx["svcSecret"].strip()},
            })
        volume_mounts.append({
            "name": "mlflow-svc", "mountPath": MLFLOW_SVC_MOUNT, "readOnly": True,
        })
        mounted.add(MLFLOW_SVC_MOUNT)

    # /dev/shm — PyTorch dataloader shared memory. Default 1Gi; shm annotation
    # overrides. Only if the developer didn't already mount /dev/shm.
    if "/dev/shm" not in mounted:
        shm_size = annots.get("shm", "1Gi")
        if "dshm" not in vol_names:
            volumes.append({"name": "dshm", "emptyDir": {"medium": "Memory", "sizeLimit": shm_size}})
        volume_mounts.append({"name": "dshm", "mountPath": "/dev/shm"})

    # Opt-in per-run workspace PVC (RWO). volumeClaimTemplates live at the
    # workflow spec level; the enhancer records the request on the step so the
    # workflow-level pass can emit it. Only if the developer asked (annotation)
    # and didn't already mount /workspace.
    if "workspace" in annots and WORKSPACE_MOUNT not in mounted:
        step.setdefault("_workspace_request", annots["workspace"])
        volume_mounts.append({"name": "workspace", "mountPath": WORKSPACE_MOUNT})


def enhance_workspace_pvc(spec: dict, steps: list) -> None:
    """If any step requested a workspace PVC, emit one workflow-level
    volumeClaimTemplate (the max requested size) mounted in every step."""
    requests = [s.pop("_workspace_request") for s in steps if "_workspace_request" in s]
    if not requests:
        return

    def _gi(size: str) -> int:
        return int(str(size).rstrip("Gi") or 0)

    size = max(requests, key=_gi)
    vcts = spec.setdefault("volumeClaimTemplates", [])
    if not any(v.get("metadata", {}).get("name") == "workspace" for v in vcts):
        vcts.append(
            {
                "metadata": {"name": "workspace"},
                "spec": {
                    "accessModes": ["ReadWriteOnce"],
                    "resources": {"requests": {"storage": size}},
                },
            }
        )


# ------------------------------------------------------------- dynamic enums


def enhance_dynamic_enums(spec: dict, ctx: dict, catalog: dict) -> None:
    """Decorate specific parameters with live enum values the developer
    could never know at authoring time. Only fills absent `enum` keys, so
    a static Choice enum derived from the config tree wins over the platform."""
    by_name = {p["name"]: p for p in spec["arguments"]["parameters"]}

    def set_enum(param_name: str, values: list) -> None:
        param = by_name.get(param_name)
        if param is None or "enum" in param:
            return
        default = param.get("default") or param.get("value")
        if default and default not in values:
            values = [default] + values
        param["enum"] = values

    if catalog:
        set_enum("data-ref", list(catalog.get("refs", [])))


def enhance_platform_group(steps: list, ctx: dict) -> None:
    """Inject the REAL platform config values into the compose step's overrides.

    `platform.*` is platform-injected runtime config (the project's lakeFS repo,
    the MLflow endpoint, the checkpoint bucket) — never a developer knob, never
    committed to a repo. The template vendors a PLACEHOLDER copy of the group so
    `python -m kubecore.compose` works on a laptop, and that copy is baked into
    the compose step's image.

    In the cluster the placeholder must lose. Without this the composed params
    carried `platform.lakefs.repository: PLACEHOLDER-repo`, so config-validation
    dutifully checked `s3://PLACEHOLDER-repo/...` and every run died on a lakeFS
    ListObjectsV2 error (live 2026-07-16).

    These go in as plain Hydra override tokens, FIRST, so a developer's own
    tokens still apply afterwards and the `platform.*` guard on the ADVANCED
    user YAML (compose.py) still refuses user edits to this group.
    """
    compose = next((s for s in steps if s["name"] == "compose-and-validate"), None)
    if compose is None:
        return  # a pipeline without the compose step — nothing to inject

    checkpoints = ctx.get("checkpoints") or {}
    overrides = {
        "platform.lakefs.repository": ctx["lakefs"]["repository"],
        "platform.lakefs.endpoint": ctx["lakefs"]["endpoint"],
        "platform.mlflow.tracking_uri": ctx["mlflow"]["trackingUri"],
    }
    if checkpoints.get("bucket"):
        overrides["platform.checkpoints.bucket"] = checkpoints["bucket"]
    if checkpoints.get("prefix"):
        overrides["platform.checkpoints.prefix"] = checkpoints["prefix"]

    args = compose["container"].setdefault("args", [])
    existing = {a.split("=", 1)[0] for a in args if isinstance(a, str)}
    tokens = [f"{k}={v}" for k, v in overrides.items() if k not in existing and v]
    compose["container"]["args"] = tokens + args


def enhance_pipeline_info(spec: dict, ctx: dict, steps: list) -> None:
    """Read-only documentation parameter, first in the form (like the
    live WFT's pipeline-info)."""
    parameters = spec["arguments"]["parameters"]
    if "pipeline-info" in _param_names(parameters):
        return
    lines = [
        "HOW TO RUN: defaults are runnable as-is — just press Submit.",
        "PER-STEP COMPUTE: {step}-class picks the node class; {step}-cpu / {step}-mem "
        "dial one run's resources.",
        "COMPUTE CLASSES on this pool:",
    ]
    for cls in ctx["computeClasses"]["all"]:
        acc = cls.get("guestAccelerator")
        acc_txt = f" + {acc['count']}x {acc['type']}" if acc else ""
        lines.append(f"  - {cls['name']} ({cls.get('tier', 'cpu')}) — {cls['machineType']}{acc_txt}")
    hpc = hpc_classes(ctx)
    if hpc:
        account = (ctx.get("hpc") or {}).get("account") or ""
        lines.append(
            "MELUXINA (HPC): pick a meluxina-* class in any {step}-class dropdown and "
            "that step runs as a Slurm job on MeluXina"
            + (f" (account {account})" if account else "")
            + " — queue time applies; the pipeline waits, stages the dataset in and the "
            "step's outputs back; hpc_time_limit in pipeline.py caps the job. Classes:")
        for c in hpc:
            lines.append(f"  - {c['name']} ({c.get('tier', 'cpu')}) — partition {c.get('partition')}"
                         f"{', ' + str(c.get('gpus')) + ' GPUs' if c.get('gpus') else ''}"
                         f", {c.get('cpus', '?')} cores, {c.get('memoryGiB', '?')} GiB"
                         f"{' — ' + c['description'] if c.get('description') else ''}")
    lines.append("STEPS: " + " -> ".join(s["name"] for s in steps))
    lines.append("(This field is informational — leave it unchanged.)")
    parameters.insert(
        0,
        {
            "name": "pipeline-info",
            "value": "\n".join(lines),
            "description": "Read-only: live pool compute classes + how to fill this form. Leave unchanged.",
        },
    )


# ---------------------------------------------------------------------- main


def enhance(wft: dict, ctx: dict, catalog: dict = None) -> dict:
    enhance_metadata(wft, ctx)
    spec = wft["spec"]
    enhance_spec_top_level(spec, ctx)
    steps = step_templates(spec)
    enhance_arguments(spec, ctx, steps)
    gpu_step_names: set = set()
    for step in steps:
        annots = platform_annotations(step)  # validates; raises on unknown keys
        # Captured BEFORE enhance_gpu_routing strips the typed request — the
        # developer's nvidia.com/gpu declaration is the capability marker.
        gpu = is_gpu_step(step)
        if gpu:
            gpu_step_names.add(step["name"])
        enhance_image(step, ctx, annots)
        enhance_env(step, ctx, annots)
        enhance_class_param(spec, step, ctx, annots)
        compute_class = resolve_compute_class(step, ctx, annots)
        enhance_sizing_knobs(spec, step, compute_class)  # before scheduling fill
        enhance_scheduling(step, ctx, annots)
        # After scheduling fill, so the final requests (developer-set or
        # platform whole-node) are what get checked against the class budget.
        validate_scheduling(step, compute_class, gpu, env_has_gpu=bool(gpu_class_names(ctx)))
        # After validation: strips the typed GPU request and installs the
        # runtime class->gpus routing (#865), so the checks above still saw
        # the full gpu-tier picture.
        enhance_gpu_routing(spec, step, ctx, annots)
        enhance_volumes(step, annots, ctx)
    enhance_workspace_pvc(spec, steps)
    enhance_platform_group(steps, ctx)
    enhance_dynamic_enums(spec, ctx, catalog or {})
    enhance_pipeline_info(spec, ctx, steps)
    # PRD-1016: the MeluXina leg — must run LAST (twin tasks copy each routed
    # step's FINAL container image/command, and the depends rewrite needs the
    # final DAG). See kubecore/meluxina.py.
    enhance_hpc(spec, ctx, steps, gpu_step_names)
    return wft


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Enhance a raw Hera WorkflowTemplate with platform wiring.")
    parser.add_argument("--raw", required=True, help="Path to the raw WorkflowTemplate YAML (from pipeline.py).")
    parser.add_argument("--context", required=True, help="Path to pipeline-context.yaml.")
    parser.add_argument("--catalog", default="", help="Optional path to the dataset-catalog probe output.")
    parser.add_argument("--output", required=True, help="Path to write the enhanced WorkflowTemplate YAML.")
    args = parser.parse_args(argv)

    raw = yaml.safe_load(Path(args.raw).read_text())
    ctx = yaml.safe_load(Path(args.context).read_text())
    catalog = {}
    if args.catalog and Path(args.catalog).exists():
        catalog = yaml.safe_load(Path(args.catalog).read_text()) or {}

    enhanced = enhance(raw, ctx, catalog)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(enhanced, sort_keys=False, default_flow_style=False))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
