"""Platform-owned enhancer. The successor of render_wft.py's platform half.

Takes the RAW WorkflowTemplate a developer's vanilla-Hera pipeline.py rendered
and a platform context (pipeline-context.yaml, sourced from the
`{app}-pipeline-context` ConfigMap the operator writes into the app namespace)
and injects everything the developer never had to think about: namespace,
service account, secrets, compute-class routing, node scheduling, image
supply-chain wiring, per-run sizing knobs ({step}-cpu/{step}-mem via
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

ANNOTATION_PREFIX = "platform.kubecore.io/"
KNOWN_ANNOTATIONS = {"compute-class", "inject", "source", "image", "shm", "workspace", "hpc"}
WORKSPACE_MOUNT = "/workspace"


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
    tier = "gpu" if is_gpu_step(step) else "cpu"
    return ctx["computeClasses"][tier]["name"]


def resolve_compute_class(step: dict, ctx: dict, annots: dict) -> dict:
    """The class config used for whole-node sizing/tolerations. Pinned class
    merges over its tier default (catalog entries omit toleration config)."""
    name = default_class_name(step, ctx, annots)
    for cls in ctx["computeClasses"]["all"]:
        if cls["name"] == name:
            tier_default = ctx["computeClasses"][cls.get("tier", "gpu" if is_gpu_step(step) else "cpu")]
            return {**tier_default, **cls}
    # tier default itself (name came straight from computeClasses.{cpu,gpu})
    return ctx["computeClasses"]["gpu" if is_gpu_step(step) else "cpu"]


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
    tier = "gpu" if is_gpu_step(step) else "cpu"
    default = ctx["computeClasses"][tier]["name"]
    options = [c["name"] for c in ctx["computeClasses"]["all"] if c.get("tier") == tier]
    param = {
        "name": param_name,
        "value": default,
        "description": (
            f"Compute class for the '{step_name}' step "
            f"(the KubePool's allowed {tier} classes)."
        ),
    }
    if len(options) > 1:
        param["enum"] = options
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

    # Bound how long this step may sit unschedulable (see the constants above).
    # fill-absent: a developer who set their own deadline keeps it.
    step.setdefault(
        "activeDeadlineSeconds",
        GPU_PENDING_DEADLINE_SECONDS if gpu else PENDING_DEADLINE_SECONDS,
    )

    tolerations = step.setdefault("tolerations", [])
    toleration_key = compute_class["tolerationKey"]
    if not any(t.get("key") == toleration_key for t in tolerations):
        tolerations.append(
            {
                "key": toleration_key,
                "operator": "Equal",
                "value": compute_class["tolerationValue"],
                "effect": "NoSchedule",
            }
        )
    if gpu and not any(t.get("key") == "nvidia.com/gpu" for t in tolerations):
        # GKE auto-taints accelerator nodes; tolerate it so the pod schedules.
        tolerations.append({"key": "nvidia.com/gpu", "operator": "Exists", "effect": "NoSchedule"})

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
    if gpu:
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


def validate_scheduling(step: dict, compute_class: dict) -> None:
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
    if is_gpu_step(step) and not compute_class.get("guestAccelerator"):
        raise EnhanceError(
            f"step '{step_name}' requests a GPU but compute class '{class_name}' "
            f"({machine}) has no accelerator — the pod would never schedule."
        )


def enhance_sizing_knobs(spec: dict, step: dict, compute_class: dict) -> None:
    """Per-run sizing knobs, mirroring the live WFT's podSpecPatch model.

    Emits {step}-cpu / {step}-mem workflow parameters (defaults = the
    developer-set requests when present, else the class's whole-node
    allocatable) and wires them into the template via podSpecPatch so a
    submitter can dial a single run down/up without re-rendering. The
    developer writes nothing for this.

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

    patch = (
        "containers:\n"
        "- name: main\n"
        "  resources:\n"
        "    requests:\n"
        f'      cpu: "{{{{workflow.parameters.{step_name}-cpu}}}}"\n'
        f'      memory: "{{{{workflow.parameters.{step_name}-mem}}}}"\n'
        "    limits:\n"
        f'      cpu: "{{{{workflow.parameters.{step_name}-cpu}}}}"\n'
        f'      memory: "{{{{workflow.parameters.{step_name}-mem}}}}"\n'
    )
    step.setdefault("podSpecPatch", patch)


# ------------------------------------------------------------------- volumes


def enhance_volumes(step: dict, annots: dict) -> None:
    """Inject /dev/shm (sizable via shm annotation) and, opt-in, a per-run
    RWO workspace PVC (workspace annotation). Raw developer volumes win
    (only-fill-absent)."""
    container = step["container"]
    volume_mounts = container.setdefault("volumeMounts", [])
    mounted = {m.get("mountPath") for m in volume_mounts}
    volumes = step.setdefault("volumes", [])
    vol_names = {v.get("name") for v in volumes}

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
    for step in steps:
        annots = platform_annotations(step)  # validates; raises on unknown keys
        enhance_image(step, ctx, annots)
        enhance_env(step, ctx, annots)
        enhance_class_param(spec, step, ctx, annots)
        compute_class = resolve_compute_class(step, ctx, annots)
        enhance_sizing_knobs(spec, step, compute_class)  # before scheduling fill
        enhance_scheduling(step, ctx, annots)
        # After scheduling fill, so the final requests (developer-set or
        # platform whole-node) are what get checked against the class budget.
        validate_scheduling(step, compute_class)
        enhance_volumes(step, annots)
    enhance_workspace_pvc(spec, steps)
    enhance_platform_group(steps, ctx)
    enhance_dynamic_enums(spec, ctx, catalog or {})
    enhance_pipeline_info(spec, ctx, steps)
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
