"""Seeded, PLATFORM-OWNED render-time helpers. DO NOT EDIT.

Gives pipeline.py the authoring surface: steps declare name, the config
sections they read, compute, dependencies, and outputs — and nothing else.
Under the hood this emits plain Hera Containers and a standard Argo DAG (the
vanilla-Hera property: thin helpers, standard YAML out), with the
compose-and-validate step as task 1:

- the compose step's args are ALL derived override tokens
  (`train.epochs={{workflow.parameters.train-epochs}}`), the ADVANCED
  override, and the union of declared reads= sections to validate;
- every other step receives the resolved params via an input parameter
  fed from compose's output — downstream steps NEVER see override tokens;
- `gpu=True` -> `Resources(gpus=1)`, which is exactly what the enhancer
  detects for GPU scheduling;
- `disk="20Gi"` -> an ephemeral-storage REQUEST (#892). Without it a
  disk-heavy step (quantization/export scratch, dataset unpack) rides at
  request 0 — best-effort on disk, first evicted under node pressure, and
  schedulable onto nearly-full nodes. The enhancer injects a platform
  default for steps that don't set it, so this knob is for steps that need
  MORE than the baseline;
- `needs=[other]` wires dependencies AND feeds every declared output of
  the needed step in as an input; a needed step that is conditional
  (`when=`) gets a skip-tolerant depends expression automatically.

This is a render-time helper, not a runtime SDK.
"""

import json
import re

from hera.workflows import DAG, Container, Parameter, Resources, WorkflowTemplate
from hera.workflows.models import ValueFrom

from kubecore import derive_tree

IMAGE = "platform-managed"  # sentinel; the platform always rewrites images

# Step names become Argo template names + the steps/<dir> folder + the
# image-<step> parameter, so they must be DNS-label-safe (lowercase alphanumeric
# and hyphens, no leading/trailing hyphen). Validating here turns a silent bad
# render (or a cryptic Hera NodeNameConflict) into a clear authoring error.
_STEP_NAME_RE = re.compile(r"^[a-z0-9]([a-z0-9-]*[a-z0-9])?$")


class AuthoringError(Exception):
    pass
COMPOSE_STEP = "compose-and-validate"
PARAMS_PATH = "/work/params.yaml"
# The compose step runs `python -m kubecore.compose` inside its own image;
# every other step runs `python -m app.entry` (the app's console-script).
COMPOSE_COMMAND = ["python", "-m", "kubecore.compose"]
STEP_COMMAND = ["python", "-m", "app.entry"]

_current = None


# Kubernetes resource quantity ("10Gi", "512Mi", "2G"); anchored so a typo
# ("10 Gi", "10GB") fails the render instead of rendering an invalid manifest.
_DISK_QUANTITY_RE = re.compile(r"^[0-9]+(Ki|Mi|Gi|Ti|K|M|G|T)?$")
_HPC_TIME_LIMIT_RE = re.compile(r"^[0-9]+[mh]?$")


class Step:
    def __init__(self, name, reads=None, gpu=False, needs=None, outputs=None, when=None, disk=None,
                 hpc_time_limit=None):
        self.name = name
        self.reads = list(reads or [])
        self.gpu = gpu
        self.needs = list(needs or [])
        self.outputs = list(outputs or [])
        self.when = when
        self.disk = disk
        self.hpc_time_limit = hpc_time_limit


def step(name, reads=None, gpu=False, needs=None, outputs=None, when=None, disk=None,
         hpc_time_limit=None) -> Step:
    if _current is None:
        raise RuntimeError("step() must be called inside `with pipeline(...):`")
    if not isinstance(name, str) or not _STEP_NAME_RE.match(name):
        raise AuthoringError(
            f"step name {name!r} is invalid — use a DNS-label name: lowercase "
            f"letters, digits and hyphens, no leading/trailing hyphen "
            f"(e.g. 'model-evaluation'). The name becomes the Argo template, the "
            f"steps/<dir> folder, and the image-<step> parameter."
        )
    if any(s.name == name for s in _current.steps):
        raise AuthoringError(
            f"duplicate step name {name!r} — every step must be uniquely named "
            f"(it is the step's identity in the DAG and its image)."
        )
    if disk is not None and not (isinstance(disk, str) and _DISK_QUANTITY_RE.match(disk)):
        raise AuthoringError(
            f"step {name!r}: disk={disk!r} is not a Kubernetes quantity — use a "
            f"string like '20Gi' (digits + optional Ki/Mi/Gi/Ti/K/M/G/T suffix)."
        )
    if hpc_time_limit is not None and not (
        isinstance(hpc_time_limit, str) and _HPC_TIME_LIMIT_RE.match(hpc_time_limit)
    ):
        raise AuthoringError(
            f"step {name!r}: hpc_time_limit={hpc_time_limit!r} must be minutes or hours as a "
            f"string like '90m', '12h' or '720' (the Slurm wall-clock limit when this GPU step "
            f"runs on the HPC target; the run is killed when it elapses)."
        )
    if hpc_time_limit is not None and not gpu:
        raise AuthoringError(
            f"step {name!r}: hpc_time_limit only applies to gpu=True steps — only GPU steps "
            f"are routed to the HPC target."
        )
    s = Step(name, reads, gpu, needs, outputs, when, disk, hpc_time_limit)
    _current.steps.append(s)
    return s


class pipeline:
    def __init__(self, name: str):
        self.name = name
        self.steps = []
        self.wt = None

    def __enter__(self):
        global _current
        _current = self
        return self

    def __exit__(self, exc_type, exc, tb):
        global _current
        _current = None
        if exc_type is None:
            self._build()
        return False

    def _build(self) -> None:
        form = derive_tree.derive()
        derive_tree.validate_reads(
            {s.name: s.reads for s in self.steps}, form.sections
        )  # render gate: reads= sections must exist in the tree
        required = sorted({sec for s in self.steps for sec in s.reads})

        with WorkflowTemplate(
            name=self.name, entrypoint="main", arguments=form.parameters
        ) as wt:
            compose = Container(
                name=COMPOSE_STEP, image=IMAGE,
                command=COMPOSE_COMMAND,
                args=form.tokens
                # render-time defaults manifest: lets the compose step
                # elide untouched leaves so group swaps take effect
                # (the WFT stays a self-contained release artifact)
                + ["--render-defaults", json.dumps(form.render_defaults, separators=(",", ":"))]
                + ["--advanced", "{{workflow.parameters.config}}"]
                + [f"--require={sec}" for sec in required]
                + ["--output", PARAMS_PATH],
                outputs=[Parameter(name="params", value_from=ValueFrom(path=PARAMS_PATH))],
            )
            containers = {}
            for s in self.steps:
                inputs = [Parameter(name="params")]
                args = ["--params", "{{inputs.parameters.params}}"]
                for need in s.needs:
                    for out in need.outputs:
                        inputs.append(Parameter(name=out, default="{}"))
                        args += [f"--{out}", f"{{{{inputs.parameters.{out}}}}}"]
                containers[s.name] = Container(
                    name=s.name, image=IMAGE,
                    command=STEP_COMMAND, args=args,
                    # Request-only for disk: the request is the eviction shield
                    # and scheduling signal; a limit would OOM-kill spiky export
                    # scratch instead (#892). The enhancer reads this as the
                    # developer-set value and keys the {step}-disk knob off it.
                    resources=Resources(
                        gpus=1 if s.gpu else None,
                        ephemeral_request=s.disk,
                    )
                    if (s.gpu or s.disk)
                    else None,
                    # Read by kubecore/meluxina.py: the Slurm time limit when
                    # this step runs on the HPC target (developer-set; the
                    # enhancer defaults it otherwise).
                    annotations={"platform.kubecore.io/hpc": s.hpc_time_limit}
                    if s.hpc_time_limit
                    else None,
                    inputs=inputs,
                    outputs=[
                        Parameter(name=out, value_from=ValueFrom(path=f"/work/output/{out}.json", default="{}"))
                        for out in s.outputs
                    ],
                )

            with DAG(name="main"):
                compose_task = compose()
                tasks = {}
                for s in self.steps:
                    arguments = {"params": compose_task.get_parameter("params")}
                    for need in s.needs:
                        for out in need.outputs:
                            arguments[out] = tasks[need.name].get_parameter(out)
                    depends = " && ".join(
                        [COMPOSE_STEP]
                        + [
                            f"({n.name}.Succeeded || {n.name}.Skipped || {n.name}.Omitted)"
                            if n.when
                            else n.name
                            for n in s.needs
                        ]
                    )
                    tasks[s.name] = containers[s.name](
                        arguments=arguments, when=s.when, depends=depends
                    )
        self.wt = wt

    def write(self, path) -> None:
        path.parent.mkdir(exist_ok=True)
        path.write_text(self.wt.to_yaml())
        print(f"wrote {path}")
