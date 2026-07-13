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
- `needs=[other]` wires dependencies AND feeds every declared output of
  the needed step in as an input; a needed step that is conditional
  (`when=`) gets a skip-tolerant depends expression automatically.

This is a render-time helper, not a runtime SDK.
"""

import json

from hera.workflows import DAG, Container, Parameter, Resources, WorkflowTemplate
from hera.workflows.models import ValueFrom

from kubecore import derive_tree

IMAGE = "platform-managed"  # sentinel; the platform always rewrites images
COMPOSE_STEP = "compose-and-validate"
PARAMS_PATH = "/work/params.yaml"
# The compose step runs `python -m kubecore.compose` inside its own image;
# every other step runs `python -m app.entry` (the app's console-script).
COMPOSE_COMMAND = ["python", "-m", "kubecore.compose"]
STEP_COMMAND = ["python", "-m", "app.entry"]

_current = None


class Step:
    def __init__(self, name, reads=None, gpu=False, needs=None, outputs=None, when=None):
        self.name = name
        self.reads = list(reads or [])
        self.gpu = gpu
        self.needs = list(needs or [])
        self.outputs = list(outputs or [])
        self.when = when


def step(name, reads=None, gpu=False, needs=None, outputs=None, when=None) -> Step:
    if _current is None:
        raise RuntimeError("step() must be called inside `with pipeline(...):`")
    s = Step(name, reads, gpu, needs, outputs, when)
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
                    resources=Resources(gpus=1) if s.gpu else None,
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
