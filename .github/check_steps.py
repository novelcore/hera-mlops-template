"""Step ↔ folder consistency check (run by the PR render workflow).

The enhancer wires every step in your pipeline.py to an image named
`image-<step>`, which the platform builds from `steps/<dir>/Dockerfile`
(dir = step name with hyphens → underscores). If you declare a step but forget
its folder/Dockerfile, the WorkflowTemplate still renders — and then fails at
RUN time with ImagePullBackOff, long after the PR merged. This catches that at
PR time instead.

For every container-backed step in the rendered WorkflowTemplate, require one of:
  - steps/<dir>/Dockerfile exists (the normal case: the platform builds it), OR
  - the step carries `platform.kubecore.io/image` (verbatim utility image — the
    sanctioned escape hatch, no build needed).

Exits non-zero with an actionable message listing every offending step.
"""

import sys
from pathlib import Path

import yaml

WFT = Path("out/workflow-template.yaml")
STEPS_DIR = Path("steps")
IMAGE_ANNOTATION = "platform.kubecore.io/image"


def main() -> int:
    wft = yaml.safe_load(WFT.read_text())
    templates = [t for t in wft["spec"]["templates"] if "container" in t]

    problems = []
    for t in templates:
        name = t["name"]
        annotations = t.get("metadata", {}).get("annotations", {})
        if IMAGE_ANNOTATION in annotations:
            continue  # verbatim utility image — no build expected
        dockerfile = STEPS_DIR / name.replace("-", "_") / "Dockerfile"
        if not dockerfile.is_file():
            problems.append((name, dockerfile))

    if problems:
        print("Step ↔ folder check FAILED — these steps are declared in your "
              "pipeline but have no buildable image:\n", file=sys.stderr)
        for name, dockerfile in problems:
            print(f"  - step '{name}': expected {dockerfile} (not found)", file=sys.stderr)
        print(
            "\nFix one of:\n"
            "  • add steps/<dir>/Dockerfile for the step (dir = step name, "
            "hyphens → underscores), or\n"
            "  • if it's a verbatim utility image, set the "
            f"'{IMAGE_ANNOTATION}' annotation on the step.\n"
            "Without this the WorkflowTemplate renders but the step "
            "ImagePullBackOffs at run time.",
            file=sys.stderr,
        )
        return 1

    print(f"Step ↔ folder check OK — all {len(templates)} step(s) are buildable.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
