"""hello-world — the example step for this template.

Every step is a small container that receives the resolved params.yaml (the
composed config tree) on --params and does its work. This one reads the
`experiment` config section and prints a value — replace the body with your
own logic, and add your runtime dependencies to the Dockerfile.
"""

import argparse

import yaml

# The config sections this step reads (must exist in the config/ tree).
READS = ["experiment"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", required=True,
                        help="Resolved params.yaml content (from compose-and-validate).")
    args, _ = parser.parse_known_args()
    cfg = yaml.safe_load(args.params) or {}
    experiment = cfg.get("experiment", {}) or {}
    print(f"👋 Hello from the hera pipeline template! experiment.name = {experiment.get('name')}")


if __name__ == "__main__":
    main()
