"""model-registration entry point: reads its declared config slice from the resolved
params.yaml produced by compose-and-validate. Everything this step may
tune lives in those sections; platform endpoints are in cfg["platform"]."""

import argparse

import yaml

READS = ["data", "model", "registration"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", required=True,
                        help="Resolved params.yaml content (from compose-and-validate).")
    args, _ = parser.parse_known_args()
    cfg = yaml.safe_load(args.params)
    config = {section: cfg[section] for section in READS}
    print(f"[model-registration] config sections {list(config)} loaded; "
          f"platform mlflow={cfg['platform']['mlflow']['tracking_uri']}")


if __name__ == "__main__":
    main()
