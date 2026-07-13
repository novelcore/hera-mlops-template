#!/usr/bin/env bash
# Local iteration loop: no cluster needed. Creates/reuses a venv, installs the
# pinned render deps, then runs the three stages:
#   pipeline.py            -> out/raw-workflow-template.yaml  (your DAG)
#   kubecore.enhance       -> out/workflow-template.yaml      (platform wiring)
#   kubecore.compose       -> out/params.yaml                 (composed defaults)
#
# out/workflow-template.yaml is what the platform releases; out/params.yaml is
# exactly what your steps receive at run time.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

if [ ! -d .venv ]; then
  python3 -m venv .venv
fi

.venv/bin/pip install -q --upgrade pip
.venv/bin/pip install -q -e .

# pipeline.py imports kubecore.* — run from the repo root so the package resolves.
PYTHONPATH="$PWD" .venv/bin/python pipeline.py
PYTHONPATH="$PWD" .venv/bin/python -m kubecore.enhance \
  --raw out/raw-workflow-template.yaml \
  --context kubecore/local-dev/pipeline-context.yaml \
  --catalog kubecore/local-dev/dataset-catalog.yaml \
  --output out/workflow-template.yaml
PYTHONPATH="$PWD" .venv/bin/python -m kubecore.compose --output out/params.yaml

echo
echo "done:"
echo "  out/raw-workflow-template.yaml  (your DAG, from pipeline.py)"
echo "  out/workflow-template.yaml      (enhanced release artifact)"
echo "  out/params.yaml                 (composed defaults — what steps receive)"
