# dataset-loading step image.
# The platform builds this on push (kaniko -> Zot) and wires the resulting
# image into the WorkflowTemplate as image-dataset-loading. The in-image command
# is `python -m app.entry` — the same module you run locally.
FROM python:3.11-slim

WORKDIR /app

# Step runtime dependencies:
#   boto3            — S3 / lakeFS (S3-compatible) object access
#   requests         — lakeFS commit-hash HTTP API (dataset lineage)
#   pydantic*        — the service's typed config + domain models
#   PyYAML           — params.yaml + data.yaml parsing
RUN pip install --no-cache-dir \
    "boto3>=1.34,<2" \
    "requests>=2.31,<3" \
    "pydantic>=2,<3" \
    "pydantic-settings>=2,<3" \
    "PyYAML>=6,<7"

COPY app/ ./app/

# Argo runs: python -m app.entry --params <resolved params.yaml content> ...
ENTRYPOINT ["python", "-m", "app.entry"]
