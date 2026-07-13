# model-quantization step image.
# The platform builds this on push (kaniko -> Zot) and wires the resulting
# image into the WorkflowTemplate as image-model-quantization. The in-image command is
# `python -m app.entry` — the same module you run locally.
FROM python:3.11-slim

WORKDIR /app

# Step runtime dependencies. Add what this step needs here.
RUN pip install --no-cache-dir PyYAML

COPY app/ ./app/

# Argo runs: python -m app.entry --params <resolved params.yaml content> ...
ENTRYPOINT ["python", "-m", "app.entry"]
