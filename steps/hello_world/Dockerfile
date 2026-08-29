# hello-world step image. The platform builds this on push and wires the
# resulting image into the WorkflowTemplate as image-hello-world. The in-image
# command is `python -m app.entry` — the same module you run locally.
FROM python:3.11-slim
WORKDIR /app
RUN pip install --no-cache-dir PyYAML
COPY app/ ./app/
ENTRYPOINT ["python", "-m", "app.entry"]
