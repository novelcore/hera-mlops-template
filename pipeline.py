"""Your pipeline: the DAG. Structure and nothing else.

Each step declares its name, the config sections it reads (reads=), whether it
needs a GPU (gpu=True), what it depends on (needs=), any small result files it
writes (outputs=), and an optional condition (when=). Every tunable parameter
lives in the config/ tree and reaches the submit form automatically — there is
no parameter wiring here.

This template ships a single example step (hello-world). Add your own steps
under steps/<name>/ and wire them here. See README.md / DEVELOPER.md.
"""

from pathlib import Path

from kubecore.authoring import pipeline, step  # platform-owned, do not edit

HERE = Path(__file__).parent

# The name you pass here is for local readability only — the platform RENAMES
# the released WorkflowTemplate to "{your-app}-pipeline" at CI render time, so
# each app gets its own uniquely-named WFT (no collisions across apps in a
# shared namespace). You don't need to change it per app.
with pipeline("ml-pipeline") as p:
    hello = step("hello-world", reads=["experiment"])

if __name__ == "__main__":
    p.write(HERE / "out" / "raw-workflow-template.yaml")
