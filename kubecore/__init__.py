"""kubecore — seeded, PLATFORM-OWNED ML-pipeline authoring helpers. DO NOT EDIT.

Vendored into the app repo (OQ1: vendored-first — offline local render, no
registry dependency). The platform replaces this directory wholesale on
upgrade; anything you change here is lost.

Public surface for pipeline.py:

    from kubecore.authoring import pipeline, step

Everything else in this package is render-time / run-time platform machinery:

    kubecore.authoring   — step()/pipeline() -> plain Hera Container + Argo DAG
    kubecore.derive_tree — Hydra config tree -> submit-form parameters
    kubecore.compose     — the compose-and-validate step (run via `python -m kubecore.compose`)
    kubecore.enhance     — platform post-processor (runs in CI, never imported by pipeline.py)

The directory is named `kubecore` (NOT `platform`) deliberately: a top-level
`platform` package shadows Python's stdlib `platform` module and breaks any
dependency that calls `platform.system()`.
"""

__version__ = "1.0.0"
