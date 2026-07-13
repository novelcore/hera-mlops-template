# compose-and-validate step image (the pipeline's first task).
#
# Unlike the other steps, this one runs PLATFORM code (`python -m
# kubecore.compose`) against YOUR config/ tree: it composes the tree with the
# submitted form values into the single resolved params.yaml every downstream
# step receives, and fails fast (pre-GPU) on any bad override.
#
# It bundles the vendored `kubecore/` package + your `config/` tree. The
# platform builds THIS step with the repo ROOT as the build context (the CI
# detects the compose-and-validate step and widens the context automatically),
# so the COPY lines below resolve kubecore/ and config/ from the repo root. The
# platform wires the built image in as image-compose-and-validate.
FROM python:3.11-slim

WORKDIR /work

# The exact pinned render/compose deps (hera not needed at compose time, but
# hydra-core + omegaconf + pyyaml are).
RUN pip install --no-cache-dir "hydra-core>=1.3,<1.4" "omegaconf>=2.3,<2.4" "PyYAML>=6,<7"

# Bundle the platform compose engine and the experiment config tree.
COPY kubecore/ ./kubecore/
COPY config/ ./config/

# Argo runs: python -m kubecore.compose <override tokens> --render-defaults … \
#            --advanced … --require … --output /work/params.yaml
ENTRYPOINT ["python", "-m", "kubecore.compose"]
