# kubecore/local-dev — platform stand-ins for LOCAL rendering only

These files let `./run.sh` render/enhance/compose on a laptop with no cluster.
**You do not own or edit them** — in the cluster the platform provides the real
versions:

- `pipeline-context.yaml` — the platform API the enhancer consumes (compute
  classes, MLflow/lakeFS endpoints, secret names, checkpoints). In CI it comes
  from your project's `{app}-pipeline-context` ConfigMap.
- `dataset-catalog.yaml` — stand-in for the `ml-dataset-catalog-probe` output
  (the lakeFS refs that become the `data-ref` dropdown).

The values here are **non-functional placeholders** (`PLACEHOLDER-*`,
`*.internal.invalid`). Real endpoints, secret names, and OIDC ids are
platform-internal — they are injected LIVE at CI render time from the
operator-written gitops `pipeline-context.yaml`, and must never be committed to
this (or any) repo. Local renders show the correct *shape*; the platform always
uses the live versions at release time.
