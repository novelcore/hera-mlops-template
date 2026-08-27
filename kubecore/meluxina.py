"""PRD-1016 — the MeluXina HPC leg the `platform.kubecore.io/hpc` annotation reserved.

`enhance_hpc` runs as the last enhancement pass. Active only when the pipeline
context carries `hpc.enabled` (Derive idiom: the operator sets it iff the
parent KubePool is HPC-ready, so non-HPC pools render byte-identically).

What it does (F-02/F-03/F-06):
  * re-emits the `target` submit-form parameter (gcp | <provider>);
  * every GPU step WITHOUT an existing task `when` (quantization-gated steps
    keep in-cluster-only behaviour this slice) gains a `-meluxina` twin DAG
    task with complementary `when:` clauses on `target`;
  * downstream `depends` tokens that reference a routed step are rewritten to
    gate on the Succeeded||Skipped twin pair (exactly one twin runs);
  * one `meluxina-run` CONTAINER template is appended (the release gate
    rejects script templates): an idempotent Slurm REST submit keyed on
    workflow.uid — a retried pod ADOPTS the queued/running job instead of
    double-submitting a full-node run — then a poll to terminal state,
    emitting `slurm-job-id` as an output.

Operational notes baked in from the live F-01/F-02 shakedown (2026-08):
  * the rotating Slurm JWT arrives via the `meluxina-jwt` Secret (PRD-1016
    F-07 ExternalSecret in the ml namespace, 5m refresh);
  * the batch script restores the Lmod init (`bash -l` + explicit source) —
    Slurm's mandatory `environment` field wipes it;
  * images pull through a digest-keyed Lustre SIF cache (D-02: cold ~8 min,
    warm ~7 s, measured on mel2107); registry auth is a best-effort
    metadata-server access token (GAR);
  * data plane (F-04/F-08): the submit pod mints a short-lived Zitadel
    bearer from the mounted machine key (the same JWT-profile recipe as the
    seeded mlflow_zitadel_auth module) and passes it in the Slurm job
    environment together with the PUBLIC MLflow/lakeFS endpoints — never the
    machine key itself (T-03: no long-lived credential leaves the cluster).
    The batch script stages the pinned dataset version from the lakeFS
    public endpoint to a Lustre cache keyed by lakeFS commit (D-04: the
    in-cluster manifest-only/S3-gateway streaming path is blocked by the
    SSO front off-cluster) and bind-mounts it read-only at
    /kubecore/dataset (KUBECORE_DATASET_DIR) for the step.
"""

import json
import re

# The submit/poll program the meluxina-run container executes via
# `python3 -c`. Plain string — no Argo tags inside (all run-time inputs
# arrive via env), so it survives every templating layer verbatim.
MELUXINA_SUBMIT_CODE = r'''
STAGEOUT = r"""
# F-05 stage-out: the step's declared outputs (/work/output/<name>.json,
# what Argo would have read as outputs.parameters in-cluster) go to lakeFS
# at hpc-outputs/{wf-uid}/{step}/ on the run's branch; the waiter fetches
# them back into the twin's outputs so downstream steps see real values.
import json, os, pathlib, urllib.parse, urllib.request
base = os.environ['LAKEFS_ENDPOINT'].rstrip('/') + '/api/v1'
H = {'Authorization': 'Bearer ' + os.environ['LAKEFS_BEARER_TOKEN']}
repo = os.environ['DATASET_REPO']; branch = os.environ.get('DATASET_REF') or 'main'
prefix = 'hpc-outputs/%s/%s/' % (os.environ['WF_UID'], os.environ['STEP_NAME'])
out = pathlib.Path(os.environ['KAOS_WORK']) / 'output'
for name in [n for n in os.environ.get('STEP_OUTPUTS', '').split(',') if n]:
    f = out / (name + '.json')
    if not f.exists():
        print('stage-out: no', f, flush=True); continue
    data = f.read_bytes()
    body = (b'--B\r\nContent-Disposition: form-data; name="content"; filename="f"\r\n'
            b'Content-Type: application/octet-stream\r\n\r\n' + data + b'\r\n--B--\r\n')
    q = urllib.parse.urlencode({'path': prefix + name + '.json'})
    req = urllib.request.Request(base + '/repositories/%s/branches/%s/objects?%s' % (repo, branch, q),
                                 data=body, method='POST',
                                 headers=dict(H, **{'Content-Type': 'multipart/form-data; boundary=B'}))
    urllib.request.urlopen(req, timeout=120).read()
    print('stage-out:', name, len(data), 'bytes ->', prefix, flush=True)
"""
import base64, json, os, shlex, signal, sys, time, urllib.parse, urllib.request

# F-04 stage-in, run ON the compute node (system python3, stdlib only):
# resolve ref -> commit, download every object of that version from the
# lakeFS PUBLIC endpoint (bearer through oauth2-proxy — the in-cluster
# S3-gateway/manifest streaming path is blocked by the SSO front
# off-cluster) into a Lustre cache keyed by commit, parallel + resumable
# (size-matched files are skipped, .part rename is atomic). Rides to the
# node base64-encoded in the job environment.
STAGEIN = """
import concurrent.futures, json, os, pathlib, sys, urllib.parse, urllib.request
EP = os.environ['LAKEFS_ENDPOINT'].rstrip('/')
H = {'Authorization': 'Bearer ' + os.environ['LAKEFS_BEARER_TOKEN']}
REPO, REF = os.environ['DATASET_REPO'], os.environ['DATASET_REF']
def get(url):
    return urllib.request.urlopen(
        urllib.request.Request(url, headers=H), timeout=120)
base = EP + '/api/v1/repositories/' + urllib.parse.quote(REPO, safe='')
# ref -> tip commit via the commit LOG (amount=1). GET /refs/{ref} does not
# exist on our lakeFS version (404, live job 5143633) — the seeded loader's
# /refs/{ref}/commits endpoint is the proven one.
commit = json.load(get(base + '/refs/' + urllib.parse.quote(REF, safe='')
                       + '/commits?amount=1'))['results'][0]['id']
dest = pathlib.Path(os.environ['DATASET_CACHE']) / REPO / commit
open(os.environ['DATASET_DIRFILE'], 'w').write(str(dest))
if (dest / '.complete').exists():
    print('stage-in: cache hit', dest, flush=True)
    sys.exit(0)
objs, after = [], ''
while True:
    page = json.load(get(base + '/refs/' + commit + '/objects/ls?'
                         + urllib.parse.urlencode(
                             {'amount': 1000, 'after': after})))
    objs += [r for r in page.get('results') or []
             if r.get('path_type') == 'object']
    pg = page.get('pagination') or {}
    if not pg.get('has_more'):
        break
    after = pg.get('next_offset') or ''
print('stage-in:', len(objs), 'objects @', commit[:12], flush=True)
def fetch(o):
    p = dest / o['path']
    if p.exists() and p.stat().st_size == o.get('size_bytes'):
        return 0
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = str(p) + '.part'
    with get(base + '/refs/' + commit + '/objects?'
             + urllib.parse.urlencode({'path': o['path']})) as r:
        with open(tmp, 'wb') as f:
            while True:
                b = r.read(1 << 20)
                if not b:
                    break
                f.write(b)
    os.replace(tmp, p)
    return 1
with concurrent.futures.ThreadPoolExecutor(8) as ex:
    n = sum(ex.map(fetch, objs))
(dest / '.complete').touch()
print('stage-in: downloaded', n, 'of', len(objs), '->', dest, flush=True)
"""

# F-08 wallet: mint a short-lived Zitadel bearer from the mounted machine
# key — the same JWT-profile recipe as the seeded mlflow_zitadel_auth
# module (assertion aud = https://{domain}; the project-audience scope is
# what makes the token acceptable to mlflow-oidc-auth AND carries the
# groups claim). PyJWT+cryptography are pip-installed at run time: the
# submit image is bare-stdlib alpine and RS256 needs real crypto. Any
# failure degrades loudly to no-wallet (job still runs, data-plane env
# omitted) — never the machine key itself into the Slurm environment
# (T-03: HPC-side env must only ever see short-lived tokens).
WALLET = None


def mint_wallet():
    keyfile = os.environ.get('ZITADEL_MACHINE_KEY_FILE') or ''
    domain = os.environ.get('ZITADEL_DOMAIN') or ''
    if not keyfile or not os.path.exists(keyfile) or not domain:
        print('wallet: machine key / domain absent, data-plane env omitted',
              flush=True)
        return ''
    try:
        import subprocess
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q',
                        '--no-warn-script-location', '--root-user-action',
                        'ignore', 'PyJWT', 'cryptography'],
                       check=True, timeout=180)
        import jwt as pyjwt
        key = json.load(open(keyfile))
        now = int(time.time())
        assertion = pyjwt.encode(
            {'iss': key['userId'], 'sub': key['userId'],
             'aud': 'https://' + domain, 'iat': now, 'exp': now + 60},
            key['key'], algorithm='RS256', headers={'kid': key['keyId']})
        scope = 'openid email profile urn:zitadel:iam:org:projects:roles'
        pid = os.environ.get('ZITADEL_MLFLOW_PROJECT_ID') or ''
        if pid:
            scope += ' urn:zitadel:iam:org:project:id:' + pid + ':aud'
        req = urllib.request.Request(
            'https://' + domain + '/oauth/v2/token',
            data=urllib.parse.urlencode({
                'grant_type': 'urn:ietf:params:oauth:grant-type:jwt-bearer',
                'scope': scope, 'assertion': assertion}).encode())
        tok = json.load(urllib.request.urlopen(req, timeout=30)).get(
            'access_token', '')
        print('wallet: minted' if tok else 'wallet: empty token response',
              flush=True)
        return tok
    except Exception as e:
        print('wallet: mint failed, data-plane env omitted:', e, flush=True)
        return ''

# OCI WORKDIR (live job 5148071): Docker honours the image's WorkingDir,
# Apptainer does NOT — `apptainer exec` starts in the host cwd, so a step
# command like `python -m app.entry` fails with "No module named 'app'".
# Step images use different WORKDIRs (/app, /work), so it is read from the
# image config in the registry at submit time and passed as `--pwd`.
# Best-effort: any failure yields '' (no --pwd), never a blocked submit.
def fetch_workdir(image, reg_token):
    try:
        ref_host, _, rest = image.partition('/')
        if '@' in rest:
            path, ref = rest.split('@', 1)
        elif ':' in rest.rsplit('/', 1)[-1]:
            path, ref = rest.rsplit(':', 1)
        else:
            path, ref = rest, 'latest'
        hdr = {'Accept': ', '.join([
            'application/vnd.oci.image.index.v1+json',
            'application/vnd.docker.distribution.manifest.list.v2+json',
            'application/vnd.oci.image.manifest.v1+json',
            'application/vnd.docker.distribution.manifest.v2+json'])}
        if reg_token and ref_host.endswith('-docker.pkg.dev'):
            hdr['Authorization'] = 'Bearer ' + reg_token
        base = 'https://' + ref_host + '/v2/' + path

        def get(url):
            return json.load(urllib.request.urlopen(
                urllib.request.Request(url, headers=hdr), timeout=20))
        m = get(base + '/manifests/' + ref)
        if 'manifests' in m:  # multi-arch index: pick linux/amd64
            pick = next((x for x in m['manifests']
                         if (x.get('platform') or {}).get('architecture') == 'amd64'
                         and (x.get('platform') or {}).get('os', 'linux') == 'linux'),
                        m['manifests'][0])
            m = get(base + '/manifests/' + pick['digest'])
        cfg = get(base + '/blobs/' + m['config']['digest'])
        wd = (cfg.get('config') or {}).get('WorkingDir') or ''
        print('workdir:', wd or '(none)', flush=True)
        return wd
    except Exception as e:
        print('workdir: lookup failed (running from host cwd):', e, flush=True)
        return ''

API = 'https://slurm-api.lxp.lu/slurm/v0.0.44'
SDB = 'https://slurm-api.lxp.lu/slurmdb/v0.0.44'
TOKEN_FILE = '/etc/meluxina-jwt/token'


def tok():
    """The MeluXina JWT, read FRESH on every request. The platform mints it
    with a 60-minute lifetime and rotates the Secret every 25 minutes; the
    mounted file follows the rotation, an env var does not (live run
    yolotrain-meluxina-toy-tvsm8: the pod's env token expired 13:30:02 while
    the job still queued — every poll 502/511 for hours, and cancel/resubmit
    would have failed the same way)."""
    try:
        with open(TOKEN_FILE) as f:
            t = f.read().strip()
            if t:
                return t
    except OSError:
        pass
    return os.environ['SLURM_TOKEN'].strip()


def hdrs():
    return {'X-SLURM-USER-NAME': 'u104378', 'X-SLURM-USER-TOKEN': tok(),
            'Content-Type': 'application/json'}


def get(url):
    return json.load(urllib.request.urlopen(
        urllib.request.Request(url, headers=hdrs()), timeout=30))

jobname = 'kaos-' + os.environ['WF_UID'] + '-' + os.environ['STEP_NAME']
img = os.environ['IMAGE_REF']
# Apptainer rejects tag@digest refs ('Docker references with both a tag and
# digest are currently not supported', live job 5140397). The platform pins
# images as name:tag@sha256:... (Zot-retention lesson) — normalize to the
# digest-only form, which Apptainer accepts and which is the stronger pin.
if '@' in img:
    name, digest = img.split('@', 1)
    if ':' in name.rsplit('/', 1)[-1]:
        name = name.rsplit(':', 1)[0]
    img = name + '@' + digest
# shlex-quote each argv token: the joined string rides through Slurm env ->
# sh -c, and a naive join destroys embedded quoting (live job 5140493:
# python -c 'import torch; ...' arrived as bare words and exited 2).
# strict=False tolerates control characters inside strings — belt-and-braces
# for any substitution path the render-side toJson escaping didn't cover.
cmd = ' '.join(shlex.quote(t) for t in json.loads(
    os.environ.get('STEP_COMMAND') or '[]', strict=False))

jid = None


def find_active():
    for j in (get(API + '/jobs').get('jobs') or []):
        st = j.get('job_state') or []
        if j.get('name') == jobname and any(
                x in ('PENDING', 'RUNNING', 'SUSPENDED') for x in st):
            return j.get('job_id')
    return None


def submit():
    reg = ''
    try:
        r = urllib.request.Request(
            'http://metadata.google.internal/computeMetadata/v1/instance/'
            'service-accounts/default/token',
            headers={'Metadata-Flavor': 'Google'})
        reg = json.load(urllib.request.urlopen(r, timeout=5)).get(
            'access_token', '')
    except Exception as e:
        print('no registry token from metadata server (anonymous pull):', e,
              flush=True)
    batch = '\n'.join([
        '#!/bin/bash -l',
        'set +e',
        # Diagnostics without SSH (live job 5143859: exit 231 with no way to
        # see WHY): on any failure, tail this job's own output file into its
        # job COMMENT — readable over REST from the waiter and by operators.
        'fail(){ scontrol update JobId=$SLURM_JOB_ID Comment="err$1:$(tail -c 700 slurm-$SLURM_JOB_ID.out 2>/dev/null | tr "\\n" "|")" 2>/dev/null; exit $1; }',
        'for f in /usr/share/lmod/lmod/init/bash /etc/profile.d/lmod.sh; do'
        ' [ -r "$f" ] && source "$f" && break; done',
        'module load Apptainer 2>/dev/null || module load apptainer 2>/dev/null',
        'command -v apptainer >/dev/null || fail 210',
        'SCR=/project/scratch/p201342',
        'export APPTAINER_CACHEDIR=$SCR/kaos-apptainer-cache'
        ' APPTAINER_TMPDIR=$SCR/kaos-tmp',
        'mkdir -p $APPTAINER_CACHEDIR $APPTAINER_TMPDIR $SCR/sif-cache',
        'KEY=$(printf %s "$IMAGE_REF" | sha256sum | cut -c1-16)',
        'SIF=$SCR/sif-cache/$KEY.sif',
        # F-04 stage-in: only when the wallet minted AND the run pins a
        # dataset. Fails LOUDLY (exit 232) — silently missing data would
        # surface as a cryptic training error hours later.
        'DATASET_DIR=""',
        'if [ -n "$LAKEFS_BEARER_TOKEN" ] && [ -n "$LAKEFS_ENDPOINT" ] &&'
        ' [ -n "$DATASET_REPO" ] && [ -n "$DATASET_REF" ] &&'
        ' [ -n "$STAGEIN_B64" ] && command -v python3 >/dev/null; then',
        '  export DATASET_CACHE=$SCR/data-cache'
        ' DATASET_DIRFILE=$SCR/kaos-tmp/ds-$SLURM_JOB_ID.dir',
        '  printf %s "$STAGEIN_B64" | base64 -d | python3 - || fail 232',
        '  DATASET_DIR=$(cat "$DATASET_DIRFILE" 2>/dev/null)',
        'fi',
        # Apptainer passes the host env through, so the wallet/endpoint vars
        # reach the step; the bind gives it the staged version read-only.
        'BIND=""',
        'if [ -n "$DATASET_DIR" ]; then'
        ' BIND="-B $DATASET_DIR:/kubecore/dataset:ro";'
        ' export APPTAINERENV_KUBECORE_DATASET_DIR=/kubecore/dataset; fi',
        # Writable /work: the SIF is read-only, but steps write /work/output
        # (declared outputs), /work/runs, /work/dataset caches. Per-job scratch
        # on Lustre, removed after a successful stage-out.
        'export KAOS_WORK=$SCR/kaos-work/$SLURM_JOB_ID; mkdir -p $KAOS_WORK/output',
        'BIND="$BIND -B $KAOS_WORK:/work"',
        'if [ ! -f "$SIF" ]; then',
        # GCP token ONLY for GAR hosts: presenting it to Zot turns an
        # anonymous-OK pull into 401 authentication required (live job
        # 5140432 vs the anonymous F-01 pull that worked on the same repo).
        '  case "$IMAGE_REF" in *-docker.pkg.dev/*)'
        ' [ -n "$REG_TOKEN" ] && export'
        ' APPTAINER_DOCKER_USERNAME=oauth2accesstoken'
        ' APPTAINER_DOCKER_PASSWORD=$REG_TOKEN;; esac',
        '  apptainer pull "$SIF" docker://$IMAGE_REF || fail 231',
        'fi',
        # Start in the image's WORKDIR (Apptainer ignores it; Docker does not).
        'PWD_OPT=""; [ -n "$STEP_WORKDIR" ] && PWD_OPT="--pwd $STEP_WORKDIR"',
        'if [ -n "$STEP_CMD" ]; then apptainer exec --nv $PWD_OPT $BIND "$SIF"'
        ' /bin/sh -c "$STEP_CMD"; else apptainer exec --nv $PWD_OPT $BIND "$SIF"'
        ' nvidia-smi -L; fi',
        'rc=$?',
        'if [ -n "$STEP_OUTPUTS" ] && [ -n "$STAGEOUT_B64" ] && [ -n "$LAKEFS_BEARER_TOKEN" ]; then'
        ' printf %s "$STAGEOUT_B64" | base64 -d | python3 - || echo "stage-out failed (rc=$rc)"; fi',
        '[ $rc -eq 0 ] && rm -rf "$KAOS_WORK"',
        '[ $rc -ne 0 ] && fail $rc',
        'exit 0',
    ])
    env = ['PATH=/usr/bin:/bin:/usr/local/bin', 'HOME=/home/users/u104378',
           'USER=u104378', 'IMAGE_REF=' + img, 'REG_TOKEN=' + reg,
           'STEP_CMD=' + cmd, 'STEP_WORKDIR=' + fetch_workdir(img, reg),
           'WF_UID=' + (os.environ.get('WF_UID') or ''),
           'STEP_NAME=' + (os.environ.get('STEP_NAME') or ''),
           'STEP_OUTPUTS=' + (os.environ.get('STEP_OUTPUTS') or '')]
    global WALLET
    wallet = WALLET = mint_wallet()
    if wallet:
        # Public endpoints only, short-lived bearer only (T-03). MLflow's
        # client honors MLFLOW_TRACKING_TOKEN natively; the stage-in and any
        # in-step lakeFS API calls ride the same bearer through oauth2-proxy.
        mlflow_url = os.environ.get('MLFLOW_EXTERNAL_URL') or ''
        lakefs_url = os.environ.get('LAKEFS_EXTERNAL_URL') or ''
        if mlflow_url:
            env += ['MLFLOW_TRACKING_URI=' + mlflow_url,
                    'MLFLOW_TRACKING_TOKEN=' + wallet]
        if lakefs_url:
            env += ['LAKEFS_ENDPOINT=' + lakefs_url,
                    'LAKEFS_BEARER_TOKEN=' + wallet,
                    'DATASET_REPO=' + (os.environ.get('DATASET_REPO') or ''),
                    'DATASET_REF=' + (os.environ.get('DATASET_REF') or ''),
                    'STAGEIN_B64='
                    + base64.b64encode(STAGEIN.encode()).decode(),
                    'STAGEOUT_B64='
                    + base64.b64encode(STAGEOUT.encode()).decode()]
    body = {'job': {'name': jobname, 'partition': 'gpu',
                    'account': 'p201342', 'qos': 'default',
                    'time_limit': int(os.environ.get('SLURM_TIME_LIMIT') or 240),
                    'current_working_directory': '/home/users/u104378',
                    'environment': env, 'tasks': 1, 'nodes': '1'},
            'script': batch}
    req = urllib.request.Request(API + '/job/submit',
                                 data=json.dumps(body).encode(),
                                 headers=hdrs(), method='POST')
    resp = json.load(urllib.request.urlopen(req, timeout=30))
    j = resp.get('job_id')
    print('submitted job', j, 'errors:', resp.get('errors'), flush=True)
    if not j:
        sys.exit(1)
    return j


def cancel(j):
    try:
        urllib.request.urlopen(urllib.request.Request(
            API + '/job/' + str(j), headers=hdrs(), method='DELETE'), timeout=30)
        print('cancelled job', j, flush=True)
    except Exception as e:
        print('cancel failed:', e, flush=True)


def on_term(signum, frame):
    # Argo sends TERM on deadline/stop. A PENDING job never started — cancel
    # it so it does not run unobserved hours later (live job 5143859 did
    # exactly that: waiter killed at the deadline mid-queue, job ran at
    # 20:0xZ with nobody watching and burned a failed pull). A RUNNING job
    # is deliberately left to finish.
    try:
        j = (get(API + '/job/' + str(jid)).get('jobs') or [{}])[0]
        if 'PENDING' in (j.get('job_state') or []):
            cancel(jid)
    except Exception:
        pass
    sys.exit(143)


signal.signal(signal.SIGTERM, on_term)

jid = find_active()
if jid is not None:
    print('adopting existing job', jid, flush=True)
else:
    jid = submit()

open('/tmp/slurm-job-id', 'w').write(str(jid))
OUTPUT_NAMES = [n for n in os.environ.get('STEP_OUTPUTS', '').split(',') if n]
os.makedirs('/tmp/outputs', exist_ok=True)
for n in OUTPUT_NAMES:
    open('/tmp/outputs/%s.json' % n, 'w').write('{}')


def fetch_outputs():
    """F-05: pull the step's staged-out outputs back into the twin."""
    base = (os.environ.get('LAKEFS_EXTERNAL_URL') or '').rstrip('/')
    repo = os.environ.get('DATASET_REPO') or ''
    if not (OUTPUT_NAMES and base and repo and WALLET):
        return
    branch = os.environ.get('DATASET_REF') or 'main'
    for n in OUTPUT_NAMES:
        q = urllib.parse.urlencode({'path': 'hpc-outputs/%s/%s/%s.json' % (
            os.environ.get('WF_UID', ''), os.environ.get('STEP_NAME', ''), n)})
        try:
            data = urllib.request.urlopen(urllib.request.Request(
                base + '/api/v1/repositories/%s/refs/%s/objects?%s' % (repo, branch, q),
                headers={'Authorization': 'Bearer ' + WALLET}), timeout=60).read()
            open('/tmp/outputs/%s.json' % n, 'wb').write(data)
            print('output', n, ':', len(data), 'bytes', flush=True)
        except Exception as e:
            print('output', n, 'not staged out (default {}):', e, flush=True)


TERMINAL = ('COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT', 'NODE_FAIL',
            'PREEMPTED', 'OUT_OF_MEMORY')
resubmitted = False
while True:
    time.sleep(30)
    try:
        job = (get(SDB + '/job/' + str(jid)).get('jobs') or [{}])[0]
        st = (job.get('state') or {}).get('current') or []
        print('state:', st, flush=True)
        if any(x in TERMINAL for x in st):
            rc = ((job.get('exit_code') or {}).get('return_code')
                  or {}).get('number')
            print('terminal:', st, 'exit:', rc, flush=True)
            print('job comment:', job.get('comment'), flush=True)
            if 'FAILED' in st and rc == 231 and not resubmitted:
                # A cache-missing pull after a LONG queue wait runs with the
                # registry token minted at submit time — GCP tokens live 1h,
                # so it can be stale (live job 5143859: 3.5h queue -> 231).
                # One resubmission mints everything fresh; queue age is lost
                # only after an actual failure.
                resubmitted = True
                print('pull failed - resubmitting once with fresh'
                      ' credentials', flush=True)
                jid = submit()
                open('/tmp/slurm-job-id', 'w').write(str(jid))
                continue
            if 'COMPLETED' in st:
                fetch_outputs()
            sys.exit(0 if ('COMPLETED' in st and rc in (0, None)) else 1)
    except SystemExit:
        raise
    except Exception as e:
        print('poll error (transient):', e, flush=True)
'''


_SUBST_TAG = re.compile(
    r"\{\{\s*((?:workflow\.parameters|tasks)\.[A-Za-z0-9_.-]+)\s*\}\}")
_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _tag_expr(path: str) -> str:
    """Convert a dotted Argo tag path to expression syntax, using bracket
    access for segments that are not bare identifiers (task and param names
    routinely carry hyphens: tasks.compose-and-validate.outputs...)."""
    segs = path.split(".")
    out = segs[0]
    for s in segs[1:]:
        out += "." + s if _IDENT.match(s) else "['%s']" % s
    return out


def _cmd_json(cmd: list) -> str:
    """Render the step-command as a JSON array that STAYS valid JSON after
    Argo's parameter substitution.

    A naive json.dumps breaks at run time (live wf mgznz 2026-08-25): Argo
    substitutes {{workflow.parameters.X}} / {{tasks.X.outputs.parameters.Y}}
    INSIDE the already-serialized string, and a multi-line value (the
    compose-and-validate params.yaml output is a whole YAML doc) lands raw
    newlines/quotes inside a JSON string — the submit pod's json.loads dies
    on 'invalid control character'. Fix: tokens carrying substitutable tags
    are emitted as {{=toJson(...)}} expressions, so Argo itself JSON-escapes
    the substituted value; literal tokens stay json.dumps'd. Mixed tokens
    rebuild the full string inside the expression via single-quoted
    concatenation."""
    parts = []
    for tok in cmd:
        if not _SUBST_TAG.search(tok):
            parts.append(json.dumps(tok))
            continue
        exprs, pos = [], 0
        for m in _SUBST_TAG.finditer(tok):
            if m.start() > pos:
                exprs.append("'%s'" % tok[pos:m.start()].replace("\\", "\\\\").replace("'", "\\'"))
            exprs.append(_tag_expr(m.group(1)))
            pos = m.end()
        if pos < len(tok):
            exprs.append("'%s'" % tok[pos:].replace("\\", "\\\\").replace("'", "\\'"))
        parts.append("{{=toJson(%s)}}" % " + ".join(exprs))
    return "[" + ", ".join(parts) + "]"


DEFAULT_TIME_LIMIT_MINUTES = 240
# Argo budget on top of the Slurm limit: the twin pod also waits in the
# MeluXina queue (evening waits run to hours), builds the SIF and stages the
# dataset in before the job's own clock starts. If the twin's deadline fires
# first, on_term leaves a RUNNING job to finish unobserved — so the deadline
# must comfortably exceed queue + prep + time limit.
QUEUE_ALLOWANCE_MINUTES = 480
HPC_ANNOTATION = "platform.kubecore.io/hpc"


def parse_time_limit(value) -> int:
    """'90' / '90m' -> 90 minutes, '12h' -> 720. Raises ValueError otherwise."""
    text = str(value).strip()
    m = re.match(r"^([0-9]+)([mh]?)$", text)
    if not m or int(m.group(1)) == 0:
        raise ValueError(
            "%s=%r: expected minutes or hours like '90m', '12h' or '720'"
            % (HPC_ANNOTATION, value))
    minutes = int(m.group(1))
    return minutes * 60 if m.group(2) == "h" else minutes


def step_time_limit(step: dict) -> int:
    """Slurm wall-clock minutes for a routed step (annotation or default)."""
    raw = ((step.get("metadata") or {}).get("annotations") or {}).get(HPC_ANNOTATION)
    return parse_time_limit(raw) if raw not in (None, "") else DEFAULT_TIME_LIMIT_MINUTES


def enhance_hpc(spec: dict, ctx: dict, steps: list, gpu_step_names: set) -> None:
    """Route GPU steps to MeluXina behind the `target` param (module doc)."""
    hpc = ctx.get("hpc") or {}
    if not hpc.get("enabled"):
        return
    provider = hpc.get("provider", "meluxina")

    parameters = spec["arguments"]["parameters"]
    if not any(p.get("name") == "target" for p in parameters):
        parameters.append({
            "name": "target", "value": "gcp", "enum": ["gcp", provider],
            "description": ("Computation target for this run. gcp = "
                            "in-cluster pools; %s = HPC burst (GPU training "
                            "runs on %s via Slurm)." % (provider, provider)),
        })

    entry = spec.get("entrypoint")
    dag_tpl = next((t for t in spec.get("templates", [])
                    if t.get("name") == entry and "dag" in t), None)
    if dag_tpl is None:
        return
    tasks = dag_tpl["dag"]["tasks"]
    by_tpl = {s["name"]: s for s in steps}

    routed = []
    outputs_union: set = set()
    for task in list(tasks):
        step = by_tpl.get(task.get("template"))
        if step is None or step["name"] not in gpu_step_names or task.get("when"):
            continue
        task["when"] = "{{=workflow.parameters.target != '%s'}}" % provider
        container = step.get("container") or {}
        # step-command: {{inputs.parameters.X}} tokens resolve against the
        # STEP template's inputs — copied verbatim into a task argument they
        # fail spec validation for the entire WorkflowTemplate (live-caught
        # 2026-08-25: one templated token bricked every submission, gcp runs
        # included). But the DAG task's OWN arguments carry each input's
        # value (a literal or a task-context-valid expression like
        # {{workflow.parameters.x}}), so substituting from them yields the
        # real invocation (F-04). Anything still step-scoped after
        # substitution is dropped; an empty result falls back to the
        # in-template nvidia-smi probe.
        argmap = {p.get("name"): str(p.get("value", ""))
                  for p in ((task.get("arguments") or {}).get("parameters")
                            or [])}
        cmd = [re.sub(r"\{\{\s*inputs\.parameters\.([A-Za-z0-9_.-]+)\s*\}\}",
                      lambda m: argmap.get(m.group(1), ""), tok)
               for tok in ((container.get("command") or [])
                           + (container.get("args") or []))]
        cmd = [t for t in cmd
               if t and "{{inputs." not in t and "{{item" not in t]
        minutes = step_time_limit(step)
        step_outputs = [o["name"] for o in
                        ((step.get("outputs") or {}).get("parameters") or [])]
        outputs_union.update(step_outputs)
        twin = {
            "name": task["name"] + "-" + provider,
            "template": "meluxina-run",
            "when": "{{=workflow.parameters.target == '%s'}}" % provider,
            "arguments": {"parameters": [
                {"name": "step-name", "value": task["name"]},
                {"name": "image", "value": container.get("image", "")},
                {"name": "step-command", "value": _cmd_json(cmd)},
                {"name": "time-limit", "value": str(minutes)},
                {"name": "deadline-seconds",
                 "value": str((minutes + QUEUE_ALLOWANCE_MINUTES) * 60)},
                {"name": "step-outputs", "value": ",".join(step_outputs)},
            ]},
        }
        if task.get("depends"):
            twin["depends"] = task["depends"]
        tasks.append(twin)
        routed.append(task["name"])

    if not routed:
        return

    # F-05: downstream steps consume {{tasks.<routed>.outputs.parameters.P}}.
    # Exactly one twin ran; the other is Skipped and its outputs resolve to
    # their declared defaults ("{}"). Pick the twin that actually ran.
    def _pick(m):
        r, p = m.group(1), m.group(2)
        return ("{{=tasks['%s'].status == 'Skipped' ? tasks['%s-%s'].outputs."
                "parameters['%s'] : tasks['%s'].outputs.parameters['%s']}}"
                % (r, r, provider, p, r, p))
    ref_re = re.compile(r"\{\{\s*tasks\.(%s)\.outputs\.parameters\.([A-Za-z0-9_.-]+)\s*\}\}"
                        % "|".join(re.escape(r) for r in routed))
    for task in tasks:
        if task["name"].endswith("-" + provider):
            continue
        for p in ((task.get("arguments") or {}).get("parameters") or []):
            v = p.get("value")
            if isinstance(v, str) and ref_re.fullmatch(v.strip()):
                p["value"] = ref_re.sub(_pick, v.strip())

    # A bare task token in Argo depends grammar means Succeeded; a routed dep
    # is now a twin pair where exactly one twin runs and the other is Skipped.
    for task in tasks:
        dep = task.get("depends")
        if not dep or task["name"].endswith("-" + provider):
            continue
        for r in routed:
            if task["name"] == r:
                continue
            pair = ("((%s.Succeeded || %s.Skipped) && "
                    "(%s-%s.Succeeded || %s-%s.Skipped))"
                    % (r, r, r, provider, r, provider))
            dep = re.sub(r"(?<![\w.-])%s(?![\w.-])" % re.escape(r), pair, dep)
        task["depends"] = dep

    # F-08 wallet inputs: the machine key mounts optional:true (same shape
    # the legacy render-wft uses on every step) so a non-OIDC deployment
    # still submits — the submit code degrades to no-wallet loudly. Dataset
    # coordinates prefer the pipeline's own workflow params (Alexandra's
    # WFTs carry lakefs-repo / data-ref) and fall back to the context
    # repository; no param -> no stage-in, batch runs the step data-less.
    mlflow_ctx = ctx.get("mlflow") or {}
    lakefs_ctx = ctx.get("lakefs") or {}
    wf_params = {p.get("name") for p in parameters}
    dataset_repo = ("{{workflow.parameters.lakefs-repo}}"
                    if "lakefs-repo" in wf_params
                    else str(lakefs_ctx.get("repository") or ""))
    dataset_ref = ("{{workflow.parameters.data-ref}}"
                   if "data-ref" in wf_params else "")
    spec["templates"].append({
        "name": "meluxina-run",
        "inputs": {"parameters": [
            {"name": "step-name"}, {"name": "image"}, {"name": "step-command"},
            {"name": "time-limit"}, {"name": "deadline-seconds"},
            {"name": "step-outputs", "value": ""}]},
        # Per-step: Slurm time limit + queue/prep allowance (see
        # QUEUE_ALLOWANCE_MINUTES). Argo resolves the parameter here.
        "activeDeadlineSeconds": "{{inputs.parameters.deadline-seconds}}",
        "metadata": {"labels": {"platform.kubecore.io/compute-type": "hpc"}},
        "volumes": [{"name": "mlflow-svc", "secret": {
            "secretName": str(mlflow_ctx.get("svcSecret") or "mlflow-svc"),
            "optional": True}},
                    # The rotated MeluXina JWT as a file: kubelet refreshes the
                    # mount on rotation, so long queue waits keep a valid token.
                    {"name": "meluxina-jwt", "secret": {"secretName": "meluxina-jwt",
                                                        "optional": True}}],
        "container": {
            "image": "python:3.12-alpine",
            "command": ["python3", "-c", MELUXINA_SUBMIT_CODE],
            "volumeMounts": [{"name": "mlflow-svc",
                              "mountPath": "/etc/mlflow-svc",
                              "readOnly": True},
                             {"name": "meluxina-jwt",
                              "mountPath": "/etc/meluxina-jwt",
                              "readOnly": True}],
            "env": [
                {"name": "SLURM_TOKEN", "valueFrom": {"secretKeyRef": {
                    "name": "meluxina-jwt", "key": "token"}}},
                {"name": "WF_UID", "value": "{{workflow.uid}}"},
                {"name": "STEP_NAME", "value": "{{inputs.parameters.step-name}}"},
                {"name": "IMAGE_REF", "value": "{{inputs.parameters.image}}"},
                {"name": "STEP_COMMAND", "value": "{{inputs.parameters.step-command}}"},
                {"name": "SLURM_TIME_LIMIT", "value": "{{inputs.parameters.time-limit}}"},
                {"name": "STEP_OUTPUTS", "value": "{{inputs.parameters.step-outputs}}"},
                {"name": "ZITADEL_MACHINE_KEY_FILE",
                 "value": "/etc/mlflow-svc/ZITADEL_MACHINE_KEY"},
                {"name": "ZITADEL_DOMAIN",
                 "value": str(mlflow_ctx.get("oidcDomain") or "")},
                {"name": "ZITADEL_MLFLOW_PROJECT_ID",
                 "value": str(mlflow_ctx.get("oidcProjectId") or "")},
                {"name": "MLFLOW_EXTERNAL_URL",
                 "value": str(mlflow_ctx.get("externalUrl") or "")},
                {"name": "LAKEFS_EXTERNAL_URL",
                 "value": str(lakefs_ctx.get("externalUrl") or "")},
                {"name": "DATASET_REPO", "value": dataset_repo},
                {"name": "DATASET_REF", "value": dataset_ref},
            ],
        },
        # slurm-job-id plus the union of every routed step's declared outputs
        # (one shared template; each twin fills its own, the rest default).
        "outputs": {"parameters": [{"name": "slurm-job-id",
                                    "valueFrom": {"path": "/tmp/slurm-job-id"}}]
                    + [{"name": n, "valueFrom": {"path": "/tmp/outputs/%s.json" % n,
                                                 "default": "{}"}}
                       for n in sorted(outputs_union)]},
    })
