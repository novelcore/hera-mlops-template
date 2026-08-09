"""compose-and-validate: what the first DAG step's container runs.

Invoked in-cluster as `python -m kubecore.compose`. Takes Hydra override
tokens (argv) plus the optional ADVANCED override YAML, composes the config
tree + overrides via the Hydra Compose API, validates, and emits the single
resolved params.yaml that determines the entire experiment. Downstream steps
receive this file's content as an input parameter; they never see override
tokens.

Override semantics (group swap vs touched leaves):
- GROUP tokens (key contains '/') are applied FIRST.
- A scalar-leaf token whose submitted value equals its RENDER-TIME
  default (--render-defaults manifest, baked into the WFT args) is
  DROPPED — an untouched form field follows whatever group you
  selected. A field you explicitly changed is applied AFTER the
  groups, so it wins.
Without this elision a full form submission would freeze every leaf at
the original defaults and silently revert any group swap.

Failure modes concentrated here, all before any GPU time:
- unknown/typo'd override key   -> Hydra strict mode rejects it
- type/enum-invalid value       -> structured-config validation
- step reads= section missing   -> --require check below
- ADVANCED override with unknown keys -> struct-mode merge rejects it
- ADVANCED override touching platform.* -> hard error (below)
- params.yaml too large         -> explicit size check (>200KB)
"""

import argparse
import json
import sys
from enum import Enum
from pathlib import Path

import yaml
from omegaconf import OmegaConf

from kubecore import derive_tree

MAX_PARAMS_BYTES = 200 * 1024

HERE = Path(__file__).parent  # kubecore/
DEFAULT_OUTPUT = HERE.parent / "out" / "params.yaml"


class ComposeError(Exception):
    pass


def _normalize_override_token(token: str) -> str:
    """Strip surrounding whitespace/control chars off an override token.

    Submitted form values travel UI -> Argo parameter -> argv, and that
    plumbing can attach a trailing CR/newline/space the submitter cannot see.
    A trailing '\\r' makes Hydra's override lexer throw
    LexerNoViableAltException on a perfectly ordinary token — live #867:
    `train.epochs=3\\r` failed EVERY numeric-tuning run with the caret
    pointing one past the digit. Surrounding whitespace is never semantic in
    a key=value token, so strip it from both the whole token and the value.
    """
    token = token.strip()
    key, sep, value = token.partition("=")
    if not sep:
        return token
    return f"{key.strip()}={value.strip()}"


def _quote_override_value(token: str) -> str:
    """Quote a scalar override's value so Hydra's override grammar accepts it.

    Form values reach us as `key=value` tokens. Hydra's override parser rejects
    a bare value containing a space or various special characters (a
    `LexerNoViableAltException`) — so a perfectly ordinary config string like
    `label=📊 Data summary` or `name=my run` would fail compose. Wrap the value
    in double quotes (escaping any it contains) unless it is already quoted or
    is a simple bare token Hydra parses fine. Group tokens (key contains '/')
    and the key itself are left untouched.

    HISTORY (#867): this helper originally shipped only as a hotfix on an
    app's VENDORED copy and was lost when the app re-synced to template HEAD
    — it lives here now so the template is the single source of truth.
    """
    key, sep, value = token.partition("=")
    if not sep or "/" in key:
        return token  # not a scalar leaf assignment (or a group swap)
    if value == "" or (value[0] in "\"'" and value[-1] == value[0]):
        return token  # empty, or already quoted
    # Bare tokens Hydra accepts as-is: alphanumerics, dot, underscore, hyphen,
    # plus/at/colon/slash for paths — anything else (space, emoji, punctuation)
    # needs quoting.
    import re
    if re.fullmatch(r"[A-Za-z0-9._+@:/\-]*", value):
        return token
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'{key}="{escaped}"'


def order_overrides(overrides: list, render_defaults: dict) -> list:
    """Groups first; untouched scalar leaves elided (see module docstring).

    Tokens are whitespace-normalized first (#867: invisible trailing CR from
    the submit plumbing), then changed scalar values are quoted so Hydra
    accepts strings with spaces / special characters."""
    groups, changed = [], []
    for token in overrides:
        token = _normalize_override_token(token)
        key, _, value = token.partition("=")
        if "/" in key:
            groups.append(token)
        elif render_defaults and key in render_defaults and value == render_defaults[key]:
            continue  # untouched -> follows the selected group
        else:
            changed.append(_quote_override_value(token))
    return groups + changed


def _plain(value):
    """Enum members (from structured schemas) -> their plain values."""
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_plain(v) for v in value]
    if isinstance(value, Enum):
        return value.value
    return value


def compose_and_validate(overrides: list, advanced: str = "", required: list = None,
                         render_defaults: dict = None) -> str:
    """Returns the resolved params.yaml text."""
    cfg = derive_tree.compose_tree(order_overrides(overrides, render_defaults or {}))

    if advanced.strip():
        parsed = yaml.safe_load(advanced)
        if isinstance(parsed, dict) and "platform" in parsed:
            raise ComposeError("ADVANCED override may not modify platform.*")
        # ADVANCED override merged LAST. Struct mode: unknown keys fail.
        cfg = OmegaConf.merge(cfg, OmegaConf.create(parsed))

    resolved = _plain(OmegaConf.to_container(cfg, resolve=True))
    for section in required or []:
        if section not in resolved:
            raise ComposeError(
                f"a step declares reads=['{section}'] but the composed config has no "
                f"'{section}' section (top-level sections: {', '.join(resolved)})"
            )

    text = OmegaConf.to_yaml(OmegaConf.create(resolved), resolve=True)
    if len(text.encode()) > MAX_PARAMS_BYTES:
        raise ComposeError(
            f"resolved params.yaml is {len(text.encode())} bytes — exceeds the "
            f"{MAX_PARAMS_BYTES}-byte Argo parameter budget; move bulk data out of config"
        )
    return text


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Compose and validate the experiment config.")
    parser.add_argument("--advanced", default="", help="ADVANCED override YAML, merged last.")
    parser.add_argument("--require", action="append", default=[],
                        help="Top-level section a step declared via reads= (repeatable).")
    parser.add_argument("--render-defaults", default="{}",
                        help="JSON manifest {override_key: render-time default} for elision.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("overrides", nargs="*", help="Hydra override tokens (key=value).")
    args = parser.parse_args(argv)

    try:
        text = compose_and_validate(
            args.overrides, args.advanced, args.require, json.loads(args.render_defaults)
        )
    except Exception as exc:  # any composition failure = hard error, pre-GPU
        message = str(exc)
        if exc.__cause__ is not None:  # e.g. enum ValidationError inside Hydra's wrapper
            message += f"\n  cause: {exc.__cause__}"
        print(f"compose-and-validate FAILED: {message}", file=sys.stderr)
        sys.exit(1)

    out = Path(args.output)
    out.parent.mkdir(exist_ok=True)
    out.write_text(text)
    print(f"wrote {out} ({len(text.encode())} bytes)")


if __name__ == "__main__":
    main()
