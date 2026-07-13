"""Platform-owned form derivation: Hydra config tree -> workflow form.

The single source of truth for the tunable surface is the experiment-scoped
config/ tree; this module composes it (Hydra Compose API) and derives the Argo
submit form with the structural rule — no allowlists:

  config GROUP   -> ONE dropdown parameter
                    (options = the yaml files in the group dir,
                     default = the defaults-list choice)
  SCALAR LEAF    -> one form field; dotted path mapped dots->dashes
                    (train.optimizer.lr -> train-optimizer-lr) with a
                    hard-error collision check
  COMPLEX NODE   -> (lists, class_name+params structures) NOT
                    flattened; changed via group selection or the
                    ADVANCED override only

Booleans are plain values: a bool leaf derives as a form field whose default is
"true"/"false" and whose override token Hydra parses back to a real bool. No
enum workaround.

The top-level `platform` section (mounted from pipeline-context) is EXCLUDED
from the form: it is platform-injected runtime config, not a developer knob.

Each derived entry also carries its Hydra override token
(`train.epochs={{workflow.parameters.train-epochs}}`) — these become the
compose step's args — plus a RENDER-DEFAULTS manifest
({override_key: rendered_default}) baked into the compose step's args so it can
elide untouched leaves at submit time: a scalar whose submitted value equals its
render-time default is dropped, letting a group swap actually swap (group tokens
apply first; only explicitly changed leaves override the selected group).

Structured-config schemas: sections listed in SCHEMA_SECTIONS are validated by
dataclasses registered on Hydra's ConfigStore — bad values
(quantization.mode=banana) hard-fail at compose time, and Enum-typed fields
derive as form dropdowns.

Platform-group searchpath resolution (production vs local dev):
  - CI/runtime: the enhancer materializes the `platform` group from
    pipeline-context and sets KUBECORE_PLATFORM_SEARCHPATH to its dir.
  - Local dev: falls back to the vendored `kubecore/config/` group (the
    same shape, real values from the last context) so `python -m ...`
    and pipeline.py work on a laptop with no cluster.
"""

import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

# Hera is a RENDER-TIME dependency (form derivation only). The compose step
# runs at RUN time in a lean image without Hera, and only calls compose_tree()
# — so Hera is imported lazily inside derive(), never at module load.

HERE = Path(__file__).parent  # kubecore/
CONFIG_DIR = HERE.parent / "config"  # the developer's experiment config tree
# Local-dev fallback platform group (vendored). Production overrides via env.
PLATFORM_SEARCHPATH = Path(
    os.environ.get("KUBECORE_PLATFORM_SEARCHPATH", str(HERE / "config"))
)

PLATFORM_SECTIONS = {"platform"}  # injected, never a form surface

# Sections with structured-config (dataclass) schemas: types + enums validated
# at compose time; Enum fields become form dropdowns. Extend this as the
# template declares more schemas (see SCHEMAS below).
SCHEMA_SECTIONS = ("data", "quantization")

ADVANCED_PARAM_KWARGS = dict(
    name="config",
    default="",
    description=(
        "ADVANCED: override YAML merged LAST over the composed config "
        "(OmegaConf.merge). Overrides everything above — intended for "
        "scripted submissions (argo submit -p config=...). May not "
        "modify platform.*."
    ),
)


class DeriveError(Exception):
    pass


# ---------------------------------------------------- structured schemas


class DataSource(Enum):
    lakefs = "lakefs"
    s3 = "s3"


class QuantizationMode(Enum):
    none = "none"
    ptq = "ptq"
    qat = "qat"


@dataclass
class DataConfig:
    ref: str = "main"
    source: DataSource = DataSource.lakefs
    version: str = ""
    path_override: str = ""
    sample_size: str = ""
    seed: int = 42


@dataclass
class QuantizationConfig:
    mode: QuantizationMode = QuantizationMode.none
    image_size: int = 640
    calibration_frames: int = 512
    calibration_seed: int = 42
    parity_frames: int = 100
    parity_max_abs_error: float = 0.05
    output_prefix: str = "quantization"


# {config-store name registered in config.yaml's defaults list: node}
SCHEMAS = {
    "data_schema": {"data": DataConfig},
    "quantization_schema": {"quantization": QuantizationConfig},
}

# Top-level sections that a structured-config schema backfills. A schema injects
# its section (with dataclass defaults) into the composed tree EVEN IF the
# section's yaml file was renamed/deleted — which would otherwise silently mask
# the reads= gate (a rename typo renders "successfully" with only schema
# defaults, and the developer's real values become an unread orphan section).
# So the gate must additionally require a real on-disk source for these.
SCHEMA_BACKED = {sec for node in SCHEMAS.values() for sec in node}


def _register_schemas() -> None:
    """Schemas the yaml sections merge ONTO (defaults list puts the schema
    entries first) — compose+overrides are then type/enum validated by
    OmegaConf. Extend SCHEMAS + config.yaml defaults to cover more sections."""
    cs = ConfigStore.instance()
    for name, node in SCHEMAS.items():
        cs.store(name=name, node=node)


_register_schemas()


@dataclass
class DerivedForm:
    parameters: list = field(default_factory=list)  # hera Parameters (groups first)
    tokens: list = field(default_factory=list)  # hydra override tokens for compose args
    sections: list = field(default_factory=list)  # top-level config sections (excl platform)
    render_defaults: dict = field(default_factory=dict)  # {override_key: rendered default}


def searchpath_override() -> str:
    return f"hydra.searchpath=[file://{PLATFORM_SEARCHPATH}]"


def _compose(overrides: list, return_hydra_config: bool = False):
    """Single choke point for Hydra compose. Reframes Hydra's
    MissingConfigException (a defaults-list entry points at a group option file
    that was deleted/renamed) as a clean, actionable DeriveError instead of a
    raw Hydra stack trace + search-path dump."""
    from hydra.errors import MissingConfigException
    try:
        with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
            return compose(
                config_name="config",
                overrides=[searchpath_override()] + list(overrides or []),
                return_hydra_config=return_hydra_config,
            )
    except MissingConfigException as exc:
        # message looks like: In 'config': Could not find 'model/yolov8n'
        raise DeriveError(
            f"config tree is broken: {exc}. A defaults-list entry in config/config.yaml "
            f"points at a group option that no longer exists — restore the option file "
            f"(e.g. config/<group>/<option>.yaml) or fix the choice in config/config.yaml."
        ) from exc


def compose_tree(overrides: list = None):
    """Compose the config tree (+ the platform group via searchpath)."""
    return _compose(overrides)


def _group_choices() -> dict:
    """{group_path: default_option} from the defaults list, via Hydra.
    Filtered to real group directories (schema entries etc. are not groups)."""
    hydra_cfg = _compose([], return_hydra_config=True)
    return {
        k: v
        for k, v in hydra_cfg.hydra.runtime.choices.items()
        if not k.startswith("hydra/") and k not in PLATFORM_SECTIONS and (CONFIG_DIR / k).is_dir()
    }


def _group_options(group: str) -> list:
    return sorted(p.stem for p in (CONFIG_DIR / group).glob("*.yaml"))


def _mapped(dotted: str) -> str:
    return dotted.replace(".", "-").replace("/", "-")


def _value_str(value) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, Enum):
        return str(value.value)
    return str(value)


def _walk_scalars(node: dict, path: str = "") -> list:
    """(dotted_path, value) for every scalar leaf; complex nodes (lists)
    are not flattened, plain dicts are recursed. Enum values (from
    structured schemas) count as scalars."""
    out = []
    for key, value in node.items():
        dotted = f"{path}.{key}" if path else str(key)
        if isinstance(value, dict):
            out.extend(_walk_scalars(value, dotted))
        elif isinstance(value, list):
            continue  # complex node: group selection or ADVANCED override only
        else:
            out.append((dotted, value))
    return out


def derive() -> DerivedForm:
    # Hera imported HERE (render time only) — keeps the run-time compose image
    # free of Hera. See module docstring / the hera-note above the imports.
    from hera.workflows import Parameter

    tree = OmegaConf.to_container(compose_tree(), resolve=True)
    for section in PLATFORM_SECTIONS:
        tree.pop(section, None)

    form = DerivedForm(sections=list(tree.keys()))
    seen = {}

    def claim(name: str, origin: str) -> None:
        if name in seen:
            raise DeriveError(
                f"form-name collision: '{name}' derived from both {seen[name]} "
                f"and {origin}; rename one of the config keys"
            )
        seen[name] = origin

    # groups -> dropdowns (form order: groups first)
    for group, default in sorted(_group_choices().items()):
        name = _mapped(group)
        claim(name, f"config group {group}/")
        options = _group_options(group)
        form.parameters.append(
            Parameter(
                name=name,
                default=default,
                enum=options,
                description=f"Config group '{group}': swaps the whole subtree "
                f"(options: {', '.join(options)}).",
            )
        )
        form.tokens.append(f"{group}={{{{workflow.parameters.{name}}}}}")
        form.render_defaults[group] = default

    # scalar leaves -> fields
    for dotted, value in _walk_scalars(tree):
        name = _mapped(dotted)
        claim(name, f"config leaf {dotted}")
        default = _value_str(value)
        enum = [str(m.value) for m in type(value)] if isinstance(value, Enum) else None
        form.parameters.append(
            Parameter(name=name, default=default, enum=enum, description=f"Config: {dotted}")
        )
        form.tokens.append(f"{dotted}={{{{workflow.parameters.{name}}}}}")
        form.render_defaults[dotted] = default

    form.parameters.append(Parameter(**ADVANCED_PARAM_KWARGS))
    return form


def _has_config_source(section: str) -> bool:
    """True if the section is backed by a real on-disk config source that
    actually contributes the section — either a `config/{section}/` group dir,
    or a `config/{section}.yaml` whose top-level content includes the `{section}`
    key. The key check matters: renaming the key INSIDE data.yaml (data: ->
    dataX:) leaves the file present but no longer contributes `data`, so the
    schema would silently backfill it. Detect that."""
    if (CONFIG_DIR / section).is_dir():
        return True
    f = CONFIG_DIR / f"{section}.yaml"
    if not f.is_file():
        return False
    try:
        import yaml as _yaml
        doc = _yaml.safe_load(f.read_text()) or {}
        return isinstance(doc, dict) and section in doc
    except Exception:
        # If we can't parse it, assume it's a real source (don't false-fire).
        return True


def validate_reads(step_reads: dict, sections: list) -> None:
    """Render gate: every section a step declares via reads= must exist as a
    top-level section of the composed tree (closes the silent-failure mode:
    tree section renamed but a step still reads it).

    For SCHEMA_BACKED sections the composed tree alone is not enough — a
    structured-config schema backfills the section even when its yaml file was
    renamed/deleted, silently masking the rename. So a schema-backed section
    also requires a real on-disk config source; otherwise the gate fires with a
    rename-specific message."""
    for step_name, reads in step_reads.items():
        for section in reads:
            if section not in sections:
                raise DeriveError(
                    f"step '{step_name}' reads config section '{section}' which does "
                    f"not exist in the config tree (top-level sections: {', '.join(sections)})"
                )
            if section in SCHEMA_BACKED and not _has_config_source(section):
                raise DeriveError(
                    f"step '{step_name}' reads config section '{section}', which has a "
                    f"structured-config schema but no config/{section}.yaml (or "
                    f"config/{section}/ group) — its yaml was likely renamed or deleted. "
                    f"The schema backfills defaults, so the render would silently drop your "
                    f"real values. Restore config/{section}.yaml (or update reads= + the schema)."
                )
