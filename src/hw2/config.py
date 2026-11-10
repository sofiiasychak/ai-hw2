"""Load YAML experiment configs with base inheritance and dotted overrides."""

from __future__ import annotations

import copy
import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import yaml

from hw2.paths import repo_root

BASE_KEY = "base"


class _Loader(yaml.SafeLoader):
    """Safe YAML loader that also reads floats like 1e-3."""


_Loader.add_implicit_resolver(
    "tag:yaml.org,2002:float",
    re.compile(r"^[-+]?(?:\d+\.?\d*|\.\d+)[eE][-+]?\d+$"),
    list("-+.0123456789"),
)


def parse_yaml(text: str) -> Any:
    """Parse YAML text safely, accepting scientific notation floats."""
    return yaml.load(text, Loader=_Loader)


def deep_merge(base: Mapping[str, Any], child: Mapping[str, Any]) -> dict[str, Any]:
    """Return a new dict with child merged recursively into base, child values winning."""
    merged = copy.deepcopy(dict(base))
    for key, value in child.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _read_mapping(path: Path) -> dict[str, Any]:
    """Read one YAML file and return its top-level mapping."""
    data = parse_yaml(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise TypeError(f"{path}: top level must be a mapping")
    return data


def _load_tree(path: Path, chain: tuple[Path, ...] = ()) -> dict[str, Any]:
    """Load a config and its base files recursively, rejecting inheritance cycles."""
    if path in chain:
        raise ValueError(f"base cycle: {' -> '.join(str(p) for p in (*chain, path))}")
    data = _read_mapping(path)
    bases = data.pop(BASE_KEY, None) or []
    merged: dict[str, Any] = {}
    for base in [bases] if isinstance(bases, str) else bases:
        merged = deep_merge(merged, _load_tree((path.parent / base).resolve(), (*chain, path)))
    return deep_merge(merged, data)


def parse_override(item: str) -> tuple[list[str], Any]:
    """Split an a.b=value override into its key path and YAML-parsed value."""
    key, sep, raw = item.partition("=")
    keys = key.strip().split(".")
    if not sep or not all(keys):
        raise ValueError(f"override must look like a.b=value, got {item!r}")
    return keys, parse_yaml(raw)


def apply_override(cfg: dict[str, Any], item: str) -> None:
    """Set one dotted override in place, creating missing nested mappings."""
    keys, value = parse_override(item)
    node = cfg
    for key in keys[:-1]:
        node = node.setdefault(key, {})
        if not isinstance(node, dict):
            raise ValueError(f"override {item!r}: {key!r} is not a mapping")
    node[keys[-1]] = value


def load_config(path: str | Path, overrides: Iterable[str] | None = None) -> dict[str, Any]:
    """Load a YAML config with recursive bases, add exp_id and exp_name, then apply overrides."""
    path = Path(path).resolve()
    cfg = _load_tree(path)
    cfg["exp_name"] = path.stem
    cfg["exp_id"] = path.stem.split("_", 1)[0]
    for item in overrides or ():
        apply_override(cfg, item)
    return cfg


def find_config(exp_id: str) -> Path:
    """Return the single configs/<family>/<exp_id>_<name>.yaml file of an experiment id."""
    matches = sorted((repo_root() / "configs").glob(f"*/{exp_id}_*.yaml"))
    if len(matches) != 1:
        raise FileNotFoundError(f"expected one config for {exp_id}, found {[str(m) for m in matches]}")
    return matches[0]
