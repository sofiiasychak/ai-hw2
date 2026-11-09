"""Merge a config's model block over model defaults and validate choices."""

from __future__ import annotations

from collections.abc import Collection, Mapping
from typing import Any


def model_settings(defaults: Mapping[str, Any], cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Return defaults updated by the config's model block, raising on keys the model does not know."""
    given = dict(cfg.get("model") or {})
    unknown = sorted(set(given) - set(defaults))
    if unknown:
        raise KeyError(f"unknown model settings {unknown}; known: {sorted(defaults)}")
    return {**defaults, **given}


def check_choice(settings: Mapping[str, Any], key: str, options: Collection[str]) -> str:
    """Return a setting's value after checking it is one of the allowed options."""
    value = settings[key]
    if value not in options:
        raise ValueError(f"{key} must be one of {sorted(options)}, got {value!r}")
    return value


def check_levels(sizes: Collection[Any], levels: int, key: str) -> None:
    """Raise when a per-level anchor list does not have one entry per feature pyramid level."""
    if len(sizes) != levels:
        raise ValueError(f"{key} needs one entry per FPN level ({levels}), got {len(sizes)}")
