"""Behaviour taxonomy loader — reads taxonomy.yaml and provides access to thresholds/severities."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import yaml


_taxonomy_cache: Optional[dict] = None  # behaviours section only
_full_cache: Optional[dict] = None  # whole YAML (behaviours + fragility + …)

DEFAULT_TAXONOMY_PATH = Path(__file__).parent / "taxonomy.yaml"

# Detector class names -> fragility tier in taxonomy.yaml. Classes absent
# here (and not direct tier names) fall back to Standard (1.0).
CLASS_TIER = {
    "cupboard": "Fragile",
    "cabinet": "Fragile",
    "appliance": "Fragile",
    "wardrobe": "Fragile",
    "furniture": "Fragile",
    "mattress": "Rugged",
    "carton": "Standard",
    "box": "Standard",
    "cardboard box": "Standard",
    "packet": "Standard",
    "package": "Standard",
    "parcel": "Standard",
    "goods": "Standard",
    "crate": "Standard",
    "bag": "Standard",
    "load": "Standard",
}


def _load_yaml(path: Optional[str | Path] = None) -> dict:
    global _full_cache
    if _full_cache is not None and path is None:
        return _full_cache
    if path is None:
        path = DEFAULT_TAXONOMY_PATH
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    if path == DEFAULT_TAXONOMY_PATH:
        _full_cache = data
    return data


def load_taxonomy(path: Optional[str | Path] = None) -> dict:
    """Load and cache the behaviour taxonomy from YAML.

    Returns dict keyed by behaviour key (e.g. "1_product_dropped")
    with values being the full behaviour config.
    """
    global _taxonomy_cache
    if _taxonomy_cache is not None:
        return _taxonomy_cache

    _taxonomy_cache = _load_yaml(path).get("behaviours", {})
    return _taxonomy_cache


def get_severity(behaviour_key: str) -> int:
    """Get the base severity for a behaviour type."""
    taxonomy = load_taxonomy()
    return taxonomy.get(behaviour_key, {}).get("severity_base", 50)


def get_thresholds(behaviour_key: str) -> dict:
    """Get detection thresholds for a behaviour type."""
    taxonomy = load_taxonomy()
    return taxonomy.get(behaviour_key, {}).get("thresholds", {})


def get_fragility(class_name: str) -> float:
    """Get fragility multiplier for a product class.

    Looks up the top-level ``fragility`` section of taxonomy.yaml
    (Fragile 1.5 / Standard 1.0 / Rugged 0.7). Detector class names such as
    "cupboard" or "mattress" map to a tier via CLASS_TIER first; unknown
    classes default to Standard (1.0).
    """
    fragility = _load_yaml().get("fragility", {})
    if class_name in fragility:
        return float(fragility[class_name])
    tier = CLASS_TIER.get(class_name)
    if tier is None:
        return 1.0
    return float(fragility.get(tier, 1.0))
