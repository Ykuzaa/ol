"""Small shared utilities."""

from pathlib import Path

import yaml


class ConfigNode(dict):
    """Dictionary with attribute access."""

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def _deep_update(base: dict, update: dict) -> dict:
    result = dict(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_update(result[key], value)
        else:
            result[key] = value
    return result


def _to_config_node(value):
    if isinstance(value, dict):
        return ConfigNode({key: _to_config_node(val) for key, val in value.items()})
    if isinstance(value, list):
        return [_to_config_node(item) for item in value]
    return value


def load_config(variant, config_dir=None):
    """Load base config and merge one variant config."""
    if config_dir is None:
        config_dir = Path(__file__).resolve().parents[2] / "configs"
    config_dir = Path(config_dir)

    with open(config_dir / "base.yaml") as handle:
        base_cfg = yaml.safe_load(handle)
    with open(config_dir / "variants" / f"{variant}.yaml") as handle:
        variant_cfg = yaml.safe_load(handle)

    merged = _deep_update(base_cfg, variant_cfg)
    return _to_config_node(merged)


def seed_everything(seed: int) -> None:
    """Set all random seeds."""
    import pytorch_lightning as pl

    pl.seed_everything(seed, workers=True)
