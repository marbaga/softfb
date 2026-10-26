# Adapted from https://github.com/facebookresearch/metamotivo

import dataclasses
from typing import Any, Dict
from collections.abc import Mapping
from pathlib import Path
from typing import Any
import json
import safetensors.torch
import yaml


def load_model(path: str, device: str | None, cls: Any):
    model_dir = Path(path)
    with (model_dir / "config.json").open() as f:
        loaded_config = json.load(f)
    if device is not None:
        loaded_config["device"] = device
    loaded_agent = cls(**loaded_config)
    safetensors.torch.load_model(loaded_agent, model_dir / "model.safetensors", device=device)
    return loaded_agent


def dict_to_config(source: Mapping, target: Any):
    target_fields = {field.name for field in dataclasses.fields(target)}
    for field in target_fields:
        if field in source.keys() and dataclasses.is_dataclass(getattr(target, field)):
            dict_to_config(source[field], getattr(target, field))
        elif field in source.keys():
            setattr(target, field, source[field])
        else:
            print(f"[WARNING] field {field} not found in source config")


def config_from_dict(source: Dict, config_class: Any):
    target = config_class()
    dict_to_config(source, target)
    return target


def update_with_overrides(config: Any, overrides: Dict):
    for k, v in overrides.items():
        assert hasattr(config, k), f'Cannot find parameter key: {k}'
        if isinstance(v, dict):
            update_with_overrides(getattr(config, k), v)
        else:
            setattr(config, k, v)


def dump(m_dict, t, path):
    with path.open("a+") as f:
        all_m = yaml.safe_load(f)
        all_m = {} if all_m is None else all_m
    for k, v in m_dict.items():
        if k not in all_m:
            all_m[k] = {'step': [], 'value': []}
        all_m[k]['step'].append(t)
        all_m[k]['value'].append(v)
    with path.open("w") as f:
        yaml.dump(all_m, f)
