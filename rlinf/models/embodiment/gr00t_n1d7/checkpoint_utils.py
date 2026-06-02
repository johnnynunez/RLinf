# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Checkpoint helpers for GR00T N1.7, aligned with LeRobot groot policy utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rlinf.models.embodiment.gr00t_n1d7.embodiment_tags import EmbodimentTag

# NVIDIA N1.7 LIBERO checkpoints use ``libero_sim`` in processor_config.json.
LIBERO_N17_PROCESSOR_TAG = "libero_sim"
# OSS LIBERO rollout replans after 8 of the decoded action horizon (see Isaac-GR00T).
LIBERO_N17_ACTION_EXECUTION_HORIZON = 8

# RLinf / legacy config names -> processor modality key (N1.7).
_EMBODIMENT_TAG_ALIASES: dict[str, str] = {
    "libero_panda": LIBERO_N17_PROCESSOR_TAG,
    "libero_franka": LIBERO_N17_PROCESSOR_TAG,
}


def resolve_processor_path(
    model_path: str | Path,
    processor_path: str | Path | None = None,
) -> Path:
    """Return the directory containing N1.7 processor assets."""
    if processor_path is not None:
        return Path(processor_path).expanduser()
    return Path(model_path).expanduser()


def _read_processor_kwargs(processor_root: Path) -> dict[str, Any] | None:
    config_path = processor_root / "processor_config.json"
    try:
        with config_path.open() as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    processor_kwargs = payload.get("processor_kwargs", {})
    if not isinstance(processor_kwargs, dict):
        return None
    return processor_kwargs


def infer_groot_n1_7_embodiment_tag(
    model_path: str | Path | None,
    processor_path: str | Path | None = None,
) -> str | None:
    """Infer the processor modality key from ``processor_config.json``."""
    if model_path is None:
        return None
    processor_root = resolve_processor_path(model_path, processor_path)
    processor_kwargs = _read_processor_kwargs(processor_root)
    if processor_kwargs is None:
        return None

    modality_configs = processor_kwargs.get("modality_configs", {})
    if not isinstance(modality_configs, dict):
        return None
    if LIBERO_N17_PROCESSOR_TAG in modality_configs:
        return LIBERO_N17_PROCESSOR_TAG
    if len(modality_configs) == 1:
        return next(iter(modality_configs))
    return None


def infer_groot_n1_7_action_horizon(
    model_path: str | Path | None,
    embodiment_tag: str | None = None,
    processor_path: str | Path | None = None,
) -> int | None:
    """Infer decoded action horizon from action ``delta_indices`` in the processor."""
    if model_path is None:
        return None
    processor_root = resolve_processor_path(model_path, processor_path)
    processor_kwargs = _read_processor_kwargs(processor_root)
    if processor_kwargs is None:
        return None

    modality_configs = processor_kwargs.get("modality_configs", {})
    if not isinstance(modality_configs, dict):
        return None

    if embodiment_tag is None:
        embodiment_tag = infer_groot_n1_7_embodiment_tag(model_path, processor_path)
    if embodiment_tag is None:
        return None

    embodiment_config = modality_configs.get(embodiment_tag, {})
    if not isinstance(embodiment_config, dict):
        return None
    action_config = embodiment_config.get("action", {})
    if not isinstance(action_config, dict):
        return None
    delta_indices = action_config.get("delta_indices", [])
    if not isinstance(delta_indices, list):
        return None
    return len(delta_indices) or None


def infer_groot_n1_7_action_execution_horizon(
    model_path: str | Path | None,
    embodiment_tag: str | None = None,
    processor_path: str | Path | None = None,
) -> int | None:
    """Horizon of actions executed per replan (matches LeRobot / OSS LIBERO)."""
    action_horizon = infer_groot_n1_7_action_horizon(
        model_path, embodiment_tag, processor_path
    )
    if action_horizon is None:
        return None
    if embodiment_tag is None:
        embodiment_tag = infer_groot_n1_7_embodiment_tag(model_path, processor_path)
    if embodiment_tag == LIBERO_N17_PROCESSOR_TAG:
        return min(action_horizon, LIBERO_N17_ACTION_EXECUTION_HORIZON)
    return action_horizon


def normalize_embodiment_tag_name(tag: str) -> str:
    """Map legacy RLinf tags to N1.7 processor modality keys."""
    return _EMBODIMENT_TAG_ALIASES.get(tag, tag)


def resolve_embodiment_tag_enum(tag: str) -> EmbodimentTag:
    """Resolve a config or checkpoint tag string to ``EmbodimentTag``."""
    normalized = normalize_embodiment_tag_name(tag)
    for member in EmbodimentTag:
        if member.value == normalized:
            return member
    raise ValueError(
        f"Unknown GR00T N1.7 embodiment tag '{tag}' (normalized: '{normalized}'). "
        f"Known tags: {[m.value for m in EmbodimentTag]}."
    )


def resolve_embodiment_tag_for_checkpoint(
    cfg_tag: str | None,
    model_path: str | Path,
    processor_path: str | Path | None,
    *,
    auto_infer: bool = True,
) -> EmbodimentTag:
    """Pick the processor embodiment key, preferring the checkpoint when enabled."""
    inferred = infer_groot_n1_7_embodiment_tag(model_path, processor_path)
    if auto_infer and inferred is not None:
        if cfg_tag is not None:
            cfg_normalized = normalize_embodiment_tag_name(cfg_tag)
            if cfg_normalized != inferred:
                raise ValueError(
                    f"Configured embodiment_tag '{cfg_tag}' does not match checkpoint "
                    f"processor tag '{inferred}'. Set auto_infer_embodiment_tag: true "
                    f"or align embodiment_tag with processor_config.json."
                )
        return resolve_embodiment_tag_enum(inferred)

    if cfg_tag is None:
        if inferred is not None:
            return resolve_embodiment_tag_enum(inferred)
        raise ValueError(
            "embodiment_tag is required when it cannot be inferred from the checkpoint."
        )

    if inferred is not None:
        cfg_normalized = normalize_embodiment_tag_name(cfg_tag)
        if cfg_normalized != inferred:
            raise ValueError(
                f"Configured embodiment_tag '{cfg_tag}' does not match checkpoint "
                f"processor tag '{inferred}'."
            )
    return resolve_embodiment_tag_enum(cfg_tag)
