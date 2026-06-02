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

"""GR00T N1.7 (``Gr00tN1d7``) RL integration.

Parallel to ``rlinf.models.embodiment.gr00t_n1d6`` but built against the
Isaac-GR00T **N1.7** package (Cosmos-Reason2-2B / Qwen3-VL backbone). The
observation path uses ``Gr00tN1d7Processor.process_observation`` instead of the
N1.6 ``SimulationContent``/messages path, and the VLM tensors are Qwen3-VL
(``input_ids`` / ``pixel_values`` / ``image_grid_thw``) rather than ``eagle_*``.

This integration is **additive**: it does not touch ``gr00t_n1d6``.
"""

import logging
import sys
import types
from pathlib import Path

import torch
from omegaconf import DictConfig, OmegaConf

# Monkey-patch: inject a stub for rlinf.envs.libero.asset_paths so that
# env workers can import it even when the file is absent from the container
# (mirrors gr00t_n1d6).
_OLD_ASSET_PATHS = sys.modules.get("rlinf.envs.libero.asset_paths")
if _OLD_ASSET_PATHS is None:
    _stub = types.ModuleType("rlinf.envs.libero.asset_paths")

    def _noop(*args, **kwargs):
        pass

    _stub.apply_standard_libero_env_vars = _noop
    sys.modules["rlinf.envs.libero.asset_paths"] = _stub


def get_model(cfg: DictConfig, torch_dtype=torch.bfloat16):
    from gr00t.configs.model.gr00t_n1d7 import Gr00tN1d7Config
    from gr00t.model.gr00t_n1d7.gr00t_n1d7 import Gr00tN1d7
    from transformers import AutoConfig, AutoModel

    # ``gr00t.model.gr00t_n1d7.gr00t_n1d7`` already calls AutoConfig/AutoModel
    # register at import time, so guard against the "already registered" error
    # (also makes get_model idempotent across multiple worker constructions).
    try:
        AutoConfig.register("Gr00tN1d7", Gr00tN1d7Config)
        AutoModel.register(Gr00tN1d7Config, Gr00tN1d7)
    except ValueError:
        pass
    logging.info("gr00t_n1d7: architecture Gr00tN1d7 registered (or already present).")

    import rlinf.hybrid_engines.fsdp.strategy.fsdp as fsdp_strategy

    if not hasattr(fsdp_strategy, "_is_gr00t_patched"):
        orig_policy = fsdp_strategy.get_fsdp_wrap_policy

        def custom_fsdp_wrap_policy(
            module, config=None, is_lora=False, model_type=None
        ):
            import functools

            from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

            # Keyword-matched so it works for both the N1.6 (Qwen3 + Siglip2)
            # and N1.7 (Qwen3-VL / Cosmos-Reason2) backbones without hard-coding
            # exact class names.
            target_keywords = [
                "DecoderLayer",
                "EncoderLayer",
                "VisionBlock",
                "DiTBlock",
                "NoiseNet",
                "ValueHead",
                "ActionHead",
                "Timestep",
            ]
            found_classes = set()
            for name, mod in module.named_modules():
                cname = mod.__class__.__name__
                if any(key in cname for key in target_keywords):
                    found_classes.add(mod.__class__)

            if found_classes:
                logging.info(
                    "\n  FSDP Slicer: %s\n", [c.__name__ for c in found_classes]
                )
                return functools.partial(
                    transformer_auto_wrap_policy, transformer_layer_cls=found_classes
                )

            return orig_policy(module, config, is_lora, model_type)

        fsdp_strategy.get_fsdp_wrap_policy = custom_fsdp_wrap_policy
        fsdp_strategy._is_gr00t_patched = True

    from rlinf.utils.patcher import Patcher

    Patcher.clear()
    Patcher.add_patch(
        "gr00t.data.embodiment_tags.EmbodimentTag",
        "rlinf.models.embodiment.gr00t_n1d7.embodiment_tags.EmbodimentTag",
    )
    Patcher.add_patch(
        "gr00t.data.embodiment_tags.EMBODIMENT_TAG_MAPPING",
        "rlinf.models.embodiment.gr00t_n1d7.embodiment_tags.EMBODIMENT_TAG_MAPPING",
    )
    Patcher.apply()

    from rlinf.models.embodiment.gr00t_n1d7.checkpoint_utils import (
        infer_groot_n1_7_action_execution_horizon,
        resolve_embodiment_tag_for_checkpoint,
        resolve_embodiment_tag_manual,
    )
    from rlinf.models.embodiment.gr00t_n1d7.gr00t_action_model import (
        GR00T_N1_7_ForRLActionPrediction,
    )
    from rlinf.models.embodiment.gr00t_n1d7.utils import replace_dropout_with_identity

    model_path = Path(cfg.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model path does not exist: {model_path}")

    processor_path = OmegaConf.select(cfg, "processor_path", default=None)
    auto_infer_tag = bool(
        OmegaConf.select(cfg, "auto_infer_embodiment_tag", default=True)
    )
    cfg_embodiment_tag = OmegaConf.select(cfg, "embodiment_tag", default=None)
    use_official_libero_sim = bool(
        OmegaConf.select(cfg, "use_official_libero_sim", default=True)
    )
    if auto_infer_tag:
        emb_tag = resolve_embodiment_tag_for_checkpoint(
            cfg_embodiment_tag,
            model_path,
            processor_path,
            auto_infer=True,
        )
    else:
        if cfg_embodiment_tag is None:
            raise ValueError(
                "embodiment_tag is required when auto_infer_embodiment_tag is false."
            )
        emb_tag = resolve_embodiment_tag_manual(
            cfg_embodiment_tag,
            use_official_libero_sim=use_official_libero_sim,
        )
    logging.info(
        "gr00t_n1d7: using embodiment tag '%s' (processor key).",
        emb_tag.value,
    )

    config = Gr00tN1d7Config.from_pretrained(str(model_path))
    _action_dim = cfg.get("action_dim")
    if _action_dim is not None:
        config.action_dim = _action_dim

    denoising_steps = cfg.get("denoising_steps")
    if denoising_steps is None:
        denoising_steps = getattr(config, "num_inference_timesteps", 4)

    num_action_chunks = cfg.get("num_action_chunks")
    auto_infer_chunks = bool(
        OmegaConf.select(cfg, "auto_infer_action_chunks", default=True)
    )
    inferred_chunks = None
    if auto_infer_chunks:
        inferred_chunks = infer_groot_n1_7_action_execution_horizon(
            model_path, emb_tag.value, processor_path
        )
    if num_action_chunks is None:
        num_action_chunks = inferred_chunks if inferred_chunks is not None else 8
        logging.info(
            "gr00t_n1d7: num_action_chunks=%s (inferred from checkpoint).",
            num_action_chunks,
        )
    elif (
        inferred_chunks is not None and int(num_action_chunks) != int(inferred_chunks)
    ):
        logging.warning(
            "gr00t_n1d7: num_action_chunks=%s differs from checkpoint execution "
            "horizon %s (LeRobot/OSS LIBERO uses %s).",
            num_action_chunks,
            inferred_chunks,
            inferred_chunks,
        )

    model = GR00T_N1_7_ForRLActionPrediction.from_pretrained(
        config=config,
        local_model_path=str(model_path),
        pretrained_model_name_or_path=str(model_path),
        torch_dtype=torch_dtype,
        embodiment_tag=emb_tag,
        denoising_steps=denoising_steps,
        output_action_chunks=num_action_chunks,
        obs_converter_type=cfg.obs_converter_type,
        rl_head_config=cfg.rl_head_config,
        processor_path=processor_path,
    )

    model.to(torch_dtype)
    if cfg.rl_head_config.add_value_head and hasattr(model.action_head, "value_head"):
        # reinitialize the value head after model loading
        model.action_head.value_head._init_weights()

    if cfg.rl_head_config.disable_dropout:
        replace_dropout_with_identity(model)

    return model


def patch_fsdp_rollout_state_dict():
    """Patch EmbodiedFSDPActor.get_rollout_state_dict to use full_state_dict=True.

    Model-agnostic (identical to the gr00t_n1d6 helper); kept here so the N1.7
    path does not need to import the N1.6 module.
    """
    try:
        from rlinf.workers.actor.fsdp_actor_worker import EmbodiedFSDPActor

        def _patched_get_rollout_state_dict(self) -> dict:
            return self.get_model_state_dict(cpu_offload=False, full_state_dict=True)

        EmbodiedFSDPActor.get_rollout_state_dict = _patched_get_rollout_state_dict
        logging.info(
            "[GR00T patch] EmbodiedFSDPActor.get_rollout_state_dict patched: "
            "full_state_dict=True for multi-GPU FSDP weight sync safety"
        )
    except Exception as e:
        logging.warning("[GR00T patch] Failed to patch get_rollout_state_dict: %s", e)
