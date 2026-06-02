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

import json

import pytest

from rlinf.models.embodiment.gr00t_n1d7.checkpoint_utils import (
    infer_groot_n1_7_action_execution_horizon,
    infer_groot_n1_7_action_horizon,
    infer_groot_n1_7_embodiment_tag,
    normalize_embodiment_tag_name,
    resolve_embodiment_tag_enum,
    resolve_embodiment_tag_for_checkpoint,
)
from rlinf.models.embodiment.gr00t_n1d7.embodiment_tags import EmbodimentTag


def _write_libero_n17_processor(tmp_path):
    processor_kwargs = {
        "modality_configs": {
            "libero_sim": {
                "action": {"delta_indices": list(range(16))},
                "video": {"delta_indices": list(range(25))},
            }
        },
        "max_action_horizon": 40,
    }
    (tmp_path / "processor_config.json").write_text(
        json.dumps({"processor_kwargs": processor_kwargs})
    )
    (tmp_path / "statistics.json").write_text("{}")
    (tmp_path / "embodiment_id.json").write_text(json.dumps({"libero_sim": 2}))


def test_infer_libero_sim_tag_and_horizons(tmp_path):
    _write_libero_n17_processor(tmp_path)
    assert infer_groot_n1_7_embodiment_tag(tmp_path) == "libero_sim"
    assert infer_groot_n1_7_action_horizon(tmp_path) == 16
    assert infer_groot_n1_7_action_execution_horizon(tmp_path) == 8


def test_libero_panda_alias_maps_to_libero_sim():
    assert normalize_embodiment_tag_name("libero_panda") == "libero_sim"
    assert resolve_embodiment_tag_enum("libero_panda") == EmbodimentTag.LIBERO_SIM


def test_resolve_embodiment_auto_infer(tmp_path):
    _write_libero_n17_processor(tmp_path)
    tag = resolve_embodiment_tag_for_checkpoint(
        "libero_sim", tmp_path, None, auto_infer=True
    )
    assert tag == EmbodimentTag.LIBERO_SIM


def test_mismatched_cfg_tag_raises(tmp_path):
    _write_libero_n17_processor(tmp_path)
    with pytest.raises(ValueError, match="does not match checkpoint"):
        resolve_embodiment_tag_for_checkpoint(
            "robocasa_panda_omron", tmp_path, None, auto_infer=True
        )
