# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""TorchAO API compatibility for the pinned SGLang 0.5.12 server.

Executed by the isolated engine Python, including multiprocessing children.
No engine packages are imported into Studio's environment.
"""

import runpy

from sglang.srt.layers import torchao_utils


_apply_torchao = torchao_utils.apply_torchao_config_to_model


def _apply_torchao_config(
    model,
    torchao_config,
    filter_fn = torchao_utils.proj_filter,
):
    if torchao_config in ("int8wo", "int4wo-32", "fp8wo"):
        # SGLang imports removed TorchAO float8 aliases even for INT8; use the current config API.
        from torchao.quantization import (
            Int8WeightOnlyConfig,
            Int4WeightOnlyConfig,
            Float8WeightOnlyConfig,
            quantize_,
        )
        from torchao.quantization.quantize_.workflows import (
            Int4PackingFormat,
            Int4ChooseQParamsAlgorithm,
        )

        config = (
            Int8WeightOnlyConfig()
            if torchao_config == "int8wo"
            else Float8WeightOnlyConfig()
            if torchao_config == "fp8wo"
            else Int4WeightOnlyConfig(
                group_size = 32,
                int4_packing_format = Int4PackingFormat.TILE_PACKED_TO_4D,
                int4_choose_qparams_algorithm = Int4ChooseQParamsAlgorithm.HQQ,
            )
        )
        quantize_(model, config, filter_fn = torchao_utils.proj_filter_conv3d)
        return model
    return _apply_torchao(model, torchao_config, filter_fn)


torchao_utils.apply_torchao_config_to_model = _apply_torchao_config

if __name__ == "__main__":
    runpy.run_module("sglang.launch_server", run_name = "__main__")
