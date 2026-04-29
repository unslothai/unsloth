from unittest.mock import patch


def _common_patches(
    model_path,
    *,
    config_bytes,
    local_bytes,
    safetensors_params = None,
    config = object(),
):
    from utils.hardware import hardware as hardware_module

    return [
        patch.object(
            hardware_module,
            "_resolve_model_identifier_for_gpu_estimate",
            return_value = model_path,
        ),
        patch.object(
            hardware_module,
            "_get_hf_safetensors_total_params",
            return_value = safetensors_params,
        ),
        patch.object(
            hardware_module,
            "_load_config_for_gpu_estimate",
            return_value = config,
        ),
        patch.object(
            hardware_module,
            "_estimate_fp16_model_size_bytes_from_config",
            return_value = config_bytes,
        ),
        patch.object(
            hardware_module,
            "_get_local_weight_size_bytes",
            return_value = local_bytes,
        ),
    ]


def _run(model_path, **kwargs):
    from utils.hardware import hardware as hardware_module
    patches = _common_patches(model_path, **kwargs)
    for p in patches:
        p.start()
    try:
        return hardware_module.estimate_fp16_model_size_bytes(model_path)
    finally:
        for p in patches:
            p.stop()


def test_local_weight_bytes_preferred_when_larger_than_config():
    bytes_, src = _run(
        "/local/vlm",
        config_bytes = 2 * (1 << 30),
        local_bytes = 20 * (1 << 30),
    )
    assert bytes_ == 20 * (1 << 30)
    assert src == "weight_bytes"


def test_config_bytes_preferred_when_larger_than_local():
    bytes_, src = _run(
        "/local/text-only",
        config_bytes = 20 * (1 << 30),
        local_bytes = 2 * (1 << 30),
    )
    assert bytes_ == 20 * (1 << 30)
    assert src == "config"


def test_config_bytes_returned_when_no_local_weights():
    bytes_, src = _run(
        "/local/no-weights",
        config_bytes = 5 * (1 << 30),
        local_bytes = None,
    )
    assert bytes_ == 5 * (1 << 30)
    assert src == "config"


def test_local_bytes_returned_when_config_resolution_fails():
    bytes_, src = _run(
        "/local/no-config",
        config_bytes = None,
        local_bytes = 7 * (1 << 30),
        config = None,
    )
    assert bytes_ == 7 * (1 << 30)
    assert src == "weight_bytes"


def test_equal_local_and_config_keeps_config_label():
    # why: tie-breaker is "local must be strictly larger" so an exact match
    # keeps the config-derived path (cheaper and more structured).
    same = 8 * (1 << 30)
    bytes_, src = _run(
        "/local/equal",
        config_bytes = same,
        local_bytes = same,
    )
    assert bytes_ == same
    assert src == "config"


def test_remote_safetensors_path_unaffected_by_local_weights():
    # When the resolver returns an HF repo id and safetensors metadata is
    # available, the function returns immediately at the safetensors branch
    # without consulting config or local weight bytes.
    from utils.hardware import hardware as hardware_module

    with (
        patch.object(
            hardware_module,
            "_resolve_model_identifier_for_gpu_estimate",
            return_value = "owner/repo",
        ),
        patch.object(
            hardware_module,
            "_get_hf_safetensors_total_params",
            return_value = 1_000_000_000,
        ),
        patch.object(
            hardware_module,
            "_load_config_for_gpu_estimate",
        ) as mock_load,
        patch.object(
            hardware_module,
            "_get_local_weight_size_bytes",
        ) as mock_local,
    ):
        bytes_, src = hardware_module.estimate_fp16_model_size_bytes("owner/repo")
        assert bytes_ == 2 * 1_000_000_000
        assert src == "safetensors"
        mock_load.assert_not_called()
        mock_local.assert_not_called()
