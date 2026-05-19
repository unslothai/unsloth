# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import builtins
import subprocess
import sys
from typing import Any
from unittest import mock

from core.training import worker


def _missing_flash_attn_import():
    real_import = builtins.__import__

    def fake_import(name, globals = None, locals = None, fromlist = (), level = 0):
        if name == "flash_attn":
            raise ImportError
        return real_import(name, globals, locals, fromlist, level)

    return fake_import


def _missing_module_import(missing: str):
    real_import = builtins.__import__

    def fake_import(name, globals = None, locals = None, fromlist = (), level = 0):
        if name == missing:
            raise ImportError
        return real_import(name, globals, locals, fromlist, level)

    return fake_import


def test_should_try_runtime_flash_attn_install_threshold_and_skip(monkeypatch):
    monkeypatch.delenv(worker._FLASH_ATTN_SKIP_ENV, raising = False)
    assert worker._should_try_runtime_flash_attn_install(16384) is False
    assert worker._should_try_runtime_flash_attn_install(
        16385
    ) is sys.platform.startswith("linux")

    monkeypatch.setenv(worker._FLASH_ATTN_SKIP_ENV, "1")
    assert worker._should_try_runtime_flash_attn_install(16385) is False


def test_runtime_flash_attn_uses_shared_optional_kernel_installer(monkeypatch):
    statuses: list[str] = []
    calls: list[tuple[Any, dict[str, Any]]] = []

    monkeypatch.delenv(worker._FLASH_ATTN_SKIP_ENV, raising = False)
    monkeypatch.setattr(worker, "has_blackwell_gpu", lambda: False)
    monkeypatch.setattr(
        worker,
        "_send_status",
        lambda queue, message: statuses.append(message),
    )

    def fake_install(spec, **kwargs):
        calls.append((spec, kwargs))
        kwargs["status"]("Installing prebuilt flash-attn wheel...")
        return True

    monkeypatch.setattr(worker, "install_optional_kernel", fake_install)

    worker._ensure_flash_attn_for_long_context(event_queue = [], max_seq_length = 16385)

    assert len(calls) == 1
    spec, kwargs = calls[0]
    assert spec is worker.FLASH_ATTN_SPEC
    assert kwargs["python_executable"] == sys.executable
    assert kwargs["use_uv"] is True
    assert kwargs["allow_pypi_fallback"] is True
    assert kwargs["run"] is worker._sp.run
    assert statuses == ["Installing prebuilt flash-attn wheel..."]


def test_runtime_flash_attn_reports_shared_installer_failure(monkeypatch):
    statuses: list[str] = []

    monkeypatch.delenv(worker._FLASH_ATTN_SKIP_ENV, raising = False)
    monkeypatch.setattr(worker, "has_blackwell_gpu", lambda: False)
    monkeypatch.setattr(
        worker,
        "_send_status",
        lambda queue, message: statuses.append(message),
    )
    monkeypatch.setattr(
        worker, "install_optional_kernel", mock.Mock(return_value = False)
    )

    worker._ensure_flash_attn_for_long_context(event_queue = [], max_seq_length = 16385)

    assert statuses == ["Continuing without flash-attn"]


def test_runtime_flash_attn_skip_env_avoids_all_install_work(monkeypatch):
    monkeypatch.setenv(worker._FLASH_ATTN_SKIP_ENV, "1")
    install_mock = mock.Mock()
    monkeypatch.setattr(worker, "install_optional_kernel", install_mock)

    worker._ensure_flash_attn_for_long_context(event_queue = [], max_seq_length = 16385)

    install_mock.assert_not_called()


def test_runtime_flash_attn_skips_on_blackwell(monkeypatch):
    statuses: list[str] = []
    install_mock = mock.Mock()

    monkeypatch.delenv(worker._FLASH_ATTN_SKIP_ENV, raising = False)
    monkeypatch.setattr(
        worker, "_should_try_runtime_flash_attn_install", lambda max_seq: True
    )
    monkeypatch.setattr(worker, "has_blackwell_gpu", lambda: True)
    monkeypatch.setattr(worker, "install_optional_kernel", install_mock)
    monkeypatch.setattr(
        worker,
        "_send_status",
        lambda queue, message: statuses.append(message),
    )

    worker._ensure_flash_attn_for_long_context(event_queue = [], max_seq_length = 65536)

    install_mock.assert_not_called()
    assert len(statuses) == 1
    assert "Blackwell" in statuses[0]


def test_causal_conv1d_fast_path_preserves_wheel_first_install_args(monkeypatch):
    install_mock = mock.Mock(return_value = True)
    monkeypatch.setattr(worker, "_install_package_wheel_first", install_mock)

    worker._ensure_causal_conv1d_fast_path(
        event_queue = [],
        model_name = "tiiuae/Falcon-H1-0.5B-Instruct",
    )

    install_mock.assert_called_once_with(
        event_queue = [],
        import_name = "causal_conv1d",
        display_name = "causal-conv1d",
        pypi_name = "causal-conv1d",
        pypi_version = worker._CAUSAL_CONV1D_PACKAGE_VERSION,
        filename_prefix = "causal_conv1d",
        release_tag = worker._CAUSAL_CONV1D_RELEASE_TAG,
        release_base_url = "https://github.com/Dao-AILab/causal-conv1d/releases/download",
    )


def test_causal_conv1d_fast_path_includes_qwen3_6_variants(monkeypatch):
    install_mock = mock.Mock(return_value = True)
    monkeypatch.setattr(worker, "_install_package_wheel_first", install_mock)

    worker._ensure_causal_conv1d_fast_path(
        event_queue = [],
        model_name = "unsloth/Qwen3.6-4B",
    )
    worker._ensure_causal_conv1d_fast_path(
        event_queue = [],
        model_name = "unsloth/Qwen3_6-4B",
    )

    assert install_mock.call_count == 2


def test_mamba_ssm_path_preserves_wheel_first_install_args(monkeypatch):
    install_mock = mock.Mock(return_value = True)
    monkeypatch.setattr(worker, "_install_package_wheel_first", install_mock)

    worker._ensure_mamba_ssm(
        event_queue = [],
        model_name = "tiiuae/Falcon-H1-0.5B-Instruct",
    )

    install_mock.assert_called_once_with(
        event_queue = [],
        import_name = "mamba_ssm",
        display_name = "mamba-ssm",
        pypi_name = "mamba-ssm",
        pypi_version = worker._MAMBA_SSM_PACKAGE_VERSION,
        filename_prefix = "mamba_ssm",
        release_tag = worker._MAMBA_SSM_RELEASE_TAG,
        release_base_url = "https://github.com/state-spaces/mamba/releases/download",
    )


def _force_missing_fla_imports(monkeypatch):
    """Make fla.modules / fla.ops.gated_delta_rule imports raise ImportError."""
    real_import = builtins.__import__

    def fake_import(name, *a, **kw):
        if name.startswith("fla.modules") or name.startswith("fla.ops"):
            raise ImportError
        return real_import(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", fake_import)


def test_flash_linear_attention_installs_pinned_pair_for_qwen3_5(monkeypatch):
    monkeypatch.setattr(worker.shutil, "which", lambda name: "/usr/bin/uv")
    run_mock = mock.Mock(return_value = mock.Mock(returncode = 0, stdout = ""))
    monkeypatch.setattr(worker._sp, "run", run_mock)
    _force_missing_fla_imports(monkeypatch)
    statuses: list[str] = []
    monkeypatch.setattr(worker, "_send_status", lambda queue, msg: statuses.append(msg))

    worker._ensure_flash_linear_attention(
        event_queue = [],
        model_name = "unsloth/Qwen3.5-2B",
    )

    run_mock.assert_called_once()
    args = run_mock.call_args[0][0]
    assert f"flash-linear-attention=={worker._FLA_PACKAGE_VERSION}" in args
    assert f"fla-core=={worker._FLA_CORE_PACKAGE_VERSION}" in args
    assert "--no-deps" in args
    assert run_mock.call_args.kwargs["timeout"] == worker._TILELANG_INSTALL_TIMEOUT_S
    assert any("flash-linear-attention" in s for s in statuses)


def test_flash_linear_attention_not_skipped_on_blackwell(monkeypatch):
    monkeypatch.setattr(worker.shutil, "which", lambda name: "/usr/bin/uv")
    monkeypatch.setattr(worker, "_model_wants_tilelang", lambda model_name: True)
    monkeypatch.setattr(worker, "_installed_torch_version_tuple", lambda: (2, 9))
    monkeypatch.setattr(
        worker,
        "has_blackwell_gpu",
        mock.Mock(side_effect = AssertionError("FLA must not use Blackwell gate")),
    )
    run_mock = mock.Mock(return_value = mock.Mock(returncode = 0, stdout = ""))
    monkeypatch.setattr(worker._sp, "run", run_mock)
    _force_missing_fla_imports(monkeypatch)
    monkeypatch.setattr(worker, "_send_status", lambda *a, **k: None)

    worker._ensure_flash_linear_attention(
        event_queue = [],
        model_name = "unsloth/Qwen3.5-2B",
    )

    run_mock.assert_called_once()


def test_flash_linear_attention_skips_for_unrelated_models(monkeypatch):
    run_mock = mock.Mock(return_value = mock.Mock(returncode = 0, stdout = ""))
    monkeypatch.setattr(worker._sp, "run", run_mock)

    worker._ensure_flash_linear_attention(
        event_queue = [],
        model_name = "meta-llama/Llama-3.2-1B-Instruct",
    )

    run_mock.assert_not_called()


def test_flash_linear_attention_skips_for_ssm_only_models(monkeypatch):
    # Nemotron-H / Falcon-H1 / Granite-H / LFM2 take the mamba_ssm path
    # and never call FLA's gated_delta_rule kernels.
    run_mock = mock.Mock(return_value = mock.Mock(returncode = 0, stdout = ""))
    monkeypatch.setattr(worker._sp, "run", run_mock)

    for name in (
        "tiiuae/Falcon-H1-0.5B-Instruct",
        "nvidia/Nemotron-H-8B-Base",
        "ibm-granite/granite-4.0-h-tiny",
        "LiquidAI/LFM2-1.2B-Instruct",
    ):
        worker._ensure_flash_linear_attention(event_queue = [], model_name = name)

    run_mock.assert_not_called()


def test_flash_linear_attention_matches_full_qwen3_family(monkeypatch):
    monkeypatch.setattr(worker.shutil, "which", lambda name: "/usr/bin/uv")
    run_mock = mock.Mock(return_value = mock.Mock(returncode = 0, stdout = ""))
    monkeypatch.setattr(worker._sp, "run", run_mock)
    _force_missing_fla_imports(monkeypatch)
    monkeypatch.setattr(worker, "_send_status", lambda *a, **k: None)
    # Hermetic discovery: pretend installed transformers ships all the Qwen GDN families.
    monkeypatch.setattr(
        worker,
        "_discover_fla_model_types",
        lambda: frozenset({"qwen3_5", "qwen3_5_moe", "qwen3_6", "qwen3_next"}),
    )

    for name in (
        "unsloth/Qwen3.5-2B",
        "unsloth/Qwen3_5-MoE-A22B",
        "unsloth/Qwen3.6-4B",
        "unsloth/Qwen3_6-4B",
        "unsloth/Qwen3-Next-80B-A3B",
        "unsloth/Qwen3_Next-80B-A3B",
    ):
        worker._ensure_flash_linear_attention(event_queue = [], model_name = name)

    assert run_mock.call_count == 6


def test_flash_linear_attention_skipped_below_python_3_10(monkeypatch):
    # sys.version_info is a structseq, not constructible; substitute a
    # plain tuple so the `< _FLA_MIN_PYTHON` comparison still works.
    monkeypatch.setattr(worker.sys, "version_info", (3, 9, 0, "final", 0))
    run_mock = mock.Mock(return_value = mock.Mock(returncode = 0, stdout = ""))
    monkeypatch.setattr(worker._sp, "run", run_mock)

    worker._ensure_flash_linear_attention(
        event_queue = [],
        model_name = "unsloth/Qwen3.5-2B",
    )

    run_mock.assert_not_called()


def test_flash_linear_attention_skipped_via_env(monkeypatch):
    monkeypatch.setenv(worker._FLA_SKIP_ENV, "1")
    run_mock = mock.Mock(return_value = mock.Mock(returncode = 0, stdout = ""))
    monkeypatch.setattr(worker._sp, "run", run_mock)

    worker._ensure_flash_linear_attention(
        event_queue = [],
        model_name = "unsloth/Qwen3.5-2B",
    )

    run_mock.assert_not_called()


def test_flash_linear_attention_skipped_below_torch_2_7(monkeypatch):
    monkeypatch.delenv(worker._FLA_SKIP_ENV, raising = False)
    monkeypatch.setattr(worker, "_installed_torch_version_tuple", lambda: (2, 5))
    run_mock = mock.Mock(return_value = mock.Mock(returncode = 0, stdout = ""))
    monkeypatch.setattr(worker._sp, "run", run_mock)
    statuses: list[str] = []
    monkeypatch.setattr(worker, "_send_status", lambda queue, msg: statuses.append(msg))

    worker._ensure_flash_linear_attention(
        event_queue = [],
        model_name = "unsloth/Qwen3.5-2B",
    )

    run_mock.assert_not_called()
    assert any("torch>=" in s for s in statuses)


def test_flash_linear_attention_install_includes_einops(monkeypatch):
    monkeypatch.delenv(worker._FLA_SKIP_ENV, raising = False)
    monkeypatch.setattr(worker.shutil, "which", lambda name: "/usr/bin/uv")
    monkeypatch.setattr(worker, "_installed_torch_version_tuple", lambda: (2, 9))
    monkeypatch.setattr(worker, "_flash_linear_attention_importable", lambda: False)
    run_mock = mock.Mock(return_value = mock.Mock(returncode = 0, stdout = ""))
    monkeypatch.setattr(worker._sp, "run", run_mock)
    monkeypatch.setattr(worker, "_send_status", lambda *a, **k: None)

    worker._ensure_flash_linear_attention(
        event_queue = [],
        model_name = "unsloth/Qwen3.5-2B",
    )

    args = run_mock.call_args[0][0]
    assert "--no-deps" in args
    # einops is declared by fla-core; packaging and triton are pulled in
    # because fla/utils.py imports them at module load but neither is
    # declared in fla-core's METADATA (an upstream FLA gap).
    assert "einops" in args
    assert "packaging" in args
    assert "triton" in args
    assert f"flash-linear-attention=={worker._FLA_PACKAGE_VERSION}" in args
    assert f"fla-core=={worker._FLA_CORE_PACKAGE_VERSION}" in args


def test_flash_linear_attention_logs_post_install_import_failure(monkeypatch):
    """pip exits 0 but `import fla.modules` still fails (missing transitive)."""
    monkeypatch.delenv(worker._FLA_SKIP_ENV, raising = False)
    monkeypatch.setattr(worker.shutil, "which", lambda name: "/usr/bin/uv")
    monkeypatch.setattr(worker, "_installed_torch_version_tuple", lambda: (2, 9))
    import_calls = {"count": 0}

    def fake_importable():
        import_calls["count"] += 1
        # First call (pre-install probe) -> False so we attempt install.
        # Second call (post-install verify) -> still False.
        return False

    monkeypatch.setattr(worker, "_flash_linear_attention_importable", fake_importable)
    run_mock = mock.Mock(return_value = mock.Mock(returncode = 0, stdout = ""))
    monkeypatch.setattr(worker._sp, "run", run_mock)
    statuses: list[str] = []
    monkeypatch.setattr(worker, "_send_status", lambda queue, msg: statuses.append(msg))

    worker._ensure_flash_linear_attention(
        event_queue = [],
        model_name = "unsloth/Qwen3.5-2B",
    )

    assert import_calls["count"] == 2
    assert any("not importable" in s for s in statuses)


def test_tilelang_backend_skipped_on_unsupported_linux_arch(monkeypatch):
    monkeypatch.delenv(worker._TILELANG_SKIP_ENV, raising = False)
    monkeypatch.setattr(worker.sys, "platform", "linux")
    import platform as _platform

    monkeypatch.setattr(_platform, "machine", lambda: "ppc64le")
    run_mock = mock.Mock(return_value = mock.Mock(returncode = 0, stdout = ""))
    monkeypatch.setattr(worker._sp, "run", run_mock)

    worker._ensure_tilelang_backend(
        event_queue = [],
        model_name = "unsloth/Qwen3.5-2B",
    )

    run_mock.assert_not_called()


def test_tilelang_backend_pins_only_binary(monkeypatch):
    monkeypatch.delenv(worker._TILELANG_SKIP_ENV, raising = False)
    monkeypatch.setattr(worker.shutil, "which", lambda name: "/usr/bin/uv")
    monkeypatch.setattr(worker, "_installed_tvm_ffi_version", lambda: None)
    monkeypatch.setattr(worker, "_tilelang_importable", lambda: False)
    run_mock = mock.Mock(return_value = mock.Mock(returncode = 0, stdout = ""))
    monkeypatch.setattr(worker._sp, "run", run_mock)
    monkeypatch.setattr(worker, "_send_status", lambda *a, **k: None)
    # Need to bypass the post-install probe too.
    probe_calls = {"count": 0}

    def fake_probe():
        probe_calls["count"] += 1
        # First probe (pre-install): False so install runs.
        # Second probe (post-install): True so success branch taken.
        return probe_calls["count"] > 1

    monkeypatch.setattr(worker, "_tilelang_importable", fake_probe)

    worker._ensure_tilelang_backend(
        event_queue = [],
        model_name = "unsloth/Qwen3.5-2B",
    )

    args = run_mock.call_args[0][0]
    assert "--only-binary=:all:" in args
    assert "--no-deps" not in args


def _force_missing_tilelang_imports(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *a, **kw):
        if name in ("tilelang", "tvm_ffi"):
            raise ImportError
        return real_import(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", fake_import)


def test_tilelang_backend_installs_pinned_pair_for_qwen3_5(monkeypatch):
    monkeypatch.delenv(worker._TILELANG_SKIP_ENV, raising = False)
    monkeypatch.setattr(worker.shutil, "which", lambda name: "/usr/bin/uv")
    monkeypatch.setattr(worker, "_installed_tvm_ffi_version", lambda: None)
    run_mock = mock.Mock(return_value = mock.Mock(returncode = 0, stdout = ""))
    monkeypatch.setattr(worker._sp, "run", run_mock)
    _force_missing_tilelang_imports(monkeypatch)
    statuses: list[str] = []
    monkeypatch.setattr(worker, "_send_status", lambda queue, msg: statuses.append(msg))

    worker._ensure_tilelang_backend(
        event_queue = [],
        model_name = "unsloth/Qwen3.5-2B",
    )

    run_mock.assert_called_once()
    args = run_mock.call_args[0][0]
    assert f"apache-tvm-ffi=={worker._APACHE_TVM_FFI_PACKAGE_VERSION}" in args
    assert f"tilelang=={worker._TILELANG_PACKAGE_VERSION}" in args
    assert run_mock.call_args.kwargs["timeout"] == worker._TILELANG_INSTALL_TIMEOUT_S
    assert any("Installing TileLang" in s for s in statuses)


def test_tilelang_backend_reinstalls_when_tvm_ffi_is_broken(monkeypatch):
    """Repair path issues TWO pip calls:

    Call 1 (repair): `--force-reinstall --no-deps apache-tvm-ffi==0.1.9`
      — surgically downgrades the broken package only. `--no-deps` here
      is REQUIRED to prevent --force-reinstall from cascading through
      apache-tvm-ffi's dep graph and replacing torch / the CUDA stack.

    Call 2 (install): plain `apache-tvm-ffi==0.1.9 tilelang==0.1.8`
      — resolves missing transitive deps (z3-solver, ml-dtypes) without
      --force-reinstall, so it never replaces already-correct packages.
    """
    monkeypatch.delenv(worker._TILELANG_SKIP_ENV, raising = False)
    monkeypatch.setattr(worker.shutil, "which", lambda name: "/usr/bin/uv")
    monkeypatch.setattr(worker, "_installed_tvm_ffi_version", lambda: "0.1.11")
    run_mock = mock.Mock(return_value = mock.Mock(returncode = 0, stdout = ""))
    monkeypatch.setattr(worker._sp, "run", run_mock)
    monkeypatch.setattr(worker, "_send_status", lambda *a, **k: None)

    worker._ensure_tilelang_backend(
        event_queue = [],
        model_name = "unsloth/Qwen3.5-2B",
    )

    assert run_mock.call_count == 2
    repair_args, install_args = (call[0][0] for call in run_mock.call_args_list)

    # Repair: --force-reinstall --no-deps, apache-tvm-ffi ONLY (no tilelang).
    assert "--force-reinstall" in repair_args
    assert (
        "--no-deps" in repair_args
    ), "Repair MUST use --no-deps to avoid replacing torch / CUDA"
    assert "--only-binary=:all:" in repair_args
    assert f"apache-tvm-ffi=={worker._APACHE_TVM_FFI_PACKAGE_VERSION}" in repair_args
    assert all(
        "tilelang" not in a for a in repair_args
    ), "Repair MUST only touch apache-tvm-ffi"

    # Install: regular dep-resolving install, NO --force-reinstall.
    assert "--force-reinstall" not in install_args
    assert "--no-deps" not in install_args
    assert "--only-binary=:all:" in install_args
    assert f"apache-tvm-ffi=={worker._APACHE_TVM_FFI_PACKAGE_VERSION}" in install_args
    assert f"tilelang=={worker._TILELANG_PACKAGE_VERSION}" in install_args


def test_tilelang_backend_skipped_below_python_3_10(monkeypatch):
    monkeypatch.delenv(worker._TILELANG_SKIP_ENV, raising = False)
    # sys.version_info is a structseq, not constructible; substitute a
    # plain tuple so the `< _FLA_MIN_PYTHON` comparison still works.
    monkeypatch.setattr(worker.sys, "version_info", (3, 9, 0, "final", 0))
    run_mock = mock.Mock(return_value = mock.Mock(returncode = 0, stdout = ""))
    monkeypatch.setattr(worker._sp, "run", run_mock)

    worker._ensure_tilelang_backend(
        event_queue = [],
        model_name = "unsloth/Qwen3.5-2B",
    )

    run_mock.assert_not_called()


def test_tilelang_backend_skipped_on_windows(monkeypatch):
    monkeypatch.delenv(worker._TILELANG_SKIP_ENV, raising = False)
    monkeypatch.setattr(worker.sys, "platform", "win32")
    run_mock = mock.Mock(return_value = mock.Mock(returncode = 0, stdout = ""))
    monkeypatch.setattr(worker._sp, "run", run_mock)

    worker._ensure_tilelang_backend(
        event_queue = [],
        model_name = "unsloth/Qwen3.5-2B",
    )

    run_mock.assert_not_called()


def test_tilelang_backend_swallows_install_timeout(monkeypatch):
    monkeypatch.delenv(worker._TILELANG_SKIP_ENV, raising = False)
    monkeypatch.setattr(worker.shutil, "which", lambda name: "/usr/bin/uv")
    monkeypatch.setattr(worker, "_installed_tvm_ffi_version", lambda: None)
    _force_missing_tilelang_imports(monkeypatch)

    def raise_timeout(*a, **kw):
        raise subprocess.TimeoutExpired(cmd = "pip", timeout = 1)

    monkeypatch.setattr(worker._sp, "run", raise_timeout)
    statuses: list[str] = []
    monkeypatch.setattr(worker, "_send_status", lambda queue, msg: statuses.append(msg))

    # Should not raise.
    worker._ensure_tilelang_backend(
        event_queue = [],
        model_name = "unsloth/Qwen3.5-2B",
    )

    assert any("timed out" in s.lower() for s in statuses)


def test_tilelang_backend_skipped_for_ssm_models(monkeypatch):
    monkeypatch.delenv(worker._TILELANG_SKIP_ENV, raising = False)
    run_mock = mock.Mock(return_value = mock.Mock(returncode = 0, stdout = ""))
    monkeypatch.setattr(worker._sp, "run", run_mock)

    # Nemotron-H / Falcon-H1 / Granite-H take the mamba_ssm path, not FLA's
    # gated_delta_rule -> tilelang has no effect on them.
    for name in (
        "tiiuae/Falcon-H1-0.5B-Instruct",
        "nvidia/Nemotron-H-8B-Base",
        "ibm-granite/granite-4.0-h-tiny",
        "meta-llama/Llama-3.2-1B-Instruct",
    ):
        worker._ensure_tilelang_backend(event_queue = [], model_name = name)

    run_mock.assert_not_called()


def test_tilelang_backend_skipped_via_env(monkeypatch):
    monkeypatch.setenv(worker._TILELANG_SKIP_ENV, "1")
    run_mock = mock.Mock(return_value = mock.Mock(returncode = 0, stdout = ""))
    monkeypatch.setattr(worker._sp, "run", run_mock)

    worker._ensure_tilelang_backend(
        event_queue = [],
        model_name = "unsloth/Qwen3.5-2B",
    )

    run_mock.assert_not_called()


def test_tilelang_backend_swallows_install_failure(monkeypatch):
    monkeypatch.delenv(worker._TILELANG_SKIP_ENV, raising = False)
    monkeypatch.setattr(worker.shutil, "which", lambda name: None)
    monkeypatch.setattr(worker, "_installed_tvm_ffi_version", lambda: None)
    run_mock = mock.Mock(return_value = mock.Mock(returncode = 1, stdout = "boom"))
    monkeypatch.setattr(worker._sp, "run", run_mock)
    _force_missing_tilelang_imports(monkeypatch)
    statuses: list[str] = []
    monkeypatch.setattr(worker, "_send_status", lambda queue, msg: statuses.append(msg))

    # Should not raise even when pip exits non-zero.
    worker._ensure_tilelang_backend(
        event_queue = [],
        model_name = "unsloth/Qwen3.5-2B",
    )

    run_mock.assert_called_once()
    assert any("failed" in s.lower() for s in statuses)


# ───────────────────────────────────────────────────────────────────
# Runtime hook on `is_flash_linear_attention_available` /
# `is_causal_conv1d_available`. These are the primary gate in
# normal operation; the substring tests above cover the
# UNSLOTH_STUDIO_SKIP_FAST_PATH_HOOKS=1 fallback.
# ───────────────────────────────────────────────────────────────────


class _FakeQueue(list):
    """List with `.put` so worker._send_status can send into it during tests."""

    def put(self, item):
        self.append(item)


def _make_fake_gate(initial_return: bool):
    """Build a callable that mimics transformers' lru_cache-decorated gates.

    Tracks call count and exposes a `cache_clear` attribute. The return
    value can be flipped to mimic install-then-True behaviour by setting
    `.next_return`.
    """

    class Gate:
        def __init__(self, initial: bool) -> None:
            self.next_return = initial
            self.call_count = 0
            self.cache_clear_count = 0

        def __call__(self) -> bool:
            self.call_count += 1
            return self.next_return

        def cache_clear(self) -> None:
            self.cache_clear_count += 1

    return Gate(initial_return)


def _patch_iu_gates(monkeypatch, fla_gate, conv_gate):
    """Drop fake gates onto transformers.utils.import_utils for the test."""
    from transformers.utils import import_utils as _iu

    monkeypatch.setattr(_iu, "is_flash_linear_attention_available", fla_gate)
    monkeypatch.setattr(_iu, "is_causal_conv1d_available", conv_gate)


def test_hook_installs_when_gate_returns_false(monkeypatch):
    fla_gate = _make_fake_gate(initial_return = False)
    conv_gate = _make_fake_gate(initial_return = False)
    _patch_iu_gates(monkeypatch, fla_gate, conv_gate)

    def _fla_install_side_effect(eq):
        fla_gate.next_return = True
        return True

    fla_install = mock.Mock(side_effect = _fla_install_side_effect)
    tile_install = mock.Mock(side_effect = lambda eq: None)

    def _conv_install_side_effect(**kw):
        conv_gate.next_return = True
        return True

    conv_install = mock.Mock(side_effect = _conv_install_side_effect)

    monkeypatch.setattr(
        worker, "_ensure_flash_linear_attention_unconditional", fla_install
    )
    monkeypatch.setattr(worker, "_ensure_tilelang_backend_unconditional", tile_install)
    monkeypatch.setattr(worker, "_install_package_wheel_first", conv_install)
    monkeypatch.delenv(worker._FAST_PATH_HOOKS_SKIP_ENV, raising = False)

    worker._install_fast_path_hooks(
        event_queue = _FakeQueue(), model_name = "unsloth/Qwen3.5-2B"
    )

    from transformers.utils import import_utils as _iu

    # Both gates are now wrapped. Call them — the hook should drive the install.
    assert _iu.is_flash_linear_attention_available() is True
    fla_install.assert_called_once()
    tile_install.assert_called_once()
    assert _iu.is_causal_conv1d_available() is True
    conv_install.assert_called_once()


def test_hook_skips_install_when_gate_already_true(monkeypatch):
    """When both gates are already True AND tilelang is healthy, the hook
    must do zero install work. (Tilelang repair on the already-True path
    is covered by test_hook_runs_tilelang_repair_when_fla_already_true.)
    """
    fla_gate = _make_fake_gate(initial_return = True)
    conv_gate = _make_fake_gate(initial_return = True)
    _patch_iu_gates(monkeypatch, fla_gate, conv_gate)

    fla_install = mock.Mock()
    tile_install = mock.Mock()
    conv_install = mock.Mock()
    monkeypatch.setattr(
        worker, "_ensure_flash_linear_attention_unconditional", fla_install
    )
    monkeypatch.setattr(worker, "_ensure_tilelang_backend_unconditional", tile_install)
    monkeypatch.setattr(worker, "_install_package_wheel_first", conv_install)
    # Tilelang healthy so the post_available path is a no-op (otherwise
    # it would call tile_install, which is correct behaviour but
    # outside the scope of this test).
    monkeypatch.setattr(worker, "_tilelang_importable", lambda: True)
    monkeypatch.setattr(worker, "_installed_tvm_ffi_version", lambda: "0.1.9")
    monkeypatch.delenv(worker._FAST_PATH_HOOKS_SKIP_ENV, raising = False)

    worker._install_fast_path_hooks(
        event_queue = _FakeQueue(), model_name = "unsloth/Qwen3.5-2B"
    )

    from transformers.utils import import_utils as _iu

    assert _iu.is_flash_linear_attention_available() is True
    assert _iu.is_causal_conv1d_available() is True
    fla_install.assert_not_called()
    tile_install.assert_not_called()
    conv_install.assert_not_called()


def test_hook_idempotent_on_repeat_call(monkeypatch):
    fla_gate = _make_fake_gate(initial_return = False)
    conv_gate = _make_fake_gate(initial_return = False)
    _patch_iu_gates(monkeypatch, fla_gate, conv_gate)

    def _fla_install_side_effect(eq):
        fla_gate.next_return = True
        return True

    fla_install = mock.Mock(side_effect = _fla_install_side_effect)
    tile_install = mock.Mock()

    def _conv_install_side_effect(**kw):
        conv_gate.next_return = True
        return True

    conv_install = mock.Mock(side_effect = _conv_install_side_effect)
    monkeypatch.setattr(
        worker, "_ensure_flash_linear_attention_unconditional", fla_install
    )
    monkeypatch.setattr(worker, "_ensure_tilelang_backend_unconditional", tile_install)
    monkeypatch.setattr(worker, "_install_package_wheel_first", conv_install)
    monkeypatch.delenv(worker._FAST_PATH_HOOKS_SKIP_ENV, raising = False)

    worker._install_fast_path_hooks(
        event_queue = _FakeQueue(), model_name = "unsloth/Qwen3.5-2B"
    )

    from transformers.utils import import_utils as _iu

    # First call: hook fires.
    _iu.is_flash_linear_attention_available()
    # Subsequent calls: must not re-trigger the installer.
    _iu.is_flash_linear_attention_available()
    _iu.is_flash_linear_attention_available()
    assert fla_install.call_count == 1
    assert tile_install.call_count == 1


def test_hook_handles_install_failure_gracefully(monkeypatch):
    fla_gate = _make_fake_gate(initial_return = False)
    conv_gate = _make_fake_gate(initial_return = True)  # bypass to focus on FLA
    _patch_iu_gates(monkeypatch, fla_gate, conv_gate)

    def raising_install(eq):
        raise RuntimeError("pip failed to fetch wheel")

    monkeypatch.setattr(
        worker, "_ensure_flash_linear_attention_unconditional", raising_install
    )
    monkeypatch.setattr(
        worker, "_ensure_tilelang_backend_unconditional", lambda eq: None
    )
    monkeypatch.setattr(worker, "_install_package_wheel_first", lambda **kw: None)
    monkeypatch.delenv(worker._FAST_PATH_HOOKS_SKIP_ENV, raising = False)

    worker._install_fast_path_hooks(
        event_queue = _FakeQueue(), model_name = "unsloth/Qwen3.5-2B"
    )

    from transformers.utils import import_utils as _iu

    # Must not raise; returns False so transformers falls back to torch loop.
    assert _iu.is_flash_linear_attention_available() is False


def test_hook_can_be_disabled_via_env(monkeypatch):
    fla_gate = _make_fake_gate(initial_return = False)
    conv_gate = _make_fake_gate(initial_return = False)
    _patch_iu_gates(monkeypatch, fla_gate, conv_gate)

    fla_install = mock.Mock()
    monkeypatch.setattr(
        worker, "_ensure_flash_linear_attention_unconditional", fla_install
    )
    monkeypatch.setenv(worker._FAST_PATH_HOOKS_SKIP_ENV, "1")

    worker._install_fast_path_hooks(
        event_queue = _FakeQueue(), model_name = "unsloth/Qwen3.5-2B"
    )

    from transformers.utils import import_utils as _iu

    # Hook should NOT have been installed; gates remain the fakes.
    assert _iu.is_flash_linear_attention_available is fla_gate
    assert _iu.is_causal_conv1d_available is conv_gate
    fla_install.assert_not_called()


def test_hook_clears_lru_cache_before_first_check(monkeypatch):
    fla_gate = _make_fake_gate(initial_return = True)
    conv_gate = _make_fake_gate(initial_return = True)
    _patch_iu_gates(monkeypatch, fla_gate, conv_gate)

    monkeypatch.setattr(
        worker, "_ensure_flash_linear_attention_unconditional", lambda eq: None
    )
    monkeypatch.setattr(
        worker, "_ensure_tilelang_backend_unconditional", lambda eq: None
    )
    monkeypatch.setattr(worker, "_install_package_wheel_first", lambda **kw: None)
    monkeypatch.delenv(worker._FAST_PATH_HOOKS_SKIP_ENV, raising = False)

    worker._install_fast_path_hooks(
        event_queue = _FakeQueue(), model_name = "unsloth/Qwen3.5-2B"
    )
    from transformers.utils import import_utils as _iu

    _iu.is_flash_linear_attention_available()
    # The wrapper called cache_clear at least once before delegating.
    assert fla_gate.cache_clear_count >= 1


def test_hook_rewrites_previously_imported_module_bindings(monkeypatch):
    """Modeling files bind `is_flash_linear_attention_available` locally
    via `from ... import is_X`. Reassigning the attribute on
    transformers.utils.import_utils alone does NOT reach those local
    bindings. The hook installer sweeps sys.modules and rebinds them.
    """
    fla_gate = _make_fake_gate(initial_return = False)
    conv_gate = _make_fake_gate(initial_return = True)
    _patch_iu_gates(monkeypatch, fla_gate, conv_gate)

    # Create a fake modeling module that did `from ... import is_flash_linear_attention_available`.
    fake_mod = sys.modules.setdefault(
        "_test_fake_modeling_qwen35", type(sys)("_test_fake_modeling_qwen35")
    )
    fake_mod.is_flash_linear_attention_available = fla_gate

    def fake_install(eq):
        fla_gate.next_return = True
        return True

    monkeypatch.setattr(
        worker, "_ensure_flash_linear_attention_unconditional", fake_install
    )
    monkeypatch.setattr(
        worker, "_ensure_tilelang_backend_unconditional", lambda eq: True
    )
    monkeypatch.setattr(worker, "_install_package_wheel_first", lambda **kw: True)
    monkeypatch.delenv(worker._FAST_PATH_HOOKS_SKIP_ENV, raising = False)

    worker._install_fast_path_hooks(
        event_queue = _FakeQueue(), model_name = "unsloth/Qwen3.5-2B"
    )

    # The fake module's local binding has been rewritten to the wrapper.
    assert fake_mod.is_flash_linear_attention_available is not fla_gate
    # Calling through the fake module's reference triggers the install.
    assert fake_mod.is_flash_linear_attention_available() is True

    del sys.modules["_test_fake_modeling_qwen35"]


def test_hook_skips_when_import_utils_unavailable(monkeypatch):
    """If transformers.utils.import_utils can't be imported, the hook
    installer must log and return cleanly rather than crash the worker."""
    real_import = builtins.__import__

    def fake_import(name, *a, **kw):
        if name == "transformers.utils" or name == "transformers.utils.import_utils":
            raise ImportError("transformers missing in worker venv")
        return real_import(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.delenv(worker._FAST_PATH_HOOKS_SKIP_ENV, raising = False)

    # Should not raise.
    worker._install_fast_path_hooks(
        event_queue = _FakeQueue(), model_name = "unsloth/Qwen3.5-2B"
    )


def test_substring_fallback_unchanged_when_hook_skipped(monkeypatch):
    """Hook disabled -> legacy gate falls back to auto-discovered model types."""
    install_mock = mock.Mock()
    monkeypatch.setattr(
        worker, "_ensure_flash_linear_attention_unconditional", install_mock
    )
    monkeypatch.setattr(
        worker, "_discover_fla_model_types", lambda: frozenset({"qwen3_5"})
    )
    monkeypatch.setenv(worker._FAST_PATH_HOOKS_SKIP_ENV, "1")

    worker._ensure_flash_linear_attention(
        event_queue = [], model_name = "unsloth/Qwen3.5-2B"
    )
    assert install_mock.call_count == 1

    worker._ensure_flash_linear_attention(
        event_queue = [], model_name = "meta-llama/Llama-3.1-8B"
    )
    assert install_mock.call_count == 1


# ───────────────────────────────────────────────────────────────────
# Regression tests for the 10-reviewer findings:
#   1. tilelang Qwen-guard on hook path (non-Qwen FLA models)
#   2. tilelang repair must not replace torch / CUDA stack
#   3. hook must trust installer's bool, not transformers metadata
#   4. causal-conv1d must stay eager for SSM models that bypass the gate
#   5. rebind sweep must not invoke lazy module __getattr__
#   6. tilelang skipped when FLA was skipped / failed
#   7. tilelang repair runs when FLA is already True
#   8. older FLA detected as stale and reinstalled
# ───────────────────────────────────────────────────────────────────


def test_hook_does_not_install_tilelang_for_model_outside_allowlist(monkeypatch):
    """A model whose name is not in the auto-discovered FLA allowlist calls
    is_flash_linear_attention_available but should NOT get tilelang."""
    fla_gate = _make_fake_gate(initial_return = False)
    conv_gate = _make_fake_gate(initial_return = True)
    _patch_iu_gates(monkeypatch, fla_gate, conv_gate)

    def _fla_install(eq):
        fla_gate.next_return = True
        return True

    fla_install = mock.Mock(side_effect = _fla_install)
    tile_install = mock.Mock(return_value = True)
    monkeypatch.setattr(
        worker, "_ensure_flash_linear_attention_unconditional", fla_install
    )
    monkeypatch.setattr(worker, "_ensure_tilelang_backend_unconditional", tile_install)
    monkeypatch.setattr(
        worker, "_install_package_wheel_first", mock.Mock(return_value = True)
    )
    monkeypatch.delenv(worker._FAST_PATH_HOOKS_SKIP_ENV, raising = False)
    # Hermetize the auto-discovered set so the test stays valid as new
    # transformers releases add FLA-using model_types (eg olmo_hybrid in
    # 5.4.0). The semantic under test is "outside-allowlist -> no tilelang".
    monkeypatch.setattr(
        worker,
        "_discover_fla_model_types",
        lambda: frozenset({"qwen3_5", "qwen3_5_moe", "qwen3_next"}),
    )

    worker._install_fast_path_hooks(
        event_queue = _FakeQueue(),
        model_name = "fake-org/Fictional-FLA-Only-Model-7B",
    )

    from transformers.utils import import_utils as _iu

    assert _iu.is_flash_linear_attention_available() is True
    fla_install.assert_called_once()
    tile_install.assert_not_called()


def test_hook_does_install_tilelang_for_qwen35(monkeypatch):
    """Positive control for finding #1: Qwen3.5 still gets tilelang."""
    fla_gate = _make_fake_gate(initial_return = False)
    conv_gate = _make_fake_gate(initial_return = True)
    _patch_iu_gates(monkeypatch, fla_gate, conv_gate)

    def _fla_install(eq):
        fla_gate.next_return = True
        return True

    fla_install = mock.Mock(side_effect = _fla_install)
    tile_install = mock.Mock(return_value = True)
    monkeypatch.setattr(
        worker, "_ensure_flash_linear_attention_unconditional", fla_install
    )
    monkeypatch.setattr(worker, "_ensure_tilelang_backend_unconditional", tile_install)
    monkeypatch.setattr(
        worker, "_install_package_wheel_first", mock.Mock(return_value = True)
    )
    monkeypatch.delenv(worker._FAST_PATH_HOOKS_SKIP_ENV, raising = False)

    worker._install_fast_path_hooks(
        event_queue = _FakeQueue(), model_name = "unsloth/Qwen3.5-2B"
    )

    from transformers.utils import import_utils as _iu

    _iu.is_flash_linear_attention_available()
    fla_install.assert_called_once()
    tile_install.assert_called_once()


def test_tilelang_repair_does_not_touch_torch_cuda_stack(monkeypatch):
    """Finding #2: the broken-tvm-ffi repair must use --no-deps on the
    forced step so --force-reinstall does not cascade through
    apache-tvm-ffi's dep graph and pull a different torch wheel.
    """
    monkeypatch.delenv(worker._TILELANG_SKIP_ENV, raising = False)
    monkeypatch.setattr(worker.shutil, "which", lambda name: "/usr/bin/uv")
    monkeypatch.setattr(worker, "_installed_tvm_ffi_version", lambda: "0.1.10")
    run_mock = mock.Mock(return_value = mock.Mock(returncode = 0, stdout = ""))
    monkeypatch.setattr(worker._sp, "run", run_mock)
    monkeypatch.setattr(worker, "_send_status", lambda *a, **k: None)

    worker._ensure_tilelang_backend(event_queue = [], model_name = "unsloth/Qwen3.5-2B")

    assert run_mock.call_count == 2
    repair_args = run_mock.call_args_list[0][0][0]
    # The forced step MUST be --no-deps so torch / CUDA stack is untouched.
    assert "--force-reinstall" in repair_args and "--no-deps" in repair_args
    # And it touches ONLY apache-tvm-ffi, not tilelang / torch.
    assert all("tilelang" not in a for a in repair_args)
    assert all("torch" not in a for a in repair_args)


def test_hook_trusts_installer_bool_not_metadata(monkeypatch):
    """Finding #3: if pip exits 0 but deep imports fail, the installer
    returns False; the hook must propagate False even if the underlying
    `original()` gate (which only checks metadata) returns True after
    pip succeeds.

    Setup mirrors the real bug:
      1. Pre-install: gate=False (FLA not present) → wrapper triggers install.
      2. Installer's `_flash_linear_attention_importable` post-probe fails,
         so the installer returns False. (pip exited 0 but `import fla.modules`
         raised because of a missing transitive dep.)
      3. Post-install: gate would return True (metadata check sees fla-core
         version) — but the wrapper must IGNORE that and use the installer's
         False so transformers takes the torch fallback.
    """
    # Gate flips True after install (simulating "metadata sees fla").
    fla_gate = _make_fake_gate(initial_return = False)
    conv_gate = _make_fake_gate(initial_return = True)
    _patch_iu_gates(monkeypatch, fla_gate, conv_gate)

    # Installer "succeeds" at pip, AND flips the gate to True (metadata
    # sees fla post-install), BUT returns False (deep import broken).
    def _bad_install(eq):
        fla_gate.next_return = True  # metadata says yes after pip
        return False  # but deep import is broken

    fake_fla_install = mock.Mock(side_effect = _bad_install)
    monkeypatch.setattr(
        worker, "_ensure_flash_linear_attention_unconditional", fake_fla_install
    )
    monkeypatch.setattr(
        worker, "_ensure_tilelang_backend_unconditional", mock.Mock(return_value = True)
    )
    monkeypatch.setattr(
        worker, "_install_package_wheel_first", mock.Mock(return_value = True)
    )
    monkeypatch.delenv(worker._FAST_PATH_HOOKS_SKIP_ENV, raising = False)

    worker._install_fast_path_hooks(
        event_queue = _FakeQueue(), model_name = "unsloth/Qwen3.5-2B"
    )

    from transformers.utils import import_utils as _iu

    # Hook MUST return False (installer's verdict), not True (metadata lies).
    assert _iu.is_flash_linear_attention_available() is False
    fake_fla_install.assert_called_once()


def test_rebind_does_not_trigger_module_getattr(monkeypatch):
    """Finding #5: the rebind sweep must use __dict__, not getattr(),
    to avoid invoking transformers' lazy module __getattr__ which spits
    out hundreds of "Accessing X from .models..." warnings.
    """
    original = object()
    replacement = object()

    class _GetattrTripwire(type(sys)):
        getattr_called = False

        def __getattr__(self, name):
            type(self).getattr_called = True
            raise AttributeError(name)

    lazy = _GetattrTripwire("_lazy_test_module")
    sys.modules["_lazy_test_module"] = lazy
    try:
        # No module-level binding to `is_flash_linear_attention_available`
        # in __dict__, so the sweep must NOT trip the tripwire.
        worker._rebind_in_already_imported_modules(
            attr_name = "is_flash_linear_attention_available",
            old_obj = original,
            new_obj = replacement,
        )
        assert (
            not _GetattrTripwire.getattr_called
        ), "Rebind sweep invoked __getattr__ — should use __dict__ probe"
    finally:
        sys.modules.pop("_lazy_test_module", None)


def test_hook_skips_tilelang_when_fla_install_is_skipped(monkeypatch):
    """Finding #6: env-skipped FLA returns False from
    _ensure_flash_linear_attention_unconditional; tilelang must NOT
    install in that case.
    """
    fla_gate = _make_fake_gate(initial_return = False)
    conv_gate = _make_fake_gate(initial_return = True)
    _patch_iu_gates(monkeypatch, fla_gate, conv_gate)

    monkeypatch.setenv(worker._FLA_SKIP_ENV, "1")
    tile_install = mock.Mock(return_value = True)
    monkeypatch.setattr(worker, "_ensure_tilelang_backend_unconditional", tile_install)
    monkeypatch.setattr(
        worker, "_install_package_wheel_first", mock.Mock(return_value = True)
    )
    monkeypatch.delenv(worker._FAST_PATH_HOOKS_SKIP_ENV, raising = False)

    worker._install_fast_path_hooks(
        event_queue = _FakeQueue(), model_name = "unsloth/Qwen3.5-2B"
    )

    from transformers.utils import import_utils as _iu

    # FLA gate stays False (env-skipped, install never ran).
    assert _iu.is_flash_linear_attention_available() is False
    tile_install.assert_not_called()


def test_hook_runs_tilelang_repair_when_fla_already_true(monkeypatch):
    """Finding #7: when FLA is already importable (gate returns True at
    first probe) but tilelang is missing or apache-tvm-ffi is on the
    broken list, the post-available action must still run tilelang.
    """
    fla_gate = _make_fake_gate(initial_return = True)
    conv_gate = _make_fake_gate(initial_return = True)
    _patch_iu_gates(monkeypatch, fla_gate, conv_gate)

    fla_install = mock.Mock(return_value = True)
    tile_install = mock.Mock(return_value = True)
    monkeypatch.setattr(
        worker, "_ensure_flash_linear_attention_unconditional", fla_install
    )
    monkeypatch.setattr(worker, "_ensure_tilelang_backend_unconditional", tile_install)
    monkeypatch.setattr(
        worker, "_install_package_wheel_first", mock.Mock(return_value = True)
    )
    # tilelang missing AND tvm-ffi is on broken list — both trigger repair.
    monkeypatch.setattr(worker, "_tilelang_importable", lambda: False)
    monkeypatch.setattr(worker, "_installed_tvm_ffi_version", lambda: "0.1.11")
    monkeypatch.delenv(worker._FAST_PATH_HOOKS_SKIP_ENV, raising = False)

    worker._install_fast_path_hooks(
        event_queue = _FakeQueue(), model_name = "unsloth/Qwen3.5-2B"
    )

    from transformers.utils import import_utils as _iu

    _iu.is_flash_linear_attention_available()
    # FLA install was NOT needed; tilelang repair WAS still triggered.
    fla_install.assert_not_called()
    tile_install.assert_called_once()


def test_fla_installer_force_reinstalls_when_older_version_present(monkeypatch):
    """Finding #8: when an older `flash-linear-attention` is importable
    but below the pin, the installer must force a reinstall (not no-op).
    """
    monkeypatch.delenv(worker._FLA_SKIP_ENV, raising = False)
    monkeypatch.setattr(worker.shutil, "which", lambda name: "/usr/bin/uv")
    monkeypatch.setattr(worker, "_installed_torch_version_tuple", lambda: (2, 9))
    # Importable but stale (current() reports False even though importable() is True).
    monkeypatch.setattr(worker, "_flash_linear_attention_importable", lambda: True)
    monkeypatch.setattr(worker, "_flash_linear_attention_current", lambda **kw: False)
    run_mock = mock.Mock(return_value = mock.Mock(returncode = 0, stdout = ""))
    monkeypatch.setattr(worker._sp, "run", run_mock)
    monkeypatch.setattr(worker, "_send_status", lambda *a, **k: None)

    worker._ensure_flash_linear_attention_unconditional(event_queue = [])

    run_mock.assert_called_once()
    args = run_mock.call_args[0][0]
    assert (
        "--force-reinstall" in args
    ), "Stale FLA must trigger --force-reinstall, otherwise pip is a no-op"
    # --no-deps still applies so torch stays untouched.
    assert "--no-deps" in args


def test_run_training_process_eagerly_installs_causal_conv1d_in_normal_mode():
    """Finding #4: SSM modeling files use `lazy_load_kernel("causal-conv1d")`
    and never call `is_causal_conv1d_available()`, so the hook would not
    fire for them. The orchestrator must always run the eager
    substring installer regardless of hook mode.

    This test reads the worker source rather than running the full
    orchestrator (which requires a configured training config). It
    asserts the eager install is OUTSIDE the if/else hook branch.
    """
    import inspect

    src = inspect.getsource(worker.run_training_process)
    # Find the orchestration block.
    assert "_ensure_causal_conv1d_fast_path(event_queue, model_name)" in src
    assert "_install_fast_path_hooks(event_queue, model_name)" in src
    # The eager causal_conv1d call must appear BEFORE the hook-mode if/else,
    # not nested inside the `if _FAST_PATH_HOOKS_SKIP_ENV` branch.
    eager_pos = src.find("_ensure_causal_conv1d_fast_path(event_queue, model_name)")
    skip_check_pos = src.find('os.getenv(_FAST_PATH_HOOKS_SKIP_ENV) == "1"')
    assert eager_pos < skip_check_pos, (
        "_ensure_causal_conv1d_fast_path must be called BEFORE the hook-mode "
        "branch, so SSM models that bypass is_causal_conv1d_available() still "
        "get the eager install"
    )


# ───────────────────────────────────────────────────────────────────
# HIP / ROCm regression coverage (h34v3nzc0dex Strix Halo report).
# tilelang 0.1.8 has no HIP GEMM backend; FLA's TileLang dispatch
# crashes mid-backward on AMD with "Unsupported target for gemm: hip".
# The fix: skip the install on HIP-built torch AND setdefault
# FLA_TILELANG=0 so already-installed tilelang doesn't get used either.
# ───────────────────────────────────────────────────────────────────


def test_tilelang_platform_unsupported_on_hip_torch(monkeypatch):
    """Strix Halo / MI300 with ROCm torch: linux + x86_64 looks
    identical to a CUDA box at the OS level, so the platform check
    must consult torch.version.hip explicitly.
    """
    monkeypatch.setattr(worker, "_torch_has_hip", lambda: True)
    assert worker._tilelang_platform_supported() is False


def test_tilelang_install_skipped_on_hip_torch(monkeypatch):
    """End-to-end: the unconditional installer must not call pip on HIP torch."""
    monkeypatch.delenv(worker._TILELANG_SKIP_ENV, raising = False)
    monkeypatch.setattr(worker, "_torch_has_hip", lambda: True)
    run_mock = mock.Mock(return_value = mock.Mock(returncode = 0, stdout = ""))
    monkeypatch.setattr(worker._sp, "run", run_mock)
    monkeypatch.setattr(worker, "_send_status", lambda *a, **k: None)

    result = worker._ensure_tilelang_backend_unconditional(event_queue = [])

    assert result is False
    run_mock.assert_not_called()


def test_install_fast_path_hooks_sets_fla_tilelang_zero_on_hip(monkeypatch):
    """When HIP torch is detected, hook installer must set
    FLA_TILELANG=0 (via setdefault — respects user override) so any
    PRE-EXISTING tilelang install isn't used by FLA's dispatcher.
    """
    import os as _os

    monkeypatch.delenv("FLA_TILELANG", raising = False)
    monkeypatch.delenv(worker._FAST_PATH_HOOKS_SKIP_ENV, raising = False)
    monkeypatch.setattr(worker, "_torch_has_hip", lambda: True)
    monkeypatch.setattr(
        worker, "_ensure_flash_linear_attention_unconditional", lambda eq: True
    )
    monkeypatch.setattr(
        worker, "_ensure_tilelang_backend_unconditional", lambda eq: True
    )
    monkeypatch.setattr(worker, "_install_package_wheel_first", lambda **kw: True)

    worker._install_fast_path_hooks(
        event_queue = _FakeQueue(), model_name = "unsloth/Qwen3.5-2B"
    )

    assert _os.environ.get("FLA_TILELANG") == "0"


def test_install_fast_path_hooks_respects_user_fla_tilelang_override(monkeypatch):
    """If the user explicitly set FLA_TILELANG (even on HIP), don't
    overwrite — they may know they have a HIP-aware tilelang fork.
    """
    import os as _os

    monkeypatch.setenv("FLA_TILELANG", "1")
    monkeypatch.delenv(worker._FAST_PATH_HOOKS_SKIP_ENV, raising = False)
    monkeypatch.setattr(worker, "_torch_has_hip", lambda: True)
    monkeypatch.setattr(
        worker, "_ensure_flash_linear_attention_unconditional", lambda eq: True
    )
    monkeypatch.setattr(
        worker, "_ensure_tilelang_backend_unconditional", lambda eq: True
    )
    monkeypatch.setattr(worker, "_install_package_wheel_first", lambda **kw: True)

    worker._install_fast_path_hooks(
        event_queue = _FakeQueue(), model_name = "unsloth/Qwen3.5-2B"
    )

    assert _os.environ["FLA_TILELANG"] == "1"


def test_install_fast_path_hooks_does_not_set_fla_tilelang_on_cuda(monkeypatch):
    """CUDA path must NOT set FLA_TILELANG (tilelang is wanted there)."""
    import os as _os

    monkeypatch.delenv("FLA_TILELANG", raising = False)
    monkeypatch.delenv(worker._FAST_PATH_HOOKS_SKIP_ENV, raising = False)
    monkeypatch.setattr(worker, "_torch_has_hip", lambda: False)
    monkeypatch.setattr(
        worker, "_ensure_flash_linear_attention_unconditional", lambda eq: True
    )
    monkeypatch.setattr(
        worker, "_ensure_tilelang_backend_unconditional", lambda eq: True
    )
    monkeypatch.setattr(worker, "_install_package_wheel_first", lambda **kw: True)

    worker._install_fast_path_hooks(
        event_queue = _FakeQueue(), model_name = "unsloth/Qwen3.5-2B"
    )

    assert _os.environ.get("FLA_TILELANG") is None


# ───────────────────────────────────────────────────────────────────
# Auto-discovery of FLA model_types from the installed transformers
# ───────────────────────────────────────────────────────────────────


def _make_fake_transformers_tree(
    tmp_path, fla_types: list[str], non_fla_types: list[str]
):
    """Lay out a tmp dir as `transformers/models/{type}/modeling_{type}.py`."""
    pkg = tmp_path / "transformers"
    models = pkg / "models"
    models.mkdir(parents = True)
    (pkg / "__init__.py").write_text("")
    for t in fla_types:
        d = models / t
        d.mkdir()
        (d / f"modeling_{t}.py").write_text(
            "from ...utils.import_utils import is_flash_linear_attention_available\n"
            "if is_flash_linear_attention_available():\n"
            "    from fla.modules import FusedRMSNormGated\n"
            "    from fla.ops.gated_delta_rule import chunk_gated_delta_rule\n"
        )
    for t in non_fla_types:
        d = models / t
        d.mkdir()
        (d / f"modeling_{t}.py").write_text("class Foo: pass\n")
    return pkg


def _reset_fla_cache(monkeypatch):
    monkeypatch.setattr(worker, "_TRANSFORMERS_FLA_MODEL_TYPES_CACHE", None)


def test_discover_fla_model_types_returns_only_fla_users(tmp_path, monkeypatch):
    pkg = _make_fake_transformers_tree(
        tmp_path,
        fla_types = ["qwen3_5", "qwen3_5_moe", "qwen3_next"],
        non_fla_types = ["llama", "gpt2", "mistral"],
    )
    fake = mock.MagicMock(__file__ = str(pkg / "__init__.py"))
    monkeypatch.setitem(sys.modules, "transformers", fake)
    _reset_fla_cache(monkeypatch)

    result = worker._discover_fla_model_types()
    assert result == frozenset({"qwen3_5", "qwen3_5_moe", "qwen3_next"})
    assert "llama" not in result
    assert "gpt2" not in result


def test_discover_fla_model_types_caches_across_calls(tmp_path, monkeypatch):
    pkg = _make_fake_transformers_tree(
        tmp_path, fla_types = ["qwen3_5"], non_fla_types = []
    )
    fake = mock.MagicMock(__file__ = str(pkg / "__init__.py"))
    monkeypatch.setitem(sys.modules, "transformers", fake)
    _reset_fla_cache(monkeypatch)

    from pathlib import Path as _Path

    read_calls = [0]
    real_read = _Path.read_text

    def counting_read(self, *a, **kw):
        read_calls[0] += 1
        return real_read(self, *a, **kw)

    monkeypatch.setattr(_Path, "read_text", counting_read)

    first = worker._discover_fla_model_types()
    after_first = read_calls[0]
    second = worker._discover_fla_model_types()

    assert first == second
    assert read_calls[0] == after_first  # cache hit: no extra disk reads


def test_discover_fla_model_types_handles_missing_transformers(monkeypatch):
    _reset_fla_cache(monkeypatch)

    real_import = builtins.__import__

    def fake_import(name, globals = None, locals = None, fromlist = (), level = 0):
        if name == "transformers":
            raise ImportError("transformers not installed")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    result = worker._discover_fla_model_types()
    assert result == frozenset()


def test_discover_fla_model_types_handles_unreadable_file(tmp_path, monkeypatch):
    pkg = _make_fake_transformers_tree(
        tmp_path, fla_types = ["qwen3_5"], non_fla_types = []
    )
    fake = mock.MagicMock(__file__ = str(pkg / "__init__.py"))
    monkeypatch.setitem(sys.modules, "transformers", fake)
    _reset_fla_cache(monkeypatch)

    from pathlib import Path as _Path

    real_read = _Path.read_text

    def boom_read(self, *a, **kw):
        if "modeling_qwen3_5.py" in str(self):
            raise OSError("permission denied")
        return real_read(self, *a, **kw)

    monkeypatch.setattr(_Path, "read_text", boom_read)
    result = worker._discover_fla_model_types()
    assert result == frozenset()  # unreadable file simply doesn't contribute


def test_model_wants_tilelang_handles_real_repo_names(monkeypatch):
    monkeypatch.setattr(
        worker,
        "_discover_fla_model_types",
        lambda: frozenset({"qwen3_5", "qwen3_5_moe", "qwen3_next"}),
    )
    cases = [
        ("unsloth/Qwen3.5-2B", True),
        ("Qwen/Qwen3.5-MoE-A3B", True),
        ("mlx-community/qwen3-next-80b", True),
        ("unsloth/qwen3_5_moe_a3b_lora", True),
        ("meta-llama/Llama-3.1-8B", False),
        ("nvidia/Nemotron-H-4B", False),
        ("mistralai/Mistral-7B-v0.3", False),
        ("", False),
    ]
    for name, expected in cases:
        assert worker._model_wants_tilelang(name) is expected, name


def test_model_wants_tilelang_empty_when_transformers_has_no_fla(monkeypatch):
    monkeypatch.setattr(worker, "_discover_fla_model_types", lambda: frozenset())
    assert worker._model_wants_tilelang("unsloth/Qwen3.5-2B") is False
    assert worker._model_wants_tilelang("meta-llama/Llama-3.1-8B") is False


def test_model_wants_tilelang_normalizes_separators(monkeypatch):
    monkeypatch.setattr(
        worker, "_discover_fla_model_types", lambda: frozenset({"qwen3_next"})
    )
    for variant in (
        "qwen3-next",
        "Qwen3.Next",
        "Qwen/Qwen3 Next",
        "anyone/qwen3_next",
        "qwen3.next-80b",
    ):
        assert worker._model_wants_tilelang(variant) is True, variant


# ────────────────────────────────────────────────────────────────────
# HIP source-build gcc-install-dir coverage (h34v3nzc0dex Strix Halo).
# Ubuntu 24.04 ships gcc-14's runtime dir without /usr/include/c++/14,
# so ROCm clang-20 picks it and fails with 'cstdlib' file not found
# when building causal-conv1d (or any other HIP source fallback).
# _hipcc_gcc_install_dir() finds a gcc dir that has both halves; the
# _install_package_wheel_first HIP branch passes it to clang via
# HIPCC_COMPILE_FLAGS_APPEND. Parallel to bbf004c's setup.sh fix for
# the llama.cpp HIP build (PR #5301).
# ────────────────────────────────────────────────────────────────────


def _isdir_for_layout(*existing: str):
    """Return an os.path.isdir replacement that only treats the given
    absolute paths as directories. Lets a test simulate exactly which
    gcc runtime dirs and C++ header dirs exist on the host."""
    valid = set(existing)

    def fake_isdir(path: str) -> bool:
        return path in valid

    return fake_isdir


def test_hipcc_gcc_install_dir_picks_highest_with_headers(monkeypatch):
    """gcc-14 has runtime but no /usr/include/c++/14; loop falls through
    to gcc-13 which has both. This is the exact Ubuntu 24.04 layout."""
    monkeypatch.setattr(sys, "platform", "linux")
    import platform as _platform

    monkeypatch.setattr(_platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(
        worker.os.path,
        "isdir",
        _isdir_for_layout(
            "/usr/lib/gcc/x86_64-linux-gnu/14/include",  # runtime present
            # but no /usr/include/c++/14 — typical Ubuntu 24.04 default
            "/usr/lib/gcc/x86_64-linux-gnu/13/include",
            "/usr/include/c++/13",  # libstdc++-13-dev installed
        ),
    )
    assert worker._hipcc_gcc_install_dir() == "/usr/lib/gcc/x86_64-linux-gnu/13"


def test_hipcc_gcc_install_dir_picks_14_when_headers_exist(monkeypatch):
    """If the user has libstdc++-14-dev installed, prefer gcc-14."""
    monkeypatch.setattr(sys, "platform", "linux")
    import platform as _platform

    monkeypatch.setattr(_platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(
        worker.os.path,
        "isdir",
        _isdir_for_layout(
            "/usr/lib/gcc/x86_64-linux-gnu/14/include",
            "/usr/include/c++/14",
        ),
    )
    assert worker._hipcc_gcc_install_dir() == "/usr/lib/gcc/x86_64-linux-gnu/14"


def test_hipcc_gcc_install_dir_returns_none_when_no_match(monkeypatch):
    """No gcc dir has both halves → return None and skip the env injection
    rather than guessing wrong and surfacing a confusing build failure."""
    monkeypatch.setattr(sys, "platform", "linux")
    import platform as _platform

    monkeypatch.setattr(_platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(worker.os.path, "isdir", lambda path: False)
    assert worker._hipcc_gcc_install_dir() is None


def test_hipcc_gcc_install_dir_returns_none_on_non_linux(monkeypatch):
    """Don't probe gcc layout on macOS / Windows — early-return."""
    monkeypatch.setattr(sys, "platform", "darwin")

    def _isdir_should_not_be_called(_path):
        raise AssertionError("isdir should not be called on non-Linux")

    monkeypatch.setattr(worker.os.path, "isdir", _isdir_should_not_be_called)
    assert worker._hipcc_gcc_install_dir() is None


def test_hipcc_gcc_install_dir_returns_none_on_non_x86_64(monkeypatch):
    """ROCm clang-20 on aarch64 has a different libstdc++ layout."""
    monkeypatch.setattr(sys, "platform", "linux")
    import platform as _platform

    monkeypatch.setattr(_platform, "machine", lambda: "aarch64")
    assert worker._hipcc_gcc_install_dir() is None


def _make_hip_install_env(monkeypatch, *, gcc_dir: str | None):
    """Common scaffolding for tests that exercise the HIP source-build
    branch of _install_package_wheel_first end-to-end. The package isn't
    installed yet, no prebuilt wheel exists, hipcc is on PATH, and the
    fake env reports an HIP torch."""
    monkeypatch.setattr(builtins, "__import__", _missing_module_import("causal_conv1d"))
    monkeypatch.setattr(
        worker,
        "probe_torch_wheel_env",
        lambda timeout = 30: {
            "hip_version": "7.13.26176",
            "python_tag": "cp312",
            "torch_mm": "2.11",
            "cxx11abi": "TRUE",
            "platform_tag": "linux_x86_64",
        },
    )
    monkeypatch.setattr(worker, "direct_wheel_url", lambda **kw: None)
    monkeypatch.setattr(
        worker.shutil,
        "which",
        lambda name: "/opt/rocm/bin/hipcc" if name == "hipcc" else None,
    )
    monkeypatch.setattr(worker, "_send_status", lambda *a, **k: None)
    monkeypatch.setattr(worker, "_hipcc_gcc_install_dir", lambda: gcc_dir)


def test_install_injects_gcc_install_dir_on_hip_source_build(monkeypatch):
    """HIP source-build with no user-set HIPCC_COMPILE_FLAGS_APPEND →
    subprocess env carries --gcc-install-dir=<detected path>."""
    monkeypatch.delenv("HIPCC_COMPILE_FLAGS_APPEND", raising = False)
    _make_hip_install_env(monkeypatch, gcc_dir = "/usr/lib/gcc/x86_64-linux-gnu/13")

    captured: dict[str, str] = {}

    def fake_run(cmd, **kwargs):
        captured.update(kwargs.get("env") or {})
        return subprocess.CompletedProcess(cmd, 0, "")

    monkeypatch.setattr(worker._sp, "run", fake_run)

    worker._install_package_wheel_first(
        event_queue = [],
        import_name = "causal_conv1d",
        display_name = "causal-conv1d",
        pypi_name = "causal-conv1d",
        pypi_version = "1.6.2.post1",
        filename_prefix = "causal_conv1d",
        release_tag = "v1.6.2.post1",
        release_base_url = "https://example.com",
    )

    assert (
        captured.get("HIPCC_COMPILE_FLAGS_APPEND")
        == "--gcc-install-dir=/usr/lib/gcc/x86_64-linux-gnu/13"
    )


def test_install_appends_to_existing_hipcc_compile_flags(monkeypatch):
    """User has HIPCC_COMPILE_FLAGS_APPEND='-O3 -DFOO' set → final value
    keeps the user's flags AND adds --gcc-install-dir at the end."""
    monkeypatch.setenv("HIPCC_COMPILE_FLAGS_APPEND", "-O3 -DFOO")
    _make_hip_install_env(monkeypatch, gcc_dir = "/usr/lib/gcc/x86_64-linux-gnu/13")

    captured: dict[str, str] = {}

    def fake_run(cmd, **kwargs):
        captured.update(kwargs.get("env") or {})
        return subprocess.CompletedProcess(cmd, 0, "")

    monkeypatch.setattr(worker._sp, "run", fake_run)

    worker._install_package_wheel_first(
        event_queue = [],
        import_name = "causal_conv1d",
        display_name = "causal-conv1d",
        pypi_name = "causal-conv1d",
        pypi_version = "1.6.2.post1",
        filename_prefix = "causal_conv1d",
        release_tag = "v1.6.2.post1",
        release_base_url = "https://example.com",
    )

    assert captured.get("HIPCC_COMPILE_FLAGS_APPEND") == (
        "-O3 -DFOO --gcc-install-dir=/usr/lib/gcc/x86_64-linux-gnu/13"
    )


def test_install_respects_user_gcc_install_dir(monkeypatch):
    """User explicitly set --gcc-install-dir=… already → don't touch it.
    Avoids two competing --gcc-install-dir flags on the clang command line."""
    monkeypatch.setenv(
        "HIPCC_COMPILE_FLAGS_APPEND",
        "--gcc-install-dir=/opt/custom/gcc-13",
    )
    _make_hip_install_env(monkeypatch, gcc_dir = "/usr/lib/gcc/x86_64-linux-gnu/13")

    captured: dict[str, str] = {}

    def fake_run(cmd, **kwargs):
        env = kwargs.get("env")
        if env is not None:
            captured.clear()
            captured.update(env)
        return subprocess.CompletedProcess(cmd, 0, "")

    monkeypatch.setattr(worker._sp, "run", fake_run)

    worker._install_package_wheel_first(
        event_queue = [],
        import_name = "causal_conv1d",
        display_name = "causal-conv1d",
        pypi_name = "causal-conv1d",
        pypi_version = "1.6.2.post1",
        filename_prefix = "causal_conv1d",
        release_tag = "v1.6.2.post1",
        release_base_url = "https://example.com",
    )

    # The sanitized child env preserves the user's explicit gcc-install-dir
    # rather than appending a second one.
    assert (
        captured.get("HIPCC_COMPILE_FLAGS_APPEND")
        == "--gcc-install-dir=/opt/custom/gcc-13"
    )


def test_install_does_not_inject_env_on_cuda(monkeypatch):
    """CUDA path (no hip_version in env) → no env override at all."""
    monkeypatch.delenv("HIPCC_COMPILE_FLAGS_APPEND", raising = False)
    monkeypatch.setattr(builtins, "__import__", _missing_module_import("causal_conv1d"))
    monkeypatch.setattr(
        worker,
        "probe_torch_wheel_env",
        lambda timeout = 30: {
            "python_tag": "cp312",
            "torch_mm": "2.11",
            "cuda_major": "12",
            "cxx11abi": "TRUE",
            "platform_tag": "linux_x86_64",
        },
    )
    monkeypatch.setattr(worker, "direct_wheel_url", lambda **kw: None)
    monkeypatch.setattr(worker.shutil, "which", lambda name: None)
    monkeypatch.setattr(worker, "_send_status", lambda *a, **k: None)
    # If _hipcc_gcc_install_dir were called on CUDA we'd want to know.
    monkeypatch.setattr(
        worker,
        "_hipcc_gcc_install_dir",
        lambda: (_ for _ in ()).throw(AssertionError("must not run on CUDA")),
    )

    captured: dict[str, Any] = {}

    def fake_run(cmd, **kwargs):
        captured["env_in_kwargs"] = "env" in kwargs
        return subprocess.CompletedProcess(cmd, 0, "")

    monkeypatch.setattr(worker._sp, "run", fake_run)

    worker._install_package_wheel_first(
        event_queue = [],
        import_name = "causal_conv1d",
        display_name = "causal-conv1d",
        pypi_name = "causal-conv1d",
        pypi_version = "1.6.2.post1",
        filename_prefix = "causal_conv1d",
        release_tag = "v1.6.2.post1",
        release_base_url = "https://example.com",
    )

    # CUDA branch still uses the sanitized child env, but never invokes the gcc helper.
    assert captured.get("env_in_kwargs") is True
