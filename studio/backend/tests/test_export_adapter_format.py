# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Adapter-format export: platform default, conversion routing, GGUF rejects, feature metadata."""

import json
import os
from unittest.mock import MagicMock

import pytest

from core.export import export as export_mod
from utils.models.checkpoints import parse_adapter_features


def _backend(monkeypatch, is_mlx, tmp_path):
    monkeypatch.setattr(export_mod, "_IS_MLX", is_mlx)
    monkeypatch.setattr(export_mod, "_export_runtime_available", lambda: True)
    monkeypatch.setattr(export_mod, "resolve_export_write_dir", lambda p: tmp_path / "out")
    monkeypatch.setattr(export_mod, "ensure_dir", lambda p: os.makedirs(p, exist_ok = True))
    backend = export_mod.ExportBackend.__new__(export_mod.ExportBackend)
    backend.current_model = MagicMock()
    backend.current_tokenizer = MagicMock()
    backend.is_peft = True
    return backend


def _peft_writer(model):
    # The real zoo converter refuses an existing destination and publishes a fresh one.
    def _save(
        path,
        adapter_config = None,
        adapter_format = "mlx",
    ):
        assert not os.path.lexists(path)
        os.makedirs(path)
        with open(os.path.join(path, "adapter_model.safetensors"), "w") as f:
            f.write(adapter_format)
        with open(os.path.join(path, "adapter_config.json"), "w") as f:
            json.dump({"r": 8}, f)

    model.save_lora_adapters = MagicMock(side_effect = _save)


@pytest.mark.parametrize(
    "is_mlx,requested,expect",
    [
        (True, None, "mlx"),
        (True, "mlx", "mlx"),
        (True, "peft", "peft"),
        (False, None, "peft"),
        (False, "peft", "peft"),
        (False, "mlx", "error"),
    ],
)
def test_six_cell_matrix(monkeypatch, tmp_path, is_mlx, requested, expect):
    backend = _backend(monkeypatch, is_mlx, tmp_path)
    if expect == "peft" and is_mlx:
        _peft_writer(backend.current_model)
    ok, message, _path = backend.export_lora_adapter(
        str(tmp_path / "dst"),
        adapter_format = requested,
    )
    if expect == "error":
        assert not ok and "MLX" in message
        backend.current_model.save_pretrained.assert_not_called()
        return
    assert ok, message
    if is_mlx:
        args, kwargs = backend.current_model.save_lora_adapters.call_args
        backend.current_model.save_pretrained.assert_not_called()
        if expect == "mlx":
            assert args == (str(tmp_path / "out"),) and kwargs == {}
        else:
            assert kwargs == {"adapter_format": "peft"}
            assert (tmp_path / "out" / "adapter_model.safetensors").read_text() == "peft"
    else:
        backend.current_model.save_pretrained.assert_called_once()
        backend.current_model.save_lora_adapters.assert_not_called()


def test_repeat_peft_export_overwrites(monkeypatch, tmp_path):
    backend = _backend(monkeypatch, True, tmp_path)
    _peft_writer(backend.current_model)
    out = tmp_path / "out"
    out.mkdir()
    (out / "adapter_model.safetensors").write_text("stale")
    for _ in range(2):
        ok, message, _ = backend.export_lora_adapter(str(out), adapter_format = "peft")
        assert ok, message
    assert (out / "adapter_model.safetensors").read_text() == "peft"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["out"]


def test_cuda_hub_push_unchanged(monkeypatch, tmp_path):
    backend = _backend(monkeypatch, False, tmp_path)
    monkeypatch.setattr(export_mod, "HfApi", lambda token = None: MagicMock())
    monkeypatch.setattr(export_mod, "_open_hub_repo", lambda hf_api, repo_id, private: repo_id)
    monkeypatch.setattr(export_mod, "_publish_unsloth_model_card", lambda *a: None)
    ok, message, _ = backend.export_lora_adapter("", push_to_hub = True, repo_id = "u/r", hf_token = "t")
    assert ok, message
    backend.current_model.push_to_hub.assert_called_once_with("u/r", token = "t", private = False)


def test_outdated_zoo_hard_error(monkeypatch, tmp_path):
    backend = _backend(monkeypatch, True, tmp_path)

    def _old_zoo_saver(path, adapter_config = None):  # no adapter_format kwarg
        raise AssertionError("an outdated saver must not be invoked")

    backend.current_model.save_lora_adapters = _old_zoo_saver
    ok, message, _ = backend.export_lora_adapter(
        str(tmp_path / "dst"),
        adapter_format = "peft",
    )
    assert not ok and "unsloth-zoo" in message

    def _modern_saver_with_internal_bug(
        path,
        adapter_config = None,
        adapter_format = "mlx",
    ):
        raise TypeError("scale must be a float")

    backend.current_model.save_lora_adapters = _modern_saver_with_internal_bug
    ok, message, _ = backend.export_lora_adapter(
        str(tmp_path / "dst2"),
        adapter_format = "peft",
    )
    assert not ok and "unsloth-zoo" not in message and "scale" in message


@pytest.mark.parametrize(
    "cfg,fs_attr,reason",
    [
        ({"alpha_pattern": {"^q_proj": 32}}, None, "alpha"),
        ({"use_rslora": True, "rank_pattern": {"^q_proj": 4}}, None, "rsLoRA"),
        ({"use_dora": True}, None, "DoRA"),
        ({"modules_to_save": ["lm_head"]}, None, "full-module state"),
        ({}, {"model.embed_tokens": "embedding_auto"}, "full-module state"),
        ({"target_parameters": ["experts.gate_up_proj"]}, None, "expert"),
    ],
)
def test_gguf_rejects(monkeypatch, tmp_path, cfg, fs_attr, reason):
    backend = _backend(monkeypatch, True, tmp_path)
    backend.current_model._unsloth_full_state_modules = fs_attr
    (tmp_path / "adapter_config.json").write_text(json.dumps(cfg))
    with pytest.raises(RuntimeError, match = reason):
        backend._convert_peft_dir_to_gguf(str(tmp_path), "q8_0", None)


def _converter_harness(monkeypatch, tmp_path, with_converter):
    import importlib.util
    import subprocess
    import sys
    import types

    llama = tmp_path / "home" / "llama.cpp"
    llama.mkdir(parents = True)
    if with_converter:
        (llama / "gguf-py").mkdir()
        (llama / "convert_lora_to_gguf.py").write_text("")
    zoo = types.ModuleType("unsloth_zoo")
    zoo.llama_cpp = types.SimpleNamespace(
        LLAMA_CPP_DEFAULT_DIR = str(llama),
        _resolve_converter_revision = lambda d: ("unslothai/llama.cpp", "b9000-mix-abc"),
    )
    monkeypatch.setitem(sys.modules, "unsloth_zoo", zoo)
    monkeypatch.setitem(sys.modules, "unsloth_zoo.llama_cpp", zoo.llama_cpp)
    calls = []

    def _run(
        cmd,
        env = None,
        **kwargs,
    ):
        calls.append((cmd, env))
        return types.SimpleNamespace(returncode = 0, stdout = "", stderr = "")

    monkeypatch.setattr(subprocess, "run", _run)
    real_find_spec = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name, *a: object() if name == "torch" else real_find_spec(name, *a),
    )
    backend = _backend(monkeypatch, True, tmp_path)
    backend.current_model._unsloth_full_state_modules = None
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text(json.dumps({"base_model_name_or_path": "o/m"}))
    return backend, str(adapter), calls


@pytest.mark.parametrize("token,expect", [(False, None), ("hf_x", "hf_x")])
def test_gguf_converter_token_env(monkeypatch, tmp_path, token, expect):
    monkeypatch.setenv("HF_TOKEN", "host-token")
    backend, adapter, calls = _converter_harness(monkeypatch, tmp_path, True)
    backend._convert_peft_dir_to_gguf(adapter, "q8_0", token)
    env = calls[-1][1]
    assert env.get("HF_TOKEN") == expect
    assert env.get("HF_HUB_DISABLE_IMPLICIT_TOKEN") == ("1" if token is False else "0")


def _fork_source_tarball(tag, members = None):
    import io
    import tarfile

    buf = io.BytesIO()
    with tarfile.open(fileobj = buf, mode = "w:gz") as tar:
        for name, data in members or [
            (f"llama.cpp-{tag}/convert_lora_to_gguf.py", b"import sys\n"),
            (f"llama.cpp-{tag}/gguf-py/gguf/__init__.py", b""),
        ]:
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def _serve_fork_release(
    monkeypatch,
    tag,
    tarball,
    sha = None,
    releases = (),
):
    """Fake GitHub: the fork release's sha256 json + source asset, and the release listing."""
    import hashlib
    import io

    from utils import llama_cpp_source

    sha = sha or hashlib.sha256(tarball).hexdigest()
    checksums = {
        "release_tag": tag,
        "upstream_tag": tag.split("-mix-")[0],
        "artifacts": {
            f"llama.cpp-source-{tag}.tar.gz": {"sha256": sha, "repo": "unslothai/llama.cpp"}
        },
    }
    urls = []

    def _open(url, timeout = 60):
        urls.append(url)
        assert "ggml-org" not in url
        if url.endswith("/llama-prebuilt-sha256.json"):
            return io.BytesIO(json.dumps(checksums).encode())
        if url.endswith(f"/llama.cpp-source-{tag}.tar.gz"):
            return io.BytesIO(tarball)
        if "/releases?" in url:
            return io.BytesIO(json.dumps(list(releases)).encode())
        raise AssertionError(url)

    monkeypatch.setattr(llama_cpp_source, "_open", _open)
    return urls


def test_gguf_converter_downloaded_from_fork_release_without_git(monkeypatch, tmp_path):
    backend, adapter, calls = _converter_harness(monkeypatch, tmp_path, False)
    tag = "b9000-mix-abc"
    urls = _serve_fork_release(monkeypatch, tag, _fork_source_tarball(tag))
    backend._convert_peft_dir_to_gguf(adapter, "q8_0", None)
    assert not any(c[0][0] == "git" for c in calls)
    assert urls == [
        f"https://github.com/unslothai/llama.cpp/releases/download/{tag}/llama-prebuilt-sha256.json",
        f"https://github.com/unslothai/llama.cpp/releases/download/{tag}/llama.cpp-source-{tag}.tar.gz",
    ]
    source = tmp_path / "home" / f"llama.cpp-source-{tag}"
    assert [c[0][1] for c in calls] == [str(source / "convert_lora_to_gguf.py")]
    assert sorted(p.name for p in (tmp_path / "home").iterdir()) == [
        "llama.cpp",
        f"llama.cpp-source-{tag}",
    ]
    # Cached: a second export makes no request.
    backend._convert_peft_dir_to_gguf(adapter, "q8_0", None)
    assert len(urls) == 2


def test_gguf_converter_maps_upstream_tag_to_fork_release(monkeypatch, tmp_path):
    import sys

    backend, adapter, calls = _converter_harness(monkeypatch, tmp_path, False)
    sys.modules["unsloth_zoo.llama_cpp"]._resolve_converter_revision = lambda d: (
        "ggml-org/llama.cpp",
        "b9000",
    )
    tag = "b9000-mix-def"
    urls = _serve_fork_release(
        monkeypatch,
        tag,
        _fork_source_tarball(tag),
        releases = [
            {"tag_name": "b9001-mix-aaa", "draft": False},
            {"tag_name": "b9000-mix-zzz", "draft": True},
            {"tag_name": tag, "draft": False},
        ],
    )
    backend._convert_peft_dir_to_gguf(adapter, "q8_0", None)
    assert calls[-1][0][1] == str(
        tmp_path / "home" / f"llama.cpp-source-{tag}" / "convert_lora_to_gguf.py"
    )
    assert any("/releases?" in u for u in urls)
    # Offline later: the fork tree fetched for that upstream tag is reused.
    sys.modules["unsloth_zoo.llama_cpp"]._converter_network_allowed = lambda: False
    backend._convert_peft_dir_to_gguf(adapter, "q8_0", None)
    assert calls[-1][0][1] == str(
        tmp_path / "home" / f"llama.cpp-source-{tag}" / "convert_lora_to_gguf.py"
    )


def test_gguf_converter_rejects_checksum_mismatch(monkeypatch, tmp_path):
    backend, adapter, calls = _converter_harness(monkeypatch, tmp_path, False)
    tag = "b9000-mix-abc"
    _serve_fork_release(monkeypatch, tag, _fork_source_tarball(tag), sha = "0" * 64)
    with pytest.raises(RuntimeError, match = "sha256 mismatch"):
        backend._convert_peft_dir_to_gguf(adapter, "q8_0", None)
    assert calls == []
    assert sorted(p.name for p in (tmp_path / "home").iterdir()) == ["llama.cpp"]


@pytest.mark.parametrize(
    "name",
    ["../evil.py", "/abs/evil.py", "llama.cpp-x/../../evil.py"],
)
def test_gguf_converter_refuses_path_traversal(monkeypatch, tmp_path, name):
    backend, adapter, calls = _converter_harness(monkeypatch, tmp_path, False)
    tag = "b9000-mix-abc"
    tarball = _fork_source_tarball(
        tag, [(f"llama.cpp-{tag}/convert_lora_to_gguf.py", b"x"), (name, b"x")]
    )
    _serve_fork_release(monkeypatch, tag, tarball)
    with pytest.raises(RuntimeError, match = "unsafe path"):
        backend._convert_peft_dir_to_gguf(adapter, "q8_0", None)
    assert calls == []
    assert not (tmp_path / "evil.py").exists() and not (tmp_path / "home" / "evil.py").exists()


def test_parse_adapter_features(tmp_path):
    def _dir(cfg):
        d = tmp_path / f"a{len(list(tmp_path.iterdir()))}"
        d.mkdir()
        (d / "adapter_config.json").write_text(json.dumps(cfg))
        return str(d)

    assert parse_adapter_features(str(tmp_path)) is None  # no config
    base = parse_adapter_features(_dir({"r": 8}))
    assert base == {
        "dora": False,
        "full_state": None,
        "moe_target_parameters": False,
        "non_uniform": False,
    }
    assert parse_adapter_features(_dir({"fine_tune_type": "lora"}))["full_state"] is None
    np = pytest.importorskip("numpy")
    save_file = pytest.importorskip("safetensors.numpy").save_file

    d_mlx = _dir({"fine_tune_type": "lora"})
    save_file(
        {"model.layers.0.self_attn.q_proj.lora_a": np.zeros((2, 2), dtype = "float32")},
        os.path.join(d_mlx, "adapters.safetensors"),
    )
    assert parse_adapter_features(d_mlx)["full_state"] is False
    save_file(
        {
            "model.layers.0.self_attn.q_proj.lora_a": np.zeros((2, 2), dtype = "float32"),
            "lm_head.bias": np.zeros((2,), dtype = "float32"),
        },
        os.path.join(d_mlx, "adapters.safetensors"),
    )
    assert parse_adapter_features(d_mlx)["full_state"] is True
    assert parse_adapter_features(_dir({"use_dora": True}))["dora"] is True
    assert parse_adapter_features(_dir({"fine_tune_type": "dora"}))["dora"] is True
    assert parse_adapter_features(_dir({"modules_to_save": ["lm_head"]}))["full_state"] is True
    assert (
        parse_adapter_features(_dir({"full_state_modules": {"lm_head": "modules_to_save"}}))[
            "full_state"
        ]
        is True
    )
    assert (
        parse_adapter_features(_dir({"target_parameters": ["experts.g"]}))["moe_target_parameters"]
        is True
    )
    assert parse_adapter_features(_dir({"rank_pattern": {"q": 4}}))["non_uniform"] is True
    assert (
        parse_adapter_features(_dir({"unsloth_mlx_lora_module_scales": {"q": 2.0}}))["non_uniform"]
        is True
    )


def test_local_dir_never_format_mixed(monkeypatch, tmp_path):
    backend = _backend(monkeypatch, True, tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    (out / "adapter_model.safetensors").write_bytes(b"x")  # other format
    ok, message, _ = backend.export_lora_adapter(str(out))
    assert not ok and "mix" in message
    backend.current_model.save_lora_adapters.assert_not_called()
    (out / "adapter_model.safetensors").unlink()
    (out / "named").mkdir()
    (out / "named" / "adapter_model.safetensors").write_bytes(b"x")
    ok, message, _ = backend.export_lora_adapter(str(out))
    assert not ok and "mix" in message
    (out / "named" / "adapter_model.safetensors").unlink()
    (out / "adapter_model.bin").write_bytes(b"x")
    ok, message, _ = backend.export_lora_adapter(str(out))
    assert not ok and "mix" in message


def test_gguf_converter_offline_refuses_download(monkeypatch, tmp_path):
    import sys

    backend, adapter, calls = _converter_harness(monkeypatch, tmp_path, False)
    sys.modules["unsloth_zoo.llama_cpp"]._converter_network_allowed = lambda: False
    with pytest.raises(RuntimeError, match = "offline"):
        backend._convert_peft_dir_to_gguf(adapter, "q8_0", None)
    assert calls == []


def test_gguf_converter_honors_scripts_dir(monkeypatch, tmp_path):
    import sys

    backend, adapter, calls = _converter_harness(monkeypatch, tmp_path, False)
    pinned = tmp_path / "pinned"
    pinned.mkdir()
    monkeypatch.setenv("UNSLOTH_LLAMA_CPP_SCRIPTS_DIR", str(pinned))
    sys.modules["unsloth_zoo.llama_cpp"]._resolve_converter_revision = None  # must not be called
    with pytest.raises(RuntimeError, match = "UNSLOTH_LLAMA_CPP_SCRIPTS_DIR"):
        backend._convert_peft_dir_to_gguf(adapter, "q8_0", None)
    (pinned / "convert_lora_to_gguf.py").write_text("")
    (pinned / "gguf-py").mkdir()
    backend._convert_peft_dir_to_gguf(adapter, "q8_0", None)
    assert [c[0][1] for c in calls] == [str(pinned / "convert_lora_to_gguf.py")]


def test_gguf_converter_uses_loaded_snapshot(monkeypatch, tmp_path):
    backend, adapter, calls = _converter_harness(monkeypatch, tmp_path, True)
    snap = tmp_path / "snapshot"
    snap.mkdir()
    (snap / "config.json").write_text("{}")
    backend.current_model._config_src_path = None
    backend.current_model._src_path = str(snap)
    backend._convert_peft_dir_to_gguf(adapter, "q8_0", None)
    cmd = calls[-1][0]
    assert cmd[cmd.index("--base") + 1] == str(snap) and "--base-model-id" not in cmd


def _fake_github(monkeypatch, routes):
    """routes: url suffix or substring -> bytes/obj (served) or Exception (raised)."""
    import io
    import urllib.error

    from utils import llama_cpp_source

    urls = []

    def _open(url, timeout = 60):
        urls.append(url)
        assert "ggml-org" not in url
        for key, value in routes.items():
            if url.endswith(key) or key in url:
                if value is None:
                    raise urllib.error.HTTPError(url, 404, "Not Found", {}, None)
                data = value if isinstance(value, bytes) else json.dumps(value).encode()
                return io.BytesIO(data)
        raise AssertionError(url)

    monkeypatch.setattr(llama_cpp_source, "_open", _open)
    return urls


def test_gguf_converter_reads_exact_commit_source_archive(monkeypatch, tmp_path):
    import hashlib

    backend, adapter, calls = _converter_harness(monkeypatch, tmp_path, False)
    tag, commit = "b9000-mix-abc", "a" * 40
    tarball = _fork_source_tarball(tag)
    asset = f"llama.cpp-source-commit-{commit}.tar.gz"
    _fake_github(
        monkeypatch,
        {
            "/llama-prebuilt-sha256.json": {
                "release_tag": tag,
                "source_commit": commit,
                "artifacts": {asset: {"sha256": hashlib.sha256(tarball).hexdigest()}},
            },
            f"/{asset}": tarball,
        },
    )
    backend._convert_peft_dir_to_gguf(adapter, "q8_0", None)
    assert calls[-1][0][1] == str(
        tmp_path / "home" / f"llama.cpp-source-{tag}" / "convert_lora_to_gguf.py"
    )


def test_gguf_converter_bare_fork_tag_maps_to_mix_release_across_pages(monkeypatch, tmp_path):
    import hashlib
    import sys

    backend, adapter, calls = _converter_harness(monkeypatch, tmp_path, False)
    sys.modules["unsloth_zoo.llama_cpp"]._resolve_converter_revision = lambda d: (
        "unslothai/llama.cpp",
        "b9000",
    )
    tag = "b9000-mix-def"
    tarball = _fork_source_tarball(tag)
    urls = _fake_github(
        monkeypatch,
        {
            "/download/b9000/llama-prebuilt-sha256.json": None,
            f"/download/{tag}/llama-prebuilt-sha256.json": {
                "release_tag": tag,
                "artifacts": {
                    f"llama.cpp-source-{tag}.tar.gz": {
                        "sha256": hashlib.sha256(tarball).hexdigest()
                    }
                },
            },
            f"/llama.cpp-source-{tag}.tar.gz": tarball,
            "&page=1": [{"tag_name": f"b{9100 + i}-mix-x", "draft": False} for i in range(100)],
            "&page=2": [{"tag_name": tag, "draft": False}],
        },
    )
    backend._convert_peft_dir_to_gguf(adapter, "q8_0", None)
    assert calls[-1][0][1] == str(
        tmp_path / "home" / f"llama.cpp-source-{tag}" / "convert_lora_to_gguf.py"
    )
    assert any("&page=2" in u for u in urls)


@pytest.mark.parametrize(
    "revision,legacy",
    [
        (("unslothai/llama.cpp", "b9000-mix-abc"), "llama.cpp-source-b9000"),
        ((None, None), "llama.cpp-source"),
    ],
)
def test_gguf_converter_reuses_tree_from_previous_exporter_offline(
    monkeypatch, tmp_path, revision, legacy
):
    import sys

    backend, adapter, calls = _converter_harness(monkeypatch, tmp_path, False)
    zoo = sys.modules["unsloth_zoo.llama_cpp"]
    zoo._resolve_converter_revision = lambda d: revision
    zoo._converter_network_allowed = lambda: False
    old = tmp_path / "home" / legacy
    (old / "gguf-py").mkdir(parents = True)
    (old / "convert_lora_to_gguf.py").write_text("")
    _fake_github(monkeypatch, {})
    backend._convert_peft_dir_to_gguf(adapter, "q8_0", None)
    assert calls[-1][0][1] == str(old / "convert_lora_to_gguf.py")


def test_gguf_converter_lookup_failure_keeps_the_revision(monkeypatch, tmp_path):
    import sys

    backend, adapter, calls = _converter_harness(monkeypatch, tmp_path, False)
    sys.modules["unsloth_zoo.llama_cpp"]._resolve_converter_revision = lambda d: (
        "ggml-org/llama.cpp",
        "b9000",
    )
    urls = _fake_github(monkeypatch, {"/releases?": None})
    with pytest.raises(RuntimeError, match = "converter sources"):
        backend._convert_peft_dir_to_gguf(adapter, "q8_0", None)
    assert not any("/latest/" in u for u in urls)
    assert calls == []


def test_gguf_converter_skips_prerelease_when_mapping_upstream_tag(monkeypatch):
    from utils import llama_cpp_source
    _fake_github(
        monkeypatch,
        {
            "&page=1": [
                {"tag_name": "b9000-mix-pre", "draft": False, "prerelease": True},
                {"tag_name": "b9000-mix-rel", "draft": False, "prerelease": False},
            ]
        },
    )
    assert llama_cpp_source._matching_fork_tag("b9000") == "b9000-mix-rel"


def test_gguf_converter_offline_without_revision_reuses_newest_fork_tree(monkeypatch, tmp_path):
    import os
    import sys

    backend, adapter, calls = _converter_harness(monkeypatch, tmp_path, False)
    zoo = sys.modules["unsloth_zoo.llama_cpp"]
    zoo._resolve_converter_revision = lambda d: (None, None)
    zoo._converter_network_allowed = lambda: False
    trees = []
    for i, tag in enumerate(["b8000-mix-old", "b9000-mix-new"]):
        tree = tmp_path / "home" / f"llama.cpp-source-{tag}"
        (tree / "gguf-py").mkdir(parents = True)
        (tree / "convert_lora_to_gguf.py").write_text("")
        os.utime(tree, (1000 + i, 1000 + i))
        trees.append(tree)
    _fake_github(monkeypatch, {})
    backend._convert_peft_dir_to_gguf(adapter, "q8_0", None)
    assert calls[-1][0][1] == str(trees[-1] / "convert_lora_to_gguf.py")


def test_gguf_converter_reads_legacy_upstream_named_source_archive(monkeypatch, tmp_path):
    import hashlib

    backend, adapter, calls = _converter_harness(monkeypatch, tmp_path, False)
    tag = "b9000-mix-abc"
    tarball = _fork_source_tarball(tag)
    asset = "llama.cpp-source-b9000.tar.gz"
    _fake_github(
        monkeypatch,
        {
            "/llama-prebuilt-sha256.json": {
                "release_tag": tag,
                "upstream_tag": "b9000",
                "artifacts": {asset: {"sha256": hashlib.sha256(tarball).hexdigest()}},
            },
            f"/{asset}": tarball,
        },
    )
    backend._convert_peft_dir_to_gguf(adapter, "q8_0", None)
    assert calls[-1][0][1] == str(
        tmp_path / "home" / f"llama.cpp-source-{tag}" / "convert_lora_to_gguf.py"
    )


def test_gguf_converter_online_bare_tag_ignores_an_unrelated_mix_cache(monkeypatch, tmp_path):
    import hashlib
    import sys

    backend, adapter, calls = _converter_harness(monkeypatch, tmp_path, False)
    sys.modules["unsloth_zoo.llama_cpp"]._resolve_converter_revision = lambda d: (
        "ggml-org/llama.cpp",
        "b9000",
    )
    stale = tmp_path / "home" / "llama.cpp-source-b9000-mix-pre"
    (stale / "gguf-py").mkdir(parents = True)
    (stale / "convert_lora_to_gguf.py").write_text("")
    tag = "b9000-mix-rel"
    tarball = _fork_source_tarball(tag)
    _fake_github(
        monkeypatch,
        {
            "&page=1": [{"tag_name": tag, "draft": False}],
            f"/download/{tag}/llama-prebuilt-sha256.json": {
                "release_tag": tag,
                "artifacts": {
                    f"llama.cpp-source-{tag}.tar.gz": {
                        "sha256": hashlib.sha256(tarball).hexdigest()
                    }
                },
            },
            f"/llama.cpp-source-{tag}.tar.gz": tarball,
        },
    )
    backend._convert_peft_dir_to_gguf(adapter, "q8_0", None)
    assert calls[-1][0][1] == str(
        tmp_path / "home" / f"llama.cpp-source-{tag}" / "convert_lora_to_gguf.py"
    )


def test_source_artifact_accepts_abbreviated_commit():
    from utils import llama_cpp_source
    name, digest = llama_cpp_source.source_artifact(
        {
            "source_commit": "abc1234",
            "artifacts": {"llama.cpp-source-commit-abc1234.tar.gz": {"sha256": "a" * 64}},
        },
        "b9000-mix-abc",
    )
    assert name == "llama.cpp-source-commit-abc1234.tar.gz"


def test_safe_extract_without_data_filter_drops_links(monkeypatch, tmp_path):
    import io
    import tarfile

    from utils import llama_cpp_source

    archive = tmp_path / "a.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        link = tarfile.TarInfo("root/a")
        link.type = tarfile.SYMTYPE
        link.linkname = "."
        tar.addfile(link)
        data = b"x"
        info = tarfile.TarInfo("root/convert_lora_to_gguf.py")
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))
    monkeypatch.delattr(tarfile, "data_filter", raising = False)
    out = tmp_path / "out"
    out.mkdir()
    llama_cpp_source.safe_extract_tar(archive, out)
    assert (out / "root" / "convert_lora_to_gguf.py").is_file()
    assert not (out / "root" / "a").exists() and not (out / "root" / "a").is_symlink()
