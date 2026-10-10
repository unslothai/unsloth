# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""unsloth.ini beside a GGUF: parsing, allowlist, location, and how a load applies it."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from core.inference import llama_model_ini as mi
from core.inference.llama_model_ini import (
    MAX_MODEL_INI_BYTES,
    NotGgufModel,
    parse_model_ini,
)
from core.inference.llama_server_args import validate_extra_args

# The issue reporter's file (unslothai/unsloth#10359), trimmed.
REPORTER_INI = """\
[*]
c = 128000
fitc = 128000
load-on-startup = false
b = 2048 ; formula: base b size * np
ub = 2048
np = 1 ; sync with router np flag
t = 6
ngl = -1
fit = off
flash-attn = on
jinja = true
ctk = q8_0
ctv = q8_0
cache-reuse = 512
reasoning = on
no-warmup = true
chat-template-kwargs = {"preserve_thinking": true}

[free-27B-Q3.8]
m = /mnt/AI/models/model.gguf
mm = /mnt/AI/models/mmproj.gguf
mmproj-offload = false
c = 56000
temp = 1
"""


def _pairs(args):
    out, i = {}, 0
    while i < len(args):
        if i + 1 < len(args) and not args[i + 1].startswith("--"):
            out[args[i]] = args[i + 1]
            i += 2
        else:
            out[args[i]] = True
            i += 1
    return out


def test_reporter_ini_global_section_compiles_and_passes_the_boundary():
    r = parse_model_ini(REPORTER_INI, quant = "Q4_K_S")
    assert r.applied_sections == ["*"]
    assert r.sections == ["*", "free-27B-Q3.8"]
    assert r.n_parallel == 1
    p = _pairs(r.args)
    assert p["--ctx-size"] == "128000"
    assert p["--fit"] == "off"
    assert p["--gpu-layers"] == "-1"
    assert p["--cache-type-k"] == "q8_0"
    assert p["--no-warmup"] is True
    assert p["--chat-template-kwargs"] == '{"preserve_thinking":true}'
    assert "--parallel" not in r.args and "-np" not in r.args
    assert r.ignored == []  # load-on-startup is router-only, skipped silently
    assert validate_extra_args(r.args) == r.args


def test_matching_section_overrides_global_and_reports_model_keys():
    r = parse_model_ini(REPORTER_INI, quant = "free-27B-Q3.8")
    assert r.applied_sections == ["*", "free-27B-Q3.8"]
    p = _pairs(r.args)
    assert p["--ctx-size"] == "56000"
    assert p["--temp"] == "1"
    assert p["--no-mmproj-offload"] is True
    assert r.args.count("--ctx-size") == 1
    assert {i["key"] for i in r.ignored} == {"m", "mm"}
    assert all(i["reason"] == "Studio supplies the model" for i in r.ignored)


def test_section_matches_gguf_stem_without_shard_suffix_case_insensitively():
    ini = "[*]\ntemp = 0.5\n[model-ud-q4_k_xl]\ntemp = 0.7\n"
    r = parse_model_ini(ini, gguf_filename = "UD-Q4_K_XL/Model-UD-Q4_K_XL-00001-of-00003.gguf")
    assert _pairs(r.args)["--temp"] == "0.7"


def test_keys_above_first_header_apply_like_star():
    r = parse_model_ini("temp = 0.3\n[other]\ntemp = 0.9\n", quant = "Q8_0")
    assert r.applied_sections == ["default"]
    assert _pairs(r.args)["--temp"] == "0.3"


@pytest.mark.parametrize(
    "key",
    ["c", "ctx-size", "--ctx-size", "-c", "LLAMA_ARG_CTX_SIZE"],
)
def test_every_spelling_of_one_option(key):
    assert parse_model_ini(f"{key} = 4096\n").args == ["--ctx-size", "4096"]


@pytest.mark.parametrize(
    "line, expected",
    [
        ("kvu = true", ["--kv-unified"]),
        ("kvu = off", ["--no-kv-unified"]),
        ("no-kv-unified = 1", ["--no-kv-unified"]),
        ("no-kv-unified = false", ["--kv-unified"]),
        ("warmup = false", ["--no-warmup"]),
        ("swa-full = false", []),
        ("jinja = enabled", ["--jinja"]),
        ("nkvo = true", ["--no-kv-offload"]),
    ],
)
def test_switch_polarity(line, expected):
    assert parse_model_ini(line).args == expected


@pytest.mark.parametrize(
    "line",
    [
        "c = big",
        "b = 0",
        "ctk = q3_weird",
        "fa = maybe",
        "top-p = 1.5",
        "chat-template-kwargs = [1, 2]",
        "ot = has space=CPU",
        "spec-type = warp-drive",
        "kvu = sometimes",
        "np = 0",
        "temp",
    ],
)
def test_bad_values_are_ignored_not_applied(line):
    r = parse_model_ini(line)
    assert r.args == [] and r.n_parallel is None
    assert len(r.ignored) == 1


@pytest.mark.parametrize(
    "line",
    [
        "rpc = 10.0.0.1:50052",
        "lora = /etc/passwd",
        "chat-template-file = /tmp/x.jinja",
        "log-file = /tmp/x.log",
        "host = 0.0.0.0",
        "port = 1",
        "api-key = secret",
        "slot-save-path = /tmp",
        "models-preset = /tmp/p.ini",
        "mlock = true",
        "reasoning-budget = 10",
        "device = CUDA0",
    ],
)
def test_keys_outside_the_allowlist_never_reach_the_command(line):
    r = parse_model_ini(line)
    assert r.args == []
    assert r.ignored[0]["reason"] == "not an allowed setting"


def test_comments_end_values_and_quotes_are_kept_like_llama_cpp():
    r = parse_model_ini("reasoning-effort = high # trailing\n; whole line\n")
    assert r.args == ["--reasoning-effort", "high"]


def test_repeated_header_starts_the_section_over():
    r = parse_model_ini("[*]\ntemp = 0.1\ntop-k = 5\n[*]\ntemp = 0.2\n")
    assert r.args == ["--temp", "0.2"]


def test_size_cap():
    with pytest.raises(ValueError):
        parse_model_ini("; " + "x" * MAX_MODEL_INI_BYTES)


# ── locate ─────────────────────────────────────────────────────────────


def test_local_ini_beside_the_selected_variant_wins_over_root(tmp_path):
    sub = tmp_path / "UD-Q4_K_XL"
    sub.mkdir()
    (sub / "M-UD-Q4_K_XL.gguf").write_bytes(b"GGUF")
    (tmp_path / "M-Q8_0.gguf").write_bytes(b"GGUF")
    (tmp_path / "unsloth.ini").write_text("temp = 0.1\n")
    (sub / "unsloth.ini").write_text("temp = 0.9\n")
    found = mi.locate_model_ini(str(sub / "M-UD-Q4_K_XL.gguf"))
    assert found.location == "local_dir" and "0.9" in found.text
    root = mi.locate_model_ini(str(tmp_path / "M-Q8_0.gguf"))
    assert "0.1" in root.text


def test_local_dir_without_ini_is_none_and_non_gguf_dir_raises(tmp_path):
    (tmp_path / "M-Q8_0.gguf").write_bytes(b"GGUF")
    assert mi.locate_model_ini(str(tmp_path)) is None
    empty = tmp_path / "hf"
    empty.mkdir()
    (empty / "config.json").write_text("{}")
    with pytest.raises(NotGgufModel):
        mi.locate_model_ini(str(empty))


def _hf(files, variants):
    seen = []

    def list_variants(repo_id, hf_token = None):
        return variants, False

    def download(repo_id, filename, hf_token, offline):
        seen.append(filename)
        return files.get(filename)

    return list_variants, download, seen


def _variant(filename, quant):
    return SimpleNamespace(filename = filename, quant = quant, size_bytes = 1)


def test_hf_variant_folder_first_then_root(tmp_path):
    folder_ini = tmp_path / "a.ini"
    folder_ini.write_text("temp = 0.9\n")
    root_ini = tmp_path / "b.ini"
    root_ini.write_text("temp = 0.1\n")
    variants = [_variant("UD-Q4_K_XL/M-UD-Q4_K_XL-00001-of-00002.gguf", "UD-Q4_K_XL")]
    lv, dl, seen = _hf(
        {"UD-Q4_K_XL/unsloth.ini": str(folder_ini), "unsloth.ini": str(root_ini)}, variants
    )
    got = mi._locate_hf("u/M-GGUF", "ud-q4_k_xl", None, False, lv, dl)
    assert got.location == "variant_folder" and "0.9" in got.text
    assert got.gguf_filename.endswith("-00001-of-00002.gguf")

    lv, dl, seen = _hf({"unsloth.ini": str(root_ini)}, variants)
    got = mi._locate_hf("u/M-GGUF", "UD-Q4_K_XL", None, False, lv, dl)
    assert got.location == "repo_root"
    assert seen == ["UD-Q4_K_XL/unsloth.ini", "unsloth.ini"]


def test_hf_root_variant_only_checks_root_and_non_gguf_repo_raises(tmp_path):
    lv, dl, seen = _hf({}, [_variant("M-Q8_0.gguf", "Q8_0")])
    assert mi._locate_hf("u/M-GGUF", "Q8_0", None, False, lv, dl) is None
    assert seen == ["unsloth.ini"]
    lv, dl, _ = _hf({}, [])
    with pytest.raises(NotGgufModel):
        mi._locate_hf("u/model", None, None, False, lv, dl)


# ── load integration ───────────────────────────────────────────────────


def _load_request(**kw):
    from models.inference import LoadRequest
    return LoadRequest(model_path = "u/M-GGUF", gguf_variant = "Q8_0", **kw)


def _patch_locate(
    monkeypatch,
    text,
    location = "repo_root",
):
    found = mi.LocatedModelIni(text, location, "Q8_0", "M-Q8_0.gguf")
    monkeypatch.setattr(mi, "locate_model_ini", lambda *a, **k: found)


def test_toggle_off_leaves_the_request_untouched(monkeypatch):
    from routes.inference import _apply_model_ini_to_request

    monkeypatch.setattr(mi, "locate_model_ini", lambda *a, **k: pytest.fail("must not read"))
    request = _load_request(llama_extra_args = ["--top-k", "3"])
    assert _apply_model_ini_to_request(request, "u/M-GGUF", "M") is request


def test_toggle_on_compiles_ini_and_moves_np_into_n_parallel(monkeypatch):
    from routes.inference import _apply_model_ini_to_request, _with_model_ini

    _patch_locate(monkeypatch, "c = 8192\nnp = 2\ntemp = 0.6\n")
    request = _apply_model_ini_to_request(
        _load_request(use_model_ini = True, llama_extra_args = ["--temp", "0.2"]), "u/M-GGUF", "M"
    )
    assert request.n_parallel == 2
    merged = _with_model_ini(request, request.llama_extra_args)
    # INI first, typed extras last: llama.cpp is last-wins, so the typed --temp wins.
    assert merged == ["--ctx-size", "8192", "--temp", "0.6", "--temp", "0.2"]


def test_toggle_on_without_file_is_a_400_and_non_gguf_is_ignored(monkeypatch):
    from fastapi import HTTPException

    from routes.inference import _apply_model_ini_to_request

    monkeypatch.setattr(mi, "locate_model_ini", lambda *a, **k: None)
    with pytest.raises(HTTPException) as exc:
        _apply_model_ini_to_request(_load_request(use_model_ini = True), "u/M-GGUF", "M")
    assert exc.value.status_code == 400 and "unsloth.ini" in exc.value.detail

    def not_gguf(*a, **k):
        raise NotGgufModel("u/M")

    monkeypatch.setattr(mi, "locate_model_ini", not_gguf)
    request = _load_request(use_model_ini = True)
    out = _apply_model_ini_to_request(request, "u/M", "M")
    assert out._model_ini_args == ()


def test_manual_gpu_mode_drops_the_ini_offload_flags(monkeypatch):
    from routes.inference import _apply_model_ini_to_request, _model_ini_tokens

    _patch_locate(monkeypatch, "ngl = 20\nfit = off\nn-cpu-moe = 4\nc = 4096\n")
    request = _apply_model_ini_to_request(
        _load_request(use_model_ini = True, gpu_memory_mode = "manual"), "u/M-GGUF", "M"
    )
    assert _model_ini_tokens(request) == ["--ctx-size", "4096"]


class _Backend:
    is_diffusion = False

    def __init__(
        self,
        stored,
        prefix,
        source = ("u/M-GGUF", "Q8_0"),
    ):
        self.extra_args = list(stored)
        self.requested_extra_args = list(stored)
        self.extra_args_source = source
        self._studio_model_ini_record = (tuple(prefix), source) if prefix else None


def test_stored_ini_prefix_is_never_inherited():
    from routes.inference import _resolve_inherited_extra_args
    import routes.inference as routes

    backend = _Backend(["--ctx-size", "8192", "--top-k", "3"], ["--ctx-size", "8192"])

    class Request:
        model_path = "u/M-GGUF"
        gguf_variant = "Q8_0"
        llama_extra_args = None
        model_fields_set = set()

    class Config:
        is_gguf = True
        gguf_variant = "Q8_0"
        identifier = "u/M-GGUF"

    original = routes.get_llama_cpp_backend
    routes.get_llama_cpp_backend = lambda: backend
    try:
        assert _resolve_inherited_extra_args(Request(), Config(), "u/M-GGUF", None) == [
            "--top-k",
            "3",
        ]
    finally:
        routes.get_llama_cpp_backend = original


def test_prefix_is_stale_once_another_load_rewrote_the_extras():
    from routes.inference import _model_ini_prefix, _without_model_ini

    other = _Backend(["--ctx-size", "8192"], ["--ctx-size", "8192"])
    other.extra_args_source = ("u/Other-GGUF", None)
    assert _model_ini_prefix(other) == ()
    assert _without_model_ini(other, other.extra_args) == ["--ctx-size", "8192"]


def test_status_reports_user_extras_and_ini_flag():
    from routes.inference import _model_ini_prefix, _without_model_ini

    backend = _Backend(["--ctx-size", "8192"], ["--ctx-size", "8192"])
    assert _model_ini_prefix(backend) == ("--ctx-size", "8192")
    assert _without_model_ini(backend, backend.requested_extra_args) == []


def test_runtime_fields_echo_model_ini_applied():
    import inspect

    import routes.inference as routes

    src = inspect.getsource(routes._llama_runtime_fields)
    assert "model_ini_applied" in src and "_without_model_ini" in src
    from models.inference import InferenceStatusResponse, LoadResponse

    assert "model_ini_applied" in LoadResponse.model_fields
    assert "model_ini_applied" in InferenceStatusResponse.model_fields


# ── GET /api/models/model-ini ──────────────────────────────────────────


def _call_route(monkeypatch, **kw):
    import routes.models as models_routes

    monkeypatch.setattr(models_routes.account_access, "managed_account", lambda: False)
    monkeypatch.setattr(models_routes, "_resolve_hub_token", lambda header, token: None)
    args = dict(
        repo_id = "u/M-GGUF",
        gguf_variant = "Q8_0",
        local_path = None,
        hf_token = None,
        hf_token_header = None,
        current_subject = "owner",
        via_api_key = False,
    )
    args.update(kw)
    return asyncio.run(models_routes.get_model_ini(**args))


def test_route_found_and_absent(monkeypatch):
    _patch_locate(monkeypatch, "[*]\nc = 4096\nrpc = x\n", location = "variant_folder")
    body = _call_route(monkeypatch)
    assert body.found and body.location == "variant_folder"
    assert body.args == ["--ctx-size", "4096"]
    assert [i.key for i in body.ignored] == ["rpc"]

    monkeypatch.setattr(mi, "locate_model_ini", lambda *a, **k: None)
    body = _call_route(monkeypatch)
    assert body.found is False and body.args == [] and body.filename == "unsloth.ini"


def test_route_unreadable_hub_reads_as_absent(monkeypatch):
    def boom(*a, **k):
        raise OSError("hub down")

    monkeypatch.setattr(mi, "locate_model_ini", boom)
    assert _call_route(monkeypatch).found is False


def test_already_loaded_dedupe_sees_the_ini_both_ways():
    """The fast path compares INI + typed extras against what the resident launched, so an
    unchanged toggle-on reload is a no-op and switching the toggle off is a reload."""
    from core.inference.llama_cpp import LlamaCppBackend
    from models.inference import LoadRequest
    from routes.inference import _active_gguf_intent

    backend = LlamaCppBackend()
    backend._process = object()
    backend._healthy = True
    backend._model_identifier = "owner/repo"
    backend._hf_variant = "Q4_K_M"
    backend._extra_args = ["--ctx-size", "8192", "--top-k", "20"]
    backend._extra_args_source = ("owner/repo", "Q4_K_M")
    backend._studio_model_ini_record = (("--ctx-size", "8192"), backend._extra_args_source)
    kwargs = dict(
        model_identifier = "owner/repo",
        chat_template_override = None,
        n_parallel = 1,
        native_grant_backed = False,
    )

    on = LoadRequest(model_path = "owner/repo", gguf_variant = "Q4_K_M")
    on._model_ini_args = ("--ctx-size", "8192")
    assert _active_gguf_intent(on, backend, **kwargs).extra_args == (
        "--ctx-size",
        "8192",
        "--top-k",
        "20",
    )
    off = LoadRequest(model_path = "owner/repo", gguf_variant = "Q4_K_M")
    assert _active_gguf_intent(off, backend, **kwargs).extra_args == ("--top-k", "20")
