# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0


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


def test_local_file_quant_selects_its_section_without_a_variant(tmp_path):
    (tmp_path / "Qwen3-0.6B-Q4_K_M.gguf").write_bytes(b"GGUF")
    (tmp_path / "unsloth.ini").write_text("[*]\ntemp = 0.4\n[Q4_K_M]\ntop-p = 0.77\n")
    found = mi.locate_model_ini(str(tmp_path / "Qwen3-0.6B-Q4_K_M.gguf"))
    compiled = mi.parse_model_ini(found.text, quant = found.quant, gguf_filename = found.gguf_filename)
    assert found.quant == "Q4_K_M"
    assert compiled.applied_sections == ["*", "Q4_K_M"] and compiled.args[-2:] == [
        "--top-p",
        "0.77",
    ]


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
        sampling = any(
            t in ("--temp", "--top-p", "--top-k", "--min-p", "--repeat-penalty") for t in prefix
        )
        self._studio_model_ini_record = (tuple(prefix), source, sampling) if prefix else None


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


def test_ini_only_extras_are_cleared_explicitly_when_the_toggle_goes_off(monkeypatch):
    # None would leave the INI tokens stored for the next load to inherit as typed extras.
    from routes.inference import _resolve_inherited_extra_args
    import routes.inference as routes

    backend = _Backend(["--ctx-size", "8192"], ["--ctx-size", "8192"])
    monkeypatch.setattr(routes, "get_llama_cpp_backend", lambda: backend)
    request = SimpleNamespace(model_path = "u/M-GGUF", gguf_variant = "Q8_0", llama_extra_args = None)
    config = SimpleNamespace(is_gguf = True, gguf_variant = "Q8_0", identifier = "u/M-GGUF")
    assert _resolve_inherited_extra_args(request, config, "u/M-GGUF", None) == []
    backend._studio_model_ini_record = None
    backend.extra_args = []
    assert _resolve_inherited_extra_args(request, config, "u/M-GGUF", None) is None


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


def _call_route(monkeypatch, **kw):
    import routes.models as models_routes

    monkeypatch.setattr(models_routes.account_access, "managed_account", lambda: False)
    monkeypatch.setattr(models_routes, "_resolve_hub_token", lambda header, token: None)
    args = dict(
        repo_id = "u/M-GGUF",
        gguf_variant = "Q8_0",
        local_path = None,
        hf_token = None,
        offline = False,
        native_path_lease = None,
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
    backend._studio_model_ini_record = (("--ctx-size", "8192"), backend._extra_args_source, False)
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


def test_ini_sampling_becomes_the_chat_defaults():
    from routes.inference import _with_model_ini_sampling

    backend = _Backend(
        ["--ctx-size", "4096", "--temp", "0.42", "--top-k", "17", "--top-p", "0.77"],
        ["--ctx-size", "4096", "--temp", "0.42", "--top-k", "17", "--top-p", "0.77"],
    )
    base = {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.01}
    assert _with_model_ini_sampling(base, backend) == {
        "temperature": 0.42,
        "top_p": 0.77,
        "top_k": 17,
        "min_p": 0.01,
    }
    assert base["temperature"] == 0.6
    assert _with_model_ini_sampling(base, _Backend(["--temp", "0.42"], [])) is base


def test_typed_sampler_after_the_ini_wins_in_the_reported_defaults():
    from routes.inference import _with_model_ini_sampling

    backend = _Backend(
        ["--temp", "0.6", "--repeat-penalty", "1.1", "--temp=0.2"],
        ["--temp", "0.6", "--repeat-penalty", "1.1"],
    )
    base = {"temperature": 0.7, "top_p": 0.95}
    assert _with_model_ini_sampling(base, backend) == {
        "temperature": 0.2,
        "top_p": 0.95,
        "repetition_penalty": 1.1,
    }
    backend = _Backend(["--temp", "0.6", "--top-k", "5"], ["--temp", "0.6"])
    assert "top_k" not in _with_model_ini_sampling(base, backend)


def test_parallel_only_ini_still_counts_as_applied(monkeypatch):
    import routes.inference as routes

    _patch_locate(monkeypatch, "np = 2\n")
    request = routes._apply_model_ini_to_request(_load_request(use_model_ini = True), "u/M-GGUF", "M")
    assert request.n_parallel == 2 and request._model_ini_args == ()
    assert request._model_ini_applied is True and request._model_ini_sampling is False
    backend = _Backend([], [])
    backend._studio_model_ini_record = routes._model_ini_record_for(
        request, backend.extra_args_source
    )
    assert routes._model_ini_resident(backend) and routes._model_ini_prefix(backend) == ()


def test_sampling_flag_tracks_whether_the_ini_set_a_sampler(monkeypatch):
    import routes.inference as routes

    _patch_locate(monkeypatch, "c = 4096\nfit = off\n")
    perf = routes._apply_model_ini_to_request(_load_request(use_model_ini = True), "u/M-GGUF", "M")
    assert perf._model_ini_applied and perf._model_ini_sampling is False
    _patch_locate(monkeypatch, "c = 4096\nmin-p = 0.05\n")
    samp = routes._apply_model_ini_to_request(_load_request(use_model_ini = True), "u/M-GGUF", "M")
    assert samp._model_ini_sampling is True
    from models.inference import InferenceStatusResponse, LoadResponse

    assert "model_ini_sampling" in LoadResponse.model_fields
    assert "model_ini_sampling" in InferenceStatusResponse.model_fields


def test_split_mode_is_left_to_studio():
    compiled = parse_model_ini(
        "[*]\nsm = tensor\n[Q8_0]\nsplit-mode = row\n", quant = "Q8_0", gguf_filename = None
    )
    assert compiled.args == []
    assert [(i["key"], "tensor parallel" in i["reason"]) for i in compiled.ignored] == [
        ("sm", True),
        ("split-mode", True),
    ]


def test_qualified_variant_matches_its_bare_quant_section():
    compiled = parse_model_ini(
        "[Q6_K]\ntemp = 0.5\n",
        quant = "distilled/ltx-2.3-22b-distilled-Q6_K",
        gguf_filename = "distilled/ltx-2.3-22b-distilled-Q6_K.gguf",
    )
    assert compiled.applied_sections == ["Q6_K"] and compiled.args == ["--temp", "0.5"]


def test_route_passes_offline_through(monkeypatch):
    seen = {}

    def locate(path, variant, **kw):
        seen.update(kw)
        return None

    monkeypatch.setattr(mi, "locate_model_ini", locate)
    _call_route(monkeypatch, offline = True)
    assert seen.get("offline") is True


def test_offline_hf_lookup_lists_variants_from_the_cache_only(monkeypatch):
    import utils.models.model_config as mc

    monkeypatch.setattr(mc, "list_gguf_variants", lambda *a, **k: pytest.fail("network listing"))
    monkeypatch.setattr(
        mc,
        "_list_gguf_variants_from_hf_cache",
        lambda repo: ([SimpleNamespace(filename = "Q8_0/M-Q8_0.gguf", quant = "Q8_0")], False),
    )
    asked = []
    mi._locate_hf("u/M-GGUF", "Q8_0", None, True, None, lambda r, f, t, off: asked.append((f, off)))
    assert asked == [("Q8_0/unsloth.ini", True), ("unsloth.ini", True)]


def test_route_reads_beside_a_native_lease(monkeypatch, tmp_path):
    import routes.models as models_routes

    gguf = tmp_path / "M-Q8_0.gguf"
    gguf.write_bytes(b"GGUF")
    (tmp_path / "unsloth.ini").write_text("c = 2048\n")
    grants = []

    def verify(lease, **kw):
        grants.append((lease, kw))
        return SimpleNamespace(canonical_path = gguf)

    monkeypatch.setattr(models_routes, "verify_native_path_lease", verify, raising = False)
    body = _call_route(monkeypatch, repo_id = "M-Q8_0.gguf", native_path_lease = "lease-1")
    assert body.found and body.args == ["--ctx-size", "2048"]
    assert grants == [
        (
            "lease-1",
            dict(
                operation = "validate-model",
                expected_kind = "model",
                expected_path_type = "file",
                allowed_suffixes = (".gguf",),
            ),
        )
    ]

    from fastapi import HTTPException
    from utils.native_path_leases import NativePathLeaseError

    def refuse(lease, **kw):
        raise NativePathLeaseError("Native path grant expired; re-select the file.")

    monkeypatch.setattr(models_routes, "verify_native_path_lease", refuse, raising = False)
    with pytest.raises(HTTPException) as exc:
        _call_route(monkeypatch, repo_id = "M-Q8_0.gguf", native_path_lease = "stale")
    assert exc.value.status_code == 400 and "re-select" in exc.value.detail


def test_load_path_resolves_the_ini_offline_when_the_hub_is_unreachable(monkeypatch):
    import contextlib

    import routes.inference as routes

    seen = {}

    def fake_locate(*a, **k):
        seen.update(k)
        return mi.LocatedModelIni("c = 4096\n", "repo_root", "Q8_0", "M-Q8_0.gguf")

    monkeypatch.setattr(mi, "locate_model_ini", fake_locate)
    monkeypatch.setattr(
        routes, "_hf_offline_if_unreachable_for", lambda _m: contextlib.nullcontext(True)
    )
    routes._apply_model_ini_to_request(_load_request(use_model_ini = True), "u/M-GGUF", "M")
    assert seen["offline"] is True


@pytest.mark.parametrize(
    "line",
    [
        "temp = 2.5",
        "top-k = 101",
        "top-k = -2",
        "repeat-penalty = 0.9",
        "repeat-penalty = 2.5",
        "presence-penalty = -0.5",
        "presence-penalty = 2.5",
    ],
)
def test_samplers_outside_the_chat_schema_are_ignored(line):
    compiled = parse_model_ini(line + "\n", quant = None, gguf_filename = None)
    assert compiled.args == [] and [i["key"] for i in compiled.ignored] == [line.split(" =")[0]]


def test_samplers_at_the_chat_schema_edges_are_kept():
    compiled = parse_model_ini(
        "temp = 2\ntop-k = -1\nrepeat-penalty = 1\npresence-penalty = 2\n",
        quant = None,
        gguf_filename = None,
    )
    assert compiled.ignored == []
    assert compiled.args == [
        "--temp",
        "2",
        "--top-k",
        "-1",
        "--repeat-penalty",
        "1",
        "--presence-penalty",
        "2",
    ]


def test_local_directory_picks_the_gguf_the_loader_picks(tmp_path):
    (tmp_path / "M-Q2_K.gguf").write_bytes(b"GGUF" + b"\0" * 100)
    (tmp_path / "M-Q8_0.gguf").write_bytes(b"GGUF" + b"\0" * 5000)
    (tmp_path / "unsloth.ini").write_text("temp = 0.4\n[Q8_0]\ntop-p = 0.8\n")
    from utils.models.model_config import detect_gguf_model

    assert Path(detect_gguf_model(str(tmp_path))).name == "M-Q8_0.gguf"
    found = mi.locate_model_ini(str(tmp_path))
    assert (found.gguf_filename, found.quant) == ("M-Q8_0.gguf", "Q8_0")


def test_oversized_remote_ini_is_refused_before_download(monkeypatch):
    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache", lambda *a, **k: None)
    monkeypatch.setattr(
        huggingface_hub,
        "get_hf_file_metadata",
        lambda url, token = None, **k: SimpleNamespace(size = MAX_MODEL_INI_BYTES + 1),
    )
    monkeypatch.setattr(
        huggingface_hub, "hf_hub_download", lambda **k: pytest.fail("downloaded an oversized INI")
    )
    with pytest.raises(ValueError, match = "larger than"):
        mi._hf_download("u/M-GGUF", "unsloth.ini", None, False)


def test_offline_lookup_never_asks_for_remote_metadata(monkeypatch):
    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache", lambda *a, **k: None)
    monkeypatch.setattr(
        huggingface_hub, "get_hf_file_metadata", lambda *a, **k: pytest.fail("network metadata")
    )

    def cache_only(**kw):
        assert kw["local_files_only"] is True
        raise type("LocalEntryNotFoundError", (Exception,), {})()

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", cache_only)
    assert mi._hf_download("u/M-GGUF", "unsloth.ini", None, True) is None


def test_runtime_fields_list_the_sampling_keys_the_ini_set():
    import routes.inference as routes

    backend = _Backend(["--temp", "0.6", "--temp", "0.3", "--top-k", "9"], ["--temp", "0.6"])
    assert routes._model_ini_sampling_keys(backend) == ["temperature"]
    assert routes._model_ini_sampling_keys(_Backend(["--ctx-size", "8"], ["--ctx-size", "8"])) == []
    from models.inference import InferenceStatusResponse, LoadResponse

    assert "model_ini_sampling_keys" in LoadResponse.model_fields
    assert "model_ini_sampling_keys" in InferenceStatusResponse.model_fields


@pytest.mark.parametrize("value, expected", [("auto", "-1"), ("all", "999"), ("ALL", "999")])
def test_gpu_layers_words_compile_to_integers_the_boundary_accepts(value, expected):
    compiled = parse_model_ini(f"ngl = {value}\n", quant = None, gguf_filename = None)
    assert compiled.args == ["--gpu-layers", expected]
    assert validate_extra_args(compiled.args) == compiled.args


def test_main_gpu_is_left_to_studios_gpu_selection():
    compiled = parse_model_ini("main-gpu = 1\nc = 4096\n", quant = None, gguf_filename = None)
    assert compiled.args == ["--ctx-size", "4096"]
    assert [(i["key"], i["reason"]) for i in compiled.ignored] == [
        ("main-gpu", "use Studio's GPU selection instead")
    ]


def test_typed_sampler_spellings_and_bounds_decide_what_is_promoted():
    from routes.inference import _with_model_ini_sampling

    base = {"temperature": 0.7, "top_p": 0.95}
    backend = _Backend(["--top-p", "0.8", "--top_p", "0.2"], ["--top-p", "0.8"])
    assert _with_model_ini_sampling(base, backend)["top_p"] == 0.2
    backend = _Backend(["--temp", "0.6", "--temp", "3"], ["--temp", "0.6"])
    assert _with_model_ini_sampling(base, backend) is base


def test_managed_offline_lookup_authorizes_offline(monkeypatch):
    import routes.models as models_routes

    _patch_locate(monkeypatch, "c = 4096\n")
    seen = {}
    monkeypatch.setattr(models_routes.account_access, "managed_account", lambda: True)
    monkeypatch.setattr(
        models_routes.account_access, "require_model_access", lambda repo, **k: seen.update(k)
    )
    monkeypatch.setattr(models_routes, "_resolve_hub_token", lambda header, token: None)
    body = asyncio.run(
        models_routes.get_model_ini(
            repo_id = "u/M-GGUF",
            gguf_variant = "Q8_0",
            local_path = None,
            hf_token = None,
            offline = True,
            native_path_lease = None,
            hf_token_header = None,
            current_subject = "member",
            via_api_key = False,
        )
    )
    assert body.found and seen == {"offline": True}


def test_hf_without_a_variant_uses_the_loaders_auto_pick(tmp_path):
    folder_ini = tmp_path / "a.ini"
    folder_ini.write_text("temp = 0.9\n")
    variants = [
        _variant("UD-Q4_K_XL/M-UD-Q4_K_XL.gguf", "UD-Q4_K_XL"),
        _variant("Q8_0/M-Q8_0.gguf", "Q8_0"),
    ]
    from utils.models.model_config import _pick_best_gguf

    best = _pick_best_gguf([v.filename for v in variants])
    folder = best.split("/")[0]
    lv, dl, seen = _hf({f"{folder}/unsloth.ini": str(folder_ini)}, variants)
    got = mi._locate_hf("u/M-GGUF", None, None, False, lv, dl)
    assert got.location == "variant_folder" and got.gguf_filename == best


def test_ini_prefix_is_stripped_after_placement_drops_one_of_its_flags():
    from routes.inference import _without_model_ini

    prefix = ["--ctx-size", "4096", "--split-mode", "row", "--temp", "0.6"]
    stored = ["--ctx-size", "4096", "--temp", "0.6", "--top-k", "3"]
    backend = _Backend(stored, prefix)
    assert _without_model_ini(backend, stored) == ["--top-k", "3"]
    assert _without_model_ini(backend, prefix + ["--seed", "1"]) == ["--seed", "1"]


def test_local_ini_symlink_leaving_the_model_folder_is_not_read(tmp_path):
    outside = tmp_path / "outside.txt"
    outside.write_text("secret = 1\n")
    model = tmp_path / "model"
    model.mkdir()
    (model / "M-Q8_0.gguf").write_bytes(b"GGUF")
    (model / "unsloth.ini").symlink_to(outside)
    assert mi.locate_model_ini(str(model / "M-Q8_0.gguf")) is None
    (model / "unsloth.ini").unlink()
    (model / "real.ini").write_text("c = 4096\n")
    (model / "unsloth.ini").symlink_to(model / "real.ini")
    assert "4096" in mi.locate_model_ini(str(model / "M-Q8_0.gguf")).text


@pytest.mark.parametrize("value", ["0,0", "1,3.5e38", "-1,2", "1e-40,1", "3e38,3e38"])
def test_degenerate_tensor_splits_are_ignored_not_fatal(value):
    compiled = parse_model_ini(f"ts = {value}\nc = 4096\n", quant = None, gguf_filename = None)
    assert compiled.args == ["--ctx-size", "4096"]
    assert [i["key"] for i in compiled.ignored] == ["ts"]
    validate_extra_args(compiled.args)


def test_frequency_penalty_is_left_to_the_chat_request():
    compiled = parse_model_ini("frequency-penalty = 0.5\n", quant = None, gguf_filename = None)
    assert compiled.args == [] and [i["key"] for i in compiled.ignored] == ["frequency-penalty"]


def test_hf_snapshot_ini_linking_into_its_repos_blobs_is_read(tmp_path):
    repo = tmp_path / "models--u--M-GGUF"
    (repo / "blobs").mkdir(parents = True)
    snap = repo / "snapshots" / "abc123"
    snap.mkdir(parents = True)
    (repo / "blobs" / "f00d").write_text("c = 4096\n")
    (repo / "blobs" / "beef").write_bytes(b"GGUF")
    (snap / "unsloth.ini").symlink_to(repo / "blobs" / "f00d")
    (snap / "M-Q8_0.gguf").symlink_to(repo / "blobs" / "beef")
    assert "4096" in mi.locate_model_ini(str(snap / "M-Q8_0.gguf")).text
    other = tmp_path / "models--x--Y" / "blobs"
    other.mkdir(parents = True)
    (other / "f00d").write_text("c = 1\n")
    (snap / "unsloth.ini").unlink()
    (snap / "unsloth.ini").symlink_to(other / "f00d")
    assert mi.locate_model_ini(str(snap / "M-Q8_0.gguf")) is None


def test_spec_type_is_left_to_studios_speculative_setting():
    compiled = parse_model_ini(
        "spec-type = draft-mtp\nspec-draft-n-max = 3\n", quant = None, gguf_filename = None
    )
    assert "--spec-type" not in compiled.args
    assert [i["key"] for i in compiled.ignored] == ["spec-type"]


def test_spec_default_is_left_to_studios_speculative_setting():
    compiled = parse_model_ini("spec-default = true\n", quant = None, gguf_filename = None)
    assert compiled.args == [] and [i["key"] for i in compiled.ignored] == ["spec-default"]


@pytest.mark.parametrize("text", ["b = 1\n", "np = 4\nb = 3\n"])
def test_batch_below_the_serving_floor_is_ignored(text):
    from core.inference.llama_server_args import check_batch_floor

    compiled = parse_model_ini(text + "c = 4096\n", quant = None, gguf_filename = None)
    assert compiled.args == ["--ctx-size", "4096"]
    assert [i["key"] for i in compiled.ignored] == ["b"]
    check_batch_floor(compiled.args, compiled.n_parallel or 1)


def test_gpu_layers_past_int32_are_ignored():
    compiled = parse_model_ini("ngl = 4294967296\n", quant = None, gguf_filename = None)
    assert compiled.args == [] and [i["key"] for i in compiled.ignored] == ["ngl"]
