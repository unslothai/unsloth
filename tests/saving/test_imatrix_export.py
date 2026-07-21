"""CPU-only tests for the GGUF imatrix export option.

Cover imatrix_file resolution (path / *.gguf_file rename / True auto-download with mocked Hub),
the upstream unsloth/<base>-GGUF repo derivation, the conditional IQ-quant gate in save_to_gguf,
and that quantize_gguf / _quantize_q2_k_l actually emit --imatrix. No GPU, no real weights, no
real Hub or llama.cpp - the heavy bits are monkeypatched.
"""

from __future__ import annotations

import inspect
import os

import pytest

import unsloth.save as S
import unsloth_zoo.llama_cpp as L

# The --imatrix wiring lives in unsloth_zoo's quantize_gguf (a companion change). Where the
# installed unsloth_zoo predates it, skip the tests that require it rather than hard-failing CI.
_ZOO_HAS_IMATRIX = "imatrix" in inspect.signature(L.quantize_gguf).parameters
_needs_zoo_imatrix = pytest.mark.skipif(
    not _ZOO_HAS_IMATRIX,
    reason = "installed unsloth_zoo quantize_gguf has no imatrix kwarg (companion change not landed)",
)


class _Cfg:
    def __init__(self, name):
        self._name_or_path = name
        self.architectures = ["LlamaForCausalLM"]


class _Model:
    def __init__(self, name = "unsloth/Llama-3.1-8B-Instruct"):
        self.config = _Cfg(name)
        self.peft_config = {}


# -- registry + signatures -----------------------------------------------------------------


def test_public_savers_accept_imatrix_file():
    for fn in (S.unsloth_save_pretrained_gguf, S.unsloth_push_to_hub_gguf):
        assert "imatrix_file" in inspect.signature(fn).parameters, fn.__name__


@_needs_zoo_imatrix
def test_quantize_gguf_accepts_imatrix():
    assert "imatrix" in inspect.signature(L.quantize_gguf).parameters


def test_imatrix_quants_registry():
    for q in ("iq2_xxs", "iq4_xs", "iq1_s", "iq3_xxs"):
        assert q in S.IMATRIX_QUANTS
        assert (
            q not in S.ALLOWED_QUANTS
        ), f"{q} must be gated, not in the always-on allow-list"


# -- _resolve_imatrix_file -----------------------------------------------------------------


def test_resolve_none_and_false_return_none(tmp_path):
    assert S._resolve_imatrix_file(_Model(), None, None, str(tmp_path)) is None
    assert S._resolve_imatrix_file(_Model(), False, None, str(tmp_path)) is None


def test_resolve_bad_type_raises_typeerror(tmp_path):
    with pytest.raises(TypeError):
        S._resolve_imatrix_file(_Model(), 123, None, str(tmp_path))


def test_resolve_missing_path_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        S._resolve_imatrix_file(
            _Model(), str(tmp_path / "nope.dat"), None, str(tmp_path)
        )


def test_resolve_plain_path_passthrough(tmp_path):
    dat = tmp_path / "my_imatrix.dat"
    dat.write_bytes(b"x" * 32)
    assert S._resolve_imatrix_file(_Model(), str(dat), None, str(tmp_path)) == str(dat)


def test_resolve_gguf_file_is_renamed_to_gguf(tmp_path):
    src = tmp_path / "imatrix_unsloth.gguf_file"
    src.write_bytes(b"x" * 32)
    dest = tmp_path / "export"
    out = S._resolve_imatrix_file(_Model(), str(src), None, str(dest))
    assert out.endswith(".gguf") and not out.endswith(".gguf_file")
    assert os.path.isfile(out)


# -- repo derivation -----------------------------------------------------------------------


def test_repo_candidates_appends_gguf():
    repos = S._gguf_repo_candidates(_Model("unsloth/Llama-3.1-8B-Instruct"))
    assert "unsloth/Llama-3.1-8B-Instruct-GGUF" in repos


def test_repo_candidates_maps_official_base_to_unsloth_org():
    # The upstream imatrix only lives in unsloth/<base>-GGUF, so an official base id must map onto
    # the unsloth org rather than deriving a non-existent meta-llama/...-GGUF repo.
    repos = S._gguf_repo_candidates(_Model("meta-llama/Llama-3.1-8B-Instruct"))
    assert "unsloth/Llama-3.1-8B-Instruct-GGUF" in repos
    assert not any(r.startswith("meta-llama/") for r in repos)


def test_repo_candidates_keeps_existing_gguf_suffix():
    repos = S._gguf_repo_candidates(_Model("unsloth/Qwen3.6-35B-A3B-GGUF"))
    assert repos == ["unsloth/Qwen3.6-35B-A3B-GGUF"]


def test_repo_candidates_skips_local_dirs(tmp_path):
    assert S._gguf_repo_candidates(_Model(str(tmp_path))) == []


# -- True: auto-download (mocked Hub) ------------------------------------------------------


class _FakeApi:
    def __init__(self, files, **kw):
        self._files = files

    def list_repo_files(self, repo_id):
        return list(self._files.get(repo_id, []))


def _patch_hub(monkeypatch, files, downloaded_dir):
    # HfApi is the module-level name in unsloth.save; hf_hub_download is imported locally inside
    # the helper, so patch it on huggingface_hub. Both must be patched to stay fully offline.
    monkeypatch.setattr(S, "HfApi", lambda **kw: _FakeApi(files))

    def _fake_download(
        repo_id,
        filename,
        token = None,
        **kw,
    ):
        os.makedirs(downloaded_dir, exist_ok = True)
        path = os.path.join(downloaded_dir, filename)
        with open(path, "wb") as f:
            f.write(b"imatrix-bytes")
        return path

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", _fake_download)


def test_resolve_true_prefers_dat(monkeypatch, tmp_path):
    cache = tmp_path / "cache"
    files = {
        "unsloth/Llama-3.1-8B-Instruct-GGUF": [
            "imatrix_unsloth.dat",
            "imatrix_unsloth.gguf_file",
            "model.Q4_K_M.gguf",
        ]
    }
    _patch_hub(monkeypatch, files, str(cache))
    out = S._resolve_imatrix_file(_Model(), True, "tok", str(tmp_path / "dest"))
    assert os.path.basename(out) == "imatrix_unsloth.dat"
    # downloaded into the caller dest, not left only in the (fake) cache
    assert os.path.dirname(out) == str(tmp_path / "dest")


def test_resolve_true_downloads_gguf_file_and_renames(monkeypatch, tmp_path):
    cache = tmp_path / "cache"
    files = {"unsloth/Llama-3.1-8B-Instruct-GGUF": ["imatrix_unsloth.gguf_file"]}
    _patch_hub(monkeypatch, files, str(cache))
    out = S._resolve_imatrix_file(_Model(), True, "tok", str(tmp_path / "dest"))
    assert os.path.basename(out) == "imatrix_unsloth.gguf"


def test_resolve_true_missing_raises(monkeypatch, tmp_path):
    _patch_hub(
        monkeypatch,
        {"unsloth/Llama-3.1-8B-Instruct-GGUF": ["model.Q4_K_M.gguf"]},
        str(tmp_path),
    )
    with pytest.raises(RuntimeError) as e:
        S._resolve_imatrix_file(_Model(), True, "tok", str(tmp_path / "dest"))
    assert "imatrix" in str(e.value).lower()


# -- IQ gate in save_to_gguf ---------------------------------------------------------------


def test_iq_quant_without_imatrix_is_rejected():
    with pytest.raises(RuntimeError) as e:
        S.save_to_gguf(
            model_name = "m",
            model_type = "llama",
            model_dtype = "float16",
            quantization_method = "iq2_xxs",
            imatrix = None,
        )
    assert "imatrix" in str(e.value).lower()


def test_unknown_quant_is_rejected():
    with pytest.raises(RuntimeError):
        S.save_to_gguf(
            model_name = "m",
            model_type = "llama",
            model_dtype = "float16",
            quantization_method = "totally_bogus",
            imatrix = None,
        )


# -- --imatrix actually reaches llama-quantize ---------------------------------------------


@_needs_zoo_imatrix
def test_quantize_gguf_emits_imatrix_flag(monkeypatch, tmp_path):
    captured = {}

    def _fake_run(command, *a, **kw):
        captured["command"] = command
        # llama-quantize would write the output; emulate so the existence check passes.
        out = command.split()[-2] if False else None
        # output_gguf is the 2nd-to-last token before quant_type/threads; just create it.
        with open(tmp_path / "out.gguf", "wb") as f:
            f.write(b"GGUF")

        class R:
            returncode = 0
            stdout = ""

        return R()

    import shlex

    monkeypatch.setattr(L.subprocess, "run", _fake_run)
    imat = str(tmp_path / "imatrix it.dat")  # space in path -> must be shell-quoted
    with open(
        imat, "wb"
    ) as f:  # quantize_gguf validates the imatrix exists before running
        f.write(b"\x00")
    L.quantize_gguf(
        input_gguf = str(tmp_path / "in.gguf"),
        output_gguf = str(tmp_path / "out.gguf"),
        quant_type = "iq4_xs",
        quantizer_location = "llama-quantize",
        n_threads = 4,
        imatrix = imat,
        print_output = False,
    )
    cmd = captured["command"]
    assert "--imatrix" in cmd
    assert "iq4_xs" in cmd
    # the path with a space must appear shell-quoted (shlex.quote), never bare
    assert f"--imatrix {shlex.quote(imat)}" in cmd


def test_quantize_gguf_no_imatrix_has_no_flag(monkeypatch, tmp_path):
    captured = {}

    def _fake_run(command, *a, **kw):
        captured["command"] = command
        with open(tmp_path / "out.gguf", "wb") as f:
            f.write(b"GGUF")

        class R:
            returncode = 0
            stdout = ""

        return R()

    monkeypatch.setattr(L.subprocess, "run", _fake_run)
    L.quantize_gguf(
        input_gguf = str(tmp_path / "in.gguf"),
        output_gguf = str(tmp_path / "out.gguf"),
        quant_type = "q4_k_m",
        quantizer_location = "llama-quantize",
        n_threads = 4,
        print_output = False,
    )
    assert "--imatrix" not in captured["command"]
