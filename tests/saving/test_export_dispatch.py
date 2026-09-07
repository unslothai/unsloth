"""CPU-only behavioral routing tests for the export API.

With the heavy save helpers monkeypatched, confirm each `save_method` / `quantization_method`
reaches the correct export path with the correct arguments. A bare object stands in for the
model, so these run on CPU-only CI with no GPU and no real weights, yet they catch routing
regressions that pure AST checks cannot (e.g. wrong scheme/suffix/outtype passed through).
"""

from __future__ import annotations

import inspect
import os

import pytest

import unsloth.save as save_mod


class _FakeModel:
    """Minimal model stand-in; routing reads nothing meaningful off it before dispatch."""

    config = type(
        "cfg", (), {"_name_or_path": "fake/model", "architectures": ["LlamaForCausalLM"]}
    )()


class _FakeTokenizer:
    chat_template = None

    def __init__(self):
        self.saved_to = []

    def save_pretrained(self, path):
        self.saved_to.append(path)


# -- merged_*  ->  compressed-tensors dispatch ---------------------------------------------


def test_merged_fp8_routes_to_compressed(monkeypatch, tmp_path):
    seen = {}
    monkeypatch.setattr(save_mod, "_unsloth_save_compressed_tensors", lambda **kw: seen.update(kw))
    monkeypatch.setattr(save_mod, "unsloth_generic_save", lambda **kw: seen.update(generic = True))
    save_mod.unsloth_generic_save_pretrained_merged(
        _FakeModel(),
        str(tmp_path),
        tokenizer = object(),
        save_method = "fp8",
    )
    assert seen.get("scheme") == "FP8_DYNAMIC"
    assert seen.get("suffix") == "fp8"
    assert seen.get("needs_calibration") is False
    assert "generic" not in seen, "compressed save_method must not fall through to the plain merge"


def test_merged_nvfp4_marks_calibration(monkeypatch, tmp_path):
    seen = {}
    monkeypatch.setattr(save_mod, "_unsloth_save_compressed_tensors", lambda **kw: seen.update(kw))
    monkeypatch.setattr(save_mod, "unsloth_generic_save", lambda **kw: None)
    save_mod.unsloth_generic_save_pretrained_merged(
        _FakeModel(),
        str(tmp_path),
        tokenizer = object(),
        save_method = "nvfp4",
    )
    assert seen.get("scheme") == "NVFP4"
    assert seen.get("needs_calibration") is True


def test_merged_16bit_does_not_route_compressed(monkeypatch, tmp_path):
    calls = {"compressed": 0, "generic": 0}
    monkeypatch.setattr(
        save_mod,
        "_unsloth_save_compressed_tensors",
        lambda **kw: calls.__setitem__("compressed", calls["compressed"] + 1),
    )
    monkeypatch.setattr(
        save_mod,
        "unsloth_generic_save",
        lambda **kw: calls.__setitem__("generic", calls["generic"] + 1),
    )
    save_mod.unsloth_generic_save_pretrained_merged(
        _FakeModel(),
        str(tmp_path),
        tokenizer = object(),
        save_method = "merged_16bit",
    )
    assert calls["compressed"] == 0, "merged_16bit must not hit the compressed export"
    assert calls["generic"] == 1, "merged_16bit must go through the normal merge path"


# -- save_method='lora'  ->  LoRA GGUF dispatch --------------------------------------------


def test_gguf_lora_passes_valid_outtype(monkeypatch, tmp_path):
    seen = {}
    monkeypatch.setattr(
        save_mod,
        "_unsloth_save_lora_gguf",
        lambda *_a, **kw: seen.update(kw),
    )
    save_mod.unsloth_save_pretrained_gguf(
        _FakeModel(),
        str(tmp_path),
        tokenizer = object(),
        save_method = "lora",
        quantization_method = "q8_0",
    )
    assert seen.get("outtype") == "q8_0"


def test_gguf_lora_invalid_outtype_falls_back_to_f16(monkeypatch, tmp_path):
    seen = {}
    monkeypatch.setattr(
        save_mod,
        "_unsloth_save_lora_gguf",
        lambda *_a, **kw: seen.update(kw),
    )
    save_mod.unsloth_save_pretrained_gguf(
        _FakeModel(),
        str(tmp_path),
        tokenizer = object(),
        save_method = "lora",
        quantization_method = "q4_k_m",
    )
    assert (
        seen.get("outtype") == "f16"
    ), "a GGUF model quant (q4_k_m) is not a valid LoRA outtype -> f16"


@pytest.mark.parametrize("token", [False, "caller-token", None])
def test_gguf_lora_forwards_the_caller_token(monkeypatch, tmp_path, token):
    """Dropping it here sends _unsloth_save_lora_gguf to get_token(), i.e. the host credential."""
    seen = {}
    monkeypatch.setattr(
        save_mod,
        "_unsloth_save_lora_gguf",
        lambda *_a, **kw: seen.update(kw),
    )
    save_mod.unsloth_save_pretrained_gguf(
        _FakeModel(),
        str(tmp_path),
        tokenizer = object(),
        save_method = "lora",
        quantization_method = "q8_0",
        token = token,
    )
    assert seen["token"] is token


def test_gguf_lora_push_to_hub_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        save_mod.unsloth_save_pretrained_gguf(
            _FakeModel(),
            "repo/id",
            tokenizer = object(),
            save_method = "lora",
            push_to_hub = True,
        )


@pytest.mark.parametrize("trailing_separator", [False, True])
def test_non_peft_gguf_uses_checkpoint_as_input_not_output(
    monkeypatch, tmp_path, trailing_separator
):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    requested = tmp_path / "export" / "model"
    requested.parent.mkdir()
    model = _FakeModel()
    model.config = type(
        "cfg",
        (),
        {
            "_name_or_path": str(checkpoint),
            "architectures": ["LlamaForCausalLM"],
            "model_type": "llama",
        },
    )()
    tokenizer = _FakeTokenizer()
    seen = {}

    monkeypatch.setattr(save_mod, "_is_vlm", lambda _model: False)
    monkeypatch.setattr(save_mod, "_is_gpt_oss", lambda _model: False)
    monkeypatch.setattr(save_mod, "fix_tokenizer_bos_token", lambda _tokenizer: (False, None))
    monkeypatch.setattr(save_mod, "_resolve_imatrix_file", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(save_mod, "dtype_from_config", lambda _config: save_mod.torch.float16)
    monkeypatch.setattr(save_mod, "create_ollama_modelfile", lambda *_args, **_kwargs: None)

    def _save_to_gguf(**kwargs):
        seen.update(kwargs)
        output = tmp_path / "export" / "model_gguf" / "model.Q8_0.gguf"
        output.parent.mkdir()
        output.write_bytes(b"GGUF")
        return [str(output)], True, False

    monkeypatch.setattr(save_mod, "save_to_gguf", _save_to_gguf)

    requested_arg = f"{requested}{os.sep}" if trailing_separator else str(requested)
    result = save_mod.unsloth_save_pretrained_gguf(
        model,
        requested_arg,
        tokenizer = tokenizer,
        quantization_method = "q8_0",
    )

    assert seen["model_directory"] == str(checkpoint)
    assert seen["gguf_directory"] == f"{requested}_gguf"
    assert result["gguf_directory"] == f"{requested}_gguf"
    assert tokenizer.saved_to == [str(checkpoint)]


# The above rejection points users at push_to_hub_gguf(save_method='lora'), so that path has to work; it is only ever
# exercised here.


def test_push_to_hub_gguf_lora_dispatches(monkeypatch):
    seen = {}
    monkeypatch.setattr(
        save_mod,
        "_unsloth_save_lora_gguf",
        lambda model, tok, sd, **kw: seen.update(kw),
    )
    save_mod.unsloth_push_to_hub_gguf(
        _FakeModel(),
        "repo/id",
        tokenizer = object(),
        save_method = "lora",
        quantization_method = "q8_0",
    )
    assert seen.get("outtype") == "q8_0"
    assert seen.get("push_to_hub") is True


def test_push_to_hub_gguf_lora_skips_non_main_process(monkeypatch):
    calls = []
    monkeypatch.setattr(
        save_mod,
        "_unsloth_save_lora_gguf",
        lambda *a, **kw: calls.append(kw),
    )
    result = save_mod.unsloth_push_to_hub_gguf(
        _FakeModel(),
        "repo/id",
        tokenizer = object(),
        save_method = "lora",
        is_main_process = False,
    )
    assert result is None
    assert calls == []


def test_push_to_hub_gguf_skips_non_main_process_before_merged_conversion(monkeypatch):
    calls = []
    monkeypatch.setattr(
        save_mod,
        "unsloth_save_pretrained_gguf",
        lambda **kw: calls.append(kw),
    )
    result = save_mod.unsloth_push_to_hub_gguf(
        _FakeModel(),
        "repo/id",
        tokenizer = object(),
        is_main_process = False,
    )
    assert result is None
    assert calls == []


def test_push_to_hub_gguf_preserves_positional_max_shard_size():
    bound = inspect.signature(save_mod.unsloth_push_to_hub_gguf).bind(
        _FakeModel(),
        "repo/id",
        object(),
        "q4_k_m",
        None,
        None,
        None,
        None,
        "token",
        "50GB",
    )
    assert bound.arguments["max_shard_size"] == "50GB"
    assert "is_main_process" not in bound.arguments


# -- torchao PTQ / QAT dispatch ------------------------------------------------------------


def test_torchao_ptq_routes_to_given_config(monkeypatch, tmp_path):
    seen = {}
    monkeypatch.setattr(
        save_mod, "_unsloth_save_torchao_with_given_config", lambda **kw: seen.update(given = True)
    )
    monkeypatch.setattr(
        save_mod,
        "_unsloth_save_torchao_with_attached_config",
        lambda **kw: seen.update(attached = True),
    )
    save_mod.unsloth_save_pretrained_torchao(
        _FakeModel(),
        str(tmp_path),
        tokenizer = object(),
        torchao_config = object(),
    )
    assert seen.get("given") and not seen.get("attached")


def test_torchao_qat_routes_to_attached_config(monkeypatch, tmp_path):
    seen = {}
    monkeypatch.setattr(
        save_mod, "_unsloth_save_torchao_with_given_config", lambda **kw: seen.update(given = True)
    )
    monkeypatch.setattr(
        save_mod,
        "_unsloth_save_torchao_with_attached_config",
        lambda **kw: seen.update(attached = True),
    )
    model = _FakeModel()
    model._torchao_config = object()  # simulates a model trained with qat_scheme
    save_mod.unsloth_save_pretrained_torchao(
        model,
        str(tmp_path),
        tokenizer = object(),
        torchao_config = None,
    )
    assert seen.get("attached") and not seen.get("given")


def test_torchao_requires_config_or_qat(tmp_path):
    # No torchao_config and no attached QAT config is a user error, surfaced eagerly.
    with pytest.raises(AssertionError):
        save_mod.unsloth_save_pretrained_torchao(
            _FakeModel(),
            str(tmp_path),
            tokenizer = object(),
            torchao_config = None,
        )


def _run_lora_gguf(monkeypatch, tmp_path, token):
    """Drive _unsloth_save_lora_gguf to the converter call and return the env it would use."""
    captured = {}

    class _FakePeft:
        config = _FakeModel.config

    class _FakePopen:
        def __init__(self, _cmd, **kwargs):
            captured["env"] = kwargs["env"]
            self.stdout = []
            self.returncode = 0

        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return False

        def wait(self):
            return 0

    llama_dir = tmp_path / "llama.cpp"
    llama_dir.mkdir()
    (llama_dir / "convert_lora_to_gguf.py").write_text("", encoding = "utf-8")

    monkeypatch.setattr(save_mod, "PeftModelForCausalLM", _FakePeft)
    monkeypatch.setattr(save_mod, "LLAMA_CPP_DEFAULT_DIR", str(llama_dir))
    monkeypatch.setattr(save_mod, "save_lora_to_custom_dir", lambda *_a: None)
    monkeypatch.setattr(save_mod, "install_llama_cpp", lambda **_kw: None)
    monkeypatch.setattr(save_mod, "_lora_base_model_id", lambda _m: "org/private-base")
    monkeypatch.setattr(save_mod, "_loaded_via_remote_code", lambda _m: False)
    monkeypatch.setattr(save_mod, "get_token", lambda: "host-ambient-token")
    monkeypatch.setattr(save_mod.subprocess, "Popen", _FakePopen)
    monkeypatch.setenv("HF_TOKEN", "host-ambient-token")
    monkeypatch.setenv("HUGGINGFACEHUB_API_TOKEN", "host-legacy-alias")
    monkeypatch.setenv("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")

    save_mod._unsloth_save_lora_gguf(
        _FakePeft(), _FakeTokenizer(), str(tmp_path / "out"), outtype = "f16", token = token
    )
    return captured["env"]


def test_lora_gguf_converter_is_denied_the_host_token(monkeypatch, tmp_path):
    env = _run_lora_gguf(monkeypatch, tmp_path, token = False)
    for key in save_mod._HF_TOKEN_ENV_KEYS:
        assert key not in env, f"{key} survived into a forced-anonymous converter"
    assert env["HF_HUB_DISABLE_IMPLICIT_TOKEN"] == "1", "the cached token is still implicit"
    # Scrubbing the env still leaves the operator's token FILE readable by get_token().
    assert env["HF_TOKEN_PATH"] == os.devnull


def test_lora_gguf_converter_gets_an_explicit_token(monkeypatch, tmp_path):
    env = _run_lora_gguf(monkeypatch, tmp_path, token = "caller-token")
    assert env["HF_TOKEN"] == "caller-token"
    assert env["HUGGING_FACE_HUB_TOKEN"] == "caller-token"
    assert "HUGGINGFACEHUB_API_TOKEN" not in env
    assert env["HF_HUB_DISABLE_IMPLICIT_TOKEN"] == "0"


def test_lora_gguf_converter_keeps_the_ambient_token_when_none(monkeypatch, tmp_path):
    env = _run_lora_gguf(monkeypatch, tmp_path, token = None)
    assert env["HF_TOKEN"] == "host-ambient-token"


def test_lora_gguf_converter_does_not_overrule_the_operator_optout(monkeypatch, tmp_path):
    # get_token() ignores the flag, so a caller who passed nothing holds the token the operator
    # switched off; only a token they supplied earns clearing it.
    env = _run_lora_gguf(monkeypatch, tmp_path, token = None)
    assert env["HF_HUB_DISABLE_IMPLICIT_TOKEN"] == "1"
    assert env["HUGGINGFACEHUB_API_TOKEN"] == "host-legacy-alias"


@pytest.mark.parametrize(
    "token,expected",
    [
        ("", None),
        ("   ", None),
        ("  hf_caller  ", "hf_caller"),
        (None, None),
        (False, False),
        (True, True),
    ],
)
def test_clean_save_token(token, expected):
    # Blank reaches HfApi as a literal "Bearer " header, which 1.x rejects. False must survive:
    # collapsing it to None is the ambient token, not anonymity.
    result = save_mod._clean_save_token(token)
    assert result is expected if expected in (None, False, True) else result == expected


@pytest.mark.parametrize("blank", ["", "   "])
def test_lora_gguf_converter_reads_a_blank_token_as_absent(monkeypatch, tmp_path, blank):
    env = _run_lora_gguf(monkeypatch, tmp_path, token = blank)
    assert env["HF_TOKEN"] == "host-ambient-token"
    assert env["HF_HUB_DISABLE_IMPLICIT_TOKEN"] == "1"
    assert env["HUGGINGFACEHUB_API_TOKEN"] == "host-legacy-alias"


def test_lora_gguf_converter_denies_the_oidc_material(monkeypatch, tmp_path):
    # hub >= 1.19 exchanges these inside get_token() ahead of HF_TOKEN, so scrubbing the aliases
    # alone still lets a denied child mint one.
    monkeypatch.setenv("HF_OIDC_RESOURCE", "https://huggingface.co")
    monkeypatch.setenv("HF_OIDC_ID_TOKEN", "operator-oidc-assertion")
    env = _run_lora_gguf(monkeypatch, tmp_path, token = False)
    assert "HF_OIDC_RESOURCE" not in env
    assert "HF_OIDC_ID_TOKEN" not in env


def test_lora_gguf_converter_honours_token_true(monkeypatch, tmp_path):
    # True means "use the cached token" and outranks the flag; falling through every branch made
    # it plain inheritance, which an ambient =1 voided.
    env = _run_lora_gguf(monkeypatch, tmp_path, token = True)
    assert env["HF_TOKEN"] == "host-ambient-token"
    assert env["HF_HUB_DISABLE_IMPLICIT_TOKEN"] == "0"


@pytest.mark.parametrize(
    "token,explicit,expected",
    [
        (False, True, {"scrubbed": True, "granted": None, "implicit": "1"}),
        ("caller", True, {"scrubbed": True, "granted": "caller", "implicit": "0"}),
        ("ambient", False, {"scrubbed": False, "granted": "ambient", "implicit": None}),
        (None, False, {"scrubbed": False, "granted": None, "implicit": None}),
    ],
)
def test_apply_token_to_child_env(token, explicit, expected):
    env = {
        "HF_TOKEN": "operator",
        "HUGGINGFACEHUB_API_TOKEN": "operator-legacy",
        "HF_OIDC_RESOURCE": "https://huggingface.co",
        "PATH": "/usr/bin",
        "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1",
    }
    save_mod._apply_token_to_child_env(env, token, explicit = explicit)

    assert env["PATH"] == "/usr/bin", "an unrelated variable was disturbed"
    if expected["scrubbed"]:
        assert "HUGGINGFACEHUB_API_TOKEN" not in env
        assert "HF_OIDC_RESOURCE" not in env
    else:
        assert env["HUGGINGFACEHUB_API_TOKEN"] == "operator-legacy"
        assert env["HF_OIDC_RESOURCE"] == "https://huggingface.co"
    if expected["granted"] is None:
        assert env.get("HF_TOKEN") in (None, "operator")
    else:
        assert env["HF_TOKEN"] == expected["granted"]
        assert env["HUGGING_FACE_HUB_TOKEN"] == expected["granted"]
    # None means "leave the inherited flag exactly as the operator set it".
    assert env["HF_HUB_DISABLE_IMPLICIT_TOKEN"] == (expected["implicit"] or "1")


def test_every_converter_child_env_goes_through_the_token_boundary():
    """No child env in save.py may be built without applying the caller boundary to it.

    Structural rather than textual: a third subprocess added next to these two would otherwise
    repeat the leak silently, which is how _unsloth_save_compressed_tensors came to have it.
    """
    import ast
    import pathlib

    def _is_environ_copy(node):
        # os.environ.copy() exactly -- not os.environ.get(...) next to some other .copy().
        return (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "copy"
            and isinstance(node.func.value, ast.Attribute)
            and node.func.value.attr == "environ"
            and isinstance(node.func.value.value, ast.Name)
            and node.func.value.value.id == "os"
        )

    tree = ast.parse(pathlib.Path(save_mod.__file__).read_text(encoding = "utf-8"))
    builders, offenders = [], []
    for func in ast.walk(tree):
        if not isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if not any(_is_environ_copy(node) for node in ast.walk(func)):
            continue
        builders.append(func.name)
        if "_apply_token_to_child_env" not in ast.dump(func):
            offenders.append(f"{func.name} (line {func.lineno})")

    assert builders, "the AST matcher found no child-env builders at all; it has drifted"
    assert not offenders, (
        "these build a child env without applying the token boundary: " + ", ".join(offenders)
    )
