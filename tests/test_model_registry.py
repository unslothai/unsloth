"""Register each model set and check the registered ids exist on the HF Hub."""

import os
import subprocess
import sys
from dataclasses import dataclass

import pytest
from huggingface_hub import HfApi
from huggingface_hub.errors import HfHubHTTPError, RepositoryNotFoundError

from unsloth.registry import register_models, search_models
from unsloth.registry._deepseek import register_deepseek_models
from unsloth.registry._gemma import register_gemma_models
from unsloth.registry._llama import LlamaModelInfo, register_llama_models
from unsloth.registry._mistral import register_mistral_models
from unsloth.registry._phi import register_phi_models
from unsloth.registry._qwen import register_qwen_models
from unsloth.registry.registry import (
    MODEL_REGISTRY,
    QUANT_TAG_MAP,
    ModelInfo,
    QuantType,
    register_model,
)

MODEL_NAMES = [
    "llama",
    "qwen",
    "mistral",
    "phi",
    "gemma",
    "deepseek",
]
MODEL_REGISTRATION_METHODS = [
    register_llama_models,
    register_qwen_models,
    register_mistral_models,
    register_phi_models,
    register_gemma_models,
    register_deepseek_models,
]


@dataclass
class ModelTestParam:
    name: str
    register_models: callable


class HubUnavailable(Exception):
    """The Hub could not answer, so the registry cannot be judged from here."""


def _model_is_missing(api: HfApi, model_id: str) -> bool:
    """Only RepositoryNotFoundError means missing; a 429, 5xx or network error must not count as missing."""
    try:
        api.model_info(model_id, expand = ["lastModified"])
    except RepositoryNotFoundError:
        return True
    except Exception as exc:
        raise HubUnavailable(f"{model_id}: {type(exc).__name__}: {exc}") from exc
    return False


def _test_model_uploaded(model_ids: list[str]):
    api = HfApi()
    missing_models = []
    for _id in model_ids:
        try:
            if _model_is_missing(api, _id):
                missing_models.append(_id)
        except HubUnavailable as exc:
            pytest.skip(f"Hugging Face Hub unavailable: {exc}")

    return missing_models


TestParams = [
    ModelTestParam(name, models) for name, models in zip(MODEL_NAMES, MODEL_REGISTRATION_METHODS)
]


@pytest.mark.parametrize("model_test_param", TestParams, ids = lambda param: param.name)
def test_model_registration(model_test_param: ModelTestParam):
    MODEL_REGISTRY.clear()
    registration_method = model_test_param.register_models
    registration_method()
    registered_models = MODEL_REGISTRY.keys()
    missing_models = _test_model_uploaded(registered_models)
    assert not missing_models, f"{model_test_param.name} missing following models: {missing_models}"


def test_all_model_registration():
    register_models()
    registered_models = MODEL_REGISTRY.keys()
    missing_models = _test_model_uploaded(registered_models)
    assert not missing_models, f"Missing following models: {missing_models}"


def test_unquantized_default_quant_type_is_usable():
    assert ModelInfo.append_quant_type("Llama-3.1-8B") == "Llama-3.1-8B"
    assert ModelInfo.append_quant_type("Llama-3.1-8B", None) == "Llama-3.1-8B"

    assert ModelInfo.append_quant_type("Llama-3.1-8B", QuantType.NONE) == "Llama-3.1-8B"
    assert (
        ModelInfo.append_quant_type("Llama-3.1-8B", QuantType.BNB)
        == "Llama-3.1-8B-" + QUANT_TAG_MAP[QuantType.BNB]
    )

    info = LlamaModelInfo(
        org = "unsloth", base_name = "Llama", version = "3.1", size = 8, instruct_tag = "Instruct"
    )
    assert info.quant_type == QuantType.NONE
    assert info.model_path == "unsloth/Llama-3.1-8B-Instruct"


def test_register_model_defaults_to_no_quantization():
    key = "unsloth/Llama-9.9-1B"
    MODEL_REGISTRY.pop(key, None)
    try:
        register_model(LlamaModelInfo, org = "unsloth", base_name = "Llama", version = "9.9", size = 1)
        assert key in MODEL_REGISTRY
        assert MODEL_REGISTRY[key].quant_type == QuantType.NONE
        assert key in [m.model_path for m in search_models(quant_types = [QuantType.NONE])]
    finally:
        MODEL_REGISTRY.pop(key, None)


def test_quant_type():
    # NOTE: for org="unsloth" models, QuantType.NONE aliases QuantType.UNSLOTH
    dynamic_quant_models = search_models(quant_types = [QuantType.UNSLOTH])
    assert all(m.quant_type == QuantType.UNSLOTH for m in dynamic_quant_models)
    quant_tag = QUANT_TAG_MAP[QuantType.UNSLOTH]
    assert all(quant_tag in m.model_path for m in dynamic_quant_models)


def _run_registry_child(body: str) -> subprocess.CompletedProcess:
    """Child imports conftest first, since unsloth.registry raises NotImplementedError on CPU-only CI."""
    tests_dir = os.path.dirname(os.path.abspath(__file__))
    prelude = (
        f"import sys; sys.path.insert(0, {tests_dir!r})\n"
        "try:\n"
        "    import conftest  # noqa: F401  GPU-free harness on no-accelerator runners\n"
        "except Exception:\n"
        "    pass\n"
    )
    return subprocess.run(
        [sys.executable, "-c", prelude + body],
        capture_output = True,
        text = True,
        check = False,
    )


_REGISTRY_LIFECYCLE = (
    "import unsloth.registry\n"
    "from unsloth.registry import register_models\n"
    "from unsloth.registry.registry import MODEL_REGISTRY\n"
    "print('REGISTRY_SIZE', len(MODEL_REGISTRY))\n"
    "register_models()\n"
    "orgs = sorted({m.org for m in MODEL_REGISTRY.values()})\n"
    "deepseek = [k for k in MODEL_REGISTRY if 'deepseek' in k.lower()]\n"
    "print('ORGS', orgs)\n"
    "print('NUM_DEEPSEEK', len(deepseek))"
)


@pytest.fixture(scope = "module")
def registry_lifecycle():
    """One module-scoped child serves both checks; nearly all its cost is import unsloth."""
    return _run_registry_child(_REGISTRY_LIFECYCLE)


def test_importing_registry_does_not_register_models(registry_lifecycle):
    """Importing unsloth.registry must not register models; _deepseek used to do so at module scope."""
    result = registry_lifecycle
    assert result.returncode == 0, (
        f"registry import subprocess exited {result.returncode}\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    size_lines = [line for line in result.stdout.splitlines() if line.startswith("REGISTRY_SIZE")]
    assert size_lines == ["REGISTRY_SIZE 0"], result.stdout + result.stderr


def test_register_models_registers_no_upstream_originals(registry_lifecycle):
    """register_models() must not leak upstream originals; an import-time deepseek call set its guards."""
    result = registry_lifecycle
    assert result.returncode == 0, (
        f"register_models subprocess exited {result.returncode}\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    out = result.stdout
    assert "ORGS ['unsloth']" in out, out + result.stderr
    deepseek_lines = [line for line in out.splitlines() if line.startswith("NUM_DEEPSEEK")]
    assert deepseek_lines and int(deepseek_lines[0].split()[1]) > 0, out + result.stderr


class _FakeApi:
    def __init__(self, error):
        self.error = error

    def model_info(
        self,
        model_id,
        expand = None,
    ):
        if self.error is not None:
            raise self.error
        return object()


def _hub_error(cls, message):
    """Skip __init__: HfHubHTTPError's signature differs between huggingface_hub 0.x and 1.x."""
    error = cls.__new__(cls)
    Exception.__init__(error, message)
    return error


def test_missing_repo_is_reported_missing():
    """A repo the Hub says does not exist is a registry error."""
    api = _FakeApi(_hub_error(RepositoryNotFoundError, "404 Client Error. Repository Not Found"))
    assert _model_is_missing(api, "unsloth/does-not-exist")


def test_present_repo_is_not_reported_missing():
    assert not _model_is_missing(_FakeApi(None), "unsloth/Qwen2.5-7B")


@pytest.mark.parametrize(
    "make_error",
    [
        lambda: _hub_error(HfHubHTTPError, "429 Client Error: Too Many Requests"),
        lambda: ConnectionError("Failed to establish a new connection"),
        lambda: TimeoutError("read timed out"),
    ],
    ids = ["rate_limited", "connection_refused", "timeout"],
)
def test_unreachable_hub_skips_instead_of_reporting_missing(monkeypatch, make_error):
    """A hub outage must not be reported as every registered model missing."""
    error = make_error()
    with pytest.raises(HubUnavailable):
        _model_is_missing(_FakeApi(error), "unsloth/Qwen2.5-7B")

    monkeypatch.setattr(sys.modules[__name__], "HfApi", lambda: _FakeApi(error))
    with pytest.raises(pytest.skip.Exception):
        _test_model_uploaded(["unsloth/Qwen2.5-7B"])


def test_qwen_2_5_coder_registered_without_dynamic_quants():
    from unsloth.registry._qwen import Qwen_2_5_CoderMeta
    from unsloth.registry.registry import _register_models

    MODEL_REGISTRY.clear()
    _register_models(Qwen_2_5_CoderMeta)
    assert "unsloth/Qwen2.5-Coder-7B-Instruct-bnb-4bit" in MODEL_REGISTRY
    assert not [k for k in MODEL_REGISTRY if "Coder" in k and k.endswith("unsloth-bnb-4bit")]
