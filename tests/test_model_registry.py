"""Register each model set and check the registered ids exist on the HF Hub."""

import os
import subprocess
import sys
from dataclasses import dataclass

import pytest
from huggingface_hub import ModelInfo as HfModelInfo

from unsloth.registry import register_models, search_models
from unsloth.registry._deepseek import register_deepseek_models
from unsloth.registry._gemma import register_gemma_models
from unsloth.registry._llama import register_llama_models
from unsloth.registry._mistral import register_mistral_models
from unsloth.registry._phi import register_phi_models
from unsloth.registry._qwen import register_qwen_models
from unsloth.registry.registry import MODEL_REGISTRY, QUANT_TAG_MAP, QuantType
from unsloth.utils.hf_hub import get_model_info

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


def _test_model_uploaded(model_ids: list[str]):
    missing_models = []
    for _id in model_ids:
        model_info: HfModelInfo = get_model_info(_id)
        if not model_info:
            missing_models.append(_id)

    return missing_models


TestParams = [
    ModelTestParam(name, models)
    for name, models in zip(MODEL_NAMES, MODEL_REGISTRATION_METHODS)
]


@pytest.mark.parametrize("model_test_param", TestParams, ids = lambda param: param.name)
def test_model_registration(model_test_param: ModelTestParam):
    MODEL_REGISTRY.clear()
    registration_method = model_test_param.register_models
    registration_method()
    registered_models = MODEL_REGISTRY.keys()
    missing_models = _test_model_uploaded(registered_models)
    assert (
        not missing_models
    ), f"{model_test_param.name} missing following models: {missing_models}"


def test_all_model_registration():
    register_models()
    registered_models = MODEL_REGISTRY.keys()
    missing_models = _test_model_uploaded(registered_models)
    assert not missing_models, f"Missing following models: {missing_models}"


def test_quant_type():
    # NOTE: for org="unsloth" models, QuantType.NONE aliases QuantType.UNSLOTH
    dynamic_quant_models = search_models(quant_types = [QuantType.UNSLOTH])
    assert all(m.quant_type == QuantType.UNSLOTH for m in dynamic_quant_models)
    quant_tag = QUANT_TAG_MAP[QuantType.UNSLOTH]
    assert all(quant_tag in m.model_path for m in dynamic_quant_models)


def _run_registry_child(body: str) -> subprocess.CompletedProcess:
    """Run ``body`` in a fresh interpreter that first imports this directory's
    ``conftest`` so it inherits the same GPU-free harness the pytest session
    uses (device_type stubs plus torch.cuda probe patches). Without it,
    ``import unsloth.registry`` raises ``NotImplementedError`` from
    ``unsloth_zoo.device_type`` on no-accelerator CI runners, so the child
    would exit non-zero and the test would fail even though the registry code
    is correct. A fresh process also keeps each check independent of any
    ``register_models()`` calls other tests make on the shared registry.
    """
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


def test_importing_registry_does_not_register_models():
    """Importing the registry must not populate MODEL_REGISTRY on its own.

    ``_deepseek`` used to call ``register_deepseek_models(...)`` at module
    scope, so merely importing ``unsloth.registry`` registered models as an
    import side effect, unlike every other family which only registers on
    demand.
    """
    result = _run_registry_child(
        "import unsloth.registry\n"
        "from unsloth.registry.registry import MODEL_REGISTRY\n"
        "print('REGISTRY_SIZE', len(MODEL_REGISTRY))"
    )
    assert result.returncode == 0, (
        f"registry import subprocess exited {result.returncode}\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    size_lines = [
        line for line in result.stdout.splitlines() if line.startswith("REGISTRY_SIZE")
    ]
    assert size_lines == ["REGISTRY_SIZE 0"], result.stdout + result.stderr


def test_register_models_registers_no_upstream_originals():
    """``register_models()`` must register each family's ``unsloth``-org models
    and must NOT leak upstream vendor "original" models.

    Before the fix, ``_deepseek``'s import-time
    ``register_deepseek_models(include_original_model = True)`` set the
    ``_IS_DEEPSEEK_*_REGISTERED`` guards, so the later default
    ``register_models()`` early-returned for deepseek and its 10 ``deepseek-ai``
    originals leaked permanently (129 -> 139). This asserts the whole registry
    is ``unsloth``-org after ``register_models()`` while deepseek is still
    registered via the normal path. Runs in a fresh interpreter so it is
    independent of other tests' registry mutations.
    """
    result = _run_registry_child(
        "import unsloth.registry\n"
        "from unsloth.registry import register_models\n"
        "from unsloth.registry.registry import MODEL_REGISTRY\n"
        "register_models()\n"
        "orgs = sorted({m.org for m in MODEL_REGISTRY.values()})\n"
        "deepseek = [k for k in MODEL_REGISTRY if 'deepseek' in k.lower()]\n"
        "print('ORGS', orgs)\n"
        "print('NUM_DEEPSEEK', len(deepseek))"
    )
    assert result.returncode == 0, (
        f"register_models subprocess exited {result.returncode}\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    out = result.stdout
    # Every registered model is unsloth-org: no upstream "original" leaked.
    assert "ORGS ['unsloth']" in out, out + result.stderr
    # Deepseek is still registered via the normal path, just without originals.
    deepseek_lines = [
        line for line in out.splitlines() if line.startswith("NUM_DEEPSEEK")
    ]
    assert deepseek_lines and int(deepseek_lines[0].split()[1]) > 0, out + result.stderr
