import importlib
import pkgutil
import pytest

# List of all model modules in the unsloth/models directory
MODULES = [
    "rl",
    "vision",
    "llama",
    "loader",
    "loader_utils",
    "mapper",
    "mistral",
    "qwen2",
    "_utils",
    "cohere",
    "dpo",
    "gemma",
    "gemma2",
    "granite",
]


@pytest.mark.parametrize("module_name", MODULES)

def test_import_module(module_name):
    """Smoke test that each model module can be imported. Skips if no NVIDIA GPU is found."""
    full_module_name = f"unsloth.models.{module_name}"
    try:
        module = importlib.import_module(full_module_name)
    except NotImplementedError as e:
        if "No NVIDIA GPU" in str(e):
            pytest.skip("Skipping test because no NVIDIA GPU available")
        else:
            raise
    assert module is not None, f"Failed to import {full_module_name}"


def test_models_module_listing():
    """Test that all expected modules are present in unsloth/models package. Skips if no NVIDIA GPU is found."""
    try:
        import unsloth.models
    except NotImplementedError as e:
        if "No NVIDIA GPU" in str(e):
            pytest.skip("Skipping test because no NVIDIA GPU available")
        else:
            raise
    found_modules = [name for _, name, _ in pkgutil.iter_modules(unsloth.models.__path__)]
    for mod in MODULES:
        assert mod in found_modules, f"Module {mod} not found in unsloth.models package"