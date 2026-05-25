import sys
import types

from unsloth_cli.commands import inference as inference_cmd


class _FakeModelConfig:
    def __init__(self, *, is_gguf: bool):
        self.is_gguf = is_gguf
        self.gguf_file = "/tmp/model.gguf" if is_gguf else None
        self.gguf_mmproj_file = None
        self.gguf_hf_repo = None
        self.gguf_variant = None
        self.identifier = "fake/model"
        self.is_vision = False


class _FakeModelConfigType:
    next_config = None

    @classmethod
    def from_identifier(cls, **kwargs):
        return cls.next_config


class _FakeStandardBackend:
    def __init__(self):
        self.loaded = None
        self.generated = None

    def load_model(self, **kwargs):
        self.loaded = kwargs
        return True

    def generate_chat_response(self, **kwargs):
        self.generated = kwargs
        yield "hello"


class _FakeLlamaBackend:
    last_instance = None

    def __init__(self):
        self.loaded = None
        self.generated = None
        _FakeLlamaBackend.last_instance = self

    def load_model(self, **kwargs):
        self.loaded = kwargs
        return True

    def generate_chat_completion(self, **kwargs):
        self.generated = kwargs
        yield "hello"


def _install_fake_modules(monkeypatch, *, model_config, standard_backend):
    core_module = types.ModuleType("studio.backend.core")
    core_module.ModelConfig = _FakeModelConfigType
    core_module.get_inference_backend = lambda: standard_backend

    llama_module = types.ModuleType("studio.backend.core.inference.llama_cpp")
    llama_module.LlamaCppBackend = _FakeLlamaBackend

    _FakeModelConfigType.next_config = model_config
    monkeypatch.setitem(sys.modules, "studio.backend.core", core_module)
    monkeypatch.setitem(
        sys.modules, "studio.backend.core.inference.llama_cpp", llama_module
    )


def test_standard_models_use_unsloth_backend(monkeypatch, capsys):
    monkeypatch.setattr(
        inference_cmd, "_reexec_cli_in_studio_venv", lambda *args, **kwargs: None
    )
    standard_backend = _FakeStandardBackend()
    _install_fake_modules(
        monkeypatch,
        model_config = _FakeModelConfig(is_gguf = False),
        standard_backend = standard_backend,
    )

    inference_cmd.inference(model = "fake/model", prompt = "hello")

    assert standard_backend.loaded is not None
    assert standard_backend.generated is not None
    assert _FakeLlamaBackend.last_instance is None
    out = capsys.readouterr().out
    assert "Assistant:" in out
    assert "hello" in out


def test_gguf_models_use_llama_cpp_backend(monkeypatch, capsys):
    monkeypatch.setattr(
        inference_cmd, "_reexec_cli_in_studio_venv", lambda *args, **kwargs: None
    )
    standard_backend = _FakeStandardBackend()
    _install_fake_modules(
        monkeypatch,
        model_config = _FakeModelConfig(is_gguf = True),
        standard_backend = standard_backend,
    )

    inference_cmd.inference(model = "fake/model-gguf", prompt = "hello")

    assert standard_backend.loaded is None
    assert standard_backend.generated is None
    llama_backend = _FakeLlamaBackend.last_instance
    assert llama_backend is not None
    assert llama_backend.loaded is not None
    assert llama_backend.loaded["gguf_path"] == "/tmp/model.gguf"
    assert llama_backend.generated is not None
    assert llama_backend.generated["messages"] == [{"role": "user", "content": "hello"}]
    out = capsys.readouterr().out
    assert "Assistant:" in out
    assert "hello" in out
