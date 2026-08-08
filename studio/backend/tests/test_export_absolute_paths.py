# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import importlib.machinery
import importlib.util
import shutil
import sys
import types
from pathlib import Path

import pytest


_BACKEND_DIR = Path(__file__).resolve().parent.parent


def _load_module(
    module_name: str,
    relative_path: str,
    monkeypatch = None,
):
    path = _BACKEND_DIR / relative_path
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    if monkeypatch is None:
        sys.modules[module_name] = module
    else:
        monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    return module


class _DummyLogger:
    def __getattr__(self, _name):
        return lambda *args, **kwargs: None


class _Router:
    def get(self, *args, **kwargs):
        return lambda fn: fn

    def post(self, *args, **kwargs):
        return lambda fn: fn

    def delete(self, *args, **kwargs):
        return lambda fn: fn


class _HTTPException(Exception):
    def __init__(
        self,
        status_code: int,
        detail: str | None = None,
    ):
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


class _LocalModelInfo:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def _identity_decorator(*_args, **_kwargs):
    return lambda fn: fn


def _install_lightweight_backend_stubs(monkeypatch):
    fastapi = types.ModuleType("fastapi")
    fastapi.APIRouter = lambda: _Router()
    fastapi.Body = lambda default = None, **_kwargs: default
    fastapi.Depends = lambda dependency = None, **_kwargs: dependency
    fastapi.Header = lambda default = None, **_kwargs: default
    fastapi.HTTPException = _HTTPException
    fastapi.Query = lambda default = None, **_kwargs: default
    fastapi.Request = object
    monkeypatch.setitem(sys.modules, "fastapi", fastapi)

    fastapi_responses = types.ModuleType("fastapi.responses")
    fastapi_responses.StreamingResponse = object
    monkeypatch.setitem(sys.modules, "fastapi.responses", fastapi_responses)

    monkeypatch.setitem(
        sys.modules,
        "structlog",
        types.SimpleNamespace(
            BoundLogger = _DummyLogger,
            get_logger = lambda *args, **kwargs: _DummyLogger(),
        ),
    )
    loggers = types.ModuleType("loggers")
    loggers.get_logger = lambda *args, **kwargs: _DummyLogger()
    monkeypatch.setitem(sys.modules, "loggers", loggers)

    auth_pkg = types.ModuleType("auth")
    auth_mod = types.ModuleType("auth.authentication")
    auth_mod.get_current_subject = lambda: None
    monkeypatch.setitem(sys.modules, "auth", auth_pkg)
    monkeypatch.setitem(sys.modules, "auth.authentication", auth_mod)

    core_pkg = types.ModuleType("core")
    core_pkg.__path__ = []
    core_export = types.ModuleType("core.export")
    core_export.get_export_backend = lambda: None
    core_inference = types.ModuleType("core.inference")
    core_inference.__path__ = []
    core_inference.get_inference_backend = lambda: None
    monkeypatch.setitem(sys.modules, "core", core_pkg)
    monkeypatch.setitem(sys.modules, "core.export", core_export)
    monkeypatch.setitem(sys.modules, "core.inference", core_inference)
    _load_module(
        "core.inference.model_ids",
        "core/inference/model_ids.py",
        monkeypatch,
    )

    utils_pkg = types.ModuleType("utils")
    utils_pkg.__path__ = []
    utils_paths = types.ModuleType("utils.paths")
    storage_roots = _load_module(
        "utils.paths.storage_roots",
        "utils/paths/storage_roots.py",
        monkeypatch,
    )
    utils_pkg.paths = utils_paths
    utils_paths.storage_roots = storage_roots
    utils_paths.is_local_path = lambda value: Path(str(value)).is_absolute()
    utils_paths.normalize_path = lambda value: value
    utils_paths.outputs_root = lambda: Path("outputs")
    utils_paths.exports_root = storage_roots.exports_root
    utils_paths.resolve_cached_repo_id_case = lambda value: value
    utils_paths.resolve_output_dir = lambda value = None: Path(value or "outputs")
    utils_paths.resolve_export_dir = storage_roots.resolve_export_dir
    monkeypatch.setitem(sys.modules, "utils", utils_pkg)
    monkeypatch.setitem(sys.modules, "utils.paths", utils_paths)

    utils_utils = types.ModuleType("utils.utils")
    utils_utils.log_and_http_error = lambda *args, **kwargs: (_ for _ in ()).throw(
        _HTTPException(kwargs.get("status_code", 500), kwargs.get("detail"))
    )
    utils_utils.canonical_model_repo_id = lambda value: value
    utils_utils.safe_error_detail = lambda value: str(value)
    monkeypatch.setitem(sys.modules, "utils.utils", utils_utils)

    utils_models = types.ModuleType("utils.models")
    utils_models.__path__ = []
    for name in (
        "scan_trained_models",
        "scan_exported_models",
        "scan_checkpoints",
        "list_gguf_variants",
    ):
        setattr(utils_models, name, lambda *args, **kwargs: [])
    for name in (
        "get_base_model_from_checkpoint",
        "get_base_model_from_lora",
        "load_model_defaults",
    ):
        setattr(utils_models, name, lambda *args, **kwargs: None)
    utils_models.is_vision_model = lambda *args, **kwargs: False
    utils_models.is_embedding_model = lambda *args, **kwargs: False
    utils_models.ModelConfig = object
    monkeypatch.setitem(sys.modules, "utils.models", utils_models)
    _load_module(
        "utils.models.model_identity",
        "utils/models/model_identity.py",
        monkeypatch,
    )

    utils_model_config = types.ModuleType("utils.models.model_config")
    utils_model_config._pick_best_gguf = lambda variants: variants[0] if variants else None
    utils_model_config._extract_quant_label = lambda value: value
    utils_model_config._is_big_endian_gguf_path = lambda *args, **kwargs: False
    utils_model_config._is_mtp_drafter = lambda *args, **kwargs: False
    utils_model_config.is_audio_input_type = lambda *args, **kwargs: None
    monkeypatch.setitem(
        sys.modules,
        "utils.models.model_config",
        utils_model_config,
    )

    models_pkg = types.ModuleType("models")
    models_pkg.__path__ = []
    for name in (
        "CheckpointInfo",
        "CheckpointListResponse",
        "LocalModelListResponse",
        "ModelCheckpoints",
        "ModelDetails",
        "LoRAScanResponse",
        "LoRAInfo",
        "ModelListResponse",
        "LoadCheckpointRequest",
        "ExportStatusResponse",
        "ExportOperationResponse",
        "ExportMergedModelRequest",
        "ExportBaseModelRequest",
        "ExportGGUFRequest",
        "ExportLoRAAdapterRequest",
    ):
        setattr(models_pkg, name, object)
    models_pkg.LocalModelInfo = _LocalModelInfo
    monkeypatch.setitem(sys.modules, "models", models_pkg)

    models_models = types.ModuleType("models.models")
    for name in (
        "BrowseEntry",
        "BrowseFoldersResponse",
        "ExportSizeResponse",
        "GgufVariantDetail",
        "GgufVariantsResponse",
        "ScanFolderInfo",
        "AddScanFolderRequest",
    ):
        setattr(models_models, name, object)
    models_models.ModelType = str
    monkeypatch.setitem(sys.modules, "models.models", models_models)

    models_responses = types.ModuleType("models.responses")
    for name in (
        "LoRABaseModelResponse",
        "VisionCheckResponse",
        "EmbeddingCheckResponse",
    ):
        setattr(models_responses, name, object)
    monkeypatch.setitem(sys.modules, "models.responses", models_responses)


def _install_pydantic_stub(monkeypatch):
    pydantic = types.ModuleType("pydantic")
    pydantic.BaseModel = object
    pydantic.Field = lambda default = None, **_kwargs: default
    pydantic.field_validator = _identity_decorator
    monkeypatch.setitem(sys.modules, "pydantic", pydantic)


def _install_export_backend_stubs(monkeypatch):
    _install_lightweight_backend_stubs(monkeypatch)

    unsloth = types.ModuleType("unsloth")
    unsloth.FastLanguageModel = object
    unsloth.FastVisionModel = object
    unsloth._IS_MLX = True
    unsloth.__spec__ = importlib.machinery.ModuleSpec("unsloth", loader = None)
    monkeypatch.setitem(sys.modules, "unsloth", unsloth)

    unsloth_zoo = types.ModuleType("unsloth_zoo")
    unsloth_zoo.__path__ = []
    unsloth_zoo.__spec__ = importlib.machinery.ModuleSpec(
        "unsloth_zoo",
        loader = None,
        is_package = True,
    )
    llama_cpp = types.ModuleType("unsloth_zoo.llama_cpp")
    llama_cpp.LLAMA_CPP_DEFAULT_DIR = str(Path("/tmp/llama.cpp"))
    llama_cpp._resolve_local_convert_script = lambda *args, **kwargs: None
    llama_cpp.__spec__ = importlib.machinery.ModuleSpec(
        "unsloth_zoo.llama_cpp",
        loader = None,
    )
    monkeypatch.setitem(sys.modules, "unsloth_zoo", unsloth_zoo)
    monkeypatch.setitem(sys.modules, "unsloth_zoo.llama_cpp", llama_cpp)

    huggingface_hub = types.ModuleType("huggingface_hub")
    huggingface_hub.HfApi = object
    huggingface_hub.ModelCard = object
    monkeypatch.setitem(sys.modules, "huggingface_hub", huggingface_hub)

    utils_hardware = types.ModuleType("utils.hardware")
    utils_hardware.clear_gpu_cache = lambda: None
    monkeypatch.setitem(sys.modules, "utils.hardware", utils_hardware)

    utils_models = sys.modules["utils.models"]
    utils_models.get_base_model_from_lora = lambda *args, **kwargs: None
    utils_models.is_vision_model = lambda *args, **kwargs: False

    utils_model_config = sys.modules["utils.models.model_config"]
    utils_model_config.detect_audio_type = lambda *args, **kwargs: None

    utils_paths = sys.modules["utils.paths"]
    utils_paths.ensure_dir = lambda path: Path(path).mkdir(parents = True, exist_ok = True)
    utils_paths.resolve_export_write_dir = lambda value = None: Path(value or "exports")
    utils_paths.resolve_output_dir = lambda value = None: Path(value or "outputs")


def test_gguf_export_keeps_a_gguf_it_could_not_relocate(tmp_path, monkeypatch):
    _install_export_backend_stubs(monkeypatch)
    export_mod = _load_module("test_core_export_backend", "core/export/export.py", monkeypatch)

    cwd = tmp_path / "cwd"
    save_dir = tmp_path / "export"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    monkeypatch.setattr(export_mod, "resolve_export_write_dir", lambda _value: save_dir)
    monkeypatch.setattr(
        export_mod.shutil,
        "move",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("move failed")),
    )

    class _Model:
        def save_pretrained_gguf(self, model_save_path, tokenizer, quantization_method):
            Path(model_save_path).mkdir(parents = True, exist_ok = True)
            (Path(model_save_path) / "model.safetensors").write_bytes(b"weights")
            output_dir = Path(f"{model_save_path}_gguf")
            output_dir.mkdir(parents = True, exist_ok = True)
            (output_dir / "converted.gguf").write_bytes(b"gguf")

    backend = export_mod.ExportBackend.__new__(export_mod.ExportBackend)
    backend.current_model = _Model()
    backend.current_tokenizer = object()
    backend.current_checkpoint = None

    success, message, output_path = backend.export_gguf(str(save_dir), "Q4_K_M")

    assert success is False
    assert "move failed" in message
    assert output_path is None
    # Preserve a completed GGUF when relocation fails.
    roots = list(save_dir.glob("_tmp_model_*"))
    assert len(roots) == 1
    assert (roots[0] / "model_gguf" / "converted.gguf").read_bytes() == b"gguf"


def test_gguf_export_fails_when_owned_output_has_no_gguf(tmp_path, monkeypatch):
    _install_export_backend_stubs(monkeypatch)
    export_mod = _load_module(
        "test_core_export_backend_empty_owned_output", "core/export/export.py", monkeypatch
    )

    save_dir = tmp_path / "export"
    checkpoint_gguf = tmp_path / "checkpoint_gguf"
    monkeypatch.setattr(export_mod, "resolve_export_write_dir", lambda _value: save_dir)

    class _Model:
        def save_pretrained_gguf(self, model_save_path, tokenizer, quantization_method):
            checkpoint_gguf.mkdir()
            (checkpoint_gguf / "misdirected.gguf").write_bytes(b"gguf")

    backend = export_mod.ExportBackend.__new__(export_mod.ExportBackend)
    backend.current_model = _Model()
    backend.current_tokenizer = object()
    backend.current_checkpoint = str(tmp_path / "checkpoint")

    success, message, output_path = backend.export_gguf(str(save_dir), "Q4_K_M")

    assert success is False
    assert "produced no files" in message
    assert output_path is None
    assert (checkpoint_gguf / "misdirected.gguf").read_bytes() == b"gguf"
    assert not (save_dir / "export_metadata.json").exists()
    assert list(save_dir.glob("_tmp_model_*")) == []


def _require_undeletable_dir_support(tmp_path):
    """Skip where directory mode bits cannot make cleanup fail."""
    probe = tmp_path / "_perm_probe"
    victim = probe / "victim"
    probe.mkdir()
    victim.write_bytes(b"x")
    probe.chmod(0o500)
    try:
        victim.unlink()
    except OSError:
        return
    finally:
        probe.chmod(0o700)
        shutil.rmtree(probe, ignore_errors = True)
    pytest.skip("this platform allows deleting files from a read-only directory")


def test_gguf_export_survives_real_owned_temp_cleanup_failure(tmp_path, monkeypatch):
    _require_undeletable_dir_support(tmp_path)
    _install_export_backend_stubs(monkeypatch)
    export_mod = _load_module(
        "test_core_export_backend_cleanup_failure", "core/export/export.py", monkeypatch
    )

    save_dir = tmp_path / "export"
    monkeypatch.setattr(export_mod, "resolve_export_write_dir", lambda _value: save_dir)
    locked_dirs = []

    class _Model:
        def save_pretrained_gguf(self, model_save_path, tokenizer, quantization_method):
            output_dir = Path(f"{model_save_path}_gguf")
            output_dir.mkdir(parents = True, exist_ok = True)
            (output_dir / "converted.gguf").write_bytes(b"gguf")
            # A read-only model directory makes the real cleanup fail.
            merged = Path(model_save_path)
            merged.mkdir()
            (merged / "model.safetensors").write_bytes(b"weights")
            merged.chmod(0o500)
            locked_dirs.append(merged)

    backend = export_mod.ExportBackend.__new__(export_mod.ExportBackend)
    backend.current_model = _Model()
    backend.current_tokenizer = object()
    backend.current_checkpoint = None

    try:
        success, message, output_path = backend.export_gguf(str(save_dir), "Q4_K_M")
    finally:
        for locked in locked_dirs:
            locked.chmod(0o700)

    assert success is True, message
    assert output_path == str(save_dir.resolve())
    assert (save_dir / "converted.gguf").read_bytes() == b"gguf"
    # The removable output is gone, but the locked model directory remains.
    assert len(locked_dirs) == 1
    model_tmp_root = locked_dirs[0].parent
    assert not (model_tmp_root / "model_gguf").exists()
    assert (locked_dirs[0] / "model.safetensors").read_bytes() == b"weights"
    assert list(save_dir.glob("_tmp_model_*")) == [model_tmp_root]


# Both PEFT and non-PEFT exports must preserve an existing checkpoint sibling.
@pytest.mark.parametrize("merges_into_model_dir", [False, True], ids = ["non_peft", "peft"])
def test_gguf_export_preserves_unowned_paths(tmp_path, monkeypatch, merges_into_model_dir):
    _install_export_backend_stubs(monkeypatch)
    export_mod = _load_module(
        "test_core_export_backend_owned_temp", "core/export/export.py", monkeypatch
    )

    cwd = tmp_path / "cwd"
    save_dir = tmp_path / "export"
    checkpoint = tmp_path / "Qwen3-8B"
    checkpoint_gguf = tmp_path / "Qwen3-8B_gguf"
    cwd.mkdir()
    checkpoint.mkdir()
    checkpoint_gguf.mkdir()
    notes = checkpoint_gguf / "notes.txt"
    imatrix = checkpoint_gguf / "imatrix.dat"
    old_gguf = checkpoint_gguf / "old.Q4_K_M.gguf"
    old_modelfile = checkpoint_gguf / "Modelfile"
    concurrent_dir = save_dir / "user-created"
    concurrent_cwd_gguf = cwd / "unrelated.gguf"
    notes.write_text("keep", encoding = "utf-8")
    imatrix.write_bytes(b"imatrix")
    old_gguf.write_bytes(b"old")
    old_modelfile.write_text("FROM old.Q4_K_M.gguf", encoding = "utf-8")
    monkeypatch.chdir(cwd)
    monkeypatch.setattr(export_mod, "resolve_export_write_dir", lambda _value: save_dir)

    class _Model:
        def save_pretrained_gguf(self, model_save_path, tokenizer, quantization_method):
            if merges_into_model_dir:
                merged = Path(model_save_path)
                merged.mkdir()
                (merged / "model.safetensors").write_bytes(b"merged")
            output_dir = Path(f"{model_save_path}_gguf")
            output_dir.mkdir(parents = True, exist_ok = True)
            (output_dir / "new.Q4_K_M.gguf").write_bytes(b"new")
            (output_dir / "Modelfile").write_text("FROM new.Q4_K_M.gguf", encoding = "utf-8")
            concurrent_dir.mkdir()
            (concurrent_dir / "notes.txt").write_text("keep", encoding = "utf-8")
            concurrent_cwd_gguf.write_bytes(b"unrelated")
            # Reported files may also appear in the owned-root scan.
            return {
                "gguf_directory": str(output_dir),
                "gguf_files": [str(output_dir / "new.Q4_K_M.gguf")],
                "modelfile_location": str(output_dir / "Modelfile"),
            }

    backend = export_mod.ExportBackend.__new__(export_mod.ExportBackend)
    backend.current_model = _Model()
    backend.current_tokenizer = object()
    backend.current_checkpoint = str(checkpoint)

    success, _, output_path = backend.export_gguf(str(save_dir), "Q4_K_M")

    assert success is True
    assert output_path == str(save_dir.resolve())
    assert (save_dir / "new.Q4_K_M.gguf").read_bytes() == b"new"
    assert (save_dir / "Modelfile").read_text(encoding = "utf-8") == "FROM new.Q4_K_M.gguf"
    assert not (save_dir / "old.Q4_K_M.gguf").exists()
    assert notes.read_text(encoding = "utf-8") == "keep"
    assert imatrix.read_bytes() == b"imatrix"
    assert old_gguf.read_bytes() == b"old"
    assert old_modelfile.read_text(encoding = "utf-8") == "FROM old.Q4_K_M.gguf"
    assert (concurrent_dir / "notes.txt").read_text(encoding = "utf-8") == "keep"
    assert concurrent_cwd_gguf.read_bytes() == b"unrelated"
    assert list(save_dir.glob("_tmp_model_*")) == []


def test_gguf_export_relocates_gguf_written_into_the_model_path(tmp_path, monkeypatch):
    # MLX writes into the save path and returns no manifest.
    _install_export_backend_stubs(monkeypatch)
    export_mod = _load_module(
        "test_core_export_backend_in_place_output", "core/export/export.py", monkeypatch
    )

    save_dir = tmp_path / "export"
    monkeypatch.setattr(export_mod, "resolve_export_write_dir", lambda _value: save_dir)

    class _Model:
        def save_pretrained_gguf(self, model_save_path, tokenizer, quantization_method):
            output_dir = Path(model_save_path)
            output_dir.mkdir(parents = True, exist_ok = True)
            (output_dir / "Qwen3-8B.Q4_K_M.gguf").write_bytes(b"converted")

    backend = export_mod.ExportBackend.__new__(export_mod.ExportBackend)
    backend.current_model = _Model()
    backend.current_tokenizer = object()
    backend.current_checkpoint = str(tmp_path / "Qwen3-8B")

    success, message, output_path = backend.export_gguf(str(save_dir), "Q4_K_M")

    assert success is True, message
    assert output_path == str(save_dir.resolve())
    assert (save_dir / "Qwen3-8B.Q4_K_M.gguf").read_bytes() == b"converted"
    assert list(save_dir.glob("_tmp_model_*")) == []


def test_gguf_export_rejects_symlink_inside_its_owned_temp_tree(tmp_path, monkeypatch):
    _install_export_backend_stubs(monkeypatch)
    export_mod = _load_module(
        "test_core_export_backend_symlinked_output", "core/export/export.py", monkeypatch
    )

    save_dir = tmp_path / "export"
    user_owned_gguf = tmp_path / "user-owned.gguf"
    user_owned_gguf.write_bytes(b"keep")
    monkeypatch.setattr(export_mod, "resolve_export_write_dir", lambda _value: save_dir)

    class _Model:
        def save_pretrained_gguf(self, model_save_path, tokenizer, quantization_method):
            output_dir = Path(f"{model_save_path}_gguf")
            output_dir.mkdir(parents = True)
            link = output_dir / "converted.gguf"
            try:
                link.symlink_to(user_owned_gguf)
            except OSError as exception:
                pytest.skip(f"symlinks unavailable: {exception}")
            return {"gguf_files": [str(link)]}

    backend = export_mod.ExportBackend.__new__(export_mod.ExportBackend)
    backend.current_model = _Model()
    backend.current_tokenizer = object()
    backend.current_checkpoint = None

    success, message, output_path = backend.export_gguf(str(save_dir), "Q4_K_M")

    assert success is False
    assert "symlink" in message
    assert output_path is None
    assert user_owned_gguf.read_bytes() == b"keep"
    assert not (save_dir / "converted.gguf").exists()
    roots = list(save_dir.glob("_tmp_model_*"))
    assert len(roots) == 1
    assert (roots[0] / "model_gguf" / "converted.gguf").is_symlink()


def test_gguf_export_relocates_only_reported_files_from_outside_the_owned_root(
    tmp_path, monkeypatch
):
    # Relocate reported files without scanning their unowned directory.
    _install_export_backend_stubs(monkeypatch)
    export_mod = _load_module(
        "test_core_export_backend_unowned_report", "core/export/export.py", monkeypatch
    )

    save_dir = tmp_path / "export"
    checkpoint_gguf = tmp_path / "Qwen3-8B_gguf"
    checkpoint_gguf.mkdir()
    (checkpoint_gguf / "notes.txt").write_text("keep", encoding = "utf-8")
    (checkpoint_gguf / "old.Q8_0.gguf").write_bytes(b"old")
    monkeypatch.setattr(export_mod, "resolve_export_write_dir", lambda _value: save_dir)

    class _Model:
        def save_pretrained_gguf(self, model_save_path, tokenizer, quantization_method):
            (checkpoint_gguf / "new.Q4_K_M.gguf").write_bytes(b"new")
            (checkpoint_gguf / "Modelfile").write_text("FROM new.Q4_K_M.gguf", encoding = "utf-8")
            return {
                "gguf_directory": str(checkpoint_gguf),
                "gguf_files": [str(checkpoint_gguf / "new.Q4_K_M.gguf")],
                "modelfile_location": str(checkpoint_gguf / "Modelfile"),
            }

    backend = export_mod.ExportBackend.__new__(export_mod.ExportBackend)
    backend.current_model = _Model()
    backend.current_tokenizer = object()
    backend.current_checkpoint = str(tmp_path / "Qwen3-8B")

    success, message, output_path = backend.export_gguf(str(save_dir), "Q4_K_M")

    assert success is True, message
    assert output_path == str(save_dir.resolve())
    assert (save_dir / "new.Q4_K_M.gguf").read_bytes() == b"new"
    assert (save_dir / "Modelfile").read_text(encoding = "utf-8") == "FROM new.Q4_K_M.gguf"
    assert not (save_dir / "old.Q8_0.gguf").exists()
    assert checkpoint_gguf.is_dir()
    assert (checkpoint_gguf / "notes.txt").read_text(encoding = "utf-8") == "keep"
    assert (checkpoint_gguf / "old.Q8_0.gguf").read_bytes() == b"old"
    assert list(save_dir.glob("_tmp_model_*")) == []


def test_save_directory_validator_rejects_windows_parent_segments(monkeypatch):
    _install_pydantic_stub(monkeypatch)
    export_models = _load_module("test_models_export", "models/export.py", monkeypatch)

    with pytest.raises(ValueError, match = r"\.\."):
        export_models._validate_save_directory(r"E:\AI\..\secret")


def test_save_directory_validator_allows_deep_absolute_paths(monkeypatch, tmp_path):
    _install_pydantic_stub(monkeypatch)
    export_models = _load_module("test_models_export_deep_path", "models/export.py", monkeypatch)

    deep_path = tmp_path
    for index in range(40):
        deep_path /= f"segment-{index:02d}"
    raw = str(deep_path)

    assert len(raw) > 255
    assert export_models._validate_save_directory(raw) == raw


def test_save_directory_validator_rejects_long_path_component(monkeypatch, tmp_path):
    _install_pydantic_stub(monkeypatch)
    export_models = _load_module(
        "test_models_export_long_component", "models/export.py", monkeypatch
    )

    with pytest.raises(ValueError, match = "path components"):
        export_models._validate_save_directory(str(tmp_path / ("a" * 256)))


def test_export_write_dir_accepts_external_absolute_but_read_dir_rejects(tmp_path, monkeypatch):
    storage_roots = _load_module(
        "test_storage_roots_accept_external",
        "utils/paths/storage_roots.py",
    )

    export_root = tmp_path / "exports"
    external = tmp_path / "external"
    export_root.mkdir()
    external.mkdir()
    monkeypatch.setattr(storage_roots, "exports_root", lambda: export_root)

    assert storage_roots.resolve_export_write_dir(str(external)) == external

    with pytest.raises(ValueError, match = "path escapes root"):
        storage_roots.resolve_export_dir(str(external))


def test_export_write_dir_accepts_expanded_home_path(tmp_path, monkeypatch):
    storage_roots = _load_module(
        "test_storage_roots_accept_home_path",
        "utils/paths/storage_roots.py",
    )

    export_root = tmp_path / "exports"
    home = tmp_path / "home"
    export_root.mkdir()
    home.mkdir()
    monkeypatch.setattr(storage_roots, "exports_root", lambda: export_root)
    if storage_roots.os.name == "nt":
        monkeypatch.setenv("USERPROFILE", str(home))
    else:
        monkeypatch.setenv("HOME", str(home))

    assert storage_roots.resolve_export_write_dir("~/exports/model") == home / "exports" / "model"


def test_resolve_export_write_dir_rejects_backslash_parent_segment():
    storage_roots = _load_module(
        "test_storage_roots_reject_parent",
        "utils/paths/storage_roots.py",
    )

    with pytest.raises(ValueError, match = r"\.\."):
        storage_roots.resolve_export_write_dir(r"exports\..\outside")


def test_export_write_dir_handles_non_native_windows_absolute_as_relative(tmp_path, monkeypatch):
    storage_roots = _load_module(
        "test_storage_roots_non_native_windows_path",
        "utils/paths/storage_roots.py",
    )

    export_root = tmp_path / "exports"
    export_root.mkdir()
    monkeypatch.setattr(storage_roots, "exports_root", lambda: export_root)

    if storage_roots.os.name == "nt":
        pytest.skip("Windows drive paths are native on Windows")

    assert (
        storage_roots.resolve_export_write_dir(r"C:\exports\model")
        == export_root / r"C:\exports\model"
    )


def test_export_details_registers_external_absolute_output(tmp_path, monkeypatch):
    _install_lightweight_backend_stubs(monkeypatch)
    export_route = _load_module(
        "test_routes_export_external",
        "routes/export.py",
        monkeypatch,
    )

    output = tmp_path / "Gemma4_26B_gguf"
    output.mkdir()
    export_root = tmp_path / "studio" / "exports"
    export_root.mkdir(parents = True)
    registered = []

    monkeypatch.setattr(
        export_route,
        "_try_register_external_export",
        lambda path: (registered.append(path) is None, str(path)),
    )
    monkeypatch.setattr(
        "utils.paths.storage_roots.exports_root",
        lambda: export_root,
    )

    details = export_route._export_details(str(output))

    assert details == {
        "output_path": str(output),
        "scan_folder_registered": True,
        "scan_folder_path": str(output),
    }
    assert registered == [output]


def test_export_details_does_not_register_contained_exports(tmp_path, monkeypatch):
    _install_lightweight_backend_stubs(monkeypatch)
    export_route = _load_module(
        "test_routes_export_contained",
        "routes/export.py",
        monkeypatch,
    )

    export_root = tmp_path / "exports"
    output = export_root / "model-gguf"
    output.mkdir(parents = True)

    monkeypatch.setattr(
        export_route,
        "_try_register_external_export",
        lambda path: pytest.fail(f"unexpected registration: {path}"),
    )
    monkeypatch.setattr(
        "utils.paths.storage_roots.exports_root",
        lambda: export_root,
    )

    assert export_route._export_details(str(output)) == {"output_path": "model-gguf"}


def test_registered_absolute_export_folder_is_discoverable(tmp_path, monkeypatch):
    _install_lightweight_backend_stubs(monkeypatch)
    models_route = _load_module("test_routes_models", "routes/models.py", monkeypatch)

    export_dir = tmp_path / "Gemma4_26B_gguf"
    export_dir.mkdir()
    gguf_file = export_dir / "Gemma4_26B.BF16-00001-of-00002.gguf"
    gguf_file.write_bytes(b"gguf")

    found = models_route._scan_models_dir(export_dir)

    assert len(found) == 1
    assert found[0].path == str(gguf_file)
    assert found[0].source == "models_dir"
