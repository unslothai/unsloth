"""Exercise resume guard control flow without importing Torch, vLLM or Unsloth."""

import ast
import gc
import os
from pathlib import Path
from types import SimpleNamespace

import pytest


REPO_ROOT = Path(os.environ.get("UNSLOTH_REPO_ROOT", Path(__file__).resolve().parents[1]))
SOURCE = REPO_ROOT / "unsloth" / "models" / "rl.py"
TREE = ast.parse(SOURCE.read_text(encoding = "utf-8"))
FUNCTION = next(node for node in TREE.body if isinstance(node, ast.FunctionDef)
                and node.name == "_maybe_prepare_vllm_for_resume")
LEGACY_CUDA_GUARD = ast.unparse(FUNCTION.body[0].test) == "not torch.cuda.is_available()"
CODE = compile(ast.Module(body = [FUNCTION], type_ignores = []), str(SOURCE), "exec")


def _engine(events, enabled = True, sleep_error = False):
    def sleep(level):
        events.append(f"sleep:{level}")
        if sleep_error:
            raise RuntimeError("synthetic sleep failure")

    config = SimpleNamespace(model_config = SimpleNamespace(enable_sleep_mode = enabled))
    return SimpleNamespace(sleep = sleep, llm_engine = SimpleNamespace(vllm_config = config))


def _invoke(monkeypatch, trainer, device_count, events):
    def unexpected_cuda_cache():
        pytest.fail("Resume must use shared cache dispatch, not direct CUDA cache cleanup")

    namespace = {
        "DEVICE_COUNT": device_count,
        "torch": SimpleNamespace(cuda = SimpleNamespace(
            is_available = lambda: False, empty_cache = unexpected_cuda_cache)),
        "clean_gpu_cache": lambda: events.append("cache"),
    }
    exec(CODE, namespace)
    with monkeypatch.context() as patch:
        patch.setattr(gc, "collect", lambda: events.append("collect"))
        namespace[FUNCTION.name](trainer)


@pytest.mark.parametrize("case", ["no-device", "no-engine", "sleep-disabled", "no-config"])
def test_resume_guard_does_not_flush_unneeded_cache(monkeypatch, case):
    events = []
    engine = _engine(events, enabled = case != "sleep-disabled")
    trainer = SimpleNamespace(llm = engine)
    if case == "no-engine":
        trainer = SimpleNamespace()
    elif case == "no-config":
        engine.llm_engine = None
    _invoke(monkeypatch, trainer, 0 if case == "no-device" else 1, events)
    assert events == []


@pytest.mark.xfail(
    condition = LEGACY_CUDA_GUARD,
    reason = "CUDA-only resume guard skips non-CUDA devices; tracked in #13438",
    raises = AssertionError,
    strict = True,
)
@pytest.mark.parametrize("case", ["direct-llm", "model-fallback", "sleep-error"])
def test_non_cuda_resume_reaches_shared_cache_dispatch(monkeypatch, case):
    events = []
    engine = _engine(events, sleep_error = case == "sleep-error")
    trainer = (SimpleNamespace(model = SimpleNamespace(vllm_engine = engine))
               if case == "model-fallback" else SimpleNamespace(llm = engine))
    _invoke(monkeypatch, trainer, 1, events)
    print({"case": case, "cuda_available": False, "device_count": 1, "events": events})
    assert events == ["sleep:1"] + ["collect", "cache"] * 3
