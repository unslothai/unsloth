"""Tests _resolve_offload_embedding in vision.py. No GPU needed.

`offload_embedding = True` on a model with tied word embeddings used to raise
NotImplementedError and abort the load. It is a VRAM optimisation, not a
correctness switch, so it should turn itself off instead, as the fast_inference
case a few lines earlier already does. Two shipped notebooks (NeMo-Gym-Sudoku,
NeMo-Gym-Multi-Environment) died this way on unsloth/Qwen2.5-1.5B-Instruct.

Every platform branch is driven explicitly, so the assertions hold on Linux,
macOS, Windows and WSL alike: the host's own os.name never decides.
"""

import ast, os
from contextlib import contextmanager

import pytest

# Skip rather than error where torch is absent.
# Only `nn.Embedding` / `nn.Linear` / `torch.device` are wanted here, no GPU, but a bare module-level import turns a
# machine without torch into a collection error, which aborts the whole pytest session instead of leaving one skipped
torch = pytest.importorskip("torch")
nn = pytest.importorskip("torch.nn")

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VISION = os.path.join(HERE, "unsloth", "models", "vision.py")

_SRC = open(VISION, encoding = "utf-8").read()


_DISTRIBUTED = [False]


def _load(*names):
    mod = ast.parse(_SRC)
    # The sentinel lives in loader_utils;
    # importing that module would drag in torch's CUDA stack, so mirror the one value these functions read.
    # `is_distributed` is driven explicitly so the assertions never depend on whether the host happens to have
    # torchrun's env vars set.
    ns = {
        "torch": torch,
        "os": os,
        "OFFLOAD_EMBEDDING_AUTO": "auto",
        "is_distributed": lambda: _DISTRIBUTED[0],
    }
    wanted = set(names)
    for node in mod.body:
        if isinstance(node, ast.FunctionDef) and node.name in wanted:
            exec(ast.get_source_segment(_SRC, node), ns)
            wanted.discard(node.name)
        elif isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "").startswith(
            "_OFFLOAD_EMBEDDING_"
        ):
            # tests below cannot drift from the shipped numbers.
            # The size thresholds the auto decision reads;
            exec(ast.get_source_segment(_SRC, node), ns)
    if wanted:
        raise AssertionError(f"not found in vision.py: {sorted(wanted)}")
    return ns


_NS = _load(
    "_embeddings_are_tied",
    "_offload_embedding_unsupported_platform",
    "_embedding_dispatch_device",
    "_embedding_is_worth_offloading",
    "_resolve_offload_embedding",
)
resolve = _NS["_resolve_offload_embedding"]
unsupported_platform = _NS["_offload_embedding_unsupported_platform"]
dispatch_device = _NS["_embedding_dispatch_device"]

_WSL_VARS = ("WSL_DISTRO_NAME", "WSL_INTEROP")


@contextmanager
def _as_platform(os_name, wsl = False):
    """Drive the platform inputs directly instead of trusting the host's."""
    saved_env = {v: os.environ.get(v) for v in _WSL_VARS}
    saved_name = os.name
    for v in _WSL_VARS:
        os.environ.pop(v, None)
    if wsl:
        os.environ["WSL_DISTRO_NAME"] = "Ubuntu"
    # os.name is read by pathlib, so keep the window as small as possible.
    os.name = os_name
    try:
        yield
    finally:
        os.name = saved_name
        for v, old in saved_env.items():
            if old is None:
                os.environ.pop(v, None)
            else:
                os.environ[v] = old


class _Model:
    def __init__(self, emb, out):
        self._emb, self._out = emb, out

    def get_input_embeddings(self):
        return self._emb

    def get_output_embeddings(self):
        return self._out


class _Opaque:
    """Some architectures refuse to expose embeddings."""

    def get_input_embeddings(self):
        raise NotImplementedError("no embeddings here")

    def get_output_embeddings(self):
        return None


def _tied_model():
    emb = nn.Embedding(32, 8)
    lm = nn.Linear(8, 32, bias = False)
    lm.weight = emb.weight
    return _Model(emb, lm)


def _untied_model():
    return _Model(nn.Embedding(32, 8), nn.Linear(8, 32, bias = False))


def test_disabled_stays_disabled():
    for os_name, wsl in (("posix", False), ("nt", False), ("posix", True)):
        with _as_platform(os_name, wsl = wsl):
            assert resolve(_untied_model(), False) is False
            assert resolve(_tied_model(), False) is False


def test_untied_model_keeps_offload():
    with _as_platform("posix"):
        assert resolve(_untied_model(), True) is True


def test_tied_model_disables_offload_instead_of_raising():
    with _as_platform("posix"):
        assert resolve(_tied_model(), True) is False


def test_opaque_model_leaves_request_alone():
    with _as_platform("posix"):
        assert resolve(_Opaque(), True) is True


def test_wsl_and_windows_disable_offload():
    # Neither can offload, and the flag also gates the multi-device hook attach, so it has to read False rather than
    for var in _WSL_VARS:
        with _as_platform("posix"):
            os.environ[var] = "1"
            assert unsupported_platform() == "WSL"
            assert resolve(_tied_model(), True) is False
            assert resolve(_untied_model(), True) is False
            assert resolve(_Opaque(), True) is False

    with _as_platform("nt"):
        assert unsupported_platform() == "Windows"
        assert resolve(_tied_model(), True) is False
        assert resolve(_untied_model(), True) is False
        assert resolve(_Opaque(), True) is False

    # Cannot inspect it, so do not guess, and do not crash.
    with _as_platform("posix"):
        assert unsupported_platform() is None


def test_platform_gate_lives_in_one_place():
    # The offload block used to re-test os.name itself;
    helper = _SRC[_SRC.index("def _offload_embedding_unsupported_platform(") :]
    helper = helper[: helper.index("\n\n\ndef ")]
    for probe in ('os.name == "nt"', "WSL_DISTRO_NAME", "WSL_INTEROP"):
        assert _SRC.count(probe) == 1, f"{probe} must be tested in exactly one place"
        assert probe in helper, f"{probe} belongs in _offload_embedding_unsupported_platform"


def test_resolved_before_multidevice_hooks():
    # Hook attach returns early while offload_embedding is still True.
    call = _SRC.index("offload_embedding = _resolve_offload_embedding(")
    # Anchor on the indented CALL, not the module-level `def`.
    hooks = _SRC.index("\n                _attach_bnb_multidevice_hooks(")
    assert call < hooks, "offload_embedding must be resolved before hook attach"


def test_no_tied_embedding_raise_remains():
    assert "is not supported for models with tied word" not in _SRC


class _Hook:
    def __init__(self, execution_device):
        self.execution_device = execution_device


def _dispatched_model(execution_device = torch.device("cuda", 0)):
    m = _untied_model()
    m.get_input_embeddings()._hf_hook = _Hook(execution_device)
    return m


def test_dispatch_device_reads_the_accelerate_hook():
    assert dispatch_device(nn.Embedding(32, 8)) is None
    assert dispatch_device(_dispatched_model().get_input_embeddings()) is not None
    assert dispatch_device(_dispatched_model(None).get_input_embeddings()) is None
    assert dispatch_device(None) is None


def test_dispatched_model_disables_offload():
    with _as_platform("posix"):
        assert resolve(_dispatched_model(), True) is False


def test_hook_without_execution_device_keeps_offload():
    with _as_platform("posix"):
        assert resolve(_dispatched_model(None), True) is True


def test_undispatched_model_keeps_offload():
    # A hook that never moves anything cannot undo the offload.
    # The single-GPU path must not lose the VRAM saving.
    # accelerate re-sends the ids to its recorded device after the offload pre-hook has sent them to the CPU weight, so
    with _as_platform("posix"):
        assert resolve(_untied_model(), True) is True


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"[PASS] {name}")
    print("all offload tied auto-disable tests passed")


# --------------------------------------------------------------------------------------
worth_offloading = _NS["_embedding_is_worth_offloading"]
MIN_BYTES = _NS["_OFFLOAD_EMBEDDING_MIN_BYTES"]
MIN_FRACTION = _NS["_OFFLOAD_EMBEDDING_MIN_FRACTION"]


class _FakeWeight:
    def __init__(
        self,
        n_bytes,
        device_type = "cuda",
        index = 0,
    ):
        self._bytes = n_bytes
        self.device = torch.device(
            f"{device_type}:{index}" if device_type == "cuda" else device_type
        )

    def numel(self):
        return self._bytes // 2

    def element_size(self):
        return 2

    def data_ptr(self):
        # Distinct per object, so the tied-weights check sees these as untied.
        return id(self)


class _FakeEmbedding:
    def __init__(self, weight):
        self.weight = weight


@contextmanager
def _card(total_bytes):
    """Drive total device memory directly; no GPU is touched."""
    saved = torch.cuda.get_device_properties
    torch.cuda.get_device_properties = lambda index = 0: type(
        "_Props", (), {"total_memory": total_bytes}
    )()
    try:
        yield
    finally:
        torch.cuda.get_device_properties = saved


def test_a_big_embedding_on_a_small_card_is_offloaded():
    """Muse Glimmer's 202048 x 6656 embedding is 2.5 GiB, 16% of a 16 GB T4. Every one of
    the four notebooks passed `offload_embedding = True` by hand for exactly this."""
    with _card(16 * 2**30):
        assert worth_offloading(_FakeEmbedding(_FakeWeight(int(2.5 * 2**30)))) is True


def test_the_same_embedding_on_a_big_card_is_left_alone():
    """3% of an 80 GB card. The PCIe traffic buys nothing there."""
    with _card(80 * 2**30):
        assert worth_offloading(_FakeEmbedding(_FakeWeight(int(2.5 * 2**30)))) is False


def test_a_small_embedding_is_never_worth_the_traffic():
    """Under the absolute floor even though it clears the fraction on a tiny card."""
    with _card(4 * 2**30):
        assert worth_offloading(_FakeEmbedding(_FakeWeight(MIN_BYTES // 2))) is False


def test_anything_unmeasurable_declines():
    """Not offloading is what every release before this did, so it is the safe answer."""
    with _card(16 * 2**30):
        assert worth_offloading(_FakeEmbedding(None)) is False
        assert worth_offloading(_FakeEmbedding(_FakeWeight(4 * 2**30, "cpu"))) is False
        assert worth_offloading(object()) is False


def test_auto_declines_a_tied_model_without_printing(capsys):
    """The tied decline explains why something a caller ASKED for is not happening. For a
    default nobody set it would be an apology in front of every tied-embedding load."""
    model = _tied_model()
    with _as_platform("posix"):
        assert resolve(model, "auto") is False
    assert capsys.readouterr().out == ""


def test_an_explicit_request_still_explains_itself(capsys):
    model = _tied_model()
    with _as_platform("posix"):
        assert resolve(model, True) is False
    assert "ties embed_tokens" in capsys.readouterr().out


def _sized_model(n_bytes):
    """An untied, undispatched model whose embedding is exactly `n_bytes` on cuda:0."""
    return _Model(_FakeEmbedding(_FakeWeight(n_bytes)), _FakeEmbedding(_FakeWeight(8)))


def test_auto_offloads_a_big_embedding_and_declines_a_small_one():
    """`resolve` must actually consult the size test, not just default to yes: a blanket
    yes would offload every model on every card and cost PCIe traffic for nothing."""
    with _as_platform("posix"), _card(16 * 2**30):
        assert resolve(_sized_model(int(2.5 * 2**30)), "auto") is True
        assert resolve(_sized_model(64 * 2**20), "auto") is False


def test_auto_declines_the_same_embedding_on_a_card_with_room():
    with _as_platform("posix"), _card(80 * 2**30):
        assert resolve(_sized_model(int(2.5 * 2**30)), "auto") is False


def test_explicit_true_and_false_are_untouched_by_the_auto_default():
    """Backwards compatibility: the size test only ever runs for `"auto"`."""
    model = _untied_model()
    with _as_platform("posix"), _card(80 * 2**30):
        # 80 GB card, so `"auto"` would decline; an explicit True must not.
        assert resolve(model, True) is True
        assert resolve(model, False) is False


@contextmanager
def _under_ddp():
    _DISTRIBUTED[0] = True
    try:
        yield
    finally:
        _DISTRIBUTED[0] = False


def test_a_distributed_launch_declines_the_offload(capsys):
    """The offload leaves embed_tokens on the CPU while the rest of the rank stays on CUDA.
    Under full finetuning that parameter is trainable, and DDP wrapping with device_ids
    refuses a module whose trainable parameters span both, so the run dies before step 1.
    The old False default kept distributed callers away from this; the new one does not."""
    with _as_platform("posix"), _card(16 * 2**30), _under_ddp():
        assert resolve(_sized_model(int(2.5 * 2**30)), "auto") is False
    assert capsys.readouterr().out == ""


def test_a_distributed_launch_also_declines_an_explicit_request(capsys):
    """Same veto for someone who asked outright, with the reason, as the other declines do.
    It is a VRAM optimisation, not a correctness switch, so turning it off beats failing."""
    with _as_platform("posix"), _card(16 * 2**30), _under_ddp():
        assert resolve(_sized_model(int(2.5 * 2**30)), True) is False
    assert "distributed launch" in capsys.readouterr().out


def test_a_single_process_run_is_unaffected():
    with _as_platform("posix"), _card(16 * 2**30):
        assert resolve(_sized_model(int(2.5 * 2**30)), "auto") is True
