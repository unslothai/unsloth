"""PR #11238, commit 7b45cc8f9: `patch_fla_autotuner_fast_path` rebinds
`fla.ops.utils.cache.CachedAutotuner.run` to launch with a latched config.

Each test compares the patched path against the ORIGINAL `CachedAutotuner.run` on the same
inputs, so "same kernel, same config, same numerics" is measured, not asserted.

    PYTHONPATH=<tree> CUDA_VISIBLE_DEVICES=<n> pytest -q test_fla_autotuner_fast_path.py
"""

import importlib

import pytest

# These runners do not all ship torch/triton/fla; skip rather than erroring at collection.
torch = pytest.importorskip("torch")
triton = pytest.importorskip("triton")
tl = pytest.importorskip("triton.language")
fla_cache = pytest.importorskip("fla.ops.utils.cache")
U = pytest.importorskip("unsloth.models._utils")

PATCHER = getattr(U, "patch_fla_autotuner_fast_path", None)
needs_patch = pytest.mark.skipif(
    PATCHER is None, reason = "base tree: no patch_fla_autotuner_fast_path"
)
CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs a GPU")


class _ReuseBestCache(dict):
    """Byte-copy of unsloth_zoo/compiler.py:4808 so the tests reproduce the real shape."""

    def __contains__(self, key):
        return len(self) > 0 or super().__contains__(key)

    def __getitem__(self, key):
        if not super().__contains__(key) and len(self) > 0:
            return next(iter(self.values()))
        return super().__getitem__(key)


@triton.jit
def _add_kernel(x_ptr, y_ptr, o_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    tl.store(
        o_ptr + offs, tl.load(x_ptr + offs, mask = mask) + tl.load(y_ptr + offs, mask = mask), mask = mask
    )


def _make(
    configs,
    key = ("n",),
    pre_hook = None,
):
    from fla.ops.utils.cache import fla_cache_autotune
    if pre_hook is not None:
        configs = [
            triton.Config(
                c.kwargs, num_warps = c.num_warps, num_stages = c.num_stages, pre_hook = pre_hook
            )
            for c in configs
        ]
    return fla_cache_autotune(configs = configs, key = list(key))(_add_kernel)


CFGS2 = [triton.Config({"BLOCK": 64}, num_warps = 2), triton.Config({"BLOCK": 128}, num_warps = 4)]
CFGS1 = [triton.Config({"BLOCK": 128}, num_warps = 4)]


def _grid(n):
    # A meta-dependent grid, as every real fla kernel uses. A hardcoded grid would silently
    # under-cover the tensor whenever the tuned BLOCK is smaller than the one assumed here,
    # which makes the harness, not the patch, the source of any mismatch.
    return lambda meta: (triton.cdiv(n, meta["BLOCK"]),)


def _run(kern, n = 4096):
    x = torch.randn(n, device = "cuda")
    y = torch.randn(n, device = "cuda")
    o = torch.empty_like(x)
    kern[_grid(n)](x, y, o, n)
    return x, y, o


def _unpatch(cls, original):
    cls.run = original


@pytest.fixture
def fresh():
    """Reload the module so each test starts from an unpatched CachedAutotuner."""
    importlib.reload(fla_cache)
    orig = fla_cache.CachedAutotuner.run
    yield fla_cache.CachedAutotuner
    fla_cache.CachedAutotuner.run = orig


# --------------------------------------------------------------------------- install
@needs_patch
def test_patch_installs_once_and_is_idempotent(fresh):
    assert not getattr(fresh.run, "_unsloth_fast_path", False)
    PATCHER()
    first = fresh.run
    assert getattr(first, "_unsloth_fast_path", False) is True
    PATCHER()
    assert fresh.run is first, "a second call must not stack another wrapper"


@needs_patch
def test_patch_declines_under_FLA_CACHE_MODE_always(fresh, monkeypatch):
    class _Mode:
        value = "always"

    monkeypatch.setattr(fla_cache, "FLA_CACHE_MODE", _Mode(), raising = False)
    PATCHER()
    assert not getattr(
        fresh.run, "_unsloth_fast_path", False
    ), "the debug mode re-reads config files on every launch; the fast path must stand down"


# --------------------------------------------------------------------------- behaviour
@CUDA
@needs_patch
def test_multi_config_plain_dict_cache_falls_back(fresh):
    kern = _make(CFGS2)
    PATCHER()
    _run(kern)
    at = kern
    while not hasattr(at, "configs"):
        at = at.fn
    assert type(at.cache) is not _ReuseBestCache
    assert (
        getattr(at, "_unsloth_fixed_config", None) is None
    ), "without _ReuseBestCache the autotuner must keep triton's own lookup"


@CUDA
@needs_patch
def test_reuse_best_cache_fast_path_matches_the_original_bit_for_bit(fresh):
    kern_a = _make(CFGS2)
    kern_b = _make(CFGS2)
    # settle both the same way with the ORIGINAL run
    _run(kern_a)
    _run(kern_b)
    at_a, at_b = kern_a, kern_b
    while not hasattr(at_a, "configs"):
        at_a = at_a.fn
    while not hasattr(at_b, "configs"):
        at_b = at_b.fn
    at_a.cache = _ReuseBestCache(at_a.cache)
    at_b.cache = _ReuseBestCache(at_b.cache)

    n = 4096
    x = torch.randn(n, device = "cuda")
    y = torch.randn(n, device = "cuda")
    o_ref = torch.empty_like(x)
    o_fast = torch.empty_like(x)
    kern_a[_grid(n)](x, y, o_ref, n)  # unpatched

    PATCHER()
    kern_b[_grid(n)](x, y, o_fast, n)  # patched

    assert getattr(at_b, "_unsloth_fixed_config", None) is not None, "fast path did not latch"
    assert at_b._unsloth_fixed_config.all_kwargs() == at_a.best_config.all_kwargs()
    assert at_b.best_config is at_b._unsloth_fixed_config
    assert torch.equal(o_ref, o_fast)
    assert torch.equal(o_ref, x + y)


@CUDA
@needs_patch
def test_single_config_fast_path_first_and_subsequent_launches(fresh):
    kern = _make(CFGS1)
    PATCHER()
    x, y, o = _run(kern)
    assert torch.equal(o, x + y)
    at = kern
    while not hasattr(at, "configs"):
        at = at.fn
    assert at._unsloth_fixed_config is at.configs[0]
    x2, y2, o2 = _run(kern)  # second launch uses the latch
    assert torch.equal(o2, x2 + y2)


@CUDA
@needs_patch
def test_per_config_pre_hook_falls_back_and_the_hook_still_fires(fresh):
    fired = []
    kern = _make(CFGS1, pre_hook = lambda nargs: fired.append(1))
    PATCHER()
    _run(kern)
    _run(kern)
    at = kern
    while not hasattr(at, "configs"):
        at = at.fn
    assert getattr(at, "_unsloth_fixed_config", None) is None
    assert len(fired) >= 2, "a per-Config pre_hook must run on every launch"


# --------------------------------------------------------------------------- absence
def test_patch_is_a_no_op_when_fla_is_missing(monkeypatch):
    """fla < 0.5.1 has no fla.ops.utils.cache; the patch must not raise."""
    if PATCHER is None:
        pytest.skip("base tree")
    import builtins

    real_import = builtins.__import__

    def _blocked(name, *a, **kw):
        if name.startswith("fla"):
            raise ImportError("no fla")
        return real_import(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", _blocked)
    PATCHER()  # must not raise


# --------------------------------------------------------------------------- cache probe
CACHE_PROBE = getattr(U, "_unsloth_cache_reuses_one_config", None)
needs_probe = pytest.mark.skipif(CACHE_PROBE is None, reason = "tree without the behaviour probe")


@needs_probe
def test_cache_probe_detects_the_reuse_cache_under_any_name():
    """unsloth_zoo defines _ReuseBestCache inside a function, so there is no symbol to import.
    The probe must key off behaviour, not the class name, or a rename silently turns the fast
    path off."""

    class RenamedTomorrow(_ReuseBestCache):
        pass

    for cls in (_ReuseBestCache, RenamedTomorrow):
        c = cls()
        assert CACHE_PROBE(c) is False, "an empty reuse cache reuses nothing yet"
        c["k"] = object()
        assert CACHE_PROBE(c) is True


@needs_probe
def test_cache_probe_rejects_a_plain_dict_and_anything_that_raises():
    assert CACHE_PROBE({}) is False
    assert CACHE_PROBE({"k": object()}) is False

    class Hostile(dict):
        def __contains__(self, key):
            raise RuntimeError("no")

    h = Hostile()
    h["k"] = object()
    assert CACHE_PROBE(h) is False, "a cache that raises must fall back, not propagate"


@needs_probe
def test_cache_probe_does_not_mutate_the_cache():
    c = _ReuseBestCache()
    c["k"] = object()
    before = dict(c)
    CACHE_PROBE(c)
    assert dict(c) == before and len(c) == 1
