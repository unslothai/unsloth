# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Remote code that reads image helpers off a model's image_processing module.

transformers 5 stopped re-exporting the generic image helpers from each model's
``image_processing_*`` module. Checkpoints whose own modeling file was written
against the 4.x layout do

    import transformers.models.siglip2.image_processing_siglip2 as siglip2_ips
    ...
    @siglip2_ips.filter_out_non_signature_kwargs()

and raise ``AttributeError`` while the class body is executing, so the model
cannot be loaded at all. Measured on transformers 5.17.0 with
microsoft/Phi-4-reasoning-vision-15B: ten of the sixteen names that file reads
off that module are gone, and ``AutoProcessor.from_pretrained`` fails with
``module 'transformers.models.siglip2.image_processing_siglip2' has no
attribute 'filter_out_non_signature_kwargs'``.

Every test here drives the real functions against the real transformers module.
Nothing asserts on a literal list of names, because the set that was dropped
differs per release and a hardcoded list would pass while the fix was doing
nothing.
"""

import importlib

import pytest

transformers = pytest.importorskip("transformers")

from unsloth.import_fixes import (  # noqa: E402
    _IMAGE_PROCESSING_MODULES,
    _IMAGE_PROCESSING_SYMBOL_HOMES,
    _IMAGE_REEXPORT_FLAG,
    _image_processing_reexports_are_missing,
    _install_legacy_image_reexports,
    _remove_legacy_image_reexports,
)

SIGLIP2 = "transformers.models.siglip2.image_processing_siglip2"


def _fresh_module(name):
    """A module with our patch fully removed, so a test sees the upstream state.

    Removing only ``__getattr__`` is not enough: the forwarder caches each hit
    with ``setattr``, so a later probe would see the names still present and
    the test would skip itself into passing.
    """
    importlib.import_module(name)
    _remove_legacy_image_reexports(name)
    return importlib.import_module(name)


@pytest.fixture
def siglip2_module():
    module = _fresh_module(SIGLIP2)
    yield module
    _fresh_module(SIGLIP2)


def test_homes_are_importable():
    """Every home the fix resolves from must exist, or it silently finds nothing."""
    for home in _IMAGE_PROCESSING_SYMBOL_HOMES:
        importlib.import_module(home)


def test_probe_matches_reality(siglip2_module):
    """The probe must agree with a direct attribute read, not with a version."""
    missing = _image_processing_reexports_are_missing(siglip2_module)
    assert missing == (not hasattr(siglip2_module, "filter_out_non_signature_kwargs"))


def test_fix_restores_every_name_the_remote_code_reads(siglip2_module):
    """Restore the names Phi-4-reasoning-vision reads, whichever are missing here."""
    if not _image_processing_reexports_are_missing(siglip2_module):
        pytest.skip("this transformers still re-exports the image helpers")

    # The exact set the checkpoint's modeling file reads off the module.
    names = [
        "BatchFeature",
        "ChannelDimension",
        "PILImageResampling",
        "convert_image_to_patches",
        "convert_to_rgb",
        "filter_out_non_signature_kwargs",
        "get_image_size_for_max_num_patches",
        "infer_channel_dimension_format",
        "make_flat_list_of_images",
        "pad_along_first_dim",
        "resize",
        "to_channel_dimension_format",
        "to_numpy_array",
        "valid_images",
        "validate_preprocess_arguments",
    ]
    before = [n for n in names if not hasattr(siglip2_module, n)]
    assert before, "nothing was missing, so this test would prove nothing"

    assert _install_legacy_image_reexports(SIGLIP2) is True
    for name in before:
        assert getattr(siglip2_module, name, None) is not None, name


def test_resolved_symbol_is_the_real_one(siglip2_module):
    """The forwarded object must be transformers' own, not a stand-in."""
    if not _image_processing_reexports_are_missing(siglip2_module):
        pytest.skip("this transformers still re-exports the image helpers")
    _install_legacy_image_reexports(SIGLIP2)
    from transformers.utils import filter_out_non_signature_kwargs as real

    assert siglip2_module.filter_out_non_signature_kwargs is real


def test_unknown_names_still_raise(siglip2_module):
    """A genuine typo must not turn into a confusing failure later on."""
    _install_legacy_image_reexports(SIGLIP2)
    with pytest.raises(AttributeError):
        siglip2_module.unsloth_definitely_not_a_transformers_symbol


def test_private_names_are_not_forwarded(siglip2_module):
    """Underscore names are never re-exports, and forwarding them hides bugs."""
    _install_legacy_image_reexports(SIGLIP2)
    with pytest.raises(AttributeError):
        siglip2_module._unsloth_definitely_not_a_transformers_symbol


def test_fix_is_idempotent(siglip2_module):
    """A second call must not stack another layer of forwarding."""
    if not _image_processing_reexports_are_missing(siglip2_module):
        pytest.skip("this transformers still re-exports the image helpers")
    assert _install_legacy_image_reexports(SIGLIP2) is True
    assert _install_legacy_image_reexports(SIGLIP2) is False


def test_fix_can_be_undone(siglip2_module):
    """Keep the original reachable, so the patch can be tested and undone."""
    if not _image_processing_reexports_are_missing(siglip2_module):
        pytest.skip("this transformers still re-exports the image helpers")
    _install_legacy_image_reexports(SIGLIP2)
    assert getattr(siglip2_module, _IMAGE_REEXPORT_FLAG, False) is True
    restored = _fresh_module(SIGLIP2)
    assert getattr(restored, _IMAGE_REEXPORT_FLAG, False) is False


def test_every_target_module_is_real():
    """A typo in the module list would make the fix quietly do nothing."""
    for name in _IMAGE_PROCESSING_MODULES:
        importlib.import_module(name)


def test_import_unsloth_does_not_pull_in_the_image_stack():
    """The fix must stay lazy: importing the siglip2 image module costs seconds."""
    import subprocess
    import sys

    code = (
        "import sys; import unsloth; "
        "print('transformers.models.siglip2.image_processing_siglip2' in sys.modules)"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output = True, text = True)
    if out.returncode != 0:
        # `import unsloth` did not finish here, so there is no import whose
        # laziness could be measured. Skip rather than fail: asserting tests the
        # runner, not the fix.
        #
        # Not keyed on one message or one raising function. Observed on CPU
        # runners: `NotImplementedError` from unsloth_zoo's `get_device_type`,
        # worded two different ways by version, and separately `ImportError:
        # cannot import name 'get_quant_type' from 'unsloth_zoo.utils'` on a
        # version-skewed runner. The stderr tail goes into the skip reason so
        # `-rs` still says which hosts opted out and why.
        pytest.skip(
            "import unsloth does not complete on this host; nothing to measure. "
            + out.stderr.strip()[-400:]
        )
    assert out.stdout.strip().splitlines()[-1] == "False", out.stdout[-2000:]


# The tests above drive `_install_legacy_image_reexports` directly, so every one
# of them still passes with the `fix_transformers5_image_processing_reexports()`
# call reverted out of `_gpu_init.py`: they prove the helper works, never that
# it is wired up. The two below close that gap and are the ones that fail when
# the functional hunk is removed.


def _unsloth_import_or_skip():
    """Skip when this host cannot finish `import unsloth` at all.

    The tests that use this assert on state `import unsloth` installs, but they
    do not own that import: the module-level `from unsloth.import_fixes import
    ...` above is what triggers it, and on a host with no supported accelerator
    it ends at `_gpu_init.py`'s device check, BEFORE the line that installs this
    fix. The assertion then reports the fix missing for a reason that has
    nothing to do with re-exports.

    Measured on GitHub's `macos-15` (Apple Silicon, mlx present but
    `is_mlx_available()` false, so neither import path completes):
    `NotImplementedError: Unsloth currently only works on NVIDIA GPUs and Intel
    GPUs` from `_gpu_init.py`, three red tests, while `ubuntu-latest`,
    `ubuntu-24.04-arm`, `windows-latest` and `macos-15-intel` were green on the
    same commit.

    Not keyed on one exception. A first version matched only the accelerator
    `NotImplementedError`, and the very next cross-platform pass failed
    `ubuntu-24.04-arm` and `windows-latest` on a different one entirely,
    `ImportError: cannot import name 'get_quant_type' from 'unsloth_zoo.utils'`,
    with the same three tests red for the same reason: no import happened. Any
    reason `import unsloth` cannot finish leaves nothing here to measure.

    This does not make the tests unfailable. The defect they exist to catch is a
    host that CAN import unsloth and still has no wrap, and that host does not
    skip. The skip reason names the exception so `-rs` shows which hosts opted
    out and why, and `test_every_import_path_installs_the_fix` keeps the wiring
    covered from any of them.
    """
    try:
        import unsloth  # noqa: F401
    except Exception as e:
        pytest.skip(
            f"import unsloth does not complete on this host "
            f"({type(e).__name__}: {str(e)[:160]}); no import to measure"
        )


def test_the_fix_is_actually_installed_on_import():
    """`import unsloth` must leave get_class_in_module wrapped."""
    from packaging.version import Version

    _unsloth_import_or_skip()

    if Version(transformers.__version__) < Version("5.0.0"):
        pytest.skip("no re-exports were dropped before transformers 5")
    from transformers import dynamic_module_utils

    assert getattr(dynamic_module_utils, "_unsloth_patched_get_class_in_module", False)
    assert hasattr(dynamic_module_utils.get_class_in_module, "__wrapped__")


def test_remote_code_reading_siglip_helpers_loads(tmp_path):
    """End to end through the entry point transformers really uses.

    A stand-in for microsoft/Phi-4-reasoning-vision-15B's own image processing
    file: same module, same decorator, same class-body timing. Driving
    `get_class_in_module` rather than importing the siglip module directly is
    the point, because the fix is deliberately lazy and only installs there.
    """
    import pathlib

    _unsloth_import_or_skip()

    from transformers import dynamic_module_utils
    from transformers.utils import HF_MODULES_CACHE

    siglip2 = _fresh_module(SIGLIP2)
    if not _image_processing_reexports_are_missing(siglip2):
        pytest.skip("this transformers still re-exports the image helpers")

    package = pathlib.Path(HF_MODULES_CACHE) / "unsloth_reexport_probe"
    package.mkdir(parents = True, exist_ok = True)
    (package / "__init__.py").write_text("")
    (package / "image_processing_probe.py").write_text(
        "import transformers.models.siglip2.image_processing_siglip2 as siglip2_ips\n"
        "\n"
        "class ProbeImageProcessor:\n"
        "    @siglip2_ips.filter_out_non_signature_kwargs()\n"
        "    def preprocess(self, images, **kwargs):\n"
        "        return siglip2_ips.to_numpy_array(images)\n"
    )
    try:
        loaded = dynamic_module_utils.get_class_in_module(
            "ProbeImageProcessor",
            "unsloth_reexport_probe/image_processing_probe.py",
            force_reload = True,
        )
        assert loaded.__name__ == "ProbeImageProcessor"
    finally:
        import shutil
        shutil.rmtree(package, ignore_errors = True)


# Helpers transformers 5 KEPT but re-specified from numpy (channel-last) to
# torch (channel-first). A module __getattr__ never fires for a name that still
# resolves, so forwarding cannot reach these: they need replacing. Phi-4's
# modeling_phi4_visionr.py builds numpy arrays at line 302 and hands them to
# both at lines 347-348.


def _numpy_image():
    np = pytest.importorskip("numpy")
    return np.arange(4 * 4 * 3, dtype = np.float32).reshape(4, 4, 3)


def test_retained_helpers_accept_the_numpy_arrays_remote_code_passes(siglip2_module):
    np = pytest.importorskip("numpy")
    _install_legacy_image_reexports(SIGLIP2)

    patches = siglip2_module.convert_image_to_patches(_numpy_image(), 2)
    assert isinstance(patches, np.ndarray)
    # 2x2 patches of 2x2x3 = 4 patches of 12 values, the transformers 4.x shape
    assert patches.shape == (4, 12)

    padded, mask = siglip2_module.pad_along_first_dim(patches, 6)
    assert isinstance(padded, np.ndarray)
    assert padded.shape == (6, 12)
    assert mask.tolist() == [1, 1, 1, 1, 0, 0]


def test_the_torch_contract_is_untouched(siglip2_module):
    """transformers' own Siglip2ImageProcessor calls these with tensors.

    Replacing them outright would fix the remote checkpoint by breaking the
    model the module is named after, so the shim dispatches on the argument.
    """
    torch = pytest.importorskip("torch")
    image = torch.arange(3 * 4 * 4, dtype = torch.float32).reshape(3, 4, 4)

    try:
        before = siglip2_module.convert_image_to_patches(image, 2).clone()
        before_pad, before_mask = siglip2_module.pad_along_first_dim(before, 6)
    except Exception:
        # transformers 4.x specified these for numpy only, so there is no torch
        # contract to preserve. Skipping rather than asserting one into existence.
        pytest.skip("this transformers has no torch contract for these helpers")

    _install_legacy_image_reexports(SIGLIP2)

    after = siglip2_module.convert_image_to_patches(image, 2)
    after_pad, after_mask = siglip2_module.pad_along_first_dim(after, 6)
    assert torch.equal(after, before)
    assert torch.equal(after_pad, before_pad)
    assert torch.equal(after_mask, before_mask)


def test_numpy_shim_is_idempotent_and_removable(siglip2_module):
    _install_legacy_image_reexports(SIGLIP2)
    once = siglip2_module.convert_image_to_patches
    _install_legacy_image_reexports(SIGLIP2)  # no-op, already flagged
    assert siglip2_module.convert_image_to_patches is once

    restored = _fresh_module(SIGLIP2)
    assert not getattr(restored.convert_image_to_patches, "_unsloth_numpy_dispatch", False)


@pytest.mark.parametrize("style", ["positional", "keyword", "legacy-keyword"])
def test_numpy_dispatch_covers_the_keyword_forms(siglip2_module, style):
    """Both helpers have a valid keyword form, and transformers renamed one.

    pad_along_first_dim's first parameter went from `array` (4.x) to `tensor`
    (5.x), so a 4.x caller using the keyword names something the current
    implementation does not accept at all.
    """
    np = pytest.importorskip("numpy")
    if not _image_processing_reexports_are_missing(siglip2_module):
        # transformers 4.x: the shim is correctly a no-op, and the 5.x spelling
        # of the first parameter does not exist there to be accepted.
        pytest.skip("this transformers still re-exports the image helpers")
    _install_legacy_image_reexports(SIGLIP2)
    image = _numpy_image()

    if style == "positional":
        patches = siglip2_module.convert_image_to_patches(image, 2)
        padded, mask = siglip2_module.pad_along_first_dim(patches, 6)
    elif style == "keyword":
        patches = siglip2_module.convert_image_to_patches(image = image, patch_size = 2)
        padded, mask = siglip2_module.pad_along_first_dim(
            tensor = patches,
            target_length = 6,
        )
    else:
        patches = siglip2_module.convert_image_to_patches(image = image, patch_size = 2)
        padded, mask = siglip2_module.pad_along_first_dim(
            array = patches,
            target_length = 6,
        )

    assert isinstance(patches, np.ndarray) and patches.shape == (4, 12)
    assert isinstance(padded, np.ndarray) and padded.shape == (6, 12)
    assert mask.tolist() == [1, 1, 1, 1, 0, 0]


def test_every_import_path_installs_the_fix():
    """Both entry points must call it, not just the CUDA one.

    `unsloth/__init__.py` returns early on Apple Silicon with MLX and never
    reaches `_gpu_init.py`, so the call added there alone left macOS unpatched.
    Caught by the macOS leg of cross-platform CI, held here so it fails
    everywhere: the running host cannot exercise the branch it is not on.
    """
    import pathlib

    root = pathlib.Path(__file__).parents[1] / "unsloth"
    for site in ("_gpu_init.py", "__init__.py"):
        source = (root / site).read_text(encoding = "utf-8")
        assert "fix_transformers5_image_processing_reexports" in source, site


def test_the_wrapper_is_reinstalled_after_a_module_reload():
    """`importlib.reload` restores upstream get_class_in_module but keeps our flag.

    Reload re-runs the module body in the EXISTING namespace, so the function
    goes back to upstream while a module attribute we added survives. A guard
    reading that attribute would then refuse to re-wrap a module that is once
    again unpatched; the guard reads the live function instead.
    """
    from packaging.version import Version

    if Version(transformers.__version__) < Version("5.0.0"):
        pytest.skip("no re-exports were dropped before transformers 5")
    _unsloth_import_or_skip()

    from unsloth.import_fixes import fix_transformers5_image_processing_reexports
    from transformers import dynamic_module_utils

    wrapped = dynamic_module_utils.get_class_in_module
    assert hasattr(wrapped, "__wrapped__")
    try:
        importlib.reload(dynamic_module_utils)
        assert not hasattr(dynamic_module_utils.get_class_in_module, "__wrapped__")
        # The module flag is exactly what survived, which is why it cannot be the guard.
        assert getattr(dynamic_module_utils, "_unsloth_patched_get_class_in_module", False)

        fix_transformers5_image_processing_reexports()
        assert hasattr(dynamic_module_utils.get_class_in_module, "__wrapped__")
    finally:
        dynamic_module_utils.get_class_in_module = wrapped
        dynamic_module_utils._unsloth_patched_get_class_in_module = True


def test_the_module_shims_are_reinstalled_after_a_module_reload(siglip2_module):
    """Reload restores the helpers the module body assigns; the flag survives.

    Measured, not assumed: `__getattr__` survives because the source never
    assigns it, while `convert_image_to_patches` and `pad_along_first_dim` are
    assigned by the body and come back as upstream torch implementations. So
    the module ends up HALF patched, and a guard reading the module flag would
    call that done and leave remote-code preprocessing broken again.
    """
    if not _image_processing_reexports_are_missing(siglip2_module):
        pytest.skip("this transformers still re-exports the image helpers")

    _install_legacy_image_reexports(SIGLIP2)
    dispatched = lambda: getattr(
        siglip2_module.convert_image_to_patches, "_unsloth_numpy_dispatch", False
    )
    assert dispatched()
    try:
        assert importlib.reload(siglip2_module) is siglip2_module
        assert getattr(siglip2_module, _IMAGE_REEXPORT_FLAG, False), "the flag survives"
        assert not dispatched(), "the numpy dispatch does not"

        assert _install_legacy_image_reexports(SIGLIP2) is True
        assert dispatched()
    finally:
        _remove_legacy_image_reexports(SIGLIP2)


# ---------------------------------------------------------------------------
# The same numpy/torch split one level up: BACKEND METHODS on the remote class.
#
# transformers 5 put a torchvision backend in every image processor's MRO, so a
# remote-code subclass that hands channel-last numpy to `self.normalize` reaches
# torchvision and raises. These tests drive the real classes; nothing here
# asserts on a version.

from unsloth.import_fixes import (  # noqa: E402
    _IMAGE_METHOD_BOUND,
    _IMAGE_METHOD_PATCH_FLAG,
    _LEGACY_NUMPY_IMAGE_METHODS,
    _install_legacy_numpy_image_methods,
    _is_remote_image_processor_class,
    _remove_legacy_numpy_image_methods,
    _resolved_image_method,
)

REMOTE_MODULE = "transformers_modules.unsloth_probe.image_processing_probe"


def _backend_module():
    """transformers 5's torchvision backend, or a skip where it cannot run.

    Two separate reasons to skip, and the second is not obvious. The module
    imports cleanly on transformers 5 even when torchvision is unusable, but it
    binds `tvF` (torchvision.transforms.v2.functional) only behind
    `is_torchvision_available()`, so its methods then raise
    `NameError: name 'tvF' is not defined` from inside transformers.

    Live on GitHub's `macos-15-intel`, where `pip install torch torchvision`
    resolves but the wheels are unusable: three tests here failed on that
    NameError, which says nothing about the shim. Asked of transformers' own
    probe rather than of `import torchvision`, which succeeds there.
    """
    module = pytest.importorskip("transformers.image_processing_backends")
    from transformers.utils import is_torchvision_available

    if not is_torchvision_available() or not hasattr(module, "tvF"):
        pytest.skip("torchvision is not usable here, so the backend cannot run")
    return module


@pytest.fixture
def remote_processor_class():
    """A subclass that looks exactly like one a checkpoint's own file defined.

    A real subclass of the real `Siglip2ImageProcessor`, with `__module__` set to
    where transformers puts remote code. Faking only the module string is the
    point: everything the classifier and the probe look at is genuine.
    """
    siglip2 = importlib.import_module(SIGLIP2)
    base = siglip2.Siglip2ImageProcessor

    cls = type("ProbeImageProcessorNoUpscale", (base,), {})
    cls.__module__ = REMOTE_MODULE
    yield cls
    _remove_legacy_numpy_image_methods(cls)


def _probe_image():
    np = pytest.importorskip("numpy")
    return np.arange(4 * 4 * 3, dtype = np.uint8).reshape(4, 4, 3)


def test_the_numpy_contract_is_restored_on_a_remote_subclass(remote_processor_class):
    """The whole point: channel-last numpy through the methods remote code calls."""
    np = pytest.importorskip("numpy")
    image_transforms = importlib.import_module("transformers.image_transforms")
    _backend_module()

    assert _install_legacy_numpy_image_methods(remote_processor_class) == ["rescale", "normalize"]
    inst = object.__new__(remote_processor_class)
    image = _probe_image()

    rescaled = inst.rescale(image = image, scale = 1.0 / 255.0, input_data_format = "channels_last")
    expected = image_transforms.rescale(
        image, scale = 1.0 / 255.0, input_data_format = "channels_last",
    )
    assert isinstance(rescaled, np.ndarray)
    assert rescaled.dtype == expected.dtype
    assert np.array_equal(rescaled, expected)

    normalized = inst.normalize(
        image = rescaled,
        mean = [0.5, 0.5, 0.5],
        std = [0.5, 0.5, 0.5],
        input_data_format = "channels_last",
    )
    expected = image_transforms.normalize(
        rescaled,
        mean = [0.5, 0.5, 0.5],
        std = [0.5, 0.5, 0.5],
        input_data_format = "channels_last",
    )
    assert isinstance(normalized, np.ndarray)
    assert normalized.dtype == expected.dtype
    assert np.array_equal(normalized, expected)


def test_rescale_is_in_scope_because_it_is_wrong_not_because_it_raises(remote_processor_class):
    """Pins the exact reason the gate cannot be "did it raise".

    `TorchvisionBackend.rescale` is `image * scale`, which numpy accepts and
    returns as float64 where transformers 4.x returned float32. Patching only
    the method that RAISES leaves the checkpoint's pixel_values float64, with
    nothing to notice.
    """
    np = pytest.importorskip("numpy")
    siglip2 = importlib.import_module(SIGLIP2)
    _backend_module()
    image = _probe_image()

    upstream = siglip2.Siglip2ImageProcessor().rescale(
        image = image, scale = 1.0 / 255.0, input_data_format = "channels_last",
    )
    if upstream.dtype == np.float32:
        pytest.skip("this transformers already returns the 4.x dtype from rescale")
    assert upstream.dtype == np.float64, "the silent half changed shape; re-verify the gate"

    _install_legacy_numpy_image_methods(remote_processor_class)
    patched = object.__new__(remote_processor_class).rescale(
        image = image, scale = 1.0 / 255.0, input_data_format = "channels_last",
    )
    assert patched.dtype == np.float32
    assert np.allclose(patched, upstream)


def test_transformers_own_image_processor_is_untouched(remote_processor_class):
    """Negative control, and the invariant the whole design rests on."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("PIL")
    np = pytest.importorskip("numpy")
    backends = _backend_module()
    siglip2 = importlib.import_module(SIGLIP2)
    from PIL import Image

    own = siglip2.Siglip2ImageProcessor
    image = Image.fromarray((np.random.RandomState(0).rand(64, 64, 3) * 255).astype(np.uint8))
    before = own()(images = [image], return_tensors = "pt")

    _install_legacy_numpy_image_methods(remote_processor_class)

    for name in _LEGACY_NUMPY_IMAGE_METHODS:
        assert name not in own.__dict__, f"{name} was set on transformers' own class"
        assert getattr(own, name) is getattr(backends.TorchvisionBackend, name)
    after = own()(images = [image], return_tensors = "pt")
    for key in before:
        assert torch.equal(
            torch.as_tensor(before[key]), torch.as_tensor(after[key]),
        ), f"{key} moved on transformers' own processor"


def test_the_probe_decides_not_the_version(remote_processor_class):
    """A class whose methods already honour numpy must be left alone.

    Built by subclassing `BaseImageProcessor` directly, skipping the torchvision
    backend, which is what the 4.x MRO looked like. If anyone swaps the probe
    for a `Version(...)` comparison, this goes red on transformers 5.
    """
    utils = importlib.import_module("transformers.image_processing_utils")
    cls = type("ProbeLegacyEraProcessor", (utils.BaseImageProcessor,), {})
    cls.__module__ = REMOTE_MODULE
    try:
        owner, _ = _resolved_image_method(cls, "normalize")
        if owner is None or "Torchvision" in owner.__name__:
            pytest.skip("BaseImageProcessor itself is torchvision-backed on this build")
        assert _install_legacy_numpy_image_methods(cls) == []
        for name in _LEGACY_NUMPY_IMAGE_METHODS:
            assert name not in cls.__dict__
    finally:
        _remove_legacy_numpy_image_methods(cls)


def test_the_torch_contract_is_untouched_on_the_patched_class(remote_processor_class):
    """Anything that is not a numpy array still reaches the inherited method."""
    torch = pytest.importorskip("torch")
    backends = _backend_module()

    _install_legacy_numpy_image_methods(remote_processor_class)
    inst = object.__new__(remote_processor_class)
    tensor = torch.arange(3 * 4 * 4, dtype = torch.float32).reshape(3, 4, 4) / 255.0

    assert torch.equal(
        inst.normalize(tensor, mean = [0.5] * 3, std = [0.5] * 3),
        backends.TorchvisionBackend.normalize(inst, tensor, mean = [0.5] * 3, std = [0.5] * 3),
    )
    assert torch.equal(
        inst.rescale(tensor, scale = 2.0),
        backends.TorchvisionBackend.rescale(inst, tensor, scale = 2.0),
    )


def test_the_classifier_rejects_everything_that_is_not_remote_remote(remote_processor_class):
    """Negative controls for `_is_remote_image_processor_class`."""
    siglip2 = importlib.import_module(SIGLIP2)
    configuration_utils = importlib.import_module("transformers.configuration_utils")
    processing_utils = importlib.import_module("transformers.processing_utils")

    class NotAClass:
        pass

    remote_config = type("RemoteConfig", (configuration_utils.PretrainedConfig,), {})
    remote_config.__module__ = REMOTE_MODULE
    remote_processor = type("RemoteProcessor", (processing_utils.ProcessorMixin,), {})
    remote_processor.__module__ = REMOTE_MODULE

    assert _is_remote_image_processor_class(remote_processor_class) is True
    for rejected in (
        42,
        "a string",
        NotAClass,
        remote_config,
        remote_processor,
        siglip2.Siglip2ImageProcessor,  # transformers' own, the one that must never match
    ):
        assert _is_remote_image_processor_class(rejected) is False, rejected


def test_a_method_the_remote_code_owns_is_never_replaced():
    """A checkpoint that wrote its own `normalize` keeps it."""
    np = pytest.importorskip("numpy")
    _backend_module()
    siglip2 = importlib.import_module(SIGLIP2)

    sentinel = object()

    def normalize(self, image, *args, **kwargs):
        return sentinel

    cls = type(
        "ProbeOwnNormalize", (siglip2.Siglip2ImageProcessor,), {"normalize": normalize},
    )
    cls.__module__ = REMOTE_MODULE
    try:
        assert _install_legacy_numpy_image_methods(cls) == ["rescale"]
        assert cls.__dict__["normalize"] is normalize
        assert object.__new__(cls).normalize(_probe_image()) is sentinel
    finally:
        _remove_legacy_numpy_image_methods(cls)


def test_install_is_idempotent_and_the_guard_reads_the_live_descriptor(remote_processor_class):
    """Second install is a no-op; a method that went back to upstream re-patches."""
    _backend_module()
    assert _install_legacy_numpy_image_methods(remote_processor_class) == ["rescale", "normalize"]
    first = remote_processor_class.__dict__["normalize"]

    assert _install_legacy_numpy_image_methods(remote_processor_class) == []
    assert remote_processor_class.__dict__["normalize"] is first

    # What a redefinition of the class body looks like from here: the flagged
    # function is gone while `_IMAGE_METHOD_BOUND` survives. A guard reading the
    # class attribute would call this done.
    delattr(remote_processor_class, "normalize")
    assert getattr(remote_processor_class, _IMAGE_METHOD_BOUND, None) is not None
    assert _install_legacy_numpy_image_methods(remote_processor_class) == ["normalize"]


def test_a_subclass_of_a_patched_class_is_not_double_wrapped(remote_processor_class):
    """It inherits one layer, and installing on it again does nothing."""
    np = pytest.importorskip("numpy")
    _backend_module()
    _install_legacy_numpy_image_methods(remote_processor_class)

    sub = type("ProbeSub", (remote_processor_class,), {})
    sub.__module__ = REMOTE_MODULE
    try:
        assert _install_legacy_numpy_image_methods(sub) == []
        assert "normalize" not in sub.__dict__
        out = object.__new__(sub).rescale(
            image = _probe_image(), scale = 1.0 / 255.0, input_data_format = "channels_last",
        )
        assert out.dtype == np.float32
    finally:
        _remove_legacy_numpy_image_methods(sub)


def test_the_method_shim_is_fully_removable(remote_processor_class):
    """Removal is delattr, because the method was always inherited."""
    backends = _backend_module()
    assert _install_legacy_numpy_image_methods(remote_processor_class) == ["rescale", "normalize"]

    assert sorted(_remove_legacy_numpy_image_methods(remote_processor_class)) == [
        "normalize", "rescale",
    ]
    for name in _LEGACY_NUMPY_IMAGE_METHODS:
        assert name not in remote_processor_class.__dict__
        assert getattr(remote_processor_class, name) is getattr(backends.TorchvisionBackend, name)
    assert not hasattr(remote_processor_class, _IMAGE_METHOD_BOUND)

    assert _install_legacy_numpy_image_methods(remote_processor_class) == ["rescale", "normalize"]


def test_the_dispatch_keeps_wraps_and_wrapped(remote_processor_class):
    """`functools.wraps` plus an explicit `__wrapped__`, and the flag survives both."""
    import inspect

    backends = _backend_module()
    _install_legacy_numpy_image_methods(remote_processor_class)

    dispatch = remote_processor_class.__dict__["normalize"]
    assert dispatch.__name__ == "normalize"
    assert dispatch.__wrapped__ is backends.TorchvisionBackend.normalize
    assert getattr(dispatch, _IMAGE_METHOD_PATCH_FLAG, False) is True, (
        "the flag must be set AFTER functools.wraps, which copies __dict__"
    )
    assert inspect.signature(dispatch) is not None


def test_no_shim_state_reaches_the_saved_config(remote_processor_class):
    """`_IMAGE_METHOD_BOUND` is a class attribute, so `to_dict` must not see it."""
    _backend_module()
    siglip2 = importlib.import_module(SIGLIP2)

    before = siglip2.Siglip2ImageProcessor().to_dict()
    _install_legacy_numpy_image_methods(remote_processor_class)
    instance = remote_processor_class()
    saved = instance.to_dict()

    assert not any(str(key).startswith("_unsloth") for key in saved), saved
    assert siglip2.Siglip2ImageProcessor().to_dict() == before


def test_the_remote_image_processor_finder_is_installed_once():
    """The unpickle path: one finder, and it answers for nothing else."""
    import sys

    from unsloth.import_fixes import (
        _REMOTE_IMAGE_FINDER_SENTINEL,
        _install_remote_image_processor_finder,
    )

    _unsloth_import_or_skip()
    _install_remote_image_processor_finder()
    _install_remote_image_processor_finder()

    installed = [
        finder for finder in sys.meta_path
        if getattr(finder, _REMOTE_IMAGE_FINDER_SENTINEL, False)
    ]
    assert len(installed) == 1
    assert installed[0].find_spec("json") is None
    assert installed[0].find_spec("transformers_modules.nope.not_here") is None


def test_remote_code_calling_the_backend_methods_on_numpy_loads_and_runs(tmp_path):
    """End to end through `get_class_in_module`, the way a checkpoint does it.

    The wiring test. Reverting the `_install_legacy_numpy_image_methods_now()`
    call out of the wrapper makes this raise the TypeError it exists to stop.
    """
    import pathlib

    np = pytest.importorskip("numpy")
    _unsloth_import_or_skip()
    _backend_module()

    from transformers import dynamic_module_utils
    from transformers.utils import HF_MODULES_CACHE

    siglip2 = _fresh_module(SIGLIP2)
    if not _image_processing_reexports_are_missing(siglip2):
        pytest.skip("this transformers still re-exports the image helpers")

    package = pathlib.Path(HF_MODULES_CACHE) / "unsloth_method_probe"
    package.mkdir(parents = True, exist_ok = True)
    (package / "__init__.py").write_text("")
    (package / "image_processing_probe.py").write_text(
        "import numpy as np\n"
        "import transformers.models.siglip2.image_processing_siglip2 as siglip2_ips\n"
        "\n"
        "class ProbeImageProcessor(siglip2_ips.Siglip2ImageProcessor):\n"
        "    def preprocess_like_2024(self, image):\n"
        "        image = self.rescale(image = image, scale = 1 / 255.0,\n"
        "                             input_data_format = 'channels_last')\n"
        "        return self.normalize(image = image, mean = [0.5] * 3, std = [0.5] * 3,\n"
        "                              input_data_format = 'channels_last')\n"
    )
    try:
        loaded = dynamic_module_utils.get_class_in_module(
            "ProbeImageProcessor",
            "unsloth_method_probe/image_processing_probe.py",
            force_reload = True,
        )
        image = np.arange(4 * 4 * 3, dtype = np.uint8).reshape(4, 4, 3)
        out = object.__new__(loaded).preprocess_like_2024(image)
        assert isinstance(out, np.ndarray)
        assert out.dtype == np.float32, "float64 here means rescale was left unpatched"
    finally:
        import shutil

        shutil.rmtree(package, ignore_errors = True)


# The unpickle path. `pickle` stores a processor as (module path, qualname), so a
# spawn-started DataLoader worker rebuilds the class by IMPORTING the remote
# module, never through `get_class_in_module`. Without the meta-path finder the
# child gets an upstream class and the first numpy `normalize` raises again.


def _remote_probe_package():
    """A real package under the remote-code root, so its module name is realistic."""
    import pathlib

    from transformers.dynamic_module_utils import init_hf_modules
    from transformers.utils import HF_MODULES_CACHE

    # Puts HF_MODULES_CACHE on sys.path and makes it a package; without it
    # `transformers_modules` is not importable and the spawn test errors on the
    # harness rather than on the fix.
    init_hf_modules()

    root = pathlib.Path(HF_MODULES_CACHE) / "transformers_modules"
    package = root / "unsloth_spawn_probe"
    package.mkdir(parents = True, exist_ok = True)
    (root / "__init__.py").touch(exist_ok = True)
    (package / "__init__.py").write_text("")
    (package / "image_processing_probe.py").write_text(
        "import transformers.models.siglip2.image_processing_siglip2 as siglip2_ips\n"
        "\n"
        "class SpawnProbeImageProcessor(siglip2_ips.Siglip2ImageProcessor):\n"
        "    def preprocess_like_2024(self, image):\n"
        "        image = self.rescale(image = image, scale = 1 / 255.0,\n"
        "                             input_data_format = 'channels_last')\n"
        "        return self.normalize(image = image, mean = [0.5] * 3, std = [0.5] * 3,\n"
        "                              input_data_format = 'channels_last')\n"
    )
    importlib.invalidate_caches()
    return package, "transformers_modules.unsloth_spawn_probe.image_processing_probe"


_SPAWN_CHILD = """
import pickle, sys
{preamble}
import numpy as np
# What a real worker already has: transformers puts HF_MODULES_CACHE on sys.path
# so a checkpoint's own module is importable. Without it the child fails on the
# harness instead of on the thing under test.
from transformers.dynamic_module_utils import init_hf_modules
init_hf_modules()
with open(sys.argv[1], "rb") as handle:
    processor = pickle.load(handle)
image = np.arange(4 * 4 * 3, dtype = np.uint8).reshape(4, 4, 3)
print("DTYPE", processor.preprocess_like_2024(image).dtype)
"""


def _run_spawn_child(pickled, preamble):
    import subprocess
    import sys

    return subprocess.run(
        [sys.executable, "-c", _SPAWN_CHILD.format(preamble = preamble), str(pickled)],
        capture_output = True,
        text = True,
    )


@pytest.fixture
def pickled_remote_processor(tmp_path):
    import pickle
    import shutil
    import sys

    _unsloth_import_or_skip()
    _backend_module()
    siglip2 = _fresh_module(SIGLIP2)
    if not _image_processing_reexports_are_missing(siglip2):
        pytest.skip("this transformers still re-exports the image helpers")

    package, module_name = _remote_probe_package()
    try:
        module = importlib.import_module(module_name)
        target = tmp_path / "processor.pkl"
        with open(target, "wb") as handle:
            pickle.dump(module.SpawnProbeImageProcessor(), handle)
        yield target
    finally:
        shutil.rmtree(package, ignore_errors = True)
        sys.modules.pop(module_name, None)


def test_a_spawn_started_worker_rebuilds_a_patched_class(pickled_remote_processor):
    """The finder's test: a fresh interpreter must still honour numpy."""
    out = _run_spawn_child(pickled_remote_processor, "import unsloth")
    if out.returncode != 0 and "get_device_type" not in out.stderr and (
        "unsloth" in out.stderr and "Error" in out.stderr and "TypeError" not in out.stderr
    ):
        pytest.skip(f"the child could not import unsloth: {out.stderr.strip()[-400:]}")
    assert out.returncode == 0, out.stderr[-2000:]
    assert "DTYPE float32" in out.stdout, (out.stdout, out.stderr[-2000:])


def test_a_spawn_started_worker_without_unsloth_is_the_documented_limit(
    pickled_remote_processor,
):
    """Negative control: proves the finder is what fixes the test above.

    Also pins the boundary honestly. A child that never imports unsloth is
    unpatched, and the failure is loud rather than a silent dtype change.
    """
    out = _run_spawn_child(pickled_remote_processor, "")
    assert out.returncode != 0, out.stdout
    assert "Functional F.normalize" in out.stderr or "numpy" in out.stderr, out.stderr[-2000:]


def test_deepcopy_and_pickle_keep_the_override_in_process(pickled_remote_processor):
    """Both keep `instance.__class__` by reference, so the patch travels with it."""
    import copy
    import pickle

    np = pytest.importorskip("numpy")
    with open(pickled_remote_processor, "rb") as handle:
        processor = pickle.load(handle)
    image = np.arange(4 * 4 * 3, dtype = np.uint8).reshape(4, 4, 3)

    assert processor.preprocess_like_2024(image).dtype == np.float32
    assert copy.deepcopy(processor).preprocess_like_2024(image).dtype == np.float32
    revived = pickle.loads(pickle.dumps(processor))
    assert revived.preprocess_like_2024(image).dtype == np.float32
