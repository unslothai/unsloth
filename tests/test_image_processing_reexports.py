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
        # `import unsloth` refuses to finish without an accelerator, so on a
        # CPU-only runner there is no import to measure the laziness of. Skip
        # rather than fail: asserting here tests the runner, not the fix.
        #
        # Keyed on the raising function, not its message: the wording is
        # unsloth_zoo's and differs by version ("cannot find any torch
        # accelerator" on current builds, "only works on NVIDIA GPUs and Intel
        # GPUs" on 2026.3.6), so a message match fails the test on the older
        # one for a reason that has nothing to do with this fix.
        if "get_device_type" in out.stderr and "NotImplementedError" in out.stderr:
            pytest.skip("import unsloth requires an accelerator; nothing to measure here")
        raise AssertionError(out.stderr[-2000:])
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

    Keyed on the raising function, not its message, for the reason
    `test_import_unsloth_does_not_pull_in_the_image_stack` spells out. Anything
    else still fails: a host that CAN import unsloth and still lacks the wrap is
    the defect these tests exist to catch. `test_every_import_path_installs_the_fix`
    keeps the wiring covered from a host that skips here.
    """
    import traceback

    try:
        import unsloth  # noqa: F401
    except NotImplementedError:
        if "get_device_type" in traceback.format_exc():
            pytest.skip("import unsloth needs an accelerator this host does not have")
        raise


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
