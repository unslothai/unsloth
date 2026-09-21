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
    assert out.returncode == 0, out.stderr[-2000:]
    assert out.stdout.strip().splitlines()[-1] == "False", out.stdout[-2000:]
