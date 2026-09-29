# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""Version skew that used to surface deep in a forward pass, named at import.

Three reports, one shape: pip installs a combination it has no specifier to
reject, and the user sees an error from the wrong layer.

#8933: transformers 5.10 beside a torch without `torch.float8_e8m0fnu`, so
`import unsloth` ended on a bare `AttributeError: module 'torch' has no
attribute 'float8_e8m0fnu'`.

#2760: a repackaged triton whose CUDA driver shim includes Python.h without
defining PY_SSIZE_T_CLEAN, so the first kernel launch raised `SystemError:
PY_SSIZE_T_CLEAN macro must be defined for '#' formats`.

#3130: one temporary patch raised a SyntaxError and took `import unsloth` down
with it, because the caller tolerated only ValueError and TypeError.
"""

import ast
import pathlib
import re
import sys
import textwrap
import types

import pytest

from unsloth import import_fixes


_UNSLOTH = pathlib.Path(import_fixes.__file__).resolve().parent


# ---------------------------------------------------------------- #8933


def _access_from(module_name, attribute):
    """Read `torch.<attribute>` from a frame that belongs to `module_name`.

    The fix keys off the module the ACCESS was made from, so the frame has to be
    real; a call through a helper defined in this file would be attributed to
    the test.
    """
    import torch

    module = types.ModuleType(module_name)
    module.__dict__["torch"] = torch
    code = compile(f"value = torch.{attribute}\n", f"<{module_name}>", "exec")
    exec(code, module.__dict__)
    return module.__dict__["value"]


@pytest.fixture
def patched_torch(monkeypatch):
    """torch with the diagnosis installed and one attribute known to be absent."""
    import torch

    original = torch.__dict__.get("__getattr__")
    assert original is not None, "torch has no module-level __getattr__ to wrap"
    assert import_fixes.patch_torch_missing_attribute_error() is True
    monkeypatch.setitem(import_fixes._TORCH_ATTRIBUTE_FLOORS, "unsloth_probe_dtype", "2.7.0")
    try:
        yield torch
    finally:
        # Restore whatever was there, so an idempotence check in another test
        # still measures a first install.
        installed = torch.__dict__.get("__getattr__")
        if getattr(installed, "__unsloth_patched__", False):
            torch.__getattr__ = installed.__unsloth_original__


def test_a_dependency_reaching_for_a_missing_dtype_gets_the_upgrade(patched_torch):
    with pytest.raises(import_fixes.UnslothTorchTooOldError) as raised:
        _access_from("transformers.integrations.finegrained_fp8", "unsloth_probe_dtype")

    message = str(raised.value)
    # The original wording stays first, so anything already matching on it, and
    # anyone searching the web for it, still finds what they expect.
    assert message.startswith("module 'torch' has no attribute 'unsloth_probe_dtype'")
    assert "transformers==" in message
    assert patched_torch.__version__ in message
    assert "first appears in torch 2.7.0" in message
    assert 'pip install --upgrade "torch>=2.7.0"' in message
    # Still an AttributeError, so no caller's except clause changes meaning.
    assert isinstance(raised.value, AttributeError)


def test_an_unknown_attribute_is_diagnosed_without_prescribing_a_direction(patched_torch):
    """The table names the release; it is not the gate. A dtype added after this release
    must still be diagnosed, but an absent attribute is not evidence that torch is the
    older half: an older dependency reaching for a RETIRED torch API lands here too, and
    telling that user to upgrade torch is the opposite remedy. So the no-floor branch
    names both directions and prescribes neither."""
    with pytest.raises(import_fixes.UnslothTorchTooOldError) as raised:
        _access_from("transformers.integrations.finegrained_fp8", "unsloth_future_dtype")

    message = str(raised.value)
    assert "first appears in torch" not in message
    assert "so this torch is the older half" not in message
    assert "either newer than this torch or removed by it" in message
    # Both remedies are offered, neither as the answer.
    assert 'pip install --upgrade "torch"' in message
    assert "install a transformers that matches this torch" in message


def test_a_known_floor_still_prescribes_the_upgrade(patched_torch):
    """NEGATIVE CONTROL: a floor IS evidence of the direction, so that branch must keep
    saying which half is out of step rather than being softened with it."""
    with pytest.raises(import_fixes.UnslothTorchTooOldError) as raised:
        _access_from("transformers.integrations.finegrained_fp8", "unsloth_probe_dtype")

    message = str(raised.value)
    assert "first appears in torch 2.7.0, so this torch is the older half" in message
    assert "either newer than this torch or removed by it" not in message
    assert 'pip install --upgrade "torch>=2.7.0"' in message


@pytest.mark.parametrize(
    "requester",
    ["__main__", "my_training_script", "torch._inductor.analysis.device_info"],
)
def test_user_and_torch_frames_are_left_exactly_as_torch_wrote_them(patched_torch, requester):
    """A typo in user code, or torch asking itself, is not a version report."""
    with pytest.raises(AttributeError) as raised:
        _access_from(requester, "unsloth_probe_dtype")

    assert not isinstance(raised.value, import_fixes.UnslothTorchTooOldError)
    assert str(raised.value) == "module 'torch' has no attribute 'unsloth_probe_dtype'"


@pytest.mark.parametrize(
    "torch_version, family",
    [
        ("2.7.0+rocm6.3", "ROCm"),
        ("2.7.0+xpu", "XPU"),
        ("2.7.0+cpu", "CPU"),
        ("2.6.0+cu124", "CUDA"),
    ],
)
def test_the_upgrade_names_the_accelerator_it_found(
    patched_torch, monkeypatch, torch_version, family
):
    """pip's --index-url defaults to pypi.org, which ships one build per release: the
    default CUDA one. So an unqualified upgrade silently moves a ROCm or XPU user off
    their accelerator, and the message has to say so. It must NOT paste the installed
    tag into --index-url: each index carries only the releases built for it, and the
    reported torch 2.6.0+cu124 case would then get a command with no candidate at all
    (download.pytorch.org/whl/cu124 stops at torch 2.6.0; 2.7 shipped on cu126/cu128)."""
    monkeypatch.setattr(patched_torch, "__version__", torch_version)

    with pytest.raises(import_fixes.UnslothTorchTooOldError) as raised:
        _access_from("transformers.integrations.finegrained_fp8", "unsloth_probe_dtype")

    message = str(raised.value)
    assert 'pip install --upgrade "torch>=2.7.0"' in message
    assert "--index-url" in message
    assert (
        f"--index-url https://download.pytorch.org/whl/{torch_version.split('+')[1]}" not in message
    )
    assert f"This torch is a {torch_version.split('+')[1]} build" in message
    assert f"pick the {family} index" in message
    assert "https://pytorch.org/get-started/locally/" in message


@pytest.mark.parametrize(
    "torch_version",
    ["2.10.0+rocm7.2.0.lw.gitb6ee5fde", "2.10.0+rocm7.2.0.gitba5c1517"],
)
def test_a_vendor_rocm_torch_is_sent_back_to_its_own_source(
    patched_torch, monkeypatch, torch_version
):
    """unsloth's own AMD extras install torch-2.10.0+rocm7.2.0.lw.gitb6ee5fde straight from
    repo.radeon.com. That tag is no index name, so there is no --index-url that would keep
    the accelerator, and an unqualified upgrade would put the default CUDA build over a
    working Radeon one. Naming the source is the only advice that holds."""
    monkeypatch.setattr(patched_torch, "__version__", torch_version)

    with pytest.raises(import_fixes.UnslothTorchTooOldError) as raised:
        _access_from("transformers.integrations.finegrained_fp8", "unsloth_probe_dtype")

    message = str(raised.value)
    assert f"This torch is a {torch_version.split('+')[1]} build" in message
    assert "no public index carries" in message
    assert "repo.radeon.com" in message
    assert "--index-url" not in message


def test_an_unrecognisable_local_tag_gets_no_accelerator_note(patched_torch, monkeypatch):
    """NEGATIVE CONTROL: a local tag naming no known family says nothing about the
    accelerator, so inventing a source for it would be worse than silence."""
    monkeypatch.setattr(patched_torch, "__version__", "2.10.0+localbuild1")

    with pytest.raises(import_fixes.UnslothTorchTooOldError) as raised:
        _access_from("transformers.integrations.finegrained_fp8", "unsloth_probe_dtype")

    message = str(raised.value)
    assert "This torch is a" not in message
    assert "no public index carries" not in message
    assert "repo.radeon.com" not in message


def test_a_conda_torch_is_pointed_at_conda_not_pip(patched_torch, monkeypatch, tmp_path):
    """conda writes no local tag, so a conda torch reads like a PyPI one by version alone.
    `pip install --upgrade torch` there overlays the conda-managed files with a PyPI wheel
    and can change the backend with them, so the remedy has to name conda instead."""
    conda_meta = tmp_path / "conda-meta"
    conda_meta.mkdir()
    (conda_meta / "pytorch-2.7.0-py3.12_cuda12.4_cudnn9_0.json").write_text("{}", encoding = "utf-8")
    monkeypatch.setattr(import_fixes.sys, "prefix", str(tmp_path))
    monkeypatch.setattr(patched_torch, "__version__", "2.7.0")

    with pytest.raises(import_fixes.UnslothTorchTooOldError) as raised:
        _access_from("transformers.integrations.finegrained_fp8", "unsloth_probe_dtype")

    message = str(raised.value)
    assert "conda installed this torch" in message
    assert "conda update pytorch" in message
    assert "--index-url" not in message


@pytest.mark.parametrize("torch_version", ["2.7.0", "2.9.0.dev20250101", "2.7.0+fbcode"])
def test_a_pypi_torch_gets_no_accelerator_note(patched_torch, monkeypatch, torch_version):
    """NEGATIVE CONTROL: no recognised backend tag means PyPI is already the right
    place, so the plain command stands alone with nothing extra to read."""
    monkeypatch.setattr(patched_torch, "__version__", torch_version)

    with pytest.raises(import_fixes.UnslothTorchTooOldError) as raised:
        _access_from("transformers.integrations.finegrained_fp8", "unsloth_probe_dtype")

    message = str(raised.value)
    assert 'pip install --upgrade "torch>=2.7.0"' in message
    assert "--index-url" not in message
    assert "get-started" not in message


def test_an_unattributable_access_keeps_torchs_own_error(patched_torch, monkeypatch):
    """sys._getframe is CPython-only and raises an audit event, so a hardened
    interpreter can refuse it. hasattr() swallows only AttributeError, so rethrowing
    the frame-lookup failure would break an ordinary feature probe."""

    def _refuse(_depth):
        raise RuntimeError("frame inspection is not allowed here")

    monkeypatch.setattr(sys, "_getframe", _refuse)

    with pytest.raises(AttributeError) as raised:
        _access_from("transformers.integrations.finegrained_fp8", "unsloth_probe_dtype")

    assert not isinstance(raised.value, import_fixes.UnslothTorchTooOldError)
    assert str(raised.value) == "module 'torch' has no attribute 'unsloth_probe_dtype'"
    assert hasattr(patched_torch, "__version__")
    assert getattr(patched_torch, "unsloth_probe_dtype", "fallback") == "fallback"


def test_hasattr_and_getattr_default_are_unaffected(patched_torch):
    """The error subclasses AttributeError precisely so these keep working;
    transformers 5.17 guards this very dtype with hasattr."""
    assert not hasattr(patched_torch, "unsloth_probe_dtype")
    assert getattr(patched_torch, "unsloth_probe_dtype", "fallback") == "fallback"


def test_a_failed_probe_does_not_re_read_the_installed_metadata(patched_torch, monkeypatch):
    """A missing attribute is not always fatal, so the wrapper has to stay cheap.

    `hasattr(torch, name)` and `getattr(torch, name, default)` are how these same
    libraries feature-probe, and both reach the wrapper. Resolving a distribution
    version walks sys.path and costs about 850 microseconds on a normal install,
    which is a thousand times a failed lookup, so it is asked once per package and
    remembered. Counted rather than timed: a timing threshold on a shared runner is
    a flake.
    """
    calls = []

    def counting_version(package):
        calls.append(package)
        return "9.9.9"

    import_fixes._installed_version.cache_clear()
    monkeypatch.setattr(import_fixes, "importlib_version", counting_version)
    try:
        for _ in range(25):
            assert not hasattr(patched_torch, "unsloth_probe_dtype")
        # transformers is not in the frames above (hasattr here is attributed to
        # this test module), so drive the diagnosing path explicitly too.
        for _ in range(25):
            with pytest.raises(import_fixes.UnslothTorchTooOldError):
                _access_from("transformers.integrations.finegrained_fp8", "unsloth_probe_dtype")
        # The message also names the companion packages that have to move with torch, and
        # each of those is a lookup too. What must hold is ONE lookup per package across 50
        # failed accesses, not one lookup in total.
        assert "transformers" in calls
        assert set(calls) <= {"transformers", "torchvision", "torchaudio"}, calls
        assert len(calls) == len(set(calls)), (
            f"the installed version was resolved more than once for some package: {calls}. "
            f"It cannot change inside a process and this path runs on every failed "
            f"attribute lookup"
        )
    finally:
        import_fixes._installed_version.cache_clear()


def test_an_unresolvable_package_still_reports_a_version_word(patched_torch, monkeypatch):
    """The word unknown rather than a traceback, and cached like any other answer."""

    def always_raises(package):
        raise RuntimeError("no metadata here")

    import_fixes._installed_version.cache_clear()
    monkeypatch.setattr(import_fixes, "importlib_version", always_raises)
    try:
        with pytest.raises(import_fixes.UnslothTorchTooOldError) as raised:
            _access_from("transformers.integrations.finegrained_fp8", "unsloth_probe_dtype")
        assert "transformers==unknown" in str(raised.value)
    finally:
        import_fixes._installed_version.cache_clear()


def test_installing_twice_wraps_once(patched_torch):
    first = patched_torch.__dict__["__getattr__"]
    assert import_fixes.patch_torch_missing_attribute_error() is True
    second = patched_torch.__dict__["__getattr__"]
    assert first is second
    assert not getattr(first.__unsloth_original__, "__unsloth_patched__", False)


def test_a_torch_without_a_module_getattr_is_left_alone(monkeypatch):
    """The wrapper has one prerequisite, so say what happens without it: nothing,
    and it says so rather than raising."""
    monkeypatch.setitem(sys.modules, "torch", types.ModuleType("torch"))
    assert import_fixes.patch_torch_missing_attribute_error() is False


def test_the_diagnosis_is_installed_before_anything_imports_transformers():
    """Ordering is the whole fix: installed after `import unsloth_zoo`, it would
    arrive after the import that fails."""
    source = (_UNSLOTH / "_gpu_init.py").read_text(encoding = "utf-8")
    install = source.find("patch_torch_missing_attribute_error()")
    # The statement, not the several comments that name it.
    zoo = re.search(r"^ {4}import unsloth_zoo$", source, re.M)
    assert install != -1, (
        "DRIFT DETECTED: patch_torch_missing_attribute_error is defined but never "
        "called in _gpu_init.py, so real imports never install it."
    )
    assert zoo is not None, "the `import unsloth_zoo` statement moved"
    assert install < zoo.start(), (
        "DRIFT DETECTED: patch_torch_missing_attribute_error is called after "
        "`import unsloth_zoo`, which is what imports transformers and raises."
    )
    assert "del patch_torch_missing_attribute_error" in source


def test_the_rocm_id_table_is_configured_before_torch_is_imported():
    """`patch_torch_missing_attribute_error` imports torch, so it has to come after the
    ROCm table. `configure_amdgpu_asic_id_table_path` sets AMDGPU_ASIC_ID_TABLE_PATH, which
    is how ROCm resolves AMD device names, and a torch that has already brought up libdrm
    would not see the discovered table."""
    source = (_UNSLOTH / "_gpu_init.py").read_text(encoding = "utf-8")
    table = source.find("configure_amdgpu_asic_id_table_path()")
    install = source.find("patch_torch_missing_attribute_error()")
    assert table != -1 and install != -1
    assert table < install, (
        "DRIFT DETECTED: the torch diagnosis, which imports torch, is installed before "
        "the ROCm ASIC id table path is configured."
    )


def test_a_conda_triton_is_repaired_through_conda(monkeypatch, tmp_path):
    """conda-forge publishes `triton`, and a pip --force-reinstall over a conda-managed one
    overlays conda's files with a wheel, leaving two managers owning the same paths."""
    conda_meta = tmp_path / "conda-meta"
    conda_meta.mkdir()
    (conda_meta / "triton-3.2.0-py312_0.json").write_text("{}", encoding = "utf-8")
    monkeypatch.setattr(import_fixes.sys, "prefix", str(tmp_path))

    spec, _driver = _fake_triton(tmp_path, _UNGUARDED_SHIM)
    monkeypatch.setattr(sys, "version_info", (3, 12, 3))
    monkeypatch.setattr(import_fixes.importlib.util, "find_spec", lambda name: spec)

    import importlib.metadata as metadata

    monkeypatch.setattr(metadata, "packages_distributions", lambda: {"triton": ["triton"]})
    monkeypatch.setattr(
        import_fixes,
        "importlib_version",
        lambda name: "3.2.0" if name == "triton" else _raise_missing(name),
    )

    logger = _CollectingLogger()
    monkeypatch.setattr(import_fixes, "logger", logger)
    import_fixes.check_triton_py_ssize_t_clean()

    message = logger.warnings[0]
    assert "conda installed this Triton" in message
    assert "conda install --force-reinstall triton=3.2.0" in message
    assert "pip install" not in message


def test_a_pip_triton_in_a_conda_prefix_still_gets_pip(monkeypatch, tmp_path):
    """NEGATIVE CONTROL: the conda-meta entry has to name this distribution at this
    version, or a conda prefix alone would redirect every pip-installed Triton."""
    conda_meta = tmp_path / "conda-meta"
    conda_meta.mkdir()
    (conda_meta / "pytorch-2.7.0-py312_0.json").write_text("{}", encoding = "utf-8")
    monkeypatch.setattr(import_fixes.sys, "prefix", str(tmp_path))

    spec, _driver = _fake_triton(tmp_path, _UNGUARDED_SHIM)
    monkeypatch.setattr(sys, "version_info", (3, 12, 3))
    monkeypatch.setattr(import_fixes.importlib.util, "find_spec", lambda name: spec)

    import importlib.metadata as metadata

    monkeypatch.setattr(metadata, "packages_distributions", lambda: {"triton": ["triton"]})
    monkeypatch.setattr(
        import_fixes,
        "importlib_version",
        lambda name: "3.2.0" if name == "triton" else _raise_missing(name),
    )

    logger = _CollectingLogger()
    monkeypatch.setattr(import_fixes, "logger", logger)
    import_fixes.check_triton_py_ssize_t_clean()

    message = logger.warnings[0]
    assert "conda installed this Triton" not in message
    assert '--force-reinstall --no-cache-dir "triton==3.2.0"' in message


# ---------------------------------------------------------------- #2760


_UNGUARDED_SHIM = textwrap.dedent(
    """
    #include "cuda.h"
    #include <Python.h>

    static PyObject *loadBinary(PyObject *self, PyObject *args) {
      const char *name;
      const char *data;
      Py_ssize_t data_size;
      int shared;
      int device;
      if (!PyArg_ParseTuple(args, "ss#ii", &name, &data, &data_size, &shared,
                            &device)) {
        return NULL;
      }
      return NULL;
    }
    """
)


def _fake_triton(
    tmp_path,
    source,
    backend = "nvidia",
):
    driver = tmp_path / "triton" / "backends" / backend / "driver.c"
    driver.parent.mkdir(parents = True)
    driver.write_text(source, encoding = "utf-8")
    spec = types.SimpleNamespace(
        submodule_search_locations = [str(tmp_path / "triton")],
    )
    return spec, driver


def test_a_shim_without_the_macro_is_named(monkeypatch, tmp_path):
    spec, driver = _fake_triton(tmp_path, _UNGUARDED_SHIM)
    monkeypatch.setattr(sys, "version_info", (3, 12, 3))
    monkeypatch.setattr(import_fixes.importlib.util, "find_spec", lambda name: spec)

    offenders = import_fixes._triton_driver_shims_missing_py_ssize_t_clean()
    assert offenders == [("nvidia", str(driver))]


def test_a_shim_with_the_macro_is_not_named(monkeypatch, tmp_path):
    fixed = "#define PY_SSIZE_T_CLEAN\n" + _UNGUARDED_SHIM
    spec, _ = _fake_triton(tmp_path, fixed)
    monkeypatch.setattr(sys, "version_info", (3, 12, 3))
    monkeypatch.setattr(import_fixes.importlib.util, "find_spec", lambda name: spec)

    assert import_fixes._triton_driver_shims_missing_py_ssize_t_clean() == []


def test_a_shim_with_no_hash_format_is_not_named(monkeypatch, tmp_path):
    """Only a '#' in a PyArg_Parse format needs the macro, so a shim without one
    is not a finding no matter what it includes."""
    spec, _ = _fake_triton(tmp_path, _UNGUARDED_SHIM.replace('"ss#ii"', '"sslii"'))
    monkeypatch.setattr(sys, "version_info", (3, 12, 3))
    monkeypatch.setattr(import_fixes.importlib.util, "find_spec", lambda name: spec)

    assert import_fixes._triton_driver_shims_missing_py_ssize_t_clean() == []


@pytest.mark.parametrize(
    "comment",
    [
        '/* legacy: PyArg_ParseTuple(args, "ss#ii", &name, &data, &size, &shared); */',
        '// legacy: PyArg_ParseTuple(args, "ss#ii", &name, &data, &size, &shared);',
    ],
    ids = ["block", "line"],
)
def test_a_hash_format_only_in_a_comment_is_not_a_finding(monkeypatch, tmp_path, comment):
    """A repackaged shim can keep the old call around as a comment. The compiler never
    sees it, so the active parser is the one without a '#' and the first kernel launch
    does not fail. Naming it would promise a failure that cannot happen and recommend a
    force reinstall for nothing."""
    source = comment + "\n" + _UNGUARDED_SHIM.replace('"ss#ii"', '"sslii"')
    spec, _ = _fake_triton(tmp_path, source)
    monkeypatch.setattr(sys, "version_info", (3, 12, 3))
    monkeypatch.setattr(import_fixes.importlib.util, "find_spec", lambda name: spec)

    assert import_fixes._triton_driver_shims_missing_py_ssize_t_clean() == []


def test_a_live_hash_format_beside_a_comment_is_still_a_finding(monkeypatch, tmp_path):
    """NEGATIVE CONTROL: masking comments must not mask the call itself, and a `//`
    inside a string literal must not blank the rest of the file."""
    source = (
        "/* the parser below still uses a length format */\n"
        'static const char *kDoc = "see //triton/backends for /* details */";\n' + _UNGUARDED_SHIM
    )
    spec, driver = _fake_triton(tmp_path, source)
    monkeypatch.setattr(sys, "version_info", (3, 12, 3))
    monkeypatch.setattr(import_fixes.importlib.util, "find_spec", lambda name: spec)

    assert import_fixes._triton_driver_shims_missing_py_ssize_t_clean() == [
        ("nvidia", str(driver)),
    ]


def test_a_hash_format_only_inside_a_string_literal_is_not_a_finding(monkeypatch, tmp_path):
    """A documentation string quoting an example call is not a call. The format has to be
    readable, so literals are kept, but the CALL has to be located in code."""
    source = (
        "static const char *kUsage =\n"
        '    "example: PyArg_ParseTuple(args, \\"ss#ii\\", &name, &data, &size, &shared)";\n'
        + _UNGUARDED_SHIM.replace('"ss#ii"', '"sslii"')
    )
    spec, _ = _fake_triton(tmp_path, source)
    monkeypatch.setattr(sys, "version_info", (3, 12, 3))
    monkeypatch.setattr(import_fixes.importlib.util, "find_spec", lambda name: spec)

    assert import_fixes._triton_driver_shims_missing_py_ssize_t_clean() == []


def test_a_vendor_tagged_triton_is_sent_back_to_its_own_source(monkeypatch, tmp_path):
    """The AMD extras install triton==3.6.0+rocm7.2.0.gitba5c1517 from a repo.radeon.com
    URL under the plain `triton` name. No public index publishes that version, so a pinned
    pip command resolves to nothing and the advice has to name the source instead."""
    spec, _driver = _fake_triton(tmp_path, _UNGUARDED_SHIM, backend = "amd")
    monkeypatch.setattr(sys, "version_info", (3, 12, 3))
    monkeypatch.setattr(import_fixes.importlib.util, "find_spec", lambda name: spec)

    import importlib.metadata as metadata

    monkeypatch.setattr(metadata, "packages_distributions", lambda: {"triton": ["triton"]})
    monkeypatch.setattr(
        import_fixes,
        "importlib_version",
        lambda name: "3.6.0+rocm7.2.0.gitba5c1517" if name == "triton" else _raise_missing(name),
    )

    logger = _CollectingLogger()
    monkeypatch.setattr(import_fixes, "logger", logger)
    import_fixes.check_triton_py_ssize_t_clean()

    assert len(logger.warnings) == 1, logger.warnings
    message = logger.warnings[0]
    assert "+rocm7.2.0.gitba5c1517 tag marks a vendor build" in message
    assert "repo.radeon.com" in message


def test_a_public_triton_still_gets_the_plain_reinstall_line(monkeypatch, tmp_path):
    """NEGATIVE CONTROL: a version PyPI does publish keeps the command as the advice."""
    spec, _driver = _fake_triton(tmp_path, _UNGUARDED_SHIM)
    monkeypatch.setattr(sys, "version_info", (3, 12, 3))
    monkeypatch.setattr(import_fixes.importlib.util, "find_spec", lambda name: spec)

    import importlib.metadata as metadata

    monkeypatch.setattr(metadata, "packages_distributions", lambda: {"triton": ["triton"]})
    monkeypatch.setattr(
        import_fixes,
        "importlib_version",
        lambda name: "3.6.0" if name == "triton" else _raise_missing(name),
    )

    logger = _CollectingLogger()
    monkeypatch.setattr(import_fixes, "logger", logger)
    import_fixes.check_triton_py_ssize_t_clean()

    message = logger.warnings[0]
    assert "vendor build" not in message
    assert '--force-reinstall --no-cache-dir "triton==3.6.0"' in message


def test_python_313_and_later_do_not_pay_for_the_probe(monkeypatch, tmp_path):
    """CPython stopped requiring the macro in 3.13, so the shim is fine there and
    the file is never read."""
    spec, _ = _fake_triton(tmp_path, _UNGUARDED_SHIM)
    monkeypatch.setattr(sys, "version_info", (3, 13, 1))

    def refuse(name):
        raise AssertionError("the probe read the filesystem on Python 3.13")

    monkeypatch.setattr(import_fixes.importlib.util, "find_spec", refuse)
    assert import_fixes._triton_driver_shims_missing_py_ssize_t_clean() == []


def test_the_installed_triton_is_not_flagged():
    """Drift detector, not a unit test: every triton on PyPI from 3.0.0 to 3.8.0
    defines the macro, and this must stay a silent check for real installs."""
    pytest.importorskip("triton")
    if sys.version_info >= (3, 13):
        pytest.skip("the probe is inert on Python 3.13 and later")
    offenders = import_fixes._triton_driver_shims_missing_py_ssize_t_clean()
    assert (
        offenders == []
    ), f"the installed triton would fail at the first kernel launch: {offenders}"


@pytest.mark.parametrize(
    "prefix,suffix,named",
    [
        ("#define PY_SSIZE_T_CLEAN\n", "", False),
        ("#  define   PY_SSIZE_T_CLEAN 1\n", "", False),
        ("/* PY_SSIZE_T_CLEAN is not needed here */\n", "", True),
        ("#define PY_SSIZE_T_CLEAN\n#undef PY_SSIZE_T_CLEAN\n", "", True),
        ("", "#define PY_SSIZE_T_CLEAN\n", True),
        ('const char *why = "PY_SSIZE_T_CLEAN";\n', "", True),
        # The compiler strips comments before the preprocessor sees a directive, so a
        # define that exists only inside one is not a definition at all.
        ("/*\n#define PY_SSIZE_T_CLEAN\n*/\n", "", True),
        ("/* x */ /*\n#  define PY_SSIZE_T_CLEAN 1\n*/\n", "", True),
        ("//#define PY_SSIZE_T_CLEAN\n", "", True),
        # A real define followed by an UNDEF hidden in a comment is still defined, which
        # is the control for blanking too much.
        ("#define PY_SSIZE_T_CLEAN\n/*\n#undef PY_SSIZE_T_CLEAN\n*/\n", "", False),
        # A comment between the define and the include changes nothing.
        ("#define PY_SSIZE_T_CLEAN\n/* now include it */\n", "", False),
    ],
)
def test_the_macro_only_counts_when_it_is_in_effect(monkeypatch, tmp_path, prefix, suffix, named):
    """CPython requires the define BEFORE Python.h. A comment, an `#undef`, a string
    literal and a define placed after the include all leave the '#' formats unsafe, so
    the substring test that accepted them suppressed the warning on a shim that still
    dies at the first kernel launch."""
    spec, driver = _fake_triton(tmp_path, prefix + _UNGUARDED_SHIM + suffix)
    monkeypatch.setattr(sys, "version_info", (3, 12, 3))
    monkeypatch.setattr(import_fixes.importlib.util, "find_spec", lambda name: spec)

    offenders = import_fixes._triton_driver_shims_missing_py_ssize_t_clean()
    assert offenders == ([("nvidia", str(driver))] if named else [])


def test_the_reinstall_command_names_the_distribution_that_owns_triton(monkeypatch, tmp_path):
    """triton-windows, pytorch-triton-rocm and pytorch-triton-xpu all provide the
    `triton` import name, so `importlib.metadata.version("triton")` raises on them (the
    "unknown" version) and `pip install --force-reinstall triton` installs a CUDA build
    over a platform one instead of repairing it."""
    spec, _driver = _fake_triton(tmp_path, _UNGUARDED_SHIM)
    monkeypatch.setattr(sys, "version_info", (3, 12, 3))
    monkeypatch.setattr(import_fixes.importlib.util, "find_spec", lambda name: spec)

    import importlib.metadata as metadata

    monkeypatch.setattr(metadata, "packages_distributions", lambda: {"triton": ["triton-windows"]})
    monkeypatch.setattr(
        import_fixes,
        "importlib_version",
        lambda name: "3.3.1.post19" if name == "triton-windows" else _raise_missing(name),
    )

    logger = _CollectingLogger()
    monkeypatch.setattr(import_fixes, "logger", logger)
    import_fixes.check_triton_py_ssize_t_clean()

    assert len(logger.warnings) == 1, logger.warnings
    message = logger.warnings[0]
    assert "triton-windows==3.3.1.post19" in message
    assert '--force-reinstall --no-cache-dir "triton-windows==3.3.1.post19"' in message
    assert "unknown" not in message


def _raise_missing(name):
    from importlib.metadata import PackageNotFoundError
    raise PackageNotFoundError(name)


def test_the_plain_triton_distribution_is_still_named(monkeypatch, tmp_path):
    """The control: on an ordinary CUDA install nothing about the message changes."""
    spec, _driver = _fake_triton(tmp_path, _UNGUARDED_SHIM)
    monkeypatch.setattr(sys, "version_info", (3, 12, 3))
    monkeypatch.setattr(import_fixes.importlib.util, "find_spec", lambda name: spec)

    import importlib.metadata as metadata

    monkeypatch.setattr(metadata, "packages_distributions", lambda: {"triton": ["triton"]})
    monkeypatch.setattr(import_fixes, "importlib_version", lambda name: "3.3.0")

    logger = _CollectingLogger()
    monkeypatch.setattr(import_fixes, "logger", logger)
    import_fixes.check_triton_py_ssize_t_clean()

    assert "triton==3.3.0" in logger.warnings[0]
    assert '--force-reinstall --no-cache-dir "triton==3.3.0"\n' in logger.warnings[0]


@pytest.mark.parametrize(
    "distribution, triton_version, torch_version, expected",
    [
        # The ordinary CUDA install: PyPI carries this wheel, so only the pin is added.
        (
            "triton",
            "3.2.0",
            "2.6.0+cu124",
            'pip install --force-reinstall --no-cache-dir "triton==3.2.0"',
        ),
        # Windows builds also come from PyPI, and are not a pytorch-triton-* provider.
        (
            "triton-windows",
            "3.3.1.post19",
            "2.7.1+cu126",
            'pip install --force-reinstall --no-cache-dir "triton-windows==3.3.1.post19"',
        ),
        # ROCm and XPU Triton exist only on download.pytorch.org, under torch's own tag.
        (
            "pytorch-triton-rocm",
            "3.3.0",
            "2.7.0+rocm6.3",
            "pip install --force-reinstall --no-cache-dir"
            ' --index-url https://download.pytorch.org/whl/rocm6.3 "pytorch-triton-rocm==3.3.0"',
        ),
        (
            "pytorch-triton-xpu",
            "3.3.0",
            "2.7.0+xpu",
            "pip install --force-reinstall --no-cache-dir"
            ' --index-url https://download.pytorch.org/whl/xpu "pytorch-triton-xpu==3.3.0"',
        ),
        # NEGATIVE CONTROL: no version to pin, so the command stays bare rather than
        # inventing a pin that would resolve to no wheel at all.
        ("triton", "unknown", "2.6.0+cu124", "pip install --force-reinstall --no-cache-dir triton"),
    ],
)
def test_the_reinstall_command_pins_the_installed_triton(
    monkeypatch, distribution, triton_version, torch_version, expected
):
    """An unpinned --force-reinstall resolves the NEWEST provider release, but torch pins
    Triton exactly (torch 2.6.0 requires triton==3.2.0), so the bare command can replace a
    broken shim with a working-but-ABI-incompatible Triton."""
    torch_module = types.ModuleType("torch")
    torch_module.__version__ = torch_version
    monkeypatch.setitem(sys.modules, "torch", torch_module)

    assert import_fixes._triton_reinstall_command(distribution, triton_version) == expected


def test_the_triton_probe_is_wired_into_gpu_init():
    source = (_UNSLOTH / "_gpu_init.py").read_text(encoding = "utf-8")
    assert "check_triton_py_ssize_t_clean()" in source, (
        "DRIFT DETECTED: check_triton_py_ssize_t_clean is defined but never called "
        "in _gpu_init.py."
    )
    assert "del check_triton_py_ssize_t_clean" in source


# ---------------------------------------------------------------- #3130


class _CollectingLogger:
    def __init__(self):
        self.warnings = []

    def warning(self, message):
        self.warnings.append(message)


def _isolated_run_temporary_patches(
    patches,
    logger,
    outcomes = None,
):
    """`_run_temporary_patches` alone, with its module globals supplied.

    Loading it out of the file rather than importing unsloth.models._utils keeps
    this a test of the control flow and not of a full model-stack import.
    """
    path = _UNSLOTH / "models" / "_utils.py"
    source = path.read_text(encoding = "utf-8")
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "_run_temporary_patches":
            segment = ast.get_source_segment(source, node)
            break
    else:
        raise AssertionError("_run_temporary_patches is gone from unsloth/models/_utils.py")

    namespace = {
        "TEMPORARY_PATCHES": patches,
        "logger": logger,
        "TEMPORARY_PATCH_OUTCOMES": {} if outcomes is None else outcomes,
    }
    exec(compile(segment, str(path), "exec"), namespace)
    return namespace["_run_temporary_patches"]


def test_a_patch_that_raises_is_skipped_not_fatal():
    called = []

    def raises_syntax_error():
        # What patch_merge_quantization_configs did: exec'd
        # `from transformers.quantizers.auto import ()`.
        raise SyntaxError("invalid syntax")

    def later_patch():
        called.append("later_patch")

    logger = _CollectingLogger()
    run = _isolated_run_temporary_patches([raises_syntax_error, later_patch], logger)

    run("init")

    assert called == ["later_patch"], "a raising patch stopped the ones after it"
    assert len(logger.warnings) == 1
    assert "raises_syntax_error" in logger.warnings[0]
    assert "SyntaxError" in logger.warnings[0]


def test_a_patch_is_called_once_and_gets_its_phase():
    calls = []

    def takes_phase(phase):
        calls.append(("takes_phase", phase))

    def takes_nothing():
        calls.append(("takes_nothing", None))

    def raises_value_error(phase):
        calls.append(("raises_value_error", phase))
        raise ValueError("a patch may legitimately raise this")

    logger = _CollectingLogger()
    run = _isolated_run_temporary_patches([takes_phase, takes_nothing, raises_value_error], logger)

    run("pre_compile")

    # The ValueError one used to be retried without its argument, because the
    # signature probe and the call shared an except clause.
    assert calls == [
        ("takes_phase", "pre_compile"),
        ("takes_nothing", None),
        ("raises_value_error", "pre_compile"),
    ]
    assert len(logger.warnings) == 1


def test_a_callable_with_no_readable_signature_is_still_called():
    logger = _CollectingLogger()
    # inspect.signature raises ValueError for some builtins, which is the case the
    # original except clause existed for.
    run = _isolated_run_temporary_patches([print], logger)
    run("init")
    assert logger.warnings == []


@pytest.mark.parametrize(
    "installed, version",
    [
        ("pytorch-triton-rocm", "3.3.0"),
        # download.pytorch.org renamed both accelerator providers. pytorch-triton-rocm
        # stops at 3.5.1 and triton-rocm carries 3.7 onwards, the same split the XPU
        # provider has, so a current ROCm host is only found under the new name.
        ("triton-rocm", "3.8.0"),
        ("triton-xpu", "3.7.1"),
    ],
)
def test_the_provider_lookup_still_works_without_packages_distributions(
    monkeypatch, installed, version
):
    """`importlib.metadata.packages_distributions` is Python 3.10+, and pyproject still
    admits 3.9. There the import raises, the mapping comes back empty, and the message
    used to report `triton==unknown` and recommend the CUDA `triton` over the platform
    build, which is the exact substitution this helper exists to prevent."""
    import importlib.metadata as metadata

    def _absent():
        raise ImportError("no packages_distributions on this interpreter")

    monkeypatch.setattr(metadata, "packages_distributions", _absent, raising = False)
    monkeypatch.setitem(sys.modules, "importlib_metadata", None)
    monkeypatch.setattr(
        import_fixes,
        "importlib_version",
        lambda name: version if name == installed else _raise_missing(name),
    )
    import_fixes._installed_version.cache_clear()
    try:
        assert import_fixes._triton_distribution() == (installed, version)
    finally:
        import_fixes._installed_version.cache_clear()


def test_the_provider_fallback_reports_unknown_when_nothing_is_installed(monkeypatch):
    """NEGATIVE CONTROL: the fallback asks each provider whether it is installed, so a
    host with none of them must not pick one anyway."""
    import importlib.metadata as metadata

    monkeypatch.setattr(metadata, "packages_distributions", lambda: {}, raising = False)
    monkeypatch.setitem(sys.modules, "importlib_metadata", None)
    monkeypatch.setattr(import_fixes, "importlib_version", _raise_missing)
    import_fixes._installed_version.cache_clear()
    try:
        assert import_fixes._triton_distribution() == ("triton", "unknown")
    finally:
        import_fixes._installed_version.cache_clear()


def test_the_mlx_branch_installs_the_torch_diagnosis():
    """_gpu_init.py is the only other installation site and the MLX branch never reaches
    it, so an Apple Silicon host with the old-torch/new-transformers pair would get the
    bare AttributeError this PR exists to replace. The branch already mirrors three other
    _gpu_init fixes for exactly this reason."""
    source = (_UNSLOTH / "__init__.py").read_text(encoding = "utf-8")
    mlx_branch = source[source.index("if _IS_MLX:") : source.index("import unsloth_zoo")]
    assert "patch_torch_missing_attribute_error" in mlx_branch, (
        "DRIFT DETECTED: the MLX branch no longer installs the torch-too-old diagnosis, "
        "so it is installed only on the GPU path."
    )


def _fake_distribution(files_by_name):
    """importlib.metadata.distribution, answering from `{name: [paths]}`."""

    class _Entry:
        def __init__(self, path):
            self._path = path

        def locate(self):
            return self._path

    class _Dist:
        def __init__(self, paths):
            self.files = [_Entry(p) for p in paths]

    def _distribution(name):
        if name not in files_by_name:
            from importlib.metadata import PackageNotFoundError
            raise PackageNotFoundError(name)
        return _Dist(files_by_name[name])

    return _distribution


@pytest.mark.parametrize(
    "order", [["triton", "pytorch-triton-xpu"], ["pytorch-triton-xpu", "triton"]]
)
def test_the_provider_that_ships_the_offending_file_is_the_one_named(monkeypatch, tmp_path, order):
    """Providers coexist: install_python_stack.py's _ensure_xpu_triton documents generic
    triton beside pytorch-triton-xpu on the same paths, and packages_distributions reports
    both without saying which one wrote the file. Taking the first entry can name the CUDA
    provider for an XPU install and hand the user a command that replaces the Triton that
    works -- the substitution this helper exists to avoid. The answer must not depend on
    the order the mapping happens to return."""
    driver = tmp_path / "backends" / "intel" / "driver.c"
    driver.parent.mkdir(parents = True)
    driver.write_text("x", encoding = "utf-8")

    import importlib.metadata as metadata

    monkeypatch.setattr(metadata, "packages_distributions", lambda: {"triton": list(order)})
    monkeypatch.setattr(
        metadata,
        "distribution",
        _fake_distribution(
            {
                "triton": [str(tmp_path / "backends" / "nvidia" / "driver.c")],
                "pytorch-triton-xpu": [str(driver)],
            }
        ),
    )
    monkeypatch.setattr(import_fixes, "importlib_version", lambda name: "3.3.0")
    import_fixes._installed_version.cache_clear()
    try:
        assert import_fixes._triton_distribution(str(driver))[0] == "pytorch-triton-xpu"
    finally:
        import_fixes._installed_version.cache_clear()


@pytest.mark.parametrize(
    "order",
    [("triton", "triton-xpu"), ("triton-xpu", "triton")],
    ids = ["generic-first", "accelerator-first"],
)
def test_a_file_claimed_by_two_providers_is_settled_by_torchs_backend(monkeypatch, tmp_path, order):
    """Coexisting providers RECORD the same paths, so both claim the offending file and the
    first entry is nothing but ordering. The harmful answer is always the one that installs
    a CUDA build over an accelerator one, so an XPU torch settles it for the XPU provider."""
    driver = tmp_path / "backends" / "intel" / "driver.c"
    driver.parent.mkdir(parents = True)
    driver.write_text("x", encoding = "utf-8")

    import importlib.metadata as metadata

    torch_module = types.ModuleType("torch")
    torch_module.__version__ = "2.10.0+xpu"
    monkeypatch.setitem(sys.modules, "torch", torch_module)
    monkeypatch.setattr(metadata, "packages_distributions", lambda: {"triton": list(order)})
    monkeypatch.setattr(
        metadata,
        "distribution",
        # Both RECORDs list the same file, which is the coexistence case.
        _fake_distribution({"triton": [str(driver)], "triton-xpu": [str(driver)]}),
    )
    monkeypatch.setattr(import_fixes, "importlib_version", lambda name: "3.7.1")
    import_fixes._installed_version.cache_clear()
    try:
        assert import_fixes._triton_distribution(str(driver))[0] == "triton-xpu"
    finally:
        import_fixes._installed_version.cache_clear()


@pytest.mark.parametrize(
    "order",
    [("pytorch-triton-xpu", "triton-xpu"), ("triton-xpu", "pytorch-triton-xpu")],
    ids = ["stale-first", "current-first"],
)
def test_a_rename_leftover_is_settled_by_what_torch_requires(monkeypatch, tmp_path, order):
    """Both spellings name the backend, so the family rule cannot separate them.

    An upgrade to a torch that installs `triton-xpu` leaves `pytorch-triton-xpu`'s
    dist-info behind, both RECORDs claim the driver, and recommending the stale one would
    install it over the provider this torch actually requires. torch declares that name
    itself, which is the only record that settles it.
    """
    driver = tmp_path / "backends" / "intel" / "driver.c"
    driver.parent.mkdir(parents = True)
    driver.write_text("x", encoding = "utf-8")

    import importlib.metadata as metadata

    torch_module = types.ModuleType("torch")
    torch_module.__version__ = "2.10.0+xpu"
    monkeypatch.setitem(sys.modules, "torch", torch_module)
    monkeypatch.setattr(
        metadata,
        "requires",
        lambda name: ['triton-xpu==3.7.1; platform_system == "Linux"'] if name == "torch" else [],
    )
    monkeypatch.setattr(metadata, "packages_distributions", lambda: {"triton": list(order)})
    monkeypatch.setattr(
        metadata,
        "distribution",
        _fake_distribution({name: [str(driver)] for name in order}),
    )
    monkeypatch.setattr(import_fixes, "importlib_version", lambda name: "3.7.1")
    import_fixes._installed_version.cache_clear()
    try:
        assert import_fixes._triton_distribution(str(driver))[0] == "triton-xpu"
    finally:
        import_fixes._installed_version.cache_clear()


def test_the_declared_requirement_is_read_by_distribution_name(monkeypatch):
    """The names are compared PEP 503 normalised, so an underscore spelling in torch's
    own metadata still matches the dist-info name, and a non-Triton requirement is not
    collected."""
    import importlib.metadata as metadata

    monkeypatch.setattr(
        metadata,
        "requires",
        lambda name: [
            "filelock",
            'pytorch_triton_xpu==3.7.1; platform_system == "Linux"',
            "sympy>=1.13.3",
        ],
    )
    assert import_fixes._torch_required_triton_distributions() == frozenset({"pytorch-triton-xpu"})


def test_two_claimants_a_cuda_torch_cannot_separate_name_the_generic_one(monkeypatch, tmp_path):
    """NEGATIVE CONTROL: with no accelerator in torch's tag there is no accelerator to
    preserve, so the generic provider is the answer rather than whichever came first."""
    driver = tmp_path / "backends" / "nvidia" / "driver.c"
    driver.parent.mkdir(parents = True)
    driver.write_text("x", encoding = "utf-8")

    import importlib.metadata as metadata

    torch_module = types.ModuleType("torch")
    torch_module.__version__ = "2.10.0+cu128"
    monkeypatch.setitem(sys.modules, "torch", torch_module)
    monkeypatch.setattr(
        metadata, "packages_distributions", lambda: {"triton": ["triton-windows", "triton"]}
    )
    monkeypatch.setattr(
        metadata,
        "distribution",
        _fake_distribution({"triton": [str(driver)], "triton-windows": [str(driver)]}),
    )
    monkeypatch.setattr(import_fixes, "importlib_version", lambda name: "3.7.1")
    import_fixes._installed_version.cache_clear()
    try:
        assert import_fixes._triton_distribution(str(driver))[0] == "triton"
    finally:
        import_fixes._installed_version.cache_clear()


def test_an_unownable_path_falls_back_to_the_previous_answer(monkeypatch, tmp_path):
    """NEGATIVE CONTROL: metadata that names no owner, or no path at all, must leave the
    ordering-based answer exactly as it was rather than reporting nothing."""
    import importlib.metadata as metadata

    monkeypatch.setattr(metadata, "packages_distributions", lambda: {"triton": ["triton-windows"]})
    monkeypatch.setattr(metadata, "distribution", _fake_distribution({}))
    monkeypatch.setattr(import_fixes, "importlib_version", lambda name: "3.3.1.post19")
    import_fixes._installed_version.cache_clear()
    try:
        assert import_fixes._triton_distribution(None)[0] == "triton-windows"
        assert import_fixes._triton_distribution(str(tmp_path / "nope.c"))[0] == "triton-windows"
    finally:
        import_fixes._installed_version.cache_clear()


def test_the_upgrade_moves_the_companions_that_pin_torch_exactly(patched_torch, monkeypatch):
    """`pip install --upgrade` upgrades the packages it is given and nothing else.

    Every torchvision wheel requires an exact `torch==X.Y.Z`, so naming torch alone moves
    torch and leaves the torchvision built against the old one: the "operator
    torchvision::nms does not exist" mismatch that `_torchvision_repair_command` in this
    same file exists to repair, created by following the remedy.
    """
    monkeypatch.setattr(
        import_fixes,
        "importlib_version",
        lambda name: "9.9.9" if name in ("transformers", "torchvision") else _raise_missing(name),
    )
    import_fixes._installed_version.cache_clear()
    try:
        with pytest.raises(import_fixes.UnslothTorchTooOldError) as raised:
            _access_from("transformers.integrations.finegrained_fp8", "unsloth_probe_dtype")
        message = str(raised.value)
        assert 'pip install --upgrade "torch>=2.7.0" "torchvision"' in message
        # NEGATIVE CONTROL: torchaudio is not installed in this case, so it is not named.
        assert "torchaudio" not in message
    finally:
        import_fixes._installed_version.cache_clear()


def test_a_bare_torch_install_gets_the_plain_upgrade(patched_torch, monkeypatch):
    """NEGATIVE CONTROL: nothing that pins torch is installed, so nothing extra is named."""
    monkeypatch.setattr(
        import_fixes,
        "importlib_version",
        lambda name: "9.9.9" if name == "transformers" else _raise_missing(name),
    )
    import_fixes._installed_version.cache_clear()
    try:
        with pytest.raises(import_fixes.UnslothTorchTooOldError) as raised:
            _access_from("transformers.integrations.finegrained_fp8", "unsloth_probe_dtype")
        assert 'pip install --upgrade "torch>=2.7.0"\n' in str(raised.value)
    finally:
        import_fixes._installed_version.cache_clear()


@pytest.mark.parametrize(
    "distribution, torch_version, expect_index",
    [
        # torch 2.10 renamed pytorch-triton-xpu to triton-xpu, and pyproject pins the new
        # name straight from download.pytorch.org, where the old prefix no longer matches.
        ("triton-xpu", "2.10.0+xpu", True),
        ("pytorch-triton-xpu", "2.7.0+xpu", True),
        ("pytorch-triton-rocm", "2.7.0+rocm6.3", True),
        ("triton-rocm", "2.10.0+rocm6.4", True),
        # NEGATIVE CONTROLS: PyPI serves these, so no index is added.
        ("triton", "2.6.0+cu124", False),
        ("triton-windows", "2.7.1+cu126", False),
    ],
)
def test_every_torch_index_triton_provider_gets_the_accelerator_index(
    monkeypatch, distribution, torch_version, expect_index
):
    torch_module = types.ModuleType("torch")
    torch_module.__version__ = torch_version
    monkeypatch.setitem(sys.modules, "torch", torch_module)

    command = import_fixes._triton_reinstall_command(distribution, "3.6.0")
    assert ("--index-url https://download.pytorch.org/whl/" in command) is expect_index
    assert f'"{distribution}==3.6.0"' in command
