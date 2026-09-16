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


def test_an_unknown_attribute_still_gets_the_diagnosis_without_a_floor(patched_torch):
    """The table names the release; it is not the gate. A dtype added after this
    release must still be diagnosed."""
    with pytest.raises(import_fixes.UnslothTorchTooOldError) as raised:
        _access_from("transformers.integrations.finegrained_fp8", "unsloth_future_dtype")

    message = str(raised.value)
    assert "first appears in torch" not in message
    assert 'pip install --upgrade "torch"' in message


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
        assert sorted(set(calls)) == ["transformers"], calls
        assert len(calls) == 1, (
            f"the installed version was resolved {len(calls)} times for one package; "
            f"it cannot change inside a process and this path runs on every failed "
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
    assert "--force-reinstall --no-cache-dir triton-windows" in message
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
    assert "--force-reinstall --no-cache-dir triton\n" in logger.warnings[0]


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


def _isolated_run_temporary_patches(patches, logger):
    """`_run_temporary_patches` alone, with its two module globals supplied.

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

    namespace = {"TEMPORARY_PATCHES": patches, "logger": logger}
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
