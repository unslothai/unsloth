# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Whether `install_llm_compressor()` reaches for pip, per install state.

The guard exists because the in-process `import llmcompressor` can fail even when the
distribution is fine -- Unsloth's transformers patches interfere in that process, while the
real quantization runs in a clean subprocess that imports it without trouble. Falling through
to pip there made it re-resolve the version-capped, torch/transformers-pinned spec and
backtrack destructively (it drags in numpy<2 built from source and the export dies).

Presence alone is not the whole question though. A distribution outside
``_LLM_COMPRESSOR_SPEC`` is exactly what that pin exists to correct, so it must still be
reinstalled rather than silently quantized against -- and neither is metadata health: an
install missing a dependency reads as present and in range, and pip is what repairs it.

No GPU and no network. What is substituted is the two boundaries the decision reads
(``importlib.metadata.version``, and ``subprocess.run`` for the clean-subprocess import
probe) and the one it acts on (``subprocess.check_call``).
"""

from __future__ import annotations

import importlib.metadata as md
import os
import subprocess
import sys

import pytest

from unsloth.save import (
    _LLM_COMPRESSOR_PROBE_SENTINEL,
    _LLM_COMPRESSOR_SPEC,
    install_llm_compressor,
)


def _pip_invoked_when(
    monkeypatch,
    reported: str | None,
    *,
    subprocess_import: object = 0,
    probe_version: str = "",
    probe_location: str = "",
    probe_noise: str = "",
    probes_out: list | None = None,
) -> bool:
    """Run the guard with llmcompressor reported as *reported*, answering "did it pip?".

    ``reported=None`` presents a genuinely absent distribution. ``subprocess_import`` is what
    the clean-subprocess probe finds: an exit status, or an exception instance to raise from
    ``subprocess.run`` (a timeout, a python that will not spawn).
    """
    calls: list[list[str]] = []
    probes: list[list[str]] = probes_out if probes_out is not None else []
    real_version = md.version

    def fake_run(cmd, *a, **k):
        probes.append(list(cmd))
        if isinstance(subprocess_import, BaseException):
            raise subprocess_import
        # The probe writes ONE tagged record, and it writes it AFTER whatever a
        # sitecustomize or an imported dependency has already put on stdout, which
        # ``probe_noise`` stands in for. The version is the module it actually IMPORTED,
        # which is not necessarily the version metadata reports for the same-named
        # distribution.
        import json

        stdout = probe_noise + (
            "\n"
            + _LLM_COMPRESSOR_PROBE_SENTINEL
            + " "
            + json.dumps({"version": probe_version, "location": probe_location})
            + "\n"
        )
        return subprocess.CompletedProcess(cmd, subprocess_import, stdout = stdout.encode())

    monkeypatch.setattr(subprocess, "run", fake_run)

    def fake_version(dist: str) -> str:
        if dist == "llmcompressor":
            if reported is None:
                raise md.PackageNotFoundError(dist)
            return reported
        return real_version(dist)

    monkeypatch.setattr(md, "version", fake_version)
    monkeypatch.setattr(subprocess, "check_call", lambda cmd, *a, **k: calls.append(list(cmd)))

    # The real in-process import must fail for the guard to be reached at all. It does when
    # llmcompressor is absent from this interpreter; when it is genuinely installed the first
    # branch returns early and there is nothing here to measure.
    try:
        import llmcompressor  # noqa: F401
    except Exception:
        pass
    else:
        pytest.skip("llmcompressor imports in this interpreter, so the guard is unreachable")

    # After a "successful" install the function re-imports and raises, which is the outcome
    # for the absent case rather than a harness failure.
    try:
        install_llm_compressor()
    except Exception:
        pass
    return bool(calls)


def test_an_absent_distribution_is_still_installed(monkeypatch):
    # Absent means the clean subprocess cannot import it either, which is the answer that
    # separates a genuinely missing package from a metadata-free checkout below.
    assert _pip_invoked_when(monkeypatch, None, subprocess_import = 1) is True


def test_a_metadata_free_checkout_the_export_can_import_is_not_reinstalled(monkeypatch):
    """No metadata is not the same as not installed.

    A source checkout on PYTHONPATH reports no version, so the presence read raises
    PackageNotFoundError. If the clean subprocess can still import it, the export would have
    worked, and pip re-resolving the capped spec over it is the destructive outcome this
    guard exists to avoid. So the probe decides here too, not the absence of metadata.
    """
    probes: list[list[str]] = []
    assert _pip_invoked_when(monkeypatch, None, probes_out = probes) is False
    assert len(probes) == 1, f"expected one clean-subprocess probe, got {probes}"
    # By FILE, the way the export runner is launched, not `-c`: `-c` puts the cwd on
    # sys.path and the runner does not, so a checkout visible only through the cwd passed
    # the probe and then failed in the runner after the whole merge.
    assert probes[0][0] == sys.executable, probes[0]
    assert probes[0][1].endswith(".py"), probes[0]
    assert "-c" not in probes[0], probes[0]


def test_a_supported_version_is_not_reinstalled(monkeypatch):
    # The reported case: installed, in range, and importable in a clean subprocess, so the
    # in-process failure was Unsloth's own patches and pip has nothing to fix.
    probes: list[list[str]] = []
    assert _pip_invoked_when(monkeypatch, "0.12.0", probes_out = probes) is False
    # And it was decided by asking, not assumed: one probe, this interpreter, -c so unsloth
    # is never imported into it.
    assert len(probes) == 1, f"expected one clean-subprocess probe, got {probes}"
    # By FILE, the way the export runner is launched, not `-c`: `-c` puts the cwd on
    # sys.path and the runner does not, so a checkout visible only through the cwd passed
    # the probe and then failed in the runner after the whole merge.
    assert probes[0][0] == sys.executable, probes[0]
    assert probes[0][1].endswith(".py"), probes[0]
    assert "-c" not in probes[0], probes[0]


def test_an_installed_version_that_cannot_import_anywhere_is_repaired(monkeypatch):
    """Metadata is not health. An incomplete distribution -- compressed-tensors missing, a
    truncated install -- also presents as present and in range, and pip is what fixes it.
    Skipping there defers the failure past the whole model merge to _compressed_quantize.py,
    which imports the same two symbols this probe does."""
    assert _pip_invoked_when(monkeypatch, "0.12.0", subprocess_import = 1) is True


@pytest.mark.parametrize(
    "unknown",
    [
        subprocess.TimeoutExpired(cmd = ["python"], timeout = 600),
        OSError("no such interpreter"),
    ],
    ids = ["timeout", "will-not-spawn"],
)
def test_an_unknowable_probe_answer_keeps_the_install_skipped(monkeypatch, unknown):
    """The destructive re-resolve is the worse outcome, so an unanswered probe is not
    evidence of a broken install and must not be treated as one."""
    assert _pip_invoked_when(monkeypatch, "0.12.0", subprocess_import = unknown) is False


def test_the_probe_asks_the_question_the_export_runner_will_ask():
    """The probe is only evidence if it imports what _compressed_quantize.py imports, in the
    way it is launched: `sys.executable <runner>` with the ambient environment, never `-m`
    and never with unsloth imported. A probe of some other symbol could pass while the
    export's own import fails, which is the failure this is supposed to catch."""
    import inspect
    from pathlib import Path

    from unsloth.save import _llm_compressor_imports_cleanly

    probe_src = inspect.getsource(_llm_compressor_imports_cleanly)
    runner_src = (
        Path(inspect.getfile(install_llm_compressor)).parent / "_compressed_quantize.py"
    ).read_text()
    for line in (
        "from llmcompressor import oneshot",
        "from llmcompressor.modifiers.quantization import QuantizationModifier",
    ):
        assert line in runner_src, f"the export runner no longer imports: {line}"
        assert line in probe_src, f"the probe no longer imports: {line}"
    assert "sys.executable" in probe_src, "the probe must use this interpreter"
    assert "timeout" in probe_src, "an unbounded probe would hang the export"
    assert sys.executable


@pytest.mark.parametrize("version", ["0.5.0", "1.0.0", "0.13.0"])
def test_a_version_outside_the_pin_is_reinstalled(monkeypatch, version):
    """Presence is not support. Skipping these would quantize against an unsupported
    release and leave the pin decorative."""
    from packaging.requirements import Requirement

    assert not Requirement(_LLM_COMPRESSOR_SPEC).specifier.contains(
        version, prereleases = True
    ), f"{version} must be outside {_LLM_COMPRESSOR_SPEC} for this case to mean anything"
    assert _pip_invoked_when(monkeypatch, version) is True


def test_the_pin_the_guard_compares_against_is_a_bounded_range():
    """A spec that stopped bounding either end would make the checks above vacuous."""
    from packaging.requirements import Requirement

    spec = Requirement(_LLM_COMPRESSOR_SPEC).specifier
    ops = {s.operator for s in spec}
    assert ops & {">=", ">", "=="}, f"no floor in {_LLM_COMPRESSOR_SPEC}"
    assert ops & {"<=", "<", "=="}, f"no ceiling in {_LLM_COMPRESSOR_SPEC}"


def _install_outcome(monkeypatch, *, subprocess_import: int):
    """Run the guard through the INSTALL path, answering "what did it do afterwards?".

    Returns the function's result, or the exception it raised.
    """
    real_version = md.version

    def fake_version(dist: str) -> str:
        if dist == "llmcompressor":
            raise md.PackageNotFoundError(dist)
        return real_version(dist)

    monkeypatch.setattr(md, "version", fake_version)
    # The install "succeeds" without installing anything, so the re-import after it fails --
    # which is exactly what it does in the environment this guard is about.
    monkeypatch.setattr(subprocess, "check_call", lambda cmd, *a, **k: None)
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda cmd, *a, **k: subprocess.CompletedProcess(cmd, subprocess_import),
    )
    try:
        import llmcompressor  # noqa: F401
    except Exception:
        pass
    else:
        pytest.skip("llmcompressor imports in this interpreter, so the guard is unreachable")
    try:
        return install_llm_compressor()
    except Exception as exc:
        return exc


def test_a_repaired_install_is_validated_where_the_export_will_use_it(monkeypatch):
    """The import after the install runs in THIS process, with Unsloth's transformers patches
    in it, while the quantizing happens in a subprocess that has none. Raising on that killed
    an export whose environment was by then perfectly good."""
    assert _install_outcome(monkeypatch, subprocess_import = 0) == (None, None)


def test_an_install_that_fixed_nothing_still_fails_loudly(monkeypatch):
    """The other half: if a clean subprocess cannot import it either, the install genuinely
    did not work and proceeding would waste the whole model merge before saying so."""
    outcome = _install_outcome(monkeypatch, subprocess_import = 1)
    assert isinstance(outcome, RuntimeError), outcome
    assert "could not be imported" in str(outcome)


def _with_importable_llmcompressor(monkeypatch, reported: str):
    """Present an llmcompressor that IMPORTS, at version *reported*.

    The other cases here rely on the interpreter not having it, which is what made them
    blind to this branch: an out-of-range release that imports fine returns from the very
    first try block, before any version check. Substituting the module is the only way to
    reach that path without installing an unsupported release.
    """
    import types

    real_version = md.version

    def fake_version(dist: str) -> str:
        if dist == "llmcompressor":
            return reported
        return real_version(dist)

    monkeypatch.setattr(md, "version", fake_version)

    pkg = types.ModuleType("llmcompressor")
    pkg.oneshot = lambda *a, **k: None
    quant = types.ModuleType("llmcompressor.modifiers.quantization")
    quant.QuantizationModifier = type("QuantizationModifier", (), {})
    mods = types.ModuleType("llmcompressor.modifiers")
    mods.quantization = quant
    pkg.modifiers = mods
    for name, module in (
        ("llmcompressor", pkg),
        ("llmcompressor.modifiers", mods),
        ("llmcompressor.modifiers.quantization", quant),
    ):
        monkeypatch.setitem(sys.modules, name, module)

    calls: list[list[str]] = []
    monkeypatch.setattr(subprocess, "check_call", lambda cmd, *a, **k: calls.append(list(cmd)))
    monkeypatch.setattr(subprocess, "run", lambda cmd, *a, **k: subprocess.CompletedProcess(cmd, 0))
    return calls


def test_a_supported_version_that_imports_is_used_directly(monkeypatch):
    """The ordinary fast path: in range and importable, so the real symbols come back and
    nothing is installed."""
    calls = _with_importable_llmcompressor(monkeypatch, "0.12.0")
    oneshot, modifier = install_llm_compressor()
    assert oneshot is not None and modifier is not None
    assert calls == [], "a supported install must not be touched"


@pytest.mark.parametrize("version", ["0.5.0", "1.0.0", "0.13.0"])
def test_an_out_of_range_version_is_not_accepted_just_because_it_imports(monkeypatch, version):
    """The gap: importability is not support. Returning the symbols here would quantize
    against a release _LLM_COMPRESSOR_SPEC explicitly excludes, and the pin would be
    decorative for exactly the installs it was written for."""
    from packaging.requirements import Requirement

    assert not Requirement(_LLM_COMPRESSOR_SPEC).specifier.contains(
        version, prereleases = True
    ), f"{version} must be outside {_LLM_COMPRESSOR_SPEC} for this case to mean anything"
    calls = _with_importable_llmcompressor(monkeypatch, version)
    try:
        install_llm_compressor()
    except Exception:
        pass
    assert calls, f"{version} imported and was accepted without being corrected"
    assert any(_LLM_COMPRESSOR_SPEC in " ".join(cmd) for cmd in calls), calls


def test_an_out_of_range_module_is_repaired_even_when_metadata_looks_fine(monkeypatch):
    """The version that matters is the one the import RESOLVES.

    importlib.metadata answers for a distribution, and a source checkout earlier on
    sys.path shadows an installed wheel. A stale in-range wheel therefore blessed an
    out-of-range checkout that both the probe and the export subprocess actually import,
    so the pin was decorative exactly where it is load bearing.
    """
    from packaging.requirements import Requirement

    assert not Requirement(_LLM_COMPRESSOR_SPEC).specifier.contains(
        "0.13.0", prereleases = True
    ), "0.13.0 must be outside the pin for this case to mean anything"

    # Metadata says the wheel is fine; the import resolves a newer checkout.
    assert (
        _pip_invoked_when(monkeypatch, "0.12.0", probe_version = "0.13.0") is True
    ), "an out-of-range module was accepted because a same-named wheel was in range"
    # And the in-range case is still skipped, so this is not just "always reinstall".
    assert _pip_invoked_when(monkeypatch, "0.12.0", probe_version = "0.12.0") is False
    # A module with no __version__ to read is unknown, and unknown stays usable: the
    # alternative is the destructive re-resolve this guard exists to avoid.
    assert _pip_invoked_when(monkeypatch, "0.12.0", probe_version = "") is False


def test_the_probe_runs_the_way_the_export_runner_is_launched():
    """`sys.executable <file>`, with sys.path[0] the runner's directory.

    `-c` puts the CURRENT DIRECTORY on sys.path, which the file-based runner never does, so
    a checkout visible only through the cwd passed the probe and then failed inside the
    runner after the whole model merge. The probe must not be more permissive than the
    thing it is evidence about.
    """
    import inspect

    from unsloth.save import _llm_compressor_imports_cleanly

    src = inspect.getsource(_llm_compressor_imports_cleanly)
    assert '"-c"' not in src, "the probe still runs with -c, which adds the cwd to sys.path"
    assert (
        "sys.path[0] = " in src
    ), "the probe does not pin sys.path[0], so it does not reproduce the runner's search path"
    assert (
        "os.path.dirname(os.path.abspath(__file__))" in src
    ), "the pinned path is not the runner's own directory"
    assert (
        "probe_path" in src and "subprocess.run" in src
    ), "the probe is no longer executed as a file"


def _module_at(location: str | None, version: str | None):
    import types

    module = types.ModuleType("llmcompressor")
    if location is not None:
        module.__file__ = location
    if version is not None:
        module.__version__ = version
    return module


def test_an_import_only_this_process_can_do_is_not_evidence(tmp_path, monkeypatch):
    """A successful in-process import does not mean the runner can repeat it.

    The export launches _compressed_quantize.py BY FILE, so its sys.path[0] is the runner's
    directory and the cwd is on its path nowhere. A checkout importable here only because
    the cwd is on OUR path is invisible there, and the export then dies after the whole
    model merge instead of being repaired first.
    """
    from unsloth.save import _llm_compressor_module_is_usable

    cwd_only = tmp_path / "cwd-checkout" / "llmcompressor" / "__init__.py"
    cwd_only.parent.mkdir(parents = True)
    cwd_only.write_text("")
    monkeypatch.chdir(cwd_only.parent.parent)
    monkeypatch.setattr(sys, "path", [""] + [p for p in sys.path if p not in ("", ".")])
    monkeypatch.delenv("PYTHONPATH", raising = False)
    assert (
        _llm_compressor_module_is_usable(_module_at(str(cwd_only), None)) is False
    ), "a checkout reachable only through the cwd was taken as evidence for the runner"

    # The same file, now on PYTHONPATH, IS on the runner's path.
    monkeypatch.setenv("PYTHONPATH", str(cwd_only.parent.parent))
    assert _llm_compressor_module_is_usable(_module_at(str(cwd_only), None)) is True

    # Nothing to locate is unknown, and unknown stays usable.
    assert _llm_compressor_module_is_usable(_module_at(None, None)) is True


def test_the_fast_path_checks_the_version_of_the_module_it_imported(monkeypatch):
    """Metadata answers for a distribution; the fast path returns a MODULE.

    A checkout shadowing an in-range wheel imports fine and metadata says 0.12.0, so the
    fast path returned symbols from code the pin excludes.
    """
    from packaging.requirements import Requirement

    from unsloth.save import _LLM_COMPRESSOR_SPEC, _llm_compressor_module_is_usable

    assert not Requirement(_LLM_COMPRESSOR_SPEC).specifier.contains(
        "0.13.0", prereleases = True
    ), "0.13.0 must be outside the pin for this case to mean anything"
    assert _llm_compressor_module_is_usable(_module_at(None, "0.13.0")) is False
    assert _llm_compressor_module_is_usable(_module_at(None, "0.12.0")) is True


def test_the_fast_path_asks_before_returning_its_symbols():
    """Wiring: the check has to gate the early return, or it changes nothing."""
    import inspect

    from unsloth.save import install_llm_compressor

    src = inspect.getsource(install_llm_compressor)
    assert (
        "if _llm_compressor_module_is_usable(llmcompressor):" in src
    ), "the fast path returns its symbols without asking whether the runner can use them"


def test_a_shadow_pip_cannot_replace_is_named_rather_than_reinstalled(monkeypatch):
    """pip leaves an already-satisfied requirement alone.

    The requirement is satisfied by the DISTRIBUTION's metadata, so an out-of-range checkout
    shadowing an in-range wheel survives `pip install llmcompressor>=...,<=...` untouched
    and the export would resolve it again. No reinstall can fix a shadow, so the install
    path verifies afterwards and says which module the subprocess actually imported.
    """
    import unsloth.save as save

    calls: list[list[str]] = []
    real_version = md.version

    def fake_version(dist: str) -> str:
        # The wheel's metadata: in range, which is what makes pip a no-op here.
        if dist == "llmcompressor":
            return "0.12.0"
        return real_version(dist)

    monkeypatch.setattr(md, "version", fake_version)
    monkeypatch.setattr(subprocess, "check_call", lambda cmd, *a, **k: calls.append(list(cmd)))
    # The probe imports the CHECKOUT: exit 0, version 0.13.0, and a path to name.
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda cmd, *a, **k: subprocess.CompletedProcess(
            cmd,
            0,
            stdout = (
                _LLM_COMPRESSOR_PROBE_SENTINEL
                + ' {"version": "0.13.0",'
                + ' "location": "/srv/checkout/llmcompressor/__init__.py"}'
            ).encode(),
        ),
    )
    try:
        import llmcompressor  # noqa: F401
    except Exception:
        pass
    else:
        pytest.skip("llmcompressor imports in this interpreter, so the guard is unreachable")

    with pytest.raises(RuntimeError) as excinfo:
        install_llm_compressor()
    message = str(excinfo.value)
    assert (
        "0.13.0" in message and "/srv/checkout/llmcompressor" in message
    ), f"the error does not name the module the export would resolve: {message}"
    assert "reinstalling cannot replace it" in message, message
    assert calls, "the install was skipped entirely, so a genuinely broken one is not repaired"


def test_a_probe_that_cannot_import_is_still_a_failed_install(monkeypatch):
    """The other half, kept apart: nothing imported is not a shadow."""
    outcome = _install_outcome(monkeypatch, subprocess_import = 1)
    assert isinstance(outcome, RuntimeError), outcome
    assert "could not be imported" in str(outcome), str(outcome)


def test_a_runtime_only_path_entry_is_not_evidence_for_the_child(tmp_path, monkeypatch):
    """A child process does not inherit its parent's mutated sys.path.

    sys.path.insert(0, "/opt/llm-compressor") makes the import work HERE, and a fresh
    `sys.executable <runner>` sees none of it, so accepting that entry passed a checkout the
    export cannot import and the failure landed after the whole merge.
    """
    from unsloth.save import _llm_compressor_module_is_usable

    inserted = tmp_path / "runtime-insert" / "llmcompressor" / "__init__.py"
    inserted.parent.mkdir(parents = True)
    inserted.write_text("")
    monkeypatch.delenv("PYTHONPATH", raising = False)
    monkeypatch.setattr(sys, "path", [str(inserted.parent.parent), *sys.path])
    assert (
        _llm_compressor_module_is_usable(_module_at(str(inserted), None)) is False
    ), "a runtime-only sys.path entry was taken as evidence about the child process"

    # The same directory named in the environment the child DOES inherit is accepted.
    monkeypatch.setenv("PYTHONPATH", str(inserted.parent.parent))
    assert _llm_compressor_module_is_usable(_module_at(str(inserted), None)) is True

    # And an ordinary installed module, which lives in the interpreter's own site
    # directory, still passes without any PYTHONPATH at all.
    import sysconfig

    monkeypatch.delenv("PYTHONPATH", raising = False)
    site_module = os.path.join(sysconfig.get_paths()["purelib"], "llmcompressor", "__init__.py")
    assert _llm_compressor_module_is_usable(_module_at(site_module, None)) is True


def test_a_probe_that_could_not_answer_does_not_skip_the_install(monkeypatch):
    """With no metadata there is no evidence a distribution exists at all.

    The probe answers True for its OWN failures -- a timeout, an interpreter that will not
    spawn -- which is the right default when metadata says something IS installed, and the
    wrong one here: an absent package skipped the install and failed in the runner after
    the whole merge.
    """
    for unknown in (
        subprocess.TimeoutExpired(cmd = ["python"], timeout = 600),
        OSError("no such interpreter"),
    ):
        assert (
            _pip_invoked_when(monkeypatch, None, subprocess_import = unknown) is True
        ), f"an unanswerable probe skipped the install for an absent package: {unknown!r}"

    # Metadata PRESENT keeps the old default, since something is installed either way.
    assert (
        _pip_invoked_when(
            monkeypatch,
            "0.12.0",
            subprocess_import = OSError("no such interpreter"),
        )
        is False
    )


def test_a_banner_on_the_probes_stdout_does_not_become_the_version(monkeypatch):
    """The probe's answer is not the only thing that can be on the child's stdout.

    A sitecustomize, or any dependency the import pulls in, can print first. Read
    positionally, that line became the "version" and the real version shifted into
    "location", so an out-of-range shadow reported a version that will not parse, which
    counts as unknown, which counts as usable: the repair was skipped for exactly the
    module the pin excludes.
    """
    from unsloth.save import _LLM_COMPRESSOR_PROBE_RESULT

    invoked = _pip_invoked_when(
        monkeypatch,
        "0.12.0",
        probe_version = "0.13.0",
        probe_location = "/opt/checkout/llmcompressor/__init__.py",
        probe_noise = "Loading sitecustomize\nwarning: cuda graphs disabled",
    )
    assert (
        invoked is True
    ), "a banner ahead of the probe's answer hid an out-of-range module from the repair"
    assert (
        _LLM_COMPRESSOR_PROBE_RESULT["version"] == "0.13.0"
    ), f"the parsed version came from the noise: {_LLM_COMPRESSOR_PROBE_RESULT!r}"
    assert (
        _LLM_COMPRESSOR_PROBE_RESULT["location"] == "/opt/checkout/llmcompressor/__init__.py"
    ), f"the parsed location came from the noise: {_LLM_COMPRESSOR_PROBE_RESULT!r}"


def test_a_checkout_the_child_reaches_first_is_not_answered_for_by_the_cached_wheel(
    tmp_path, monkeypatch
):
    """Reachable is not resolved, and the difference is a real configuration.

    An in-range wheel in site-packages is what THIS process imported and cached; PYTHONPATH
    is then pointed at a different checkout. A fresh child searches PYTHONPATH first, so it
    imports the checkout while a membership test blessed the wheel and skipped both the
    probe and the repair.
    """
    import sysconfig

    from unsloth.save import _llm_compressor_module_is_usable

    def _provide(root: str):
        package = tmp_path / root / "llmcompressor"
        package.mkdir(parents = True)
        (package / "__init__.py").write_text("")
        return package / "__init__.py"

    wheel = _provide("site-packages")
    _provide("checkout")
    monkeypatch.setattr(
        sysconfig, "get_paths", lambda: {"purelib": str(tmp_path / "site-packages")}
    )

    monkeypatch.setenv("PYTHONPATH", str(tmp_path / "checkout"))
    assert (
        _llm_compressor_module_is_usable(_module_at(str(wheel), "0.12.0")) is False
    ), "the cached wheel answered for a checkout the child reaches before it"

    # With no competing entry, the same cached wheel IS what the child gets.
    monkeypatch.delenv("PYTHONPATH", raising = False)
    assert _llm_compressor_module_is_usable(_module_at(str(wheel), "0.12.0")) is True

    # And an entry that provides no llmcompressor at all does not get to veto it: an empty
    # directory on PYTHONPATH is a normal thing, and forcing a probe for it would pay the
    # subprocess on every export.
    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.setenv("PYTHONPATH", str(empty))
    assert _llm_compressor_module_is_usable(_module_at(str(wheel), "0.12.0")) is True


def test_an_empty_pythonpath_component_is_the_current_directory(tmp_path, monkeypatch):
    """`PYTHONPATH=/opt/lib:` puts the CWD on the child's path, at that position.

    Verified on 3.13: the empty component is absolutized into the child's sys.path where it
    sits in PYTHONPATH, so dropping it mismodelled the child and a cwd checkout it would
    import ahead of the cached wheel went unnoticed. A trailing separator is the ordinary
    way a PYTHONPATH ends up with one.
    """
    import subprocess as sp
    import sysconfig

    from unsloth.save import _llm_compressor_module_is_usable

    def _provide(root: str):
        package = tmp_path / root / "llmcompressor"
        package.mkdir(parents = True)
        (package / "__init__.py").write_text("")
        return package / "__init__.py"

    # The premise, measured rather than assumed, with a FILE-based run like the export's.
    probe = tmp_path / "probe.py"
    probe.write_text("import os, sys\nprint(os.getcwd() in sys.path[1:])\n")
    seen = sp.run(
        [sys.executable, str(probe)],
        cwd = str(tmp_path),
        env = {**os.environ, "PYTHONPATH": str(tmp_path / "other") + os.pathsep},
        stdout = sp.PIPE,
        text = True,
    )
    assert (
        seen.stdout.strip() == "True"
    ), "an empty PYTHONPATH component no longer names the cwd, so this case is stale"

    wheel = _provide("site-packages")
    _provide("cwd-checkout")
    monkeypatch.setattr(
        sysconfig, "get_paths", lambda: {"purelib": str(tmp_path / "site-packages")}
    )
    monkeypatch.chdir(tmp_path / "cwd-checkout")
    monkeypatch.setenv("PYTHONPATH", str(tmp_path / "unrelated") + os.pathsep)
    assert (
        _llm_compressor_module_is_usable(_module_at(str(wheel), "0.12.0")) is False
    ), "a cwd checkout the child reaches through an empty PYTHONPATH component was ignored"

    # Without that component the cwd is on no path the child has, so the wheel answers.
    monkeypatch.setenv("PYTHONPATH", str(tmp_path / "unrelated"))
    assert _llm_compressor_module_is_usable(_module_at(str(wheel), "0.12.0")) is True


def test_a_user_site_checkout_outranks_the_system_site_wheel(tmp_path, monkeypatch):
    """site.main() adds the user site BEFORE the system site directories.

    So a metadata-free checkout dropped in the user site is what a fresh child imports, and
    appending the user site last let the ordered scan stop at the cached system-site wheel
    and skip both the probe and the repair.
    """
    import site
    import sysconfig

    from unsloth.save import _llm_compressor_module_is_usable

    def _provide(root: str):
        package = tmp_path / root / "llmcompressor"
        package.mkdir(parents = True)
        (package / "__init__.py").write_text("")
        return package / "__init__.py"

    wheel = _provide("system-site")
    _provide("user-site")
    monkeypatch.setattr(sysconfig, "get_paths", lambda: {"purelib": str(tmp_path / "system-site")})
    monkeypatch.setattr(site, "getsitepackages", lambda: [])
    monkeypatch.setattr(site, "getusersitepackages", lambda: str(tmp_path / "user-site"))
    monkeypatch.delenv("PYTHONPATH", raising = False)

    monkeypatch.setattr(site, "ENABLE_USER_SITE", True)
    assert (
        _llm_compressor_module_is_usable(_module_at(str(wheel), "0.12.0")) is False
    ), "the system-site wheel answered for a user-site checkout the child reaches first"

    # With the user site DISABLED the child never sees it, so the wheel is the answer again.
    monkeypatch.setattr(site, "ENABLE_USER_SITE", False)
    assert _llm_compressor_module_is_usable(_module_at(str(wheel), "0.12.0")) is True


def test_a_stale_module_is_evicted_before_the_reimport(monkeypatch):
    """importlib.invalidate_caches() clears the finders, not sys.modules.

    An out-of-range llmcompressor imported earlier in the same process was returned
    untouched after the install, so the guard did the repair and then handed back exactly
    the symbols the pin excludes.
    """
    import types

    import unsloth.save as save

    stale = types.ModuleType("llmcompressor")
    stale.__version__ = "0.13.0"
    stale.oneshot = object()
    quant = types.ModuleType("llmcompressor.modifiers.quantization")
    quant.QuantizationModifier = object()
    monkeypatch.setitem(sys.modules, "llmcompressor", stale)
    monkeypatch.setitem(sys.modules, "llmcompressor.modifiers", types.ModuleType("x"))
    monkeypatch.setitem(sys.modules, "llmcompressor.modifiers.quantization", quant)

    real_version = md.version

    def fake_version(dist: str) -> str:
        # Out of range, so the install runs rather than the fast path.
        if dist == "llmcompressor":
            return "0.13.0"
        return real_version(dist)

    monkeypatch.setattr(md, "version", fake_version)
    monkeypatch.setattr(subprocess, "check_call", lambda cmd, *a, **k: None)
    # The probe answers that the install worked, so the shadow error does not fire.
    monkeypatch.setattr(save, "_llm_compressor_imports_cleanly", lambda: True)

    try:
        save.install_llm_compressor()
    except Exception:
        # The re-import after eviction fails in this interpreter, which is the point: it went
        # to disk instead of returning the stale module.
        pass
    assert (
        "llmcompressor" not in sys.modules or sys.modules["llmcompressor"] is not stale
    ), "the repair returned the very module the pin excludes"


def test_a_zip_on_the_path_is_a_provider_too(tmp_path, monkeypatch):
    """A path entry is not always a directory: the zipimporter searches .zip and .egg.

    A shape check missed an archived llmcompressor entirely, so a checkout zipped onto
    PYTHONPATH -- which the child imports first -- let the cached wheel satisfy the fast path
    and skip both the probe and the repair.
    """
    import zipfile

    from unsloth.save import _llm_compressor_module_is_usable, _path_entry_provides_llm_compressor

    archive = tmp_path / "shadow.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("llmcompressor/__init__.py", "__version__ = '0.13.0'\n")
    assert (
        _path_entry_provides_llm_compressor(str(archive)) is True
    ), "an archived llmcompressor on the path was not seen as a provider"

    # The premise, measured: a child really does import it from there.
    probe = tmp_path / "probe.py"
    probe.write_text("import llmcompressor, sys\nprint(llmcompressor.__version__)\n")
    seen = subprocess.run(
        [sys.executable, str(probe)],
        env = {**os.environ, "PYTHONPATH": str(archive)},
        stdout = subprocess.PIPE,
        stderr = subprocess.DEVNULL,
        text = True,
    )
    assert (
        seen.stdout.strip() == "0.13.0"
    ), "the interpreter no longer imports a package out of a zip on PYTHONPATH"

    # So the cached wheel does not answer for it.
    import sysconfig

    site_package = tmp_path / "site-packages" / "llmcompressor"
    site_package.mkdir(parents = True)
    (site_package / "__init__.py").write_text("")
    monkeypatch.setattr(
        sysconfig, "get_paths", lambda: {"purelib": str(tmp_path / "site-packages")}
    )
    monkeypatch.setenv("PYTHONPATH", str(archive))
    assert (
        _llm_compressor_module_is_usable(_module_at(str(site_package / "__init__.py"), "0.12.0"))
        is False
    ), "the cached wheel answered for an archived checkout the child reaches first"

    # An empty directory still provides nothing, so it does not force a probe.
    empty = tmp_path / "empty"
    empty.mkdir()
    assert _path_entry_provides_llm_compressor(str(empty)) is False
    # Nor does a same-named directory with no __init__, which is a namespace portion.
    namespace = tmp_path / "ns" / "llmcompressor"
    namespace.mkdir(parents = True)
    assert _path_entry_provides_llm_compressor(str(tmp_path / "ns")) is False
