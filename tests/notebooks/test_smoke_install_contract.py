# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team.
"""Pins the two contracts the notebooks-ci smoke job kept breaking silently.

The job had never once reached the import check it exists for: 872 matrix legs
across 109 scheduled runs and 112 days, all red. Two independent causes:

  * The runner pinned Python 3.12 while the Colab snapshot beside it had been
    refreshed to a 3.13 image. `audioop-lts` requires 3.13, so the bulk install
    failed in 8 seconds every run and fell back to 682 one-at-a-time installs
    that spent the whole cap. Nothing tied the two files together.

  * The workflow rebuilt the converted script's filename in shell instead of
    asking the converter, and the copy was wrong for every row of the matrix.

Neither is noticeable otherwise: a leg over `timeout-minutes` is scored
`cancelled`, and `cancelled` outranks `failure` in GitHub's run rollup.
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
WORKFLOW = REPO / ".github" / "workflows" / "notebooks-ci.yml"
MAPPING = REPO / "scripts" / "data" / "colab_to_cpu_pin.json"
FREEZE = REPO / "scripts" / "data" / "colab_pip_freeze.gpu.txt"

sys.path.insert(0, str(REPO / "scripts"))
from notebook_to_python import converted_filename  # noqa: E402

JOB = "smoke-install"


def _job() -> dict:
    doc = yaml.safe_load(WORKFLOW.read_text(encoding = "utf-8"))
    job = doc["jobs"].get(JOB)
    assert job, f"{JOB} is gone from {WORKFLOW.name}; this file checks nothing"
    return job


def _notebooks() -> list[str]:
    nbs = _job()["strategy"]["matrix"]["notebook"]
    assert nbs, "the smoke matrix is empty; this guard checks nothing"
    return nbs


def _mapping() -> dict:
    return json.loads(MAPPING.read_text(encoding = "utf-8"))


def _freeze_names() -> set[str]:
    """The pinned package names in the Colab snapshot, lowercased.

    Every rule below is scoped to what the freeze actually pins, so a Colab rotation
    that drops a package cannot fail these on a name that is no longer there.
    """
    return {
        m.group(1).lower()
        for line in FREEZE.read_text(encoding = "utf-8").splitlines()
        if (m := re.match(r"^([A-Za-z0-9._-]+)\s*==", line.strip()))
    }


# --- the interpreter the snapshot was taken on ---------------------------------------


def test_the_snapshot_records_the_interpreter_it_was_captured_on():
    assert re.fullmatch(r"3\.\d+", _mapping().get("python_version", "")), (
        "colab_to_cpu_pin.json must record python_version, the interpreter the freeze "
        "beside it came from. Without it nothing connects a Colab rotation to the "
        "runner pin, which is how the job came to install a 3.13 environment on 3.12."
    )


def test_the_smoke_job_runs_the_interpreter_the_snapshot_names():
    want = _mapping()["python_version"]
    pins = [
        str((s.get("with") or {}).get("python-version"))
        for s in _job()["steps"]
        if "setup-python" in str(s.get("uses", ""))
    ]
    assert pins, f"{JOB} does not pin an interpreter at all"
    assert set(pins) == {want}, (
        f"{JOB} pins Python {pins} but the Colab snapshot was captured on {want}. A pin "
        f"carrying a Requires-Python floor above the runner cannot resolve, and one "
        f"such pin fails the whole bulk install."
    )


def test_the_freeze_resolves_against_the_interpreter_the_snapshot_names():
    """A pin whose Requires-Python floor is above the runner can never resolve.

    `audioop-lts` is the live example: it exists only for 3.13+, so its presence in
    the freeze is itself evidence of the image's interpreter.
    """
    want = _mapping()["python_version"]
    names = {
        m.group(1).lower()
        for line in FREEZE.read_text(encoding = "utf-8").splitlines()
        if (m := re.match(r"^([A-Za-z0-9._-]+)\s*==", line.strip()))
    }
    if "audioop-lts" in names:
        assert want == "3.13" or tuple(map(int, want.split("."))) >= (3, 13), (
            f"the freeze pins audioop-lts, which requires Python >= 3.13, but "
            f"python_version says {want}"
        )


# --- the converted script's name ------------------------------------------------------


@pytest.mark.parametrize("notebook", _notebooks())
def test_every_matrix_notebook_maps_to_one_converted_script(notebook):
    name = converted_filename(Path(notebook).name)
    assert name.endswith(".py") and not name.endswith("_.py"), (
        f"{notebook} converts to {name!r}. A trailing underscore is the signature of "
        f"rebuilding the name in shell, where basename's newline becomes one."
    )


def test_converted_names_do_not_collide_across_the_matrix():
    """The job takes 'the one .py in the output directory', so a collision hides a leg."""
    seen: dict[str, list[str]] = {}
    for nb in _notebooks():
        seen.setdefault(converted_filename(Path(nb).name), []).append(nb)
    clashes = {k: v for k, v in seen.items() if len(v) > 1}
    assert not clashes, f"these matrix notebooks convert to the same filename: {clashes}"


@pytest.mark.parametrize(
    "filename,expected",
    [
        ("Gemma3_(4B)-Vision.ipynb", "Gemma3_4B_Vision.py"),
        ("Whisper.ipynb", "Whisper.py"),
        # A dot survives; the shell copy mapped it to `_`.
        ("Llama3.1_(8B)-GRPO.ipynb", "Llama3.1_8B_GRPO.py"),
        ("gpt-oss-(20B)-Fine-tuning.ipynb", "gpt_oss_20B_Fine_tuning.py"),
    ],
)
def test_the_naming_rule_itself(filename, expected):
    assert converted_filename(filename) == expected


def _shell(job) -> str:
    """Every `run:` body in the job, comments stripped.

    The steps quote the old broken pipeline to explain why it went, and a rule about
    what the shell DOES must not read that prose as code.
    """
    lines = []
    for step in job["steps"]:
        for line in str(step.get("run", "")).splitlines():
            if not line.lstrip().startswith("#"):
                lines.append(line)
    return "\n".join(lines)


def test_the_workflow_asks_the_converter_instead_of_rebuilding_the_name():
    """Anti-regression: the checks above read the matrix, not the step body, so
    only this one can see a reintroduced `tr` pipeline."""
    body = _shell(_job())
    assert "tr -c '[:alnum:]_'" not in body, (
        "the smoke job is rebuilding the converted filename in shell again. Call "
        "scripts/notebook_to_python.py on the one notebook and take the file it wrote."
    )
    assert "notebook_to_python.py" in body, (
        "the smoke job should convert its own matrix notebook with the converter "
        "directly, so the name it looks for is the name that was written"
    )


# --- the install itself ---------------------------------------------------------------


def test_the_seed_install_refuses_source_builds():
    """Sdist-only pins need system libraries the runner lacks; without --only-binary
    pip spends 20-90s per package on a doomed build."""
    body = _shell(_job())
    installs = [ln for ln in body.splitlines() if re.search(r"\bpip install\b", ln)]
    offenders = [
        ln.strip() for ln in installs if "--upgrade pip" not in ln and "--only-binary" not in ln
    ]
    assert not offenders, "these seed installs allow source builds:\n  " + "\n  ".join(offenders)


def test_the_known_unbuildable_pins_are_skipped():
    """Each failed a source build in the 2026-08-31 run, and one is enough to fail
    the bulk resolve for the whole set."""
    skip = set(_mapping()["skip"])
    # name -> the system dependency whose absence killed its build in that run.
    system_bound = {
        "cyipopt": "ipopt",
        "dbus-python": "dbus-1",
        "dlib": "cmake",
        "gdal": "gdal-config",
        "pycairo": "cairo",
        "pygobject": "girepository",
        "python-apt": "apt",
        "rpy2": "R_HOME",
    }
    missing = sorted(set(system_bound) - skip)
    assert not missing, (
        f"these pins cannot build on ubuntu-latest on any interpreter and only cost "
        f"build time, so they belong in the skip list: {missing}"
    )


def _seeded_pins() -> list[str]:
    """The pins the seed step would hand pip, applying the same transformations it does."""
    mapping = _mapping()
    skip = set(mapping["skip"])
    spoof = set(mapping["module_spoof"])
    out = []
    for line in FREEZE.read_text(encoding = "utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = re.match(r"^([A-Za-z0-9._-]+)\s*==\s*(.+)$", line)
        if not m:
            continue
        name, ver = m.group(1).lower(), m.group(2)
        if name in skip or name in spoof:
            continue
        ver = re.sub(r"[+\-].+$", "", ver)
        ver = re.sub(r"\.dev\d+$", "", ver)
        out.append(f"{name}=={ver}")
    return out


def test_no_seeded_pin_asks_pypi_for_a_version_only_a_distro_has():
    """The freeze is a snapshot of an image, so it carries versions as the image labels them,
    and a distro build labels itself `.devN`. PyPI has no such release, and one unresolvable
    pin fails the whole bulk resolve: the Ubuntu 24.04 rotation brought in `Mako==1.3.2.dev0`
    and every leg of the matrix died on

        ERROR: No matching distribution found for mako==1.3.2.dev0

    The seed strips the marker to the base version, which PyPI does have. Local versions
    (`+cu128`) are covered by the rewrite map and stripped the same way.
    """
    offenders = [pin for pin in _seeded_pins() if ".dev" in pin or "+" in pin]
    assert not offenders, (
        "these pins would be sent to PyPI carrying a marker only the Colab image uses, and "
        f"one of them fails the resolve for all of them: {offenders}"
    )


def test_the_seed_still_pins_a_package_whose_marker_was_stripped():
    """The strip must not become a skip: the image carries Mako, so the venv this job builds
    has to carry it too, just at the version PyPI publishes."""
    seeded = {pin.split("==")[0]: pin.split("==")[1] for pin in _seeded_pins()}
    raw = dict(
        re.match(r"^([A-Za-z0-9._-]+)\s*==\s*(.+)$", line.strip()).groups()
        for line in FREEZE.read_text(encoding = "utf-8").splitlines()
        if re.match(r"^([A-Za-z0-9._-]+)\s*==\s*(.+)$", line.strip())
    )
    marked = {name.lower(): ver for name, ver in raw.items() if ".dev" in ver}
    if not marked:
        pytest.skip("the current snapshot carries no .devN pin to check")
    for name, ver in marked.items():
        assert name in seeded, f"{name} was dropped rather than having its marker stripped"
        assert (
            seeded[name] == ver.split(".dev")[0]
        ), f"{name} seeded as {seeded[name]}, expected {ver.split('.dev')[0]}"


# --- the cache key has to represent what the job installs ----------------------------


def _restore_step() -> dict:
    steps = [s for s in _job()["steps"] if "pip-cache-restore" in str(s.get("uses", ""))]
    assert len(steps) == 1, f"{JOB} should restore the pip cache exactly once, got {len(steps)}"
    return steps[0]["with"]


def test_every_file_the_seed_step_reads_is_a_cache_key_input():
    """Otherwise an edit changes the install while the key stays put.

    #11270 dropped 39 pins from colab_to_cpu_pin.json and freed nothing, because the
    mapping was not a key input: the key hash did not move, the restore hit exactly,
    and pip-cache-save is gated on `cache-hit != 'true'`, so the entry holding the
    removed wheels was never rewritten. A stale entry cannot serve wrong CONTENT --
    pip's cache is addressed by URL and hash -- but it pins the entry's SIZE to a pin
    set that no longer exists.

    The rule is mechanical: whatever the shell opens, the key must hash.
    """
    files = set(_restore_step()["key-files"].split())
    # Only checked-in files under the job's checkout prefix. The seed step also
    # opens /tmp scratch and the converted _smoke.py, which are its OUTPUTS: they
    # are derived from the inputs below and cannot be edited into the repo.
    opened = set(re.findall(r"""open\(\s*["'](unsloth/[^"']+)["']""", _shell(_job())))
    assert opened, "found no repo files being read by the seed step; the pattern has drifted"
    missing = sorted(opened - files)
    assert not missing, (
        f"the seed step reads {missing} but they are not in key-files {sorted(files)}, so "
        f"editing them changes what the job installs without minting a new cache key"
    )


def test_the_mapping_is_a_cache_key_input():
    """Named explicitly, so deleting the rule above cannot quietly drop the one file
    that caused the bug."""
    assert "unsloth/scripts/data/colab_to_cpu_pin.json" in _restore_step()["key-files"].split()


def test_the_cache_key_inputs_exist():
    """A glob that matches nothing makes hashFiles return empty, which pip-cache-restore
    fails on by design -- but it fails in CI, not here, and only on the next run."""
    for rel in _restore_step()["key-files"].split():
        # key-files resolve from GITHUB_WORKSPACE and this job checks out under
        # `unsloth/`, which is the repo root from this test's point of view.
        assert rel.startswith("unsloth/"), f"{rel} is not prefixed for this job's checkout layout"
        assert (
            REPO / rel[len("unsloth/") :]
        ).exists(), f"key-files names {rel}, which does not exist"


# --- what the skip list costs and what it must not spend -----------------------------


def test_the_cuda_only_wheels_are_skipped():
    """The skip list is the only lever on this job's pip cache, which measured 6.97 GB
    per generation on 2026-09-18 -- 30% of the repo's 50 GiB Actions budget across its
    two generations, while the repo sat at 92% full and evicted other families' live
    entries. These cannot execute without a GPU, so caching them buys nothing at all.
    """
    skip = set(_mapping()["skip"])
    cuda_only = {
        "libcudf-cu12",
        "libcuml-cu12",
        "cudf-cu12",
        "cuml-cu12",
        "rmm-cu12",
        "pylibcudf-cu12",
        "pylibraft-cu12",
        "raft-dask-cu12",
        "ucxx-cu12",
        "dask-cuda",
        "numba-cuda",
        "cuda-bindings",
        "cupy-cuda12x",
        "jax-cuda12-pjrt",
        "jax-cuda12-plugin",
        "nvidia-nvshmem-cu12",
        "nvidia-cuda-nvcc-cu12",
        "nvidia-nccl-cu13",
    }
    missing = sorted((cuda_only & _freeze_names()) - skip)
    assert not missing, (
        f"these are CUDA-only wheels the CPU runner can never load, so they are pure "
        f"cache weight: {missing}"
    )


def test_the_backends_transformers_detects_stay_installed():
    """TensorFlow and Flax are 761 MiB that nothing in this repo imports, which makes
    them look like the obvious next thing to skip. They are not.

    Transformers imports either backend merely because it is INSTALLED, via
    processing_utils -> image_transforms, so their presence changes what
    `import unsloth` does. That is the subject of
    tests/test_broken_tf_does_not_break_import.py, and Colab ships them, so a seed env
    without them stops reproducing the interaction this job exists to catch.

    Fabricating .dist-info metadata without the wheel is worse than either choice: a
    find_spec hit whose import fails is the BROKEN-TF path, not Colab's healthy TF.
    """
    skip = set(_mapping()["skip"])
    detected = {"tensorflow", "flax", "jax", "jaxlib", "tf-keras"}
    wrongly_skipped = sorted((detected & _freeze_names()) & skip)
    assert not wrongly_skipped, (
        f"{wrongly_skipped} are detected-if-installed backends. Skipping them saves "
        f"cache at the cost of the fidelity this job is for; see "
        f"tests/test_broken_tf_does_not_break_import.py"
    )


def test_skipped_pins_are_not_also_marked_no_binary():
    """Dead config. The seed step only passes --no-binary for pins still present after
    the skip filter, so an entry in both lists is silently ignored and reads as though
    the package were still being built.
    """
    mapping = _mapping()
    both = sorted(set(mapping["skip"]) & set(mapping.get("no_binary", [])))
    assert not both, f"these are in skip and no_binary at once, so no_binary is dead: {both}"


def test_the_skip_list_is_closed_under_the_freezes_dependencies():
    """A skip only saves the download if nothing retained requires it.

    The seed step installs bare `name==ver`, so pip re-resolves any dropped package a
    KEPT pin depends on and downloads it anyway, unpinned -- a saving that is not one,
    and the failure mode is invisible because the install still succeeds. These edges
    were read off the freeze's own metadata; each pair is `child: parents`, and skipping
    the child obliges skipping the parents.
    """
    skip = set(_mapping()["skip"])
    names = _freeze_names()
    edges = {
        "cupy-cuda12x": {"cudf-cu12", "cuml-cu12", "dask-cudf-cu12"},
        "libcudf-cu12": {"pylibcudf-cu12"},
        "pylibcudf-cu12": {"cudf-polars-cu12"},
        "libcuml-cu12": {"cuml-cu12"},
        "rmm-cu12": {"ucxx-cu12"},
        "ucxx-cu12": {"distributed-ucxx-cu12"},
        "numba-cuda": {"dask-cuda", "distributed-ucxx-cu12"},
        "cuda-bindings": {"numba-cuda"},
        "pylibraft-cu12": {"cuml-cu12", "raft-dask-cu12"},
        "pyspark": {"dataproc-spark-connect"},
        "intel-openmp": {"mkl"},
        "tbb": {"mkl"},
        "nvidia-nccl-cu13": {"xgboost"},
    }
    leaks = {
        child: sorted(parents & names - skip)
        for child, parents in edges.items()
        if child in skip and (parents & names - skip)
    }
    assert not leaks, (
        f"each of these is skipped while a retained pin still requires it, so pip "
        f"downloads it anyway and the skip saves nothing: {leaks}"
    )


# --- the torchcodec placeholder --------------------------------------------------------


def _probe_in_a_venv_without_torchcodec(mode: str) -> str:
    """Run the two things transformers does at import time, in a subprocess whose sys.path has
    no real torchcodec, and return what it printed.

    A subprocess because the question is about interpreter state (sys.modules, sys.path,
    distribution metadata) that a stub cannot be un-installed from cleanly, and because this
    repo's own venv HAS torchcodec, which would answer both probes for the wrong reason.
    """
    import subprocess
    import textwrap

    script = textwrap.dedent(
        """
        import importlib.metadata, importlib.util, sys
        mode, tests_dir = sys.argv[1], sys.argv[2]
        sys.path.insert(0, tests_dir)
        if mode == "bare":
            import types
            sys.modules["torchcodec"] = types.ModuleType("torchcodec")
        elif mode == "spec":
            import importlib.machinery, types
            m = types.ModuleType("torchcodec")
            m.__spec__ = importlib.machinery.ModuleSpec("torchcodec", None)
            sys.modules["torchcodec"] = m
        elif mode == "dist":
            import _torchcodec_stub as t
            t.install()
        # is_torchcodec_available() -> _is_package_available(name)[0]; transformers 5.16.1
        # passes no return_version, so presence is decided by the spec alone.
        try:
            available = importlib.util.find_spec("torchcodec") is not None
        except Exception as e:
            print("AVAILABLE_RAISED", type(e).__name__); raise SystemExit(0)
        print("AVAILABLE", available)
        if available:
            # audio_utils.py:61, at import time.
            try:
                print("VERSION", importlib.metadata.version("torchcodec"))
            except Exception as e:
                print("VERSION_RAISED", type(e).__name__)
        """
    )
    # -S skips site-packages, so a real torchcodec on THIS interpreter cannot answer the
    # probes for the wrong reason. That is what the runner looks like, and it lets the test
    # run everywhere instead of skipping wherever the wheel happens to be installed.
    out = subprocess.run(
        [sys.executable, "-S", "-c", script, mode, str(REPO / "tests")],
        capture_output = True,
        text = True,
        env = {"PATH": os.environ.get("PATH", ""), "PYTHONNOUSERSITE": "1"},
        cwd = str(REPO),
    )
    return out.stdout


@pytest.mark.parametrize(
    "mode, expected",
    [
        # The shape on main: __spec__ is None and find_spec raises rather than returning None.
        ("bare", "AVAILABLE_RAISED ValueError"),
        # A hand-made ModuleSpec gets past find_spec and straight into the metadata lookup
        # that audio_utils does at import time, which is the second failure and the reason
        # the placeholder is a distribution rather than a sys.modules entry.
        ("spec", "VERSION_RAISED PackageNotFoundError"),
        # The placeholder: both probes answer.
        ("dist", "VERSION 0.0.0"),
    ],
    ids = ["bare ModuleType", "ModuleType with a spec", "the placeholder distribution"],
)
def test_the_placeholder_survives_both_probes_transformers_makes(mode, expected):
    """transformers asks two questions while importing audio_utils, and a stub has to answer
    both. is_torchcodec_available() reads find_spec, and line 61 then reads the distribution
    version. Answering only the first turns ValueError into PackageNotFoundError.
    """
    printed = _probe_in_a_venv_without_torchcodec(mode)
    assert (
        "AVAILABLE_RAISED ValueError" in printed or "AVAILABLE" in printed
    ), f"the probe subprocess produced nothing usable for {mode!r}: {printed!r}"
    assert expected in printed, printed


def _load_stub_helper():
    """Load tests/_torchcodec_stub.py by path, without touching sys.path.

    `sys.path.insert(0, REPO / "tests")` is process-wide and permanent, and `tests/` holds a
    `utils/` package that then shadows studio/backend's `utils` for every test that runs
    afterwards in the same worker. That is how this file turned
    tests/test_studio_root_resilience.py red with
    `ModuleNotFoundError: No module named 'utils.native_path_leases'` while being green itself.
    """
    import importlib.util

    path = REPO / "tests" / "_torchcodec_stub.py"
    spec = importlib.util.spec_from_file_location("_unsloth_torchcodec_stub_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_placeholder_version_stays_under_the_backend_floor():
    """load_audio resolves "auto" to torchcodec only at >= 0.3.0, so a placeholder claiming a
    newer version would be chosen as the decoder and then fail on the first real call. Below
    the floor it is visible, importable and never selected."""
    _torchcodec_stub = _load_stub_helper()

    floor = (0, 3, 0)
    actual = tuple(int(part) for part in _torchcodec_stub.VERSION.split("."))
    assert actual < floor, (
        f"the placeholder claims {_torchcodec_stub.VERSION}, at or above transformers' "
        "0.3.0 torchcodec floor, so load_audio(backend='auto') would select it"
    )


def test_the_placeholder_never_displaces_a_real_torchcodec():
    """A machine that does have the wheel must keep it: the placeholder is for the CPU runner,
    and shadowing a genuine install would be the opposite of what it is for."""
    import types

    _torchcodec_stub = _load_stub_helper()

    real = types.ModuleType(_torchcodec_stub.NAME)
    saved = sys.modules.get(_torchcodec_stub.NAME)
    sys.modules[_torchcodec_stub.NAME] = real
    try:
        assert _torchcodec_stub.install() is None, "it wrote a placeholder over a live module"
        assert sys.modules[_torchcodec_stub.NAME] is real
    finally:
        if saved is None:
            sys.modules.pop(_torchcodec_stub.NAME, None)
        else:
            sys.modules[_torchcodec_stub.NAME] = saved


def test_both_smoke_steps_stub_torchcodec_through_the_shared_helper():
    """Two steps stub it, and they used to carry their own copy of the bare-ModuleType form.
    One fixed copy is how this comes back."""
    shell = _shell(_job())
    assert "_torchcodec_stub" in shell, "the smoke job no longer uses the shared placeholder"
    assert shell.count("import _torchcodec_stub") == 2, (
        "both the install-cell step and the import-verification step must install the "
        f"placeholder; found {shell.count('import _torchcodec_stub')} site(s)"
    )
    assert 'types.ModuleType("torchcodec")' not in shell, (
        "a hand-rolled torchcodec stub is back in the workflow; its __spec__ is None and "
        "importlib.util.find_spec raises on it"
    )


def test_every_helper_the_smoke_steps_import_is_a_path_trigger():
    """A helper the job executes is part of the job. `tests/_torchcodec_stub.py` was added as a
    shared stub and imported by both smoke steps while the workflow's `pull_request.paths`
    still listed only its sibling, so a PR touching nothing else would have merged a broken
    helper without the smoke matrix or this file ever running.

    Derived from the steps rather than hand-listed, so the next shared helper is caught by
    this test instead of by a silent green run.
    """
    shell = _shell(_job())
    imported = set(re.findall(r"import\s+(_\w+)", shell))
    assert imported, "no helper imports found in the smoke steps; this guard checks nothing"

    doc = yaml.safe_load(WORKFLOW.read_text(encoding = "utf-8"))
    # `on` is the YAML 1.1 boolean True once parsed, which is why this is not doc["on"].
    triggers = doc[True] if True in doc else doc["on"]
    paths = set(triggers["pull_request"]["paths"])

    for helper in sorted(imported):
        candidate = REPO / "tests" / f"{helper}.py"
        if not candidate.is_file():
            continue  # a stdlib or third-party name that merely starts with an underscore
        entry = f"tests/{helper}.py"
        assert entry in paths, (
            f"{entry} is imported by a smoke step but is not in the workflow's "
            f"pull_request.paths, so a PR changing only that file would not run this job"
        )


def test_loading_the_stub_helper_leaves_sys_path_alone():
    """`tests/` holds a `utils/` package, so a module that puts that directory at the FRONT of
    sys.path shadows studio/backend's `utils` for every test after it in the same worker. This
    file did exactly that to reach the helper, and the casualty was another file entirely:
    tests/test_studio_root_resilience.py, red with
    `ModuleNotFoundError: No module named 'utils.native_path_leases'`.

    The claim is that loading the helper changes nothing, not that `tests/` is absent from
    sys.path: pytest's own prepend import mode puts the basedir of every collected test module
    there, so absence was never true to begin with and asserting it failed in CI for a reason
    that had nothing to do with this file.
    """
    before = list(sys.path)
    _load_stub_helper()
    assert sys.path == before, (
        "loading the helper mutated sys.path: "
        f"added {[p for p in sys.path if p not in before]!r}"
    )

    # Assembled rather than written out, so the needle does not match this line itself.
    needle = "sys.path" + ".insert(0, str(REPO / " + chr(34) + "tests" + chr(34) + "))"
    source = Path(__file__).read_text(encoding = "utf-8")
    assert needle not in source, "a sys.path insert of the tests dir is back in this file"
