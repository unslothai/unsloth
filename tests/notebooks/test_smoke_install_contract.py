# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team.
"""Pins the runner's Python to the Colab snapshot, and the converted script name to the converter."""

from __future__ import annotations

import json
import os
import re
import sys
import textwrap
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
    """Lowercased names the Colab freeze pins; checks scoped to these survive a dropped package."""
    return {
        m.group(1).lower()
        for line in FREEZE.read_text(encoding = "utf-8").splitlines()
        if (m := re.match(r"^([A-Za-z0-9._-]+)\s*==", line.strip()))
    }


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
    """A pin needing a newer Python than the runner never resolves; audioop-lts requires 3.13+."""
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
    """Every run: body of the job with comment lines dropped, so comment prose is not read as code."""
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
    # name -> the system dependency whose absence killed its build.
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


def _seed_script() -> str:
    """Lifts the seed step's Python out of the workflow so tests run the production code, not a copy."""
    shell = None
    for step in _job()["steps"]:
        if str(step.get("name", "")).startswith("Seed Colab-shaped venv"):
            shell = step["run"]
            break
    assert shell, "the seed step is gone from the smoke job; this guard checks nothing"
    body = re.search(r"<<'PY'\n(.*?)\n\s*PY\n", shell, re.S)
    assert body, "could not find the seed heredoc; the step's shape changed"
    return textwrap.dedent(body.group(1))


def _run_seed(tmp_path, freeze_text = None) -> list[str]:
    """Runs the seed script and returns its pins; python_version is rewritten to the running interpreter."""
    import subprocess

    data = tmp_path / "unsloth" / "scripts" / "data"
    data.mkdir(parents = True)
    mapping = _mapping()
    mapping["python_version"] = "%d.%d" % sys.version_info[:2]
    (data / "colab_to_cpu_pin.json").write_text(json.dumps(mapping), encoding = "utf-8")
    (data / "colab_pip_freeze.gpu.txt").write_text(
        freeze_text if freeze_text is not None else FREEZE.read_text(encoding = "utf-8"),
        encoding = "utf-8",
    )
    # Rewrite the script's two fixed output paths to a private directory.
    script = _seed_script()
    script = script.replace("/tmp/seed_torch.txt", str(tmp_path / "seed_torch.txt"))
    script = script.replace("/tmp/seed_pins.txt", str(tmp_path / "seed_pins.txt"))
    script = script.replace("/tmp/seed_no_binary.txt", str(tmp_path / "seed_no_binary.txt"))
    run = subprocess.run(
        [sys.executable, "-c", script],
        cwd = str(tmp_path),
        capture_output = True,
        text = True,
    )
    assert run.returncode == 0, f"the seed script failed: {run.stdout}\n{run.stderr}"
    pins = (tmp_path / "seed_pins.txt").read_text(encoding = "utf-8").split()
    torch_pins = (tmp_path / "seed_torch.txt").read_text(encoding = "utf-8").split()
    return pins + torch_pins


def test_no_declared_distro_marker_survives_the_seed(tmp_path):
    """Declared distro_dev_version pins must not survive the seed; a +local suffix on any pin fails."""
    seeded = _run_seed(tmp_path)
    declared = {
        name: rule["from"] for name, rule in _mapping().get("distro_dev_version", {}).items()
    }
    stale = [
        pin
        for pin in seeded
        if pin.split("==", 1)[0] in declared
        and pin.split("==", 1)[1] == declared[pin.split("==", 1)[0]]
    ]
    assert not stale, (
        "these pins kept a version only the Colab image uses, and one of them fails the "
        f"resolve for all of them: {stale}"
    )

    local = [pin for pin in seeded if "+" in pin]
    assert not local, f"the seed left a local version on: {local}"


def test_every_dev_pin_in_the_freeze_has_been_judged():
    """Each .devN pin in the freeze must be judged: a distro label to rewrite, or a real prerelease."""
    mapping = _mapping()
    rewrites = mapping.get("distro_dev_version", {})
    allowed = {entry.lower() for entry in mapping.get("published_prerelease", [])}

    unjudged = []
    for line in FREEZE.read_text(encoding = "utf-8").splitlines():
        m = re.match(r"^([A-Za-z0-9._-]+)\s*==\s*(.+)$", line.strip())
        if not m or ".dev" not in m.group(2):
            continue
        name, ver = m.group(1).lower(), m.group(2)
        declared = rewrites.get(name, {}).get("from") == ver
        if not declared and f"{name}=={ver}" not in allowed:
            unjudged.append(f"{name}=={ver}")

    assert not unjudged, (
        "these pins carry a .devN version that nothing has judged. Add a distro_dev_version "
        "entry if the image is labelling a distro build, or list it under "
        f"published_prerelease if PyPI really publishes it: {unjudged}"
    )


def test_a_rewritten_pin_is_still_installed_at_the_published_version(tmp_path):
    """The rewrite must not become a skip: the image carries Mako, so the venv this job builds
    has to carry it too, at the version PyPI publishes."""
    rewrites = _mapping().get("distro_dev_version", {})
    if not rewrites:
        pytest.skip("no distro .devN rewrites are declared")
    seeded = dict(pin.split("==", 1) for pin in _run_seed(tmp_path) if "==" in pin)
    for name, rule in rewrites.items():
        assert name in seeded, f"{name} was dropped rather than rewritten"
        assert seeded[name] == rule["to"], f"{name} seeded as {seeded[name]}, expected {rule['to']}"


def test_an_undeclared_dev_pin_is_left_alone_rather_than_guessed_at(tmp_path):
    """Undeclared .devN pins pass through untouched, since stripping them would change a real prerelease."""
    freeze = FREEZE.read_text(encoding = "utf-8") + "\nunsloth-not-a-real-pin==2.0.dev3\n"
    seeded = dict(pin.split("==", 1) for pin in _run_seed(tmp_path, freeze) if "==" in pin)
    assert (
        seeded.get("unsloth-not-a-real-pin") == "2.0.dev3"
    ), "an undeclared .devN pin was rewritten; only the mapping may decide that"


def _restore_step() -> dict:
    steps = [s for s in _job()["steps"] if "pip-cache-restore" in str(s.get("uses", ""))]
    assert len(steps) == 1, f"{JOB} should restore the pip cache exactly once, got {len(steps)}"
    return steps[0]["with"]


def test_every_file_the_seed_step_reads_is_a_cache_key_input():
    """Every repo file the seed step opens must hash into the pip cache key, or a stale entry is
    restored."""
    files = set(_restore_step()["key-files"].split())
    # Only checked-in files; /tmp scratch and the converted _smoke.py are outputs.
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
        # key-files resolve from GITHUB_WORKSPACE and this job checks out under `unsloth/`.
        assert rel.startswith("unsloth/"), f"{rel} is not prefixed for this job's checkout layout"
        assert (
            REPO / rel[len("unsloth/") :]
        ).exists(), f"key-files names {rel}, which does not exist"


def test_the_cuda_only_wheels_are_skipped():
    """CUDA-only wheels cannot run without a GPU, so caching them only spends the pip cache budget."""
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
    """Detected backends stay: transformers imports any installed TF/Flax, which changes import unsloth."""
    skip = set(_mapping()["skip"])
    detected = {"tensorflow", "flax", "jax", "jaxlib", "tf-keras"}
    wrongly_skipped = sorted((detected & _freeze_names()) & skip)
    assert not wrongly_skipped, (
        f"{wrongly_skipped} are detected-if-installed backends. Skipping them saves "
        f"cache at the cost of the fidelity this job is for; see "
        f"tests/test_broken_tf_does_not_break_import.py"
    )


def test_skipped_pins_are_not_also_marked_no_binary():
    """An entry in both skip and no_binary is dead config: the seed step never passes --no-binary for it."""
    mapping = _mapping()
    both = sorted(set(mapping["skip"]) & set(mapping.get("no_binary", [])))
    assert not both, f"these are in skip and no_binary at once, so no_binary is dead: {both}"


def test_the_skip_list_is_closed_under_the_freezes_dependencies():
    """A skipped package must not be required by any kept pin, or pip re-downloads it unpinned."""
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


def _probe_in_a_venv_without_torchcodec(mode: str) -> str:
    """Runs transformers' torchcodec probes in a subprocess, since this repo's venv has real torchcodec."""
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
    # -S skips site-packages so an installed torchcodec cannot answer the probes, like the runner.
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
        # The shape on main: __spec__ is None and find_spec raises.
        ("bare", "AVAILABLE_RAISED ValueError"),
        # A hand-made ModuleSpec still fails audio_utils' metadata lookup, hence a real distribution.
        ("spec", "VERSION_RAISED PackageNotFoundError"),
        ("dist", "VERSION 0.0.0"),
    ],
    ids = ["bare ModuleType", "ModuleType with a spec", "the placeholder distribution"],
)
def test_the_placeholder_survives_both_probes_transformers_makes(mode, expected):
    """Stub must answer find_spec and the version lookup, or ValueError becomes PackageNotFoundError."""
    printed = _probe_in_a_venv_without_torchcodec(mode)
    assert (
        "AVAILABLE_RAISED ValueError" in printed or "AVAILABLE" in printed
    ), f"the probe subprocess produced nothing usable for {mode!r}: {printed!r}"
    assert expected in printed, printed


def _load_stub_helper():
    """Loads the stub by path: a sys.path insert of tests/ would shadow studio/backend's utils package."""
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
    """Each helper the smoke steps import must be a workflow path trigger, or it can merge untested."""
    shell = _shell(_job())
    imported = set(re.findall(r"import\s+(_\w+)", shell))
    assert imported, "no helper imports found in the smoke steps; this guard checks nothing"

    doc = yaml.safe_load(WORKFLOW.read_text(encoding = "utf-8"))
    # PyYAML parses `on` as the boolean True.
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
    """Loading the stub must not change sys.path; pytest's prepend mode already adds tests/ to it."""
    before = list(sys.path)
    _load_stub_helper()
    assert sys.path == before, (
        "loading the helper mutated sys.path: "
        f"added {[p for p in sys.path if p not in before]!r}"
    )

    # Assembled so the needle does not match this line itself.
    needle = "sys.path" + ".insert(0, str(REPO / " + chr(34) + "tests" + chr(34) + "))"
    source = Path(__file__).read_text(encoding = "utf-8")
    assert needle not in source, "a sys.path insert of the tests dir is back in this file"
