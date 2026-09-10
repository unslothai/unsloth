# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""The runtime's idea of a complete sidecar matches setup's.

setup.sh and setup.ps1 install tiktoken best-effort and record a sidecar without it as
complete; the runtime validator used to require it, delete the sidecar and retry the
install that had just failed.
"""

import pathlib
import shutil
import sys
import types as _types
from pathlib import Path

# The backend uses "from utils..." imports; ensure the backend dir is on sys.path.
_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

# Stub the custom logger before importing the module under test, as the sibling tests do.
_loggers_stub = _types.ModuleType("loggers")
_loggers_stub.get_logger = lambda name: __import__("logging").getLogger(name)
sys.modules.setdefault("loggers", _loggers_stub)

import utils.transformers_version as tv


def _sidecar(
    root,
    *names,
    versions = None,
):
    versions = versions or {}
    for name in names:
        (root / name).mkdir(parents = True)
        version = versions.get(name)
        if version:
            info = root / f"{name}-{version}.dist-info"
            info.mkdir()
            (info / "METADATA").write_text(f"Name: {name}\nVersion: {version}\n", encoding = "utf-8")
    (root / tv._STUDIO_OWNED_MARKER).touch()


def test_a_failed_optional_install_leaves_no_partial_payload(tmp_path, monkeypatch):
    """pip or uv exiting nonzero after copying part of tiktoken: the sidecar is still
    accepted (the package is optional), but nothing of the package may stay, or the
    half-copied tree ahead of site-packages shadows the ambient one at tokenization."""
    root = tmp_path / ".venv_t5_550"
    monkeypatch.setattr(tv, "_venv_dir_is_valid_and_undamaged", lambda *a, **k: False)
    optional = [p for p in tv._VENV_T5_550_PACKAGES if p.startswith("tiktoken")]
    assert optional and tv._sidecar_package_is_optional(optional[0])

    def fake_install(pkg, target):
        if pkg.startswith("tiktoken"):
            (Path(target) / "tiktoken").mkdir()
            (Path(target) / "tiktoken" / "__init__.py").write_text("", encoding = "utf-8")
            (Path(target) / "tiktoken_ext").mkdir()
            (Path(target) / "tiktoken_ext" / "openai_public.py").write_text("", encoding = "utf-8")
            (Path(target) / "tiktoken-0.9.0.dist-info").mkdir()
            (Path(target) / "tiktoken-0.9.0.dist-info" / "METADATA").write_text(
                "", encoding = "utf-8"
            )
            return False
        return True

    monkeypatch.setattr(tv, "_install_to_dir", fake_install)
    assert tv._ensure_venv_dir(str(root), tv._VENV_T5_550_PACKAGES, "test sidecar") is True
    assert not (root / "tiktoken").exists()
    # The plugin namespace too: a partial tiktoken_ext ahead of site-packages shadows
    # the ambient one's openai_public module by itself.
    assert not (root / "tiktoken_ext").exists()
    assert not list(root.glob("tiktoken-*.dist-info"))
    assert tv._optional_package_absent(str(root), "tiktoken") is True


def test_a_staging_move_that_fails_part_way_rolls_the_moved_entries_back(tmp_path, monkeypatch):
    """os.replace failing on the second entry used to leave the first one in the sidecar
    with no dist-info for the whole retry backoff; the package must read as absent."""
    root = tmp_path / ".venv_t5_550"
    root.mkdir()

    def fake_install(pkg, target):
        (Path(target) / "tiktoken").mkdir()
        (Path(target) / "tiktoken" / "__init__.py").write_text("", encoding = "utf-8")
        (Path(target) / "tiktoken_ext").mkdir()
        (Path(target) / "tiktoken_ext" / "openai_public.py").write_text("", encoding = "utf-8")
        (Path(target) / "tiktoken-0.9.0.dist-info").mkdir()
        (Path(target) / "tiktoken-0.9.0.dist-info" / "RECORD").write_text("", encoding = "utf-8")
        return True

    monkeypatch.setattr(tv, "_install_to_dir", fake_install)
    real_replace = tv.os.replace
    moved = []

    def flaky_replace(source, target):
        if ".top-up-staging" in str(source) and len(moved) == 1:
            raise OSError("disk full")
        moved.append(target)
        return real_replace(source, target)

    monkeypatch.setattr(tv.os, "replace", flaky_replace)
    assert tv._stage_optional_package("tiktoken", str(root)) is False
    assert len(moved) == 1
    assert not (root / "tiktoken").exists()
    assert not (root / "tiktoken_ext").exists()
    assert not list(root.glob("tiktoken-*.dist-info"))
    assert not (root / ".top-up-staging").exists()
    assert tv._optional_package_absent(str(root), "tiktoken") is True


def test_a_sidecar_without_tiktoken_is_still_valid(tmp_path):
    root = tmp_path / ".venv_t5_550"
    _sidecar(
        root,
        "transformers",
        "huggingface_hub",
        "hf_xet",
        versions = {
            "transformers": tv.TRANSFORMERS_550_VERSION,
            "huggingface_hub": "1.8.0",
            "hf_xet": "1.4.2",
        },
    )
    assert tv._venv_dir_is_valid(str(root), tv._VENV_T5_550_PACKAGES) is True
    assert (
        tv._venv_dir_is_valid(str(root), tv._venv_t5_latest_packages(tv.TRANSFORMERS_550_VERSION))
        is True
    )


def test_a_sidecar_without_a_required_package_is_still_invalid(tmp_path):
    root = tmp_path / ".venv_t5_550"
    _sidecar(
        root,
        "transformers",
        "huggingface_hub",
        "tiktoken",
        versions = {"transformers": tv.TRANSFORMERS_550_VERSION, "huggingface_hub": "1.8.0"},
    )
    assert tv._venv_dir_is_valid(str(root), tv._VENV_T5_550_PACKAGES) is False


def test_a_runtime_repair_survives_a_tiktoken_that_will_not_install(tmp_path, monkeypatch):
    root = tmp_path / ".venv_t5_550"
    installed = []

    def fake_install(pkg, target_dir):
        installed.append(pkg)
        return not pkg.startswith("tiktoken")

    monkeypatch.setattr(tv, "_install_to_dir", fake_install)
    monkeypatch.setattr(tv, "_venv_dir_is_valid_and_undamaged", lambda *a, **k: False)
    assert tv._ensure_venv_dir(str(root), tv._VENV_T5_550_PACKAGES, "test sidecar") is True
    assert installed == list(tv._VENV_T5_550_PACKAGES)

    def fake_install_failing_required(pkg, target_dir):
        return not pkg.startswith("hf_xet")

    monkeypatch.setattr(tv, "_install_to_dir", fake_install_failing_required)
    assert tv._ensure_venv_dir(str(root), tv._VENV_T5_550_PACKAGES, "test sidecar") is False


def test_a_present_tiktoken_is_held_to_its_record_like_any_other(tmp_path, monkeypatch):
    """Absence is what is optional. A tiktoken that is present but whose RECORD names a
    file that is not there (an interrupted install, a native extension left from an
    older interpreter) is a tokenizer that fails at import, and the scan says so."""
    root = tmp_path / ".venv_t5_550"
    _sidecar(
        root,
        "transformers",
        "huggingface_hub",
        "hf_xet",
        versions = {
            "transformers": tv.TRANSFORMERS_550_VERSION,
            "huggingface_hub": "1.8.0",
            "hf_xet": "1.4.2",
        },
    )
    (root / "transformers" / "__init__.py").write_text("", encoding = "utf-8")
    info = root / "tiktoken-0.9.0.dist-info"
    info.mkdir()
    (info / "METADATA").write_text("Name: tiktoken\nVersion: 0.9.0\n", encoding = "utf-8")
    (info / "RECORD").write_text("tiktoken/__init__.py,sha256=abc,1234\n", encoding = "utf-8")
    monkeypatch.delenv(tv._SIDECAR_FILE_CHECK_ENV, raising = False)
    assert tv._sidecar_damaged_files(str(root)) != []
    assert tv._venv_dir_is_valid_and_undamaged(str(root), tv._VENV_T5_550_PACKAGES) is False
    # With the payload the RECORD names in place, the sidecar is whole.
    (root / "tiktoken").mkdir()
    (root / "tiktoken" / "__init__.py").write_bytes(b"x" * 1234)
    assert tv._sidecar_damaged_files(str(root)) == []
    assert tv._venv_dir_is_valid_and_undamaged(str(root), tv._VENV_T5_550_PACKAGES) is True
    # A required package's RECORD is held to the disk too.
    hub = root / "huggingface_hub-1.8.0.dist-info"
    (hub / "RECORD").write_text("huggingface_hub/gone.py,sha256=abc,12\n", encoding = "utf-8")
    assert tv._sidecar_damaged_files(str(root)) != []


def test_a_valid_sidecar_missing_tiktoken_is_topped_up_once(tmp_path, monkeypatch):
    """A transient failure while a sidecar was built (the latest sidecar in particular,
    which no setup top-up visits) left tiktoken out for good: every later check accepted
    the sidecar and returned before another install. The top-up adds it without touching
    the rest, and asks once per process when the wheel is unavailable."""
    root = tmp_path / ".venv_t5_550"
    root.mkdir()
    monkeypatch.setattr(tv, "_venv_dir_is_valid_and_undamaged", lambda *a, **k: True)
    installed: list[tuple[str, str]] = []
    monkeypatch.setattr(
        tv, "_install_to_dir", lambda pkg, target: installed.append((pkg, target)) or False
    )
    tv._OPTIONAL_TOP_UP_ATTEMPTED.clear()
    assert tv._ensure_venv_dir(str(root), tv._VENV_T5_550_PACKAGES, "test sidecar") is True
    assert installed == [("tiktoken", str(root / ".top-up-staging"))]
    assert root.is_dir(), "the top-up must not wipe the sidecar"
    assert tv._ensure_venv_dir(str(root), tv._VENV_T5_550_PACKAGES, "test sidecar") is True
    assert len(installed) == 1, "an unavailable wheel is asked for once per process"
    # A tiktoken that is there is left alone.
    tv._OPTIONAL_TOP_UP_ATTEMPTED.clear()
    (root / tv._OPTIONAL_TOP_UP_FAILED).unlink()
    (root / "tiktoken").mkdir()
    (root / "tiktoken" / "__init__.py").write_text("", encoding = "utf-8")
    (root / "tiktoken-0.9.0.dist-info").mkdir()
    (root / "tiktoken-0.9.0.dist-info" / "RECORD").write_text("", encoding = "utf-8")
    assert tv._ensure_venv_dir(str(root), tv._VENV_T5_550_PACKAGES, "test sidecar") is True
    assert len(installed) == 1


def test_a_failed_top_up_is_remembered_across_processes_and_retried_later(tmp_path, monkeypatch):
    """Each job spawns its own workers, and every one of them used to run the same
    doomed install: the failure is written beside the sidecar, read by the next process,
    and forgotten after a while in case it was the network."""
    root = tmp_path / ".venv_t5_550"
    root.mkdir()
    packages = tv._VENV_T5_550_PACKAGES
    installed = []
    monkeypatch.setattr(tv, "_install_to_dir", lambda pkg, target: installed.append(pkg) or False)
    tv._OPTIONAL_TOP_UP_ATTEMPTED.clear()
    tv._top_up_optional_packages(str(root), packages)
    assert installed == ["tiktoken"]
    # A fresh process: the in-memory set is empty, the file is not.
    tv._OPTIONAL_TOP_UP_ATTEMPTED.clear()
    tv._top_up_optional_packages(str(root), packages)
    assert installed == ["tiktoken"]
    # Old enough to try again.
    failures = tv._read_top_up_failures(str(root))
    failures["tiktoken"] -= tv._OPTIONAL_TOP_UP_RETRY_SECONDS + 1
    tv._write_top_up_failures(str(root), failures)
    tv._OPTIONAL_TOP_UP_ATTEMPTED.clear()
    tv._top_up_optional_packages(str(root), packages)
    assert installed == ["tiktoken", "tiktoken"]
    # A success clears the record.
    monkeypatch.setattr(tv, "_install_to_dir", lambda pkg, target: installed.append(pkg) or True)
    failures = tv._read_top_up_failures(str(root))
    failures["tiktoken"] -= tv._OPTIONAL_TOP_UP_RETRY_SECONDS + 1
    tv._write_top_up_failures(str(root), failures)
    tv._OPTIONAL_TOP_UP_ATTEMPTED.clear()
    tv._top_up_optional_packages(str(root), packages)
    assert not (root / tv._OPTIONAL_TOP_UP_FAILED).exists()


def test_an_interrupted_top_up_is_finished_by_the_next_one(tmp_path, monkeypatch):
    """A process killed between moving tiktoken/ in and its dist-info left a payload no
    scan could judge and no activation would complete: a package counts as present only
    once its dist-info has landed, and the next top-up replaces the partial entries."""
    root = tmp_path / ".venv_t5_550"
    root.mkdir()
    (root / "tiktoken").mkdir()
    (root / "tiktoken" / "__init__.py").write_text("", encoding = "utf-8")
    assert tv._optional_package_absent(str(root), "tiktoken") is True
    (root / "tiktoken-0.9.0.dist-info").mkdir()
    assert tv._optional_package_absent(str(root), "tiktoken") is True
    (root / "tiktoken-0.9.0.dist-info" / "RECORD").write_text("", encoding = "utf-8")
    assert tv._optional_package_absent(str(root), "tiktoken") is False
    shutil.rmtree(root / "tiktoken-0.9.0.dist-info")

    def fake_install(pkg, target):
        for d in ("tiktoken", "tiktoken_ext", "tiktoken-0.9.0.dist-info"):
            (pathlib.Path(target) / d).mkdir()
            (pathlib.Path(target) / d / "RECORD").write_text(d, encoding = "utf-8")
        (pathlib.Path(target) / "tiktoken" / "__init__.py").write_text("", encoding = "utf-8")
        return True

    monkeypatch.setattr(tv, "_install_to_dir", fake_install)
    tv._OPTIONAL_TOP_UP_ATTEMPTED.clear()
    tv._top_up_optional_packages(str(root), tv._VENV_T5_550_PACKAGES)
    assert tv._optional_package_absent(str(root), "tiktoken") is False
    assert (root / "tiktoken_ext" / "RECORD").is_file()


def test_latest_sidecar_activation_tops_up_a_missing_tiktoken(tmp_path, monkeypatch):
    """Model activation goes through _ensure_venv_t5_latest_exists, not the provisioning
    path; a healthy latest sidecar without tiktoken has to get its top-up there too, or a
    latest-tier Qwen model keeps failing to tokenize after a transient install failure."""
    latest = tmp_path / ".venv_t5_latest"
    latest.mkdir()
    packages = ("transformers==9.9.9", "huggingface_hub==1.8.0", "hf_xet==1.4.2", "tiktoken")
    monkeypatch.setattr(tv, "_VENV_T5_LATEST_DIR", str(latest))
    monkeypatch.setattr(tv, "_latest_pin_data", lambda: {"version": "9.9.9", "packages": packages})
    monkeypatch.setattr(tv, "_venv_dir_health", lambda *a, **k: (True, True))
    installed = []
    monkeypatch.setattr(
        tv, "_install_to_dir", lambda pkg, target: installed.append((pkg, target)) or True
    )
    tv._OPTIONAL_TOP_UP_ATTEMPTED.clear()
    assert tv._ensure_venv_t5_latest_exists() is True
    assert installed == [("tiktoken", str(latest / ".top-up-staging"))]


def test_the_top_up_stays_home_offline_and_yields_to_another_process(tmp_path, monkeypatch):
    """Workers activate tiers on their own: offline, none of them should sit through
    network retries for a best-effort package, and two finding it absent at once must
    not both write into the shared sidecar."""
    root = tmp_path / ".venv_t5_550"
    root.mkdir()
    packages = tv._VENV_T5_550_PACKAGES
    installed = []
    monkeypatch.setattr(tv, "_install_to_dir", lambda pkg, target: installed.append(pkg) or True)
    tv._OPTIONAL_TOP_UP_ATTEMPTED.clear()
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    tv._top_up_optional_packages(str(root), packages)
    assert installed == []
    monkeypatch.delenv("HF_HUB_OFFLINE")
    monkeypatch.setenv("UV_OFFLINE", "1")
    tv._top_up_optional_packages(str(root), packages)
    assert installed == []
    # Every spelling uv accepts, or an offline miss would be remembered as a failure.
    for spelling in ("t", "y", "on", "TRUE"):
        monkeypatch.setenv("UV_OFFLINE", spelling)
        tv._OPTIONAL_TOP_UP_ATTEMPTED.clear()
        tv._top_up_optional_packages(str(root), packages)
        assert installed == [], spelling
    assert not (root / tv._OPTIONAL_TOP_UP_FAILED).exists()
    monkeypatch.delenv("UV_OFFLINE")
    # Another process holds the sidecar's top-up lock: this one waits for it, up to the
    # bound, and leaves the package to the holder when the bound passes.
    monkeypatch.setattr(tv, "_OPTIONAL_TOP_UP_WAIT_SECONDS", 0.5)
    with tv._optional_top_up_lock(str(root)) as held:
        assert held is True
        tv._OPTIONAL_TOP_UP_ATTEMPTED.clear()
        tv._top_up_optional_packages(str(root), packages)
        assert installed == []
    tv._OPTIONAL_TOP_UP_ATTEMPTED.clear()
    tv._top_up_optional_packages(str(root), packages)
    assert installed == ["tiktoken"]


def test_the_top_up_is_staged_and_lands_dist_info_last(tmp_path, monkeypatch):
    """A scan by another worker must never meet a RECORD whose files have not landed:
    the package is built beside the sidecar and moved in, payload first, dist-info last."""
    root = tmp_path / ".venv_t5_550"
    root.mkdir()
    order = []

    def fake_install(pkg, target):
        assert target != str(root), "installed straight into the shared sidecar"
        for d in ("tiktoken", "tiktoken_ext", "tiktoken-0.9.0.dist-info"):
            (pathlib.Path(target) / d).mkdir()
            (pathlib.Path(target) / d / "marker").write_text(d, encoding = "utf-8")
        return True

    real_replace = tv.os.replace

    def replacing(src, dst):
        order.append(pathlib.Path(dst).name)
        return real_replace(src, dst)

    monkeypatch.setattr(tv, "_install_to_dir", fake_install)
    monkeypatch.setattr(tv.os, "replace", replacing)
    assert tv._stage_optional_package("tiktoken", str(root)) is True
    assert order[-1] == "tiktoken-0.9.0.dist-info"
    assert set(order) == {"tiktoken", "tiktoken_ext", "tiktoken-0.9.0.dist-info"}
    assert (root / "tiktoken" / "marker").is_file() and not (root / ".top-up-staging").exists()


def test_the_top_up_removes_the_recordless_dist_info_an_interrupted_install_left(
    tmp_path, monkeypatch
):
    """uv cannot uninstall a dist-info with no RECORD and lands the new version beside
    it; both validators skip the recordless one, but importlib.metadata would keep
    answering its version. Every other dist-info of the project goes before the new
    metadata lands, and a dist-info of another project stays."""
    root = tmp_path / ".venv_t5_550"
    root.mkdir()
    stale = root / "tiktoken-0.7.0.dist-info"
    stale.mkdir()
    (stale / "METADATA").write_text("Name: tiktoken\nVersion: 0.7.0\n", encoding = "utf-8")
    other = root / "regex-2024.11.6.dist-info"
    other.mkdir()
    (other / "RECORD").write_text("", encoding = "utf-8")

    def fake_install(pkg, target):
        for d in ("tiktoken", "tiktoken-0.9.0.dist-info"):
            (pathlib.Path(target) / d).mkdir()
            (pathlib.Path(target) / d / "marker").write_text(d, encoding = "utf-8")
        return True

    monkeypatch.setattr(tv, "_install_to_dir", fake_install)
    assert tv._stage_optional_package("tiktoken", str(root)) is True
    assert not stale.exists()
    assert (root / "tiktoken-0.9.0.dist-info" / "marker").is_file()
    assert (other / "RECORD").is_file()


def test_a_recordless_record_beside_a_complete_install_is_removed_without_a_top_up(
    tmp_path, monkeypatch
):
    """A retry that succeeded after an interrupted install leaves both; the package is
    present, so no top-up runs, and the stale metadata must still go."""
    root = tmp_path / ".venv_t5_550"
    (root / "tiktoken").mkdir(parents = True)
    (root / "tiktoken" / "__init__.py").write_text("", encoding = "utf-8")
    good = root / "tiktoken-0.9.0.dist-info"
    good.mkdir()
    (good / "RECORD").write_text("", encoding = "utf-8")
    stale = root / "tiktoken-0.7.0.dist-info"
    stale.mkdir()
    (stale / "METADATA").write_text("Name: tiktoken\nVersion: 0.7.0\n", encoding = "utf-8")
    monkeypatch.setattr(tv, "_env_offline", lambda: False)
    monkeypatch.delenv("UV_OFFLINE", raising = False)
    monkeypatch.setattr(
        tv, "_install_to_dir", lambda *a, **k: pytest.fail("installed over a present package")
    )
    tv._OPTIONAL_TOP_UP_ATTEMPTED.clear()
    tv._top_up_optional_packages(str(root), ("tiktoken",))
    assert not stale.exists()
    assert (good / "RECORD").is_file()


def test_an_offline_session_does_not_wipe_a_sidecar_it_cannot_rebuild(tmp_path, monkeypatch):
    """`studio update` under UV_OFFLINE leaves a stale tier for the next online update;
    the runtime repair used to delete that tier and then reach for the network. With a
    cold cache the tree stays exactly as it was; with a warm one the replacement is
    built beside it and swapped in whole."""
    root = tmp_path / ".venv_t5_550"
    root.mkdir()
    (root / "keep.txt").write_text("", encoding = "utf-8")
    # Valid only once every package landed in it: the live tree never does here.
    built = {}
    monkeypatch.setattr(
        tv,
        "_venv_dir_is_valid_and_undamaged",
        lambda d, packages, *a, **k: built.get(str(d)) == len(packages),
    )
    installed = []
    monkeypatch.setattr(tv, "_install_to_dir", lambda pkg, target: installed.append(pkg) and False)
    monkeypatch.delenv("HF_HUB_OFFLINE", raising = False)
    siblings = lambda: sorted(p.name for p in tmp_path.iterdir() if p.name.startswith(".venv_t5_550."))
    for value in ("1", "t", "Y", "true", "on"):
        monkeypatch.setenv("UV_OFFLINE", value)
        installed.clear()
        assert tv._ensure_venv_dir(str(root), tv._VENV_T5_550_PACKAGES, "test sidecar") is False
        assert (root / "keep.txt").is_file()
        # The cache was asked, beside the tree, never in it.
        assert installed == [tv._VENV_T5_550_PACKAGES[0]]
        assert siblings() == []
    # A warm cache: every package installs into the staging tree and the swap is whole.
    staged_into = []

    def warm_install(pkg, target):
        staged_into.append(target)
        built[target] = built.get(target, 0) + 1
        return True

    monkeypatch.setattr(tv, "_install_to_dir", warm_install)
    monkeypatch.setenv("UV_OFFLINE", "1")
    assert tv._ensure_venv_dir(str(root), tv._VENV_T5_550_PACKAGES, "test sidecar") is True
    assert len(set(staged_into)) == 1
    assert set(staged_into) == {str(tmp_path / f".venv_t5_550.offline-staging-{tv.os.getpid()}")}
    assert not (root / "keep.txt").exists()
    assert (root / tv._STUDIO_OWNED_MARKER).is_file()
    assert siblings() == []
    # A staging tree that installed every package but does not validate never swaps in.
    (root / "keep.txt").write_text("", encoding = "utf-8")
    built.clear()
    monkeypatch.setattr(tv, "_install_to_dir", lambda pkg, target: True)
    assert tv._ensure_venv_dir(str(root), tv._VENV_T5_550_PACKAGES, "test sidecar") is False
    assert (root / "keep.txt").is_file()
    assert siblings() == []
    monkeypatch.setattr(tv, "_venv_dir_is_valid_and_undamaged", lambda *a, **k: False)
    root.mkdir(exist_ok = True)
    (root / "keep.txt").write_text("", encoding = "utf-8")
    installed.clear()
    monkeypatch.setattr(tv, "_install_to_dir", lambda pkg, target: installed.append(pkg) or True)
    # The HF offline switches turn off Hub access, not the package index: a damaged tier
    # is still rebuilt under them, or the tier stays unusable while PyPI answers.
    monkeypatch.setenv("UV_OFFLINE", "0")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    assert tv._runtime_repair_is_offline() is False
    assert tv._ensure_venv_dir(str(root), tv._VENV_T5_550_PACKAGES, "test sidecar") is True
    assert installed == list(tv._VENV_T5_550_PACKAGES)


def test_an_empty_directory_is_still_built_offline_from_the_cache(tmp_path, monkeypatch):
    """The offline rule protects a tree that may still serve. The latest sidecar's staging
    directory and a first install have nothing to lose, and uv installs from a warm cache
    without the network, so they are attempted; only the pip fallback stays out."""
    root = tmp_path / ".venv_t5_latest.staging"
    root.mkdir()
    monkeypatch.setattr(tv, "_venv_dir_is_valid_and_undamaged", lambda *a, **k: False)
    installed = []
    monkeypatch.setattr(tv, "_install_to_dir", lambda pkg, target: installed.append(pkg) or True)
    monkeypatch.setenv("UV_OFFLINE", "1")
    assert tv._ensure_venv_dir(str(root), tv._VENV_T5_550_PACKAGES, "latest staging") is True
    assert installed == list(tv._VENV_T5_550_PACKAGES)
    # A missing directory is a first install, not a repair.
    installed.clear()
    assert tv._ensure_venv_dir(str(tmp_path / "fresh"), tv._VENV_T5_550_PACKAGES, "fresh") is True
    assert installed == list(tv._VENV_T5_550_PACKAGES)


def test_the_pip_fallback_stays_out_offline(tmp_path, monkeypatch):
    calls = []

    class _Result:
        returncode = 1
        stdout = "no cached wheel"

    def fake_run(cmd, **kwargs):
        calls.append(list(cmd))
        return _Result()

    monkeypatch.setattr(tv.shutil, "which", lambda name: "/usr/bin/uv" if name == "uv" else None)
    monkeypatch.setattr(tv.subprocess, "run", fake_run)
    monkeypatch.setenv("UV_OFFLINE", "1")
    assert tv._install_to_dir("tiktoken", str(tmp_path)) is False
    assert len(calls) == 1 and calls[0][0] == "uv"
    # Without uv there is no cache to answer from, and pip is still not asked.
    calls.clear()
    monkeypatch.setattr(tv.shutil, "which", lambda name: None)
    assert tv._install_to_dir("tiktoken", str(tmp_path)) is False
    assert calls == []
    monkeypatch.setattr(tv.shutil, "which", lambda name: "/usr/bin/uv" if name == "uv" else None)
    monkeypatch.setenv("UV_OFFLINE", "0")
    calls.clear()
    assert tv._install_to_dir("tiktoken", str(tmp_path)) is False
    assert [c[0] for c in calls] == ["uv", tv.sys.executable]


def test_an_unreadable_sidecar_counts_as_content(tmp_path, monkeypatch):
    """A listing that fails with anything but "not there" cannot say the tree is empty,
    and the path taken for "empty" begins with deleting it."""
    root = tmp_path / ".venv_t5_550"
    root.mkdir()
    (root / "keep.txt").write_text("", encoding = "utf-8")
    assert tv._sidecar_has_content(str(tmp_path / "absent")) is False
    real_listdir = tv.os.listdir

    def denied(path):
        if str(path) == str(root):
            raise PermissionError(13, "Permission denied", str(path))
        return real_listdir(path)

    monkeypatch.setattr(tv.os, "listdir", denied)
    assert tv._sidecar_has_content(str(root)) is True
    monkeypatch.setattr(tv, "_venv_dir_is_valid_and_undamaged", lambda *a, **k: False)
    monkeypatch.setattr(tv, "_install_to_dir", lambda pkg, target: False)
    monkeypatch.setenv("UV_OFFLINE", "1")
    assert tv._ensure_venv_dir(str(root), tv._VENV_T5_550_PACKAGES, "test sidecar") is False
    assert (root / "keep.txt").is_file()


def test_a_sidecar_stranded_by_an_interrupted_swap_is_restored_first(tmp_path, monkeypatch):
    """Killed between the swap's two renames, the live path is empty and the preserved
    tree sits under the retired name; the next call puts it back before it decides
    anything, rather than reading the empty path as a first install."""
    root = tmp_path / ".venv_t5_550"
    retired = tmp_path / ".venv_t5_550.offline-old-4242"
    retired.mkdir()
    (retired / "keep.txt").write_text("", encoding = "utf-8")
    (retired / tv._STUDIO_OWNED_MARKER).write_text("", encoding = "utf-8")
    monkeypatch.setattr(
        tv, "_venv_dir_is_valid_and_undamaged", lambda d, *a, **k: (Path(d) / "keep.txt").is_file()
    )
    monkeypatch.setattr(tv, "_top_up_optional_packages", lambda *a, **k: None)
    installed = []
    monkeypatch.setattr(tv, "_install_to_dir", lambda pkg, target: installed.append(pkg) and False)
    monkeypatch.setenv("UV_OFFLINE", "1")
    assert tv._ensure_venv_dir(str(root), tv._VENV_T5_550_PACKAGES, "test sidecar") is True
    assert (root / "keep.txt").is_file()
    assert not retired.exists()
    assert installed == []
    # With the live tree in place, a retired copy is a leftover and goes.
    leftover = tmp_path / ".venv_t5_550.offline-old-99"
    leftover.mkdir()
    (leftover / "x").write_text("", encoding = "utf-8")
    assert tv._ensure_venv_dir(str(root), tv._VENV_T5_550_PACKAGES, "test sidecar") is True
    assert (root / "keep.txt").is_file()
    assert not leftover.exists()


def test_a_failed_install_keeps_a_tree_another_process_completed_meanwhile(tmp_path, monkeypatch):
    """Two workers repairing one tier: the late cleanup of the failing one must not
    delete the complete tree the other just finished."""
    root = tmp_path / ".venv_t5_550"
    root.mkdir()
    verdicts = iter([False, True])
    monkeypatch.setattr(tv, "_venv_dir_is_valid_and_undamaged", lambda *a, **k: next(verdicts))
    monkeypatch.setattr(tv, "_install_to_dir", lambda pkg, target: False)
    monkeypatch.delenv("UV_OFFLINE", raising = False)
    assert tv._ensure_venv_dir(str(root), tv._VENV_T5_550_PACKAGES, "test sidecar") is True
    assert root.is_dir()


def test_a_failed_first_install_leaves_nothing_the_offline_guard_would_keep(tmp_path, monkeypatch):
    """The owned marker lands before the first package; a first install that then fails
    must not leave a marker-only or partial tree that the next offline call reads as a
    tree worth keeping and never asks the cache for again."""
    root = tmp_path / ".venv_t5_550"
    monkeypatch.setattr(tv, "_venv_dir_is_valid_and_undamaged", lambda *a, **k: False)
    monkeypatch.setattr(tv, "_install_to_dir", lambda pkg, target: False)
    monkeypatch.setenv("UV_OFFLINE", "1")
    assert tv._ensure_venv_dir(str(root), tv._VENV_T5_550_PACKAGES, "test sidecar") is False
    assert not root.exists()
    # The cache is populated now: the next call asks again instead of keeping a husk.
    installed = []
    monkeypatch.setattr(tv, "_install_to_dir", lambda pkg, target: installed.append(pkg) or True)
    assert tv._ensure_venv_dir(str(root), tv._VENV_T5_550_PACKAGES, "test sidecar") is True
    assert installed == list(tv._VENV_T5_550_PACKAGES)
    # A marker-only directory is not content either.
    root2 = tmp_path / "marker-only"
    root2.mkdir()
    (root2 / tv._STUDIO_OWNED_MARKER).write_text("", encoding = "utf-8")
    assert tv._sidecar_has_content(str(root2)) is False
    (root2 / "keep.txt").write_text("", encoding = "utf-8")
    assert tv._sidecar_has_content(str(root2)) is True
