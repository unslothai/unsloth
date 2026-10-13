# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

pytest.importorskip("datasets")

BACKEND = Path(__file__).resolve().parent.parent

# Each line names one way a Hub card reaches a host file, and whether datasets resolved it there.
PROBE = """
import sys
from utils.security import hub_data_files
{before}
import datasets.data_files
import datasets.load
from datasets import DownloadConfig
from datasets.utils.metadata import MetadataConfigs
{after}
owner, local_base = sys.argv[1], sys.argv[2]
hub = "hf://datasets/someone/repo@main"
secret = owner + "/private.csv"

def show(label, resolve):
    try:
        print(label, "read" if secret in str(resolve()) else "other")
    except FileNotFoundError:
        print(label, "refused")

for label, pattern in (("ABSOLUTE", secret), ("FILE_URL", "file://" + secret),
                       ("LOCAL_URL", "local://" + secret), ("CHAINED", "simplecache::file://" + secret)):
    show(label, lambda: datasets.data_files.resolve_pattern(pattern, base_path = hub))

def card(data_dir):
    configs, _ = datasets.load.create_builder_configs_from_metadata_configs(
        "datasets.packaged_modules.csv.csv",
        MetadataConfigs({{"default": {{"data_dir": data_dir, "data_files": "private.csv"}}}}),
        base_path = hub,
    )
    configs[0]._resolve_data_files(base_path = hub, download_config = DownloadConfig())
    return dict(configs[0].data_files)

show("DATA_DIR", lambda: card(owner))
# A cached Hub snapshot is loaded as a local folder; its card is still the Hub's.
snapshot = sys.argv[3]
show("SNAPSHOT", lambda: datasets.data_files.resolve_pattern(secret, base_path = snapshot))
own_rows = datasets.data_files.resolve_pattern("train.csv", base_path = snapshot)
print("SNAPSHOT_OWN", "read" if own_rows and own_rows[0].endswith("train.csv") else "other")
# A local dataset may still name its own files by absolute path.
show(
    "LOCAL_DATASET", lambda: datasets.data_files.resolve_pattern(secret, base_path = local_base)
)
wrapped = getattr(datasets.data_files.resolve_pattern, "__wrapped__", None)
print("DOUBLE", getattr(wrapped, "_unsloth_hub_card_guard", False))
print("FINDERS", sum(isinstance(f, hub_data_files._Finder) for f in sys.meta_path))
"""

HUB_ROUTES = ("ABSOLUTE", "FILE_URL", "LOCAL_URL", "CHAINED", "DATA_DIR", "SNAPSHOT")


def _probe(tmp_path, before, after):
    owner = tmp_path / "owner"
    owner.mkdir()
    (owner / "private.csv").write_text("secret\n1\n")
    snapshot = tmp_path / "hub" / "datasets--org--repo" / "snapshots" / "abc123"
    snapshot.mkdir(parents = True)
    (snapshot / "train.csv").write_text("a\n1\n")
    code = textwrap.dedent(PROBE.format(before = before, after = after))
    out = subprocess.run(
        [sys.executable, "-c", code, str(owner), str(tmp_path), str(snapshot)],
        cwd = BACKEND,
        capture_output = True,
        text = True,
        timeout = 300,
        env = {"PYTHONPATH": str(BACKEND), "HF_HUB_OFFLINE": "1", "PATH": "/usr/bin:/bin"},
    )
    assert out.returncode == 0, out.stderr
    return dict(line.split(" ", 1) for line in out.stdout.splitlines() if " " in line)


# Installed before datasets loads (Studio's startup), after it, and twice: one wrapper, one finder.
@pytest.mark.parametrize(
    "before,after",
    [
        ("hub_data_files.install()", ""),
        ("", "hub_data_files.install()"),
        ("hub_data_files.install(); hub_data_files.install()", "hub_data_files.install()"),
    ],
)
def test_a_hub_card_cannot_reach_a_host_file(tmp_path, before, after):
    seen = _probe(tmp_path, before, after)

    assert {route: seen[route] for route in HUB_ROUTES} == dict.fromkeys(HUB_ROUTES, "refused")
    assert seen["LOCAL_DATASET"] == "read"
    assert seen["SNAPSHOT_OWN"] == "read"
    assert seen["DOUBLE"] == "False"
    assert seen["FINDERS"] == "1"


def test_without_the_guard_datasets_reads_the_host_file(tmp_path):
    # The behaviour the guard exists for, so a datasets release that changes it shows up here.
    seen = _probe(tmp_path, "", "")

    # datasets 4.3.0 already refuses the chained form; the guard keeps refusing it if that changes.
    open_routes = [route for route in HUB_ROUTES if route != "CHAINED"]
    assert {route: seen[route] for route in open_routes} == dict.fromkeys(open_routes, "read")
