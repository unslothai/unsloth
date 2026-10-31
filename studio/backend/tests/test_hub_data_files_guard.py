# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

pytest.importorskip("datasets")

BACKEND = Path(__file__).resolve().parent.parent

# Each line names one way a Hub card reaches a host file, and whether datasets resolved it there.
PROBE = """
import os
import pathlib
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
secret = os.path.join(owner, "private.csv")

def show(label, resolve):
    try:
        # private.csv is the only file of that name, so it marks the host file on any OS.
        print(label, "read" if "private.csv" in str(resolve()) else "other")
    except FileNotFoundError:
        print(label, "refused")

posix = pathlib.Path(secret).as_posix()
for label, pattern in (("ABSOLUTE", secret), ("FILE_URL", pathlib.Path(secret).as_uri()),
                       ("LOCAL_URL", "local://" + posix), ("CHAINED", "simplecache::file://" + posix)):
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
climb = os.path.relpath(secret, snapshot)
show("TRAVERSAL", lambda: datasets.data_files.resolve_pattern(climb, base_path = snapshot))
# An owner's own folder merely named like a snapshot, outside the Hub cache, stays local.
lookalike = sys.argv[4]
show("LOOKALIKE", lambda: datasets.data_files.resolve_pattern(secret, base_path = lookalike))
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

HUB_ROUTES = (
    "ABSOLUTE",
    "FILE_URL",
    "LOCAL_URL",
    "CHAINED",
    "DATA_DIR",
    "SNAPSHOT",
    "TRAVERSAL",
)


def _probe(tmp_path, before, after):
    owner = tmp_path / "owner"
    owner.mkdir()
    (owner / "private.csv").write_text("secret\n1\n")
    snapshot = tmp_path / "hub" / "datasets--org--repo" / "snapshots" / "abc123"
    snapshot.mkdir(parents = True)
    lookalike = tmp_path / "mine" / "datasets--archive" / "snapshots" / "current"
    lookalike.mkdir(parents = True)
    (snapshot / "train.csv").write_text("a\n1\n")
    code = textwrap.dedent(PROBE.format(before = before, after = after))
    out = subprocess.run(
        [sys.executable, "-c", code, str(owner), str(tmp_path), str(snapshot), str(lookalike)],
        cwd = BACKEND,
        capture_output = True,
        text = True,
        timeout = 300,
        # The full environment: Windows cannot load sockets without SYSTEMROOT and friends.
        env = {
            **os.environ,
            "PYTHONPATH": str(BACKEND),
            "HF_HUB_OFFLINE": "1",
            "HF_HUB_CACHE": str(tmp_path / "hub"),
        },
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
    assert seen["LOOKALIKE"] == "read"
    assert seen["DOUBLE"] == "False"
    assert seen["FINDERS"] == "1"


@pytest.mark.skipif(sys.platform == "win32", reason = "records datasets' POSIX URL parsing")
def test_without_the_guard_datasets_reads_the_host_file(tmp_path):
    # The behaviour the guard exists for, so a datasets release that changes it shows up here.
    seen = _probe(tmp_path, "", "")

    # datasets 4.3.0 already refuses the chained form; the guard keeps refusing it if that changes.
    open_routes = [route for route in HUB_ROUTES if route != "CHAINED"]
    assert {route: seen[route] for route in open_routes} == dict.fromkeys(open_routes, "read")


# A card whose data_files is an http(s) URL makes the server dial it, following redirects. The owner may
# load such cards as before; a managed account may not, or it could read loopback, LAN or metadata services.
URL_PROBE = """
import http.server
import os
import sys
import threading
from utils.security import hub_data_files
hub_data_files.install()
from datasets import load_dataset
from utils.account_context import AccountContext, run_as

hits = []

class Internal(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        hits.append(self.path)
        if self.path.startswith("/public.csv"):
            self.send_response(302)
            self.send_header("Location", "/internal-secret")
            self.end_headers()
            return
        body = b"secret\\nINTERNAL_ONLY\\n"
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    do_HEAD = do_GET

    def log_message(self, *args):
        pass

server = http.server.HTTPServer(("127.0.0.1", 0), Internal)
threading.Thread(target = server.serve_forever, daemon = True).start()
snapshot = sys.argv[1]
with open(os.path.join(snapshot, "README.md"), "w") as f:
    f.write(
        "---\\nconfigs:\\n- config_name: default\\n  data_files:\\n  - split: train\\n"
        f"    path: http://127.0.0.1:{server.server_port}/public.csv\\n---\\n"
    )

def rows():
    return list(load_dataset(snapshot, split = "train", streaming = True).take(2))

for label, account in (("OWNER", None), ("MANAGED", AccountContext("acct-bob", "bob"))):
    hits.clear()
    try:
        got = rows() if account is None else run_as(account, rows)
        print(label, "read" if "INTERNAL_ONLY" in str(got) else "other", len(hits))
    except FileNotFoundError:
        print(label, "refused", len(hits))
"""


def test_a_managed_account_cannot_make_a_hub_card_dial_a_url(tmp_path):
    snapshot = tmp_path / "hub" / "datasets--org--card" / "snapshots" / "abc123"
    snapshot.mkdir(parents = True)
    out = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(URL_PROBE), str(snapshot)],
        cwd = BACKEND,
        capture_output = True,
        text = True,
        timeout = 300,
        env = {
            **os.environ,
            "PYTHONPATH": str(BACKEND),
            "HF_HUB_CACHE": str(tmp_path / "hub"),
            "NO_PROXY": "127.0.0.1",
            "no_proxy": "127.0.0.1",
        },
    )
    assert out.returncode == 0, out.stderr
    seen = dict(line.split(" ", 1) for line in out.stdout.splitlines() if " " in line)

    assert seen["OWNER"].split()[0] == "read"
    assert seen["MANAGED"] == "refused 0"
