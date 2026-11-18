# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The remote-code-scan route reads each Hub config once per request; nothing else changes. Only a
downloaded path or the Hub's own 404 is shared, only inside one scope, one context, one token
and one endpoint, and the load-time gates in the workers never share at all."""

import asyncio
import json
import threading
from pathlib import Path

import pytest
from huggingface_hub.errors import EntryNotFoundError, LocalEntryNotFoundError

import huggingface_hub
import utils.hf_probe as hf_probe
import utils.security.remote_code_scan as rcs
from utils.security.consent import (
    _load_remote_code_configs,
    evaluate_remote_code_consent_for_targets,
)
from utils.security.remote_code_scan import (
    external_auto_map_repos,
    repo_remote_code_files,
    shared_config_reads,
)

_BACKEND = Path(__file__).resolve().parents[1]
_CONFIG = json.dumps(
    {"auto_map": {"AutoModel": "modeling_x.M", "AutoTokenizer": "ext/repo--tok.T"}}
)


class _Hub:
    """In-process Hub: files per repo, a log of every probe and download, scripted failures."""

    def __init__(self, tmp_path, monkeypatch):
        self.root = tmp_path
        self.repos = {
            "org/model": {
                "config.json": _CONFIG,
                "tokenizer_config.json": "{}",
                "modeling_x.py": "class M:\n    pass\n",
            },
            "ext/repo": {"tok.py": "class T:\n    pass\n"},
        }
        self.calls = []
        self.fail = {}  # (repo, file) -> list of exceptions raised by the next downloads
        self.version = 0
        monkeypatch.setattr(huggingface_hub, "hf_hub_download", self.download)
        monkeypatch.setattr(huggingface_hub, "list_repo_files", self.listing)
        monkeypatch.setattr(hf_probe, "hf_file_definitely_absent", self.absent)

    def absent(
        self,
        repo,
        filename,
        *,
        token = None,
        **_kw,
    ):
        self.calls.append(("probe", repo, filename, token))
        return filename not in self.repos.get(repo, {})

    def download(
        self,
        repo_id = None,
        filename = None,
        *args,
        token = None,
        cache_dir = None,
        **_kw,
    ):
        if args:
            filename = args[0]
        self.calls.append(("download", repo_id, filename, token))
        pending = self.fail.get((repo_id, filename))
        if pending:
            raise pending.pop(0)
        files = self.repos.get(repo_id, {})
        if filename not in files:
            raise EntryNotFoundError(filename)
        self.version += 1
        path = self.root / f"v{self.version}" / repo_id / filename
        path.parent.mkdir(parents = True, exist_ok = True)
        path.write_text(files[filename], encoding = "utf-8")
        return str(path)

    def listing(
        self,
        repo_id,
        token = None,
        **_kw,
    ):
        self.calls.append(("list", repo_id, None, token))
        return list(self.repos.get(repo_id, {}))

    def config_reads(self, kind = None):
        return [
            c
            for c in self.calls
            if c[2] in rcs.REMOTE_CODE_CONFIG_FILES and (kind is None or c[0] == kind)
        ]


@pytest.fixture
def hub(tmp_path, monkeypatch):
    monkeypatch.delenv("HF_HUB_OFFLINE", raising = False)
    return _Hub(tmp_path, monkeypatch)


def _three_passes(token = None):
    """What the route runs per target: external refs, the auto_map check, then the scan."""
    refs = external_auto_map_repos("org/model", token)
    configs = _load_remote_code_configs("org/model", token)
    files = repo_remote_code_files("org/model", hf_token = token)
    return refs, configs, files


def test_one_scope_reads_each_config_once(hub):
    with shared_config_reads():
        shared = _three_passes()
    shared_reads = hub.config_reads()
    hub.calls.clear()
    fresh = _three_passes()
    assert shared == fresh
    per_file = len(rcs.REMOTE_CODE_CONFIG_FILES)
    # Present: one probe + one download; absent: one probe.
    assert len(shared_reads) == per_file + 2
    assert len(hub.config_reads()) == 3 * len(shared_reads)
    # The code itself is still downloaded and listed fresh by the scan.
    assert [c for c in hub.calls if c[0] == "list"] == [
        ("list", "org/model", None, None),
        ("list", "ext/repo", None, None),
    ]


def test_outside_a_scope_nothing_is_shared(hub):
    _three_passes()
    first = len(hub.config_reads())
    _three_passes()
    assert len(hub.config_reads()) == 2 * first
    assert rcs._CONFIG_READS.get() is None


def test_a_failed_read_is_never_kept(hub):
    """A transient error, an auth error and a local cache miss (offline/timeout) are all retried by
    the next pass, exactly as without the scope."""
    hub.fail[("org/model", "config.json")] = [
        RuntimeError("503"),
        LocalEntryNotFoundError("timeout, nothing cached"),
    ]
    with shared_config_reads():
        assert external_auto_map_repos("org/model") == set()  # best effort: skips the failure
        # The auto_map check sees the local miss as absence, as it always has.
        assert _load_remote_code_configs("org/model") == [{}]
        files = repo_remote_code_files("org/model")
    downloads = [c for c in hub.config_reads("download") if c[2] == "config.json"]
    assert len(downloads) == 3
    assert set(files) == {"modeling_x.py", "ext/repo--tok.py"}


def test_a_confirmed_404_is_kept_for_the_scope_only(hub):
    with shared_config_reads():
        _three_passes()
        _three_passes()
    probes = [c for c in hub.config_reads("probe") if c[2] == "processor_config.json"]
    assert len(probes) == 1
    with shared_config_reads():
        _three_passes()
    probes = [c for c in hub.config_reads("probe") if c[2] == "processor_config.json"]
    assert len(probes) == 2


def test_a_file_changed_between_two_actions_is_read_again(hub):
    with shared_config_reads():
        first = external_auto_map_repos("org/model")
    hub.repos["org/model"]["config.json"] = json.dumps(
        {"auto_map": {"AutoModel": "other/repo--evil.M"}}
    )
    hub.repos["other/repo"] = {"evil.py": "import os\nos.system('id')\n"}
    with shared_config_reads():
        second = external_auto_map_repos("org/model")
        files = repo_remote_code_files("org/model")
    assert first == {"ext/repo"} and second == {"other/repo"}
    assert "other/repo--evil.py" in files


def test_a_config_purged_from_the_cache_mid_scan_is_downloaded_again(hub):
    with shared_config_reads():
        external_auto_map_repos("org/model")
        for c in hub.root.glob("v*/org/model/config.json"):
            c.unlink()
        files = repo_remote_code_files("org/model")
    assert "ext/repo--tok.py" in files
    assert len([c for c in hub.config_reads("download") if c[2] == "config.json"]) == 2


def test_tokens_never_share_reads(hub):
    with shared_config_reads():
        _load_remote_code_configs("org/model", None)
        _load_remote_code_configs("org/model", "hf_a")
        _load_remote_code_configs("org/model", "hf_b")
        _load_remote_code_configs("org/model", "hf_a")
    tokens = [c[3] for c in hub.config_reads("download") if c[2] == "config.json"]
    assert tokens == [None, "hf_a", "hf_b"]


def test_the_endpoint_and_source_are_part_of_the_key(hub, monkeypatch):
    from huggingface_hub import constants
    with shared_config_reads():
        _load_remote_code_configs("org/model")
        monkeypatch.setattr(constants, "ENDPOINT", "https://mirror.example")
        _load_remote_code_configs("org/model")
        monkeypatch.setattr("utils.hub_settings.active_source", lambda: "modelscope")
        _load_remote_code_configs("org/model")
    assert len([c for c in hub.config_reads("download") if c[2] == "config.json"]) == 3


def test_threads_and_concurrent_requests_do_not_share(hub):
    seen = {}

    def other_thread():
        seen["scope"] = rcs._CONFIG_READS.get()
        _load_remote_code_configs("org/model")

    with shared_config_reads():
        _load_remote_code_configs("org/model")
        t = threading.Thread(target = other_thread)
        t.start()
        t.join()
    assert seen["scope"] is None
    assert len([c for c in hub.config_reads("download") if c[2] == "config.json"]) == 2

    hub.calls.clear()
    scopes = []

    async def request():
        with shared_config_reads():
            scopes.append(rcs._CONFIG_READS.get())
            await asyncio.sleep(0)
            _load_remote_code_configs("org/model")
            await asyncio.sleep(0)
            _load_remote_code_configs("org/model")

    async def two_requests():
        await asyncio.gather(request(), request())

    asyncio.run(two_requests())
    assert scopes[0] is not scopes[1]
    assert len([c for c in hub.config_reads("download") if c[2] == "config.json"]) == 2


def test_the_scope_is_cleared_on_exit_and_on_error(hub):
    with pytest.raises(RuntimeError):
        with shared_config_reads():
            _load_remote_code_configs("org/model")
            raise RuntimeError("route failed")
    assert rcs._CONFIG_READS.get() is None
    with shared_config_reads():
        outer = rcs._CONFIG_READS.get()
        with shared_config_reads():
            assert rcs._CONFIG_READS.get() is outer
        assert rcs._CONFIG_READS.get() is outer
    assert rcs._CONFIG_READS.get() is None


def test_decision_and_fingerprint_match_the_unshared_gate(hub):
    hub.repos["org/model"]["modeling_x.py"] = (
        "import subprocess\nsubprocess.Popen('id', shell=True)\n"
    )
    fresh = evaluate_remote_code_consent_for_targets(["org/model"], trust_remote_code = True)
    with shared_config_reads():
        external_auto_map_repos("org/model")
        shared = evaluate_remote_code_consent_for_targets(["org/model"], trust_remote_code = True)
    assert shared.response_payload() == fresh.response_payload()
    assert shared.fingerprint and shared.fingerprint == fresh.fingerprint


def test_only_the_advisory_route_shares_reads():
    """The load-time gates are authoritative, so they always read the Hub fresh."""
    route = (_BACKEND / "routes" / "models.py").read_text(encoding = "utf-8")
    assert "with shared_config_reads():" in route
    for rel in (
        "core/inference/worker.py",
        "core/training/worker.py",
        "core/export/worker.py",
        "utils/security/consent.py",
    ):
        src = (_BACKEND / rel).read_text(encoding = "utf-8")
        assert "shared_config_reads" not in src, rel
