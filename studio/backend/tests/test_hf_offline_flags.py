# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Effective-offline handling for the embedding preflight.

``huggingface_hub`` honors only ``HF_HUB_OFFLINE``; ``TRANSFORMERS_OFFLINE`` expresses the
same intent but does not stop a fetch. Studio treats either as offline (``hf_env_offline``)
and makes it real by passing ``local_files_only`` to the loader, which lets the Hub security
scan skip to its fail-open instead of burning both timeouts. That skip is sound only while
the loader is pinned to the local cache, so the coupling is pinned here as an invariant.
"""

from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

import pytest

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)


def _maybe_stub(name: str, builder):
    # Stub only if the real module is unavailable, so this file never shadows real packages.
    try:
        importlib.import_module(name)
    except ImportError:
        sys.modules[name] = builder()


def _build_loggers_stub():
    m = types.ModuleType("loggers")
    m.get_logger = lambda name: __import__("logging").getLogger(name)
    return m


def _build_structlog_stub():
    m = types.ModuleType("structlog")
    m.get_logger = lambda *a, **k: __import__("logging").getLogger("stub")
    return m


_maybe_stub("loggers", _build_loggers_stub)
_maybe_stub("structlog", _build_structlog_stub)

from utils.utils import hf_env_offline  # noqa: E402


@pytest.fixture(autouse = True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("HF_HUB_OFFLINE", raising = False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising = False)
    yield


# ── flag semantics ──


def test_neither_flag_is_online():
    assert hf_env_offline() is False


@pytest.mark.parametrize("var", ["HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"])
def test_either_flag_means_offline(monkeypatch, var):
    monkeypatch.setenv(var, "1")
    assert hf_env_offline() is True


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on", "  1  "])
def test_truthy_values_parse(monkeypatch, value):
    monkeypatch.setenv("HF_HUB_OFFLINE", value)
    assert hf_env_offline() is True


@pytest.mark.parametrize("value", ["", "0", "false", "no", "off", "maybe"])
def test_non_truthy_values_do_not_parse(monkeypatch, value):
    monkeypatch.setenv("HF_HUB_OFFLINE", value)
    assert hf_env_offline() is False


# ── the offline security short-circuit, and the invariant it rests on ──


def _fake_hub(monkeypatch, calls):
    fake = types.ModuleType("huggingface_hub")

    def _model_info(*a, **k):
        calls.append(1)
        raise RuntimeError("network unreachable")

    fake.model_info = _model_info
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake)


@pytest.mark.parametrize("var", ["HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"])
def test_shared_gate_still_scans_when_offline_by_default(monkeypatch, var):
    # Shared gate: an offline env var alone must never disable the scan for a fetching loader.
    import utils.security.file_security as fs

    calls: list = []
    _fake_hub(monkeypatch, calls)
    monkeypatch.setenv(var, "1")
    assert fs._fetch_security_status("org/model", None) == (None, None)
    assert calls, f"{var} alone must NOT bypass the shared malware gate"


def test_local_only_load_fails_closed_offline_never_hitting_the_hub(monkeypatch, tmp_path):
    # A local-only (offline) load never hits the Hub; it is evaluated fail-CLOSED against the
    # cached files: a pickle-free (safetensors) cache is allowed, a cached pickle is blocked.
    import utils.models.model_config as mc
    import utils.security.file_security as fs

    calls: list = []
    _fake_hub(monkeypatch, calls)

    def _evaluate(dir_path):
        monkeypatch.setattr(mc, "_active_snapshot_dir", lambda name: dir_path)
        decision = fs.evaluate_file_security("org/model", None, local_only_load = True)
        assert calls == [], "a local-only load must not hit the Hub"
        return decision.blocked

    def _snap(name, files: dict):
        d = tmp_path / name / "aaa"
        d.mkdir(parents = True)
        for fname, body in files.items():
            (d / fname).write_bytes(body if isinstance(body, bytes) else body.encode())
        return d

    # A genuinely inert (unsharded safetensors) cache is allowed.
    assert _evaluate(_snap("safe", {"model.safetensors": b"\0"})) is False

    # A cached pickle weight with no loadable safetensors alternative is blocked.
    assert _evaluate(_snap("bad", {"pytorch_model.bin": b"\0"})) is True

    # A pickle beside a bare ADAPTER (not a base weight) still loads the pickle -> blocked:
    # from_pretrained cannot use adapter_model.safetensors as the base checkpoint.
    assert (
        _evaluate(
            _snap(
                "adapter",
                {
                    "pytorch_model.bin": b"\0",
                    "adapter_model.safetensors": b"\0",
                },
            )
        )
        is True
    )

    # A pickle beside a lone ORPHAN shard (no index) still loads the pickle -> blocked:
    # a sharded safetensors load needs the index the loader reads to locate every shard.
    assert (
        _evaluate(
            _snap(
                "orphan",
                {
                    "pytorch_model.bin": b"\0",
                    "model-00001-of-00002.safetensors": b"\0",
                },
            )
        )
        is True
    )

    # A COMPLETE indexed safetensors shard set is what the loader picks instead of the
    # pickle -> allowed.
    index = (
        '{"weight_map": {"a": "model-00001-of-00002.safetensors", '
        '"b": "model-00002-of-00002.safetensors"}}'
    )
    assert (
        _evaluate(
            _snap(
                "sharded",
                {
                    "pytorch_model.bin": b"\0",
                    "model-00001-of-00002.safetensors": b"\0",
                    "model-00002-of-00002.safetensors": b"\0",
                    "model.safetensors.index.json": index,
                },
            )
        )
        is False
    )

    # A stray pickle in a NON-load subdir (no config.json) is never deserialized by
    # from_pretrained, so it must not block -- matching the online scan's load-path scope.
    stray = tmp_path / "stray" / "aaa"
    (stray / "archive").mkdir(parents = True)
    (stray / "model.safetensors").write_bytes(b"\0")
    (stray / "archive" / "pytorch_model.bin").write_bytes(b"\0")
    assert _evaluate(stray) is False

    # A pickle in a real MODULE load root (a modules.json-declared Transformer subdir) and no
    # safetensors there is a live vector -> blocked.
    modroot = tmp_path / "modroot" / "aaa"
    (modroot / "0_Transformer").mkdir(parents = True)
    (modroot / "modules.json").write_text(
        '[{"idx": 0, "name": "0_Transformer", "path": "0_Transformer", '
        '"type": "sentence_transformers.models.Transformer"}]'
    )
    (modroot / "0_Transformer" / "config.json").write_bytes(b"{}")
    (modroot / "0_Transformer" / "pytorch_model.bin").write_bytes(b"\0")
    assert _evaluate(modroot) is True

    # An UNREFERENCED nested checkpoint (its own config.json + pickle) is NOT a load root: no
    # modules.json / router_config declares it and from_pretrained never descends into it, so it
    # must not block a model the loader reads from a clean safetensors root -- matching the online
    # scan, which ignores the same unindexed subdir pickle. A stray config.json does not make a
    # subdir a load root.
    unref = tmp_path / "unref" / "aaa"
    (unref / "checkpoint-500").mkdir(parents = True)
    (unref / "config.json").write_bytes(b"{}")
    (unref / "model.safetensors").write_bytes(b"\0")
    (unref / "checkpoint-500" / "config.json").write_bytes(b"{}")
    (unref / "checkpoint-500" / "pytorch_model.bin").write_bytes(b"\0")
    assert _evaluate(unref) is False

    # A pickle in a modules.json-declared module dir WITHOUT a config.json (a WordEmbeddings
    # module: wordembedding_config.json + pytorch_model.bin) is still deserialized by the ST
    # loader, so it must be scanned -> blocked, not skipped for lacking config.json.
    we = tmp_path / "wordemb" / "aaa"
    (we / "0_WordEmbeddings").mkdir(parents = True)
    (we / "modules.json").write_text(
        '[{"idx": 0, "name": "0_WordEmbeddings", "path": "0_WordEmbeddings", '
        '"type": "sentence_transformers.models.WordEmbeddings"}]'
    )
    (we / "0_WordEmbeddings" / "wordembedding_config.json").write_bytes(b"{}")
    (we / "0_WordEmbeddings" / "pytorch_model.bin").write_bytes(b"\0")
    assert _evaluate(we) is True

    # A PEFT adapter pickle is a SEPARATE vector: from_pretrained auto-detects adapter_config.json
    # and deserializes adapter_model.bin ON TOP of the base, so an inert safetensors base does not
    # cover it -> blocked.
    adbin = tmp_path / "adapterbin" / "aaa"
    adbin.mkdir(parents = True)
    (adbin / "config.json").write_bytes(b"{}")
    (adbin / "model.safetensors").write_bytes(b"\0")
    (adbin / "adapter_config.json").write_bytes(b"{}")
    (adbin / "adapter_model.bin").write_bytes(b"\0")
    assert _evaluate(adbin) is True

    # The same adapter shipped as safetensors is inert -> allowed.
    adsafe = tmp_path / "adaptersafe" / "aaa"
    adsafe.mkdir(parents = True)
    (adsafe / "config.json").write_bytes(b"{}")
    (adsafe / "model.safetensors").write_bytes(b"\0")
    (adsafe / "adapter_config.json").write_bytes(b"{}")
    (adsafe / "adapter_model.safetensors").write_bytes(b"\0")
    assert _evaluate(adsafe) is False

    # A root pickle index whose weight_map points shards into a SUBDIR: from_pretrained follows
    # the index and deserializes them, so they must be scanned even though the subdir is not a
    # load root of its own (no config.json).
    idxsub = tmp_path / "idxsub" / "aaa"
    (idxsub / "sharded").mkdir(parents = True)
    (idxsub / "config.json").write_bytes(b"{}")
    (idxsub / "pytorch_model.bin.index.json").write_text(
        '{"weight_map": {"a": "sharded/pytorch_model-00001-of-00002.bin", '
        '"b": "sharded/pytorch_model-00002-of-00002.bin"}}'
    )
    (idxsub / "sharded" / "pytorch_model-00001-of-00002.bin").write_bytes(b"\0")
    (idxsub / "sharded" / "pytorch_model-00002-of-00002.bin").write_bytes(b"\0")
    assert _evaluate(idxsub) is True

    # Those subdir shards are covered when a loadable base safetensors sits at the index root.
    idxsafe = tmp_path / "idxsafe" / "aaa"
    (idxsafe / "sharded").mkdir(parents = True)
    (idxsafe / "config.json").write_bytes(b"{}")
    (idxsafe / "model.safetensors").write_bytes(b"\0")
    (idxsafe / "pytorch_model.bin.index.json").write_text(
        '{"weight_map": {"a": "sharded/pytorch_model-00001-of-00002.bin"}}'
    )
    (idxsafe / "sharded" / "pytorch_model-00001-of-00002.bin").write_bytes(b"\0")
    assert _evaluate(idxsafe) is False

    # A complete safetensors shard set that a safetensors index names in a SUBDIR covers a legacy
    # pytorch_model.bin: the loader reads the inert safetensors, so this must be allowed (the
    # index shards resolve relative to the index dir, not by basename).
    stsub = tmp_path / "stsub" / "aaa"
    (stsub / "weights").mkdir(parents = True)
    (stsub / "config.json").write_bytes(b"{}")
    (stsub / "pytorch_model.bin").write_bytes(b"\0")
    (stsub / "model.safetensors.index.json").write_text(
        '{"weight_map": {"a": "weights/model-00001-of-00002.safetensors", '
        '"b": "weights/model-00002-of-00002.safetensors"}}'
    )
    (stsub / "weights" / "model-00001-of-00002.safetensors").write_bytes(b"\0")
    (stsub / "weights" / "model-00002-of-00002.safetensors").write_bytes(b"\0")
    assert _evaluate(stsub) is False

    # A SentenceTransformer Router (legacy Asym) declares its child sub-modules ONLY in
    # router_config.json (never the top-level modules.json), and Router.load() deserializes each
    # child's weights from its own subdir. A config.json-less child (query_0_WordEmbeddings:
    # wordembedding_config.json + pytorch_model.bin, no safetensors) is a live pickle vector and
    # must be scanned -> blocked, not skipped for lacking a config.json or a modules.json entry.
    router = tmp_path / "router" / "aaa"
    (router / "query_0_WordEmbeddings").mkdir(parents = True)
    (router / "document_0_Transformer").mkdir(parents = True)
    (router / "modules.json").write_text(
        '[{"idx": 0, "name": "0", "path": "", "type": "sentence_transformers.models.Router"}]'
    )
    (router / "router_config.json").write_text(
        '{"types": {"query_0_WordEmbeddings": '
        '"sentence_transformers.models.WordEmbeddings", '
        '"document_0_Transformer": "sentence_transformers.models.Transformer"}, '
        '"structure": {"query": ["query_0_WordEmbeddings"], '
        '"document": ["document_0_Transformer"]}, "parameters": {}}'
    )
    (router / "query_0_WordEmbeddings" / "wordembedding_config.json").write_bytes(b"{}")
    (router / "query_0_WordEmbeddings" / "pytorch_model.bin").write_bytes(b"\0")
    (router / "document_0_Transformer" / "config.json").write_bytes(b"{}")
    (router / "document_0_Transformer" / "model.safetensors").write_bytes(b"\0")
    assert _evaluate(router) is True

    # The same Router whose child ships model.safetensors beside the pickle is inert: the child is
    # scoped as a load root, but the loader reads the safetensors it prefers -> allowed (the fix
    # scopes the child without over-blocking a genuinely safe one).
    routersafe = tmp_path / "routersafe" / "aaa"
    (routersafe / "query_0_WordEmbeddings").mkdir(parents = True)
    (routersafe / "modules.json").write_text(
        '[{"idx": 0, "name": "0", "path": "", "type": "sentence_transformers.models.Router"}]'
    )
    (routersafe / "router_config.json").write_text(
        '{"types": {"query_0_WordEmbeddings": "sentence_transformers.models.WordEmbeddings"}}'
    )
    (routersafe / "query_0_WordEmbeddings" / "wordembedding_config.json").write_bytes(b"{}")
    (routersafe / "query_0_WordEmbeddings" / "pytorch_model.bin").write_bytes(b"\0")
    (routersafe / "query_0_WordEmbeddings" / "model.safetensors").write_bytes(b"\0")
    assert _evaluate(routersafe) is False

    # The Router can itself sit in a modules.json-declared subfolder; the child scan must follow
    # router_config.json from THAT dir, not only the snapshot root.
    routersub = tmp_path / "routersub" / "aaa"
    (routersub / "1_Router" / "query_0_WordEmbeddings").mkdir(parents = True)
    (routersub / "modules.json").write_text(
        '[{"idx": 0, "name": "1_Router", "path": "1_Router", '
        '"type": "sentence_transformers.models.Router"}]'
    )
    (routersub / "1_Router" / "router_config.json").write_text(
        '{"types": {"query_0_WordEmbeddings": "sentence_transformers.models.WordEmbeddings"}}'
    )
    (routersub / "1_Router" / "query_0_WordEmbeddings" / "wordembedding_config.json").write_bytes(
        b"{}"
    )
    (routersub / "1_Router" / "query_0_WordEmbeddings" / "pytorch_model.bin").write_bytes(b"\0")
    assert _evaluate(routersub) is True

    # A modules.json declaring a TRAVERSING module path ("0/../evil") is resolved by the ST loader
    # to evil/, which it deserializes; the gate must scope the NORMALIZED dir and block
    # evil/pytorch_model.bin -- recording the raw snap/"0/../evil" (which never equals the real
    # snap/evil rglob yields) would let a malicious repo slip a pickle past the offline gate.
    trav = tmp_path / "trav" / "aaa"
    (trav / "evil").mkdir(parents = True)
    (trav / "modules.json").write_text(
        '[{"idx": 0, "name": "m", "path": "0/../evil", '
        '"type": "sentence_transformers.models.WordEmbeddings"}]'
    )
    (trav / "evil" / "wordembedding_config.json").write_bytes(b"{}")
    (trav / "evil" / "pytorch_model.bin").write_bytes(b"\0")
    assert _evaluate(trav) is True

    # The same traversal through a Router child path ("0/../evil") must also be normalized and
    # blocked, not skipped for containing "..".
    rtrav = tmp_path / "rtrav" / "aaa"
    (rtrav / "evil").mkdir(parents = True)
    (rtrav / "modules.json").write_text(
        '[{"idx": 0, "name": "0", "path": "", "type": "sentence_transformers.models.Router"}]'
    )
    (rtrav / "router_config.json").write_text(
        '{"types": {"0/../evil": "sentence_transformers.models.WordEmbeddings"}}'
    )
    (rtrav / "evil" / "wordembedding_config.json").write_bytes(b"{}")
    (rtrav / "evil" / "pytorch_model.bin").write_bytes(b"\0")
    assert _evaluate(rtrav) is True


def test_security_scan_runs_when_online(monkeypatch):
    import utils.security.file_security as fs

    calls: list = []
    _fake_hub(monkeypatch, calls)
    assert fs._fetch_security_status("org/model", None) == (None, None)
    assert calls, "online must attempt the Hub"


def _read_backend(rel: str) -> str:
    return (Path(__file__).resolve().parents[1] / rel).read_text(encoding = "utf-8")


def test_embedding_loader_forces_local_only_when_offline():
    """The invariant the RAG opt-in rests on: embeddings.py may pass local_only_load
    because its loader is pinned to the local cache by the SAME value, read ONCE and shared
    (two hf_env_offline() reads can disagree since _hf_offline_if_dns_dead() flips the vars).
    Checked at source level because importing the loader drags in sentence_transformers.
    """
    src = _read_backend("core/rag/embeddings.py")
    assert (
        "local_only = hf_env_offline()" in src
    ), "core/rag/embeddings.py must capture the offline state once in _get()"
    assert (
        "_guard_model_security(load_name, local_only)" in src
    ), "the security guard must receive the captured value, not re-read the env"
    assert "local_files_only = local_only" in src, (
        "SentenceTransformer must be pinned with the SAME captured value; a second "
        "hf_env_offline() read can flip to False and fetch the unscanned repo"
    )
    guard = src.split("def _guard_model_security", 1)[1].split("\ndef ", 1)[0]
    assert "hf_env_offline()" not in guard, (
        "_guard_model_security must take local_only_load as an argument so it "
        "cannot observe a different offline state than the loader"
    )


def test_only_the_rag_embedding_path_opts_into_the_bypass():
    """No other loader may claim local-only without constraining its loader: the
    MLX/inference, training and export gates call from_pretrained without a local-only
    argument, so passing local_only_load there would disable the gate for a fetching path.
    """
    allowed = {"core/rag/embeddings.py", "routes/settings.py"}
    callers = [
        "core/inference/worker.py",
        "core/training/worker.py",
        "core/export/worker.py",
        "routes/models.py",
        "routes/inference.py",
    ]
    for rel in callers:
        assert rel not in allowed
        assert "local_only_load" not in _read_backend(rel), (
            f"{rel} passes local_only_load but does not pin its loader to the "
            "local cache; that would disable the malware gate for a fetching path"
        )


# ── FIX 1: the safetensors CREDIT is case-SENSITIVE and only credits names the loader loads ──
#
# The offline gate skips a pickle only when a safetensors weight the loader reads INSTEAD is
# present. transformers (SAFE_WEIGHTS_NAME) and sentence-transformers (Module.load_torch_weights)
# both look up the EXACT name ``model.safetensors`` (then its index); on a case-sensitive
# filesystem a mixed-case decoy is NOT that file, so the loader falls back to the pickle. Folding
# the decoy's name into the credit (or crediting a name the loader never loads) fails OPEN.


def _snap_dir(tmp_path, name: str, files: dict):
    d = tmp_path / name / "aaa"
    d.mkdir(parents = True)
    for fname, body in files.items():
        (d / fname).write_bytes(body if isinstance(body, bytes) else body.encode())
    return d


def _offline_blocked(monkeypatch, snap) -> bool:
    """Run the offline (local-only) gate against a prebuilt cache snapshot, asserting it never
    reaches the Hub, and return whether the load is blocked."""
    import utils.models.model_config as mc
    import utils.security.file_security as fs

    calls: list = []
    _fake_hub(monkeypatch, calls)
    monkeypatch.setattr(mc, "_active_snapshot_dir", lambda name: snap)
    decision = fs.evaluate_file_security("org/model", None, local_only_load = True)
    assert calls == [], "a local-only load must not hit the Hub"
    return decision.blocked


def test_offline_mixed_case_base_safetensors_does_not_credit_pickle(monkeypatch, tmp_path):
    # ``Model.SafeTensors`` is not the exact ``model.safetensors`` the loader reads, so it cannot
    # vouch for a live ``pytorch_model.bin`` -- the loader deserializes the pickle -> BLOCKED.
    snap = _snap_dir(
        tmp_path, "mixedbase", {"Model.SafeTensors": b"\0", "pytorch_model.bin": b"\0"}
    )
    assert _offline_blocked(monkeypatch, snap) is True


def test_offline_exact_case_base_safetensors_credits_pickle(monkeypatch, tmp_path):
    # Control: the exact ``model.safetensors`` is what the loader reads instead of the pickle.
    snap = _snap_dir(
        tmp_path, "exactbase", {"model.safetensors": b"\0", "pytorch_model.bin": b"\0"}
    )
    assert _offline_blocked(monkeypatch, snap) is False


def test_offline_pytorch_model_safetensors_decoy_does_not_credit(monkeypatch, tmp_path):
    # transformers/ST never look up ``pytorch_model.safetensors``; an inert file of that name
    # must not vouch for the live ``pytorch_model.bin`` -> BLOCKED.
    snap = _snap_dir(
        tmp_path, "decoy", {"pytorch_model.safetensors": b"\0", "pytorch_model.bin": b"\0"}
    )
    assert _offline_blocked(monkeypatch, snap) is True


def test_offline_mixed_case_adapter_safetensors_does_not_credit(monkeypatch, tmp_path):
    # ``Adapter_Model.SafeTensors`` is not the exact ``adapter_model.safetensors`` PEFT loads,
    # so the live ``adapter_model.bin`` stays a separate RCE vector -> BLOCKED (an inert
    # safetensors base does not cover the adapter pickle).
    snap = _snap_dir(
        tmp_path,
        "mixedadapter",
        {
            "config.json": b"{}",
            "model.safetensors": b"\0",
            "adapter_config.json": b"{}",
            "adapter_model.bin": b"\0",
            "Adapter_Model.SafeTensors": b"\0",
        },
    )
    assert _offline_blocked(monkeypatch, snap) is True


def test_offline_exact_case_adapter_safetensors_credits(monkeypatch, tmp_path):
    # Control: the exact ``adapter_model.safetensors`` covers the adapter pickle -> allowed.
    snap = _snap_dir(
        tmp_path,
        "exactadapter",
        {
            "config.json": b"{}",
            "model.safetensors": b"\0",
            "adapter_config.json": b"{}",
            "adapter_model.bin": b"\0",
            "adapter_model.safetensors": b"\0",
        },
    )
    assert _offline_blocked(monkeypatch, snap) is False


# ── FIX 2: the ONLINE scan canonicalizes repo-controlled ``..`` paths, matching the offline gate ──
#
# The offline gate resolves a traversing declared path / weight_map shard (``0/../evil`` ->
# ``evil``) so a pickle the loader deserializes there is blocked. The online scan compares the
# Hub's CANONICAL flagged path against repo-controlled load-subdirs (a ``modules.json`` module
# ``path`` the RAG guard unions in) and ``weight_map`` shards; without collapsing ``..`` the raw
# ``0/../evil`` never matches ``evil/...`` and the flagged pickle slips through.


def test_online_load_subdir_prefixes_are_canonicalized():
    import utils.security.file_security as fs

    assert fs._index_prefixes(("0/../evil",)) == ("", "evil/")
    assert fs._load_relative_path("evil/pytorch_model.bin", ("0/../evil",)) == "pytorch_model.bin"
    # A non-traversing subdir is unchanged.
    assert fs._load_relative_path("LLM/pytorch_model.bin", ("LLM",)) == "pytorch_model.bin"
    # An escaping subdir (leading ..) is dropped -- it can never name a repo file.
    assert fs._load_relative_path("evil/x.bin", ("../evil",)) == "evil/x.bin"


def test_online_scan_blocks_flagged_pickle_under_traversing_module_subdir(monkeypatch):
    # A repo-controlled ``modules.json`` path ``0/../evil`` reaches the online scan (unioned in
    # by the RAG guard). The loader resolves it to ``evil/`` and deserializes
    # ``evil/pytorch_model.bin``; the gate must scope the canonical dir and block it root-level.
    import utils.security.file_security as fs

    monkeypatch.setattr(
        fs,
        "_fetch_security_status",
        lambda name, token: (
            {"filesWithIssues": [{"path": "evil/pytorch_model.bin", "level": "unsafe"}]},
            "commitsha",
        ),
    )
    # Definitive "no weight index" -> a missed shard is SKIPPED (allowed) on the pristine gate;
    # the fix instead treats evil/ as a load root so the pickle is root-level and blocks.
    monkeypatch.setattr(fs, "_indexed_shard_paths", lambda *a, **k: set())
    decision = fs.evaluate_file_security(
        "org/model", None, load_subdirs = ("0/../evil",), local_only_load = False
    )
    assert decision.blocked is True


def test_online_scan_blocks_flagged_pickle_under_plain_module_subdir(monkeypatch):
    # Control: a non-traversing declared subdir still blocks a flagged pickle directly under it.
    import utils.security.file_security as fs

    monkeypatch.setattr(
        fs,
        "_fetch_security_status",
        lambda name, token: (
            {"filesWithIssues": [{"path": "0_Transformer/pytorch_model.bin", "level": "unsafe"}]},
            "commitsha",
        ),
    )
    monkeypatch.setattr(fs, "_indexed_shard_paths", lambda *a, **k: set())
    decision = fs.evaluate_file_security(
        "org/model", None, load_subdirs = ("0_Transformer",), local_only_load = False
    )
    assert decision.blocked is True


def test_online_indexed_shard_paths_canonicalize_weight_map_traversal(monkeypatch, tmp_path):
    # A repo-controlled ``weight_map`` shard that traverses (``sub/../evil/pytorch_model.bin``)
    # must be recorded at the canonical ``evil/pytorch_model.bin`` the Hub reports for the
    # flagged file, so the online maybe-shard check matches it and blocks the load.
    import utils.security.file_security as fs

    fake = types.ModuleType("huggingface_hub")
    fake_utils = types.ModuleType("huggingface_hub.utils")

    class _EntryNotFound(Exception):
        pass

    fake_utils.EntryNotFoundError = _EntryNotFound
    fake.utils = fake_utils

    index_file = tmp_path / "pytorch_model.bin.index.json"
    index_file.write_text('{"weight_map": {"w": "sub/../evil/pytorch_model.bin"}}')

    def _download(
        model_name,
        filename,
        token = None,
    ):
        if filename == "pytorch_model.bin.index.json":
            return str(index_file)
        raise _EntryNotFound()

    fake.hf_hub_download = _download
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake)
    monkeypatch.setitem(sys.modules, "huggingface_hub.utils", fake_utils)

    paths = fs._indexed_shard_paths("org/model", None)
    assert "evil/pytorch_model.bin" in paths
    assert "sub/../evil/pytorch_model.bin" not in paths
