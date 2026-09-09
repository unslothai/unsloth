# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""`spark serve` has to launch the layout the planner recommends, and size the model it will load.

Three separate wrong answers met in this command. The fit class `single-or-replicas` means one
copy fits each Spark and a second does not fit BESIDE it on the same node; read as a split, it
forced the 0.92x cross-node layout onto the size range where two replicas measured 1.30x to
1.91x, and said "two copies do not fit across the pair" while one fits on each of two nodes.
The RPC server was required before the topology was known, so a replica deployment that never
speaks RPC was refused on a bundle without it. And the size came from summing every weight file
in sight, so a directory of alternative quantizations was planned as their total.
"""

from __future__ import annotations

import importlib.util
import struct
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]


def _cluster():
    spec = importlib.util.spec_from_file_location(
        "spark_cluster_for_serve", REPO / "studio" / "spark_cluster.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def cluster():
    return _cluster()


def _write(path: Path, size: int) -> Path:
    path.parent.mkdir(parents = True, exist_ok = True)
    with open(path, "wb") as handle:
        handle.truncate(size)
    return path


# ── sizing ──────────────────────────────────────────────────────────────────────

GIB = 2**30


def test_alternative_quantizations_are_refused_not_summed(cluster, tmp_path) -> None:
    """The catch: three 40 GiB quants of a model that fits each Spark read as 120 GiB."""
    _write(tmp_path / "m-Q4_K_M.gguf", 4096)
    _write(tmp_path / "m-Q8_0.gguf", 8192)
    report = cluster.model_size_report(str(tmp_path))
    assert report["gib"] is None
    assert "alternative models" in report["why"], report


def test_a_sharded_gguf_is_one_model_and_is_summed(cluster, tmp_path) -> None:
    _write(tmp_path / "m-00001-of-00002.gguf", 4096)
    _write(tmp_path / "m-00002-of-00002.gguf", 4096)
    assert cluster.model_size_report(str(tmp_path))["gib"] == pytest.approx(8192 / GIB)


def test_the_legacy_bin_beside_safetensors_is_not_counted_twice(cluster, tmp_path) -> None:
    """A repo that ships both formats holds one model written twice; transformers reads one."""
    _write(tmp_path / "model.safetensors", 4096)
    _write(tmp_path / "pytorch_model.bin", 4096)
    assert cluster.model_size_report(str(tmp_path))["gib"] == pytest.approx(4096 / GIB)


def test_only_the_current_hf_snapshot_is_sized(cluster, tmp_path, monkeypatch) -> None:
    """An updated repo keeps its superseded blobs; summing the tree counted them as weights."""
    monkeypatch.setenv("HOME", str(tmp_path))
    repo = tmp_path / ".cache" / "huggingface" / "hub" / "models--org--m"
    (repo / "refs").mkdir(parents = True)
    (repo / "refs" / "main").write_text("newsha", encoding = "utf-8")
    _write(repo / "snapshots" / "oldsha" / "model.safetensors", 8192)
    _write(repo / "snapshots" / "newsha" / "model.safetensors", 4096)
    assert cluster.model_size_report("org/m")["gib"] == pytest.approx(4096 / GIB)


def test_a_single_file_is_still_its_own_size(cluster, tmp_path) -> None:
    """No regression: naming one file is the unambiguous case and must stay exact."""
    one = _write(tmp_path / "m.gguf", 4096)
    assert cluster.model_size_gib(str(one)) == pytest.approx(4096 / GIB)


# ── GGUF header and KV size ─────────────────────────────────────────────────────


def _kv(key: str, value_type: int, payload: bytes) -> bytes:
    return struct.pack("<Q", len(key)) + key.encode() + struct.pack("<I", value_type) + payload


def _u32(value: int) -> bytes:
    return struct.pack("<I", value)


def _gguf(path: Path, pairs) -> Path:
    body = b"".join(pairs)
    with open(path, "wb") as handle:
        handle.write(struct.pack("<4sIQQ", b"GGUF", 3, 0, len(pairs)))
        handle.write(body)
    return path


def test_kv_per_token_comes_from_the_header(cluster, tmp_path) -> None:
    model = _gguf(
        tmp_path / "m.gguf",
        [
            _kv("general.architecture", 8, struct.pack("<Q", 5) + b"llama"),
            _kv("llama.block_count", 4, _u32(40)),
            _kv("llama.attention.head_count_kv", 4, _u32(8)),
            _kv("llama.attention.key_length", 4, _u32(128)),
            _kv("llama.attention.value_length", 4, _u32(128)),
        ],
    )
    meta = cluster.gguf_metadata(str(model))
    assert meta["general.architecture"] == "llama"
    # 40 layers * 8 kv heads * (128 + 128) * 2 bytes.
    assert cluster.gguf_kv_bytes_per_token(meta) == 40 * 8 * 256 * 2


def test_key_length_is_derived_when_absent(cluster, tmp_path) -> None:
    model = _gguf(
        tmp_path / "m.gguf",
        [
            _kv("general.architecture", 8, struct.pack("<Q", 5) + b"llama"),
            _kv("llama.block_count", 4, _u32(2)),
            _kv("llama.attention.head_count_kv", 4, _u32(4)),
            _kv("llama.attention.head_count", 4, _u32(8)),
            _kv("llama.embedding_length", 4, _u32(1024)),
        ],
    )
    assert cluster.gguf_kv_bytes_per_token(cluster.gguf_metadata(str(model))) == 2 * 4 * 256 * 2


def test_a_header_that_does_not_say_returns_none(cluster, tmp_path) -> None:
    """None, never a default: the number decides whether a context fits."""
    model = _gguf(
        tmp_path / "m.gguf", [_kv("general.architecture", 8, struct.pack("<Q", 5) + b"llama")]
    )
    assert cluster.gguf_kv_bytes_per_token(cluster.gguf_metadata(str(model))) is None
    assert cluster.gguf_metadata(str(_write(tmp_path / "not.gguf", 64))) == {}


def test_kv_per_user_scales_with_the_context(cluster, tmp_path) -> None:
    model = _gguf(
        tmp_path / "m.gguf",
        [
            _kv("general.architecture", 8, struct.pack("<Q", 5) + b"llama"),
            _kv("llama.block_count", 4, _u32(40)),
            _kv("llama.attention.head_count_kv", 4, _u32(8)),
            _kv("llama.attention.key_length", 4, _u32(128)),
            _kv("llama.attention.value_length", 4, _u32(128)),
        ],
    )
    per_user = cluster.serving_kv_gib_per_user(str(model), 8192)["gib"]
    assert per_user == pytest.approx(40 * 8 * 256 * 2 * 8192 / GIB)


# ── which layout `serve` prints ─────────────────────────────────────────────────


def _serve_fixture(
    cluster,
    monkeypatch,
    tmp_path,
    size_gib: float,
    *,
    rpc: bool = True,
):
    monkeypatch.setattr(cluster, "is_dgx_spark", lambda: True)
    monkeypatch.setattr(cluster, "peer_ip_for", lambda *a, **k: "192.168.200.13")
    server = _write(tmp_path / "bin" / "llama-server", 16)
    monkeypatch.setattr(cluster, "llama_server_binary", lambda: str(server))
    monkeypatch.setattr(cluster, "model_size_report", lambda target: {"gib": size_gib, "why": ""})
    monkeypatch.setattr(
        cluster, "serving_kv_gib_per_user", lambda model, ctx: {"gib": None, "why": "no header"}
    )
    rpc_binary = str(_write(tmp_path / "bin" / "ggml-rpc-server", 16))
    monkeypatch.setattr(
        cluster,
        "rpc_cluster_plan",
        lambda port = 0: {
            "ok": rpc,
            "problems": [] if rpc else ["no ggml-rpc-server binary"],
            "rpc_server": rpc_binary if rpc else None,
            "peer_ip": "192.168.200.13",
        },
    )
    monkeypatch.setattr(
        cluster,
        "rpc_protocol_preflight",
        lambda peer, port: {"problems": [], "notes": [], "peer": {"rpc_server": "/peer/rpc"}},
    )


def test_one_copy_per_node_is_served_as_replicas(cluster, monkeypatch, tmp_path, capsys) -> None:
    """80 GiB fits one Spark and not twice on one Spark: two nodes, one copy each."""
    _serve_fixture(cluster, monkeypatch, tmp_path, 80.0)
    assert cluster.plan_deployment(80.0, two_sparks = True)["topology"] == "single-or-replicas"
    assert cluster._cmd_serve("m.gguf", slots = 16) == 0
    out = capsys.readouterr().out
    assert "INDEPENDENT REPLICAS" in out, out
    assert "two copies do not fit" not in out
    assert "--rpc" not in out


def test_replicas_do_not_require_the_rpc_server(cluster, monkeypatch, tmp_path, capsys) -> None:
    """The bundle without ggml-rpc-server refused a deployment that never speaks RPC."""
    _serve_fixture(cluster, monkeypatch, tmp_path, 80.0, rpc = False)
    assert cluster._cmd_serve("m.gguf", slots = 16) == 0
    assert "INDEPENDENT REPLICAS" in capsys.readouterr().out


def test_a_model_that_does_not_fit_still_layer_splits(
    cluster, monkeypatch, tmp_path, capsys
) -> None:
    """No regression: the split is the only way to run an oversized model."""
    _serve_fixture(cluster, monkeypatch, tmp_path, 150.0)
    assert cluster._cmd_serve("m.gguf", slots = 16) == 0
    out = capsys.readouterr().out
    assert "layer-split" in out
    assert "/peer/rpc" in out, "the peer's own path, not ours mapped onto it"


def test_an_oversized_model_without_the_rpc_server_is_refused(
    cluster, monkeypatch, tmp_path, capsys
) -> None:
    _serve_fixture(cluster, monkeypatch, tmp_path, 150.0, rpc = False)
    assert cluster._cmd_serve("m.gguf", slots = 16) == 1
    assert "cannot layer-split" in capsys.readouterr().out


def test_a_model_too_large_for_the_pair_is_refused(cluster, monkeypatch, tmp_path, capsys) -> None:
    _serve_fixture(cluster, monkeypatch, tmp_path, 400.0)
    assert cluster._cmd_serve("m.gguf", slots = 16) == 1


def test_an_unsizable_model_is_refused_rather_than_split(
    cluster, monkeypatch, tmp_path, capsys
) -> None:
    """Falling through printed a two-engine split recipe for a model of unknown size."""
    _serve_fixture(cluster, monkeypatch, tmp_path, 80.0)
    monkeypatch.setattr(
        cluster, "model_size_report", lambda target: {"gib": None, "why": "two of them"}
    )
    assert cluster._cmd_serve("m.gguf", slots = 16) == 1
    out = capsys.readouterr().out
    assert "two of them" in out
    assert "llama-server" not in out


def test_kv_that_does_not_fit_sends_a_fitting_model_to_a_split(
    cluster, monkeypatch, tmp_path, capsys
) -> None:
    """The weights fit; the KV for 16 users at this context does not, and only the split has
    the room. Sizing on weights alone printed a replica pair that dies at load."""
    _serve_fixture(cluster, monkeypatch, tmp_path, 80.0)
    monkeypatch.setattr(
        cluster, "serving_kv_gib_per_user", lambda model, ctx: {"gib": 4.0, "why": ""}
    )
    assert cluster._cmd_serve("m.gguf", slots = 16) == 0
    assert "layer-split" in capsys.readouterr().out


# ── discovery, state and the GPU verdict ────────────────────────────────────────


def test_the_configured_node_count_survives_a_switched_setup(cluster, monkeypatch) -> None:
    """`setup --nodes 3 --switched` saves rail plans and no `peers` key, so discovery with
    mDNS off came back with two nodes and planned for two."""
    monkeypatch.setattr(
        cluster,
        "load_config",
        lambda: {
            "enabled": True,
            "n_nodes": 3,
            "peer_rails": [{"address": "192.168.200.13"}, {"address": "192.168.201.13"}],
            "other_rails": [[{"address": "192.168.200.14"}, {"address": "192.168.201.14"}]],
        },
    )
    peers = cluster.configured_peers()
    assert [p["address"] for p in peers] == ["192.168.200.13", "192.168.200.14"]
    merged = cluster.merge_peers([])
    assert [p["address"] for p in merged] == ["192.168.200.13", "192.168.200.14"]
    assert [p["index"] for p in merged] == [1, 2]


def test_an_explicit_peers_list_still_wins(cluster, monkeypatch) -> None:
    monkeypatch.setattr(
        cluster,
        "load_config",
        lambda: {
            "enabled": True,
            "peers": [{"hostname": "spark-b", "address": "10.0.0.2"}],
            "peer_rails": [{"address": "192.168.200.13"}],
        },
    )
    assert [p["address"] for p in cluster.configured_peers()] == ["10.0.0.2"]


def test_the_same_host_from_two_sources_is_one_node(cluster, monkeypatch) -> None:
    """Counting a peer twice would size the cluster for a Spark that is not there."""
    monkeypatch.setattr(
        cluster,
        "load_config",
        lambda: {"enabled": True, "peer_rails": [{"address": "192.168.200.13"}]},
    )
    merged = cluster.merge_peers(
        [{"hostname": "spark-b.local", "address": "192.168.200.13", "source": "mdns"}]
    )
    assert len(merged) == 1
    assert merged[0]["hostname"] == "spark-b.local"


def test_a_half_applied_netplan_is_not_configured(cluster, monkeypatch) -> None:
    """NCCL is handed both PCIe functions, so one addressed rail is not a configured pair."""
    monkeypatch.setattr(cluster, "is_dgx_spark", lambda: True)
    monkeypatch.setattr(cluster, "load_config", lambda: {"enabled": True})
    monkeypatch.setattr(
        cluster,
        "cabled_rails",
        lambda: [
            {"ib_device": "a", "netdev": "n0", "ipv4": ["192.168.200.12"], "mtu": 9000},
            {"ib_device": "b", "netdev": "n1", "ipv4": [], "mtu": 9000},
        ],
    )
    assert cluster.cluster_state() == "unconfigured"


def test_both_rails_addressed_is_configured(cluster, monkeypatch) -> None:
    monkeypatch.setattr(cluster, "is_dgx_spark", lambda: True)
    monkeypatch.setattr(cluster, "load_config", lambda: {"enabled": True})
    monkeypatch.setattr(
        cluster,
        "cabled_rails",
        lambda: [
            {"ib_device": "a", "netdev": "n0", "ipv4": ["192.168.200.12"], "mtu": 9000},
            {"ib_device": "b", "netdev": "n1", "ipv4": ["192.168.201.12"], "mtu": 9000},
        ],
    )
    assert cluster.cluster_state() == "configured"


def test_plan_drift_names_the_rail_and_what_differs(cluster, monkeypatch) -> None:
    monkeypatch.setattr(
        cluster,
        "load_config",
        lambda: {
            "enabled": True,
            "rails": [
                {"netdev": "n0", "address": "192.168.200.12", "mtu": "9000"},
                {"netdev": "n1", "address": "192.168.201.12", "mtu": "9000"},
            ],
        },
    )
    monkeypatch.setattr(
        cluster,
        "cabled_rails",
        lambda: [
            {"ib_device": "a", "netdev": "n0", "ipv4": ["192.168.200.99"], "mtu": 9000},
            {"ib_device": "b", "netdev": "n1", "ipv4": ["192.168.201.12"], "mtu": 1500},
        ],
    )
    problems = " | ".join(cluster.cluster_config_problems())
    assert "192.168.200.99" in problems and "MTU 1500" in problems, problems


def test_only_cuinit_100_is_the_reboot_fault(cluster) -> None:
    """`dead-engine` is a reboot-only diagnosis, and 100 is the code it was written for. A
    driver mismatch, and the synthesised -1 for an unloadable libcuda, are not that."""
    classify = cluster.classify_cuda_state
    assert classify(0, True) == "ok"
    assert classify(100, True) == "dead-engine"
    assert classify(803, True) == "cuda-error"
    assert classify(-1, True) == "unknown"
    assert classify(100, False) == "unknown"


def test_an_unrelated_local_listener_does_not_block_a_split(cluster, monkeypatch) -> None:
    """The split runs ggml-rpc-server on the PEER; nothing local binds that port."""
    monkeypatch.setattr(
        cluster, "compare_llama_bundles", lambda a, b: {"ok": True, "problems": [], "notes": []}
    )
    monkeypatch.setattr(cluster, "llama_bundle_identity", lambda: {"present": True})
    monkeypatch.setattr(cluster, "peer_llama_bundle_identity", lambda ip: {"present": True})

    def probe(
        host,
        port,
        timeout = 2.0,
    ):
        state = "garbled" if host == "127.0.0.1" else "ok"
        version = None if state == "garbled" else (6, 0, 0)
        return {"state": state, "version": version, "host": host}

    monkeypatch.setattr(cluster, "rpc_hello_probe_detail", probe)
    result = cluster.rpc_protocol_preflight("192.168.200.13", 50052)
    assert result["problems"] == [], result
    assert any("127.0.0.1" in note for note in result["notes"])


def test_a_bad_peer_listener_still_blocks(cluster, monkeypatch) -> None:
    """No regression: the peer's port is the one the deployment uses."""
    monkeypatch.setattr(
        cluster, "compare_llama_bundles", lambda a, b: {"ok": True, "problems": [], "notes": []}
    )
    monkeypatch.setattr(cluster, "llama_bundle_identity", lambda: {"present": True})
    monkeypatch.setattr(cluster, "peer_llama_bundle_identity", lambda ip: {"present": True})
    monkeypatch.setattr(
        cluster,
        "rpc_hello_probe_detail",
        lambda host, port, timeout = 2.0: {
            "state": "closed" if host != "127.0.0.1" else "refused",
            "version": None,
            "host": host,
        },
    )
    assert cluster.rpc_protocol_preflight("192.168.200.13", 50052)["problems"]


def test_a_split_is_refused_when_the_kv_does_not_fit_across_the_pair(cluster) -> None:
    """The aggregate gate counted weights only, so a model whose KV pushes the pair over its
    total was still planned as a layer split and `serve` printed an RPC launch that OOMs at
    load. Context is never divided: every one of the requested users gets the full context, so
    the KV that follows is priced at full and refused when it does not fit."""
    budget = cluster.SPARK_USABLE_GIB - cluster.SERVE_OVERHEAD_GIB
    weights = budget * 1.5  # does not fit one node, fits the pair on weights alone
    plan = cluster.plan_deployment(weights, two_sparks = True, concurrency = 1)
    assert plan["topology"] == "layer-split" and plan["fits"] is True

    per_user = (budget * 2 - weights) / 8 + 1.0
    tight = cluster.plan_deployment(
        weights, two_sparks = True, concurrency = 8, kv_gib_per_user = per_user
    )
    assert tight["topology"] == "too-large" and tight["fits"] is False
    # The refusal has to say which of the two it was.
    assert "KV for 8" in tight["summary"]
    assert tight["split_need_gib"] > 2 * budget

    # A single node is unaffected: it returns before the aggregate gate and keeps its
    # weight-only class, so nothing changes for one Spark.
    solo = cluster.plan_deployment(
        weights, two_sparks = False, concurrency = 8, kv_gib_per_user = per_user
    )
    assert solo["topology"] == "single"
