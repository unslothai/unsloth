# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import asyncio
import hashlib
import io
import json
import time
import zipfile

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from auth.authentication import get_current_subject
from core.flows import engine as engine_mod
from routes import flows as flows_routes
from utils.account_context import AccountContext, OWNER, bind_account, reset_account

from ._flows_helpers import doc, edge, node
from .test_flows_package import PNG, PNG_SHA

ALICE = AccountContext("alice-flows", "alice")
BOB = AccountContext("bob-flows", "bob")
ACCOUNTS = {"owner": OWNER, "alice": ALICE, "bob": BOB}


@pytest.fixture
def client(monkeypatch):
    eng = engine_mod.FlowEngine()
    eng.mint_key = lambda account: ("sk-unsloth-test", None)
    eng.revoke_key = lambda key_id: None

    async def text(ctx):
        return {"text": str(ctx.params.get("value") or "")}

    async def result(ctx):
        return {"value": ctx.inputs.get("value")}

    eng.handlers = {"input.text": text, "output.result": result}
    monkeypatch.setattr(engine_mod, "ENGINE", eng)

    app = FastAPI()

    @app.middleware("http")
    async def as_account(request: Request, call_next):
        token = bind_account(ACCOUNTS[request.headers.get("x-test-account", "owner")])
        try:
            return await call_next(request)
        finally:
            reset_account(token)

    app.include_router(flows_routes.router, prefix = "/api/flows")
    app.dependency_overrides[get_current_subject] = lambda: "tester"
    with TestClient(app) as test_client:
        yield test_client


def _simple(flow_id = "flow1") -> dict:
    return doc(
        [node("a", "input.text", value = "hello"), node("o", "output.result", name = "greeting")],
        [edge("a", "text", "o", "value")],
        id = flow_id,
    )


def test_catalog_lists_every_spec(client):
    body = client.get("/api/flows/nodes").json()
    assert {n["type"] for n in body["nodes"]} >= {
        "llm.generate",
        "image.generate",
        "story.scene",
        "world.npc",
    }


def test_crud_conflict_and_tombstone(client):
    created = client.post("/api/flows", json = {"doc": _simple()}).json()
    assert created["id"] == "flow1"
    assert client.get("/api/flows").json()["flows"][0]["nodeCount"] == 2
    renamed = {**created["doc"], "name": "Renamed"}
    saved = client.put(
        "/api/flows/flow1", json = {"doc": renamed, "baseUpdatedAt": created["updatedAt"]}
    )
    assert saved.status_code == 200 and saved.json()["name"] == "Renamed"
    stale = client.put(
        "/api/flows/flow1", json = {"doc": renamed, "baseUpdatedAt": created["updatedAt"]}
    )
    assert stale.status_code == 409
    assert client.put("/api/flows/other", json = {"doc": renamed}).status_code == 400
    assert client.delete("/api/flows/flow1").status_code == 204
    assert client.get("/api/flows/flow1").status_code == 404
    assert client.put("/api/flows/flow1", json = {"doc": renamed}).status_code == 410
    assert client.post("/api/flows", json = {"doc": _simple()}).json()["id"] != "flow1"


def test_invalid_documents_are_refused(client):
    bad = client.post("/api/flows", json = {"doc": {**_simple(), "exec": "x"}})
    assert bad.status_code == 400
    result = client.post("/api/flows/validate", json = {"doc": doc([node("s", "text.split")])}).json()
    assert result["ok"] is False and result["errors"][0]["node_id"] == "s"


def _wait_terminal(
    client,
    run_id: str,
    headers = None,
) -> dict:
    for _ in range(100):
        snap = client.get(f"/api/flows/runs/{run_id}", headers = headers).json()
        if snap["run"]["status"] in ("completed", "failed", "cancelled"):
            return snap
        time.sleep(0.05)
    raise AssertionError("run did not finish")


def _events(text: str) -> list[dict]:
    return [json.loads(line[5:]) for line in text.splitlines() if line.startswith("data:")]


def test_run_snapshot_outputs_and_event_replay(client):
    client.post("/api/flows", json = {"doc": _simple()})
    run_id = client.post("/api/flows/flow1/runs", json = {}).json()["run_id"]
    snap = _wait_terminal(client, run_id)
    assert snap["run"]["outputs"] == {"greeting": "hello"}
    assert snap["nodes"]["a"]["status"] == "done"
    replay = _events(client.post(f"/api/flows/runs/{run_id}/events", params = {"after": 0}).text)
    assert replay[0]["seq"] == 1 and replay[-1]["type"] == "run.completed"
    tail = _events(
        client.get(
            f"/api/flows/runs/{run_id}/events", headers = {"Last-Event-ID": str(replay[-3]["seq"])}
        ).text
    )
    assert [e["seq"] for e in tail] == [e["seq"] for e in replay[-2:]]
    snapshot_only = _events(client.post(f"/api/flows/runs/{run_id}/events").text)
    assert snapshot_only[0]["type"] == "snapshot"
    runs = client.get("/api/flows/flow1/runs").json()["runs"]
    assert runs[0]["id"] == run_id and runs[0]["status"] == "completed"
    latest = client.get("/api/flows/flow1/latest-outputs").json()["nodes"]
    assert latest["a"]["outputs"] == {"text": "hello"}
    assert client.post("/api/flows/runs/nope/cancel").status_code == 404


def test_run_with_unsaved_editor_doc_and_compile_errors(client):
    client.post("/api/flows", json = {"doc": _simple()})
    edited = _simple()
    edited["nodes"][0]["params"]["value"] = "edited"
    run_id = client.post("/api/flows/flow1/runs", json = {"doc": edited}).json()["run_id"]
    assert _wait_terminal(client, run_id)["run"]["outputs"] == {"greeting": "edited"}
    broken = client.post(
        "/api/flows/flow1/runs", json = {"doc": doc([node("s", "text.split")], id = "flow1")}
    )
    assert broken.status_code == 400
    assert broken.json()["detail"]["errors"][0]["node_id"] == "s"


def test_flows_and_runs_are_per_account(client):
    alice = {"x-test-account": "alice"}
    bob = {"x-test-account": "bob"}
    client.post("/api/flows", json = {"doc": _simple("shared-id")}, headers = alice)
    assert client.get("/api/flows/shared-id", headers = bob).status_code == 404
    assert client.get("/api/flows", headers = bob).json()["flows"] == []
    run_id = client.post("/api/flows/shared-id/runs", json = {}, headers = alice).json()["run_id"]
    _wait_terminal(client, run_id, alice)
    assert client.get(f"/api/flows/runs/{run_id}", headers = bob).status_code == 404
    assert client.post(f"/api/flows/runs/{run_id}/events", headers = bob).status_code == 404
    assert client.post(f"/api/flows/runs/{run_id}/cancel", headers = bob).status_code == 404
    client.put("/api/flows/shared-id/sessions/slot1", json = {"state": {"scene": "a"}}, headers = alice)
    assert client.get("/api/flows/shared-id/sessions/slot1", headers = bob).status_code == 404


def test_sessions_round_trip(client):
    client.post("/api/flows", json = {"doc": _simple()})
    saved = client.put(
        "/api/flows/flow1/sessions/auto",
        json = {"state": {"scene": "s2", "vars": {"hp": 3}}, "label": "Cave"},
    )
    assert saved.status_code == 200
    assert client.get("/api/flows/flow1/sessions").json()["sessions"][0]["label"] == "Cave"
    assert client.get("/api/flows/flow1/sessions/auto").json()["state"]["vars"] == {"hp": 3}
    assert client.put("/api/flows/flow1/sessions/../x", json = {"state": {}}).status_code in (
        400,
        404,
    )
    assert client.delete("/api/flows/flow1/sessions/auto").status_code == 204
    assert client.get("/api/flows/flow1/sessions/auto").status_code == 404


def test_documents_upload_and_come_back_as_plain_text(client):
    pdf = client.post("/api/flows/assets?name=report.pdf", content = b"%PDF-1.4 tiny").json()["media"]
    assert (
        pdf["kind"] == "document"
        and pdf["mime"] == "application/pdf"
        and pdf["name"] == "report.pdf"
    )
    page = b"<script>alert(1)</script>"
    html = client.post("/api/flows/assets?name=page.html", content = page).json()["media"]
    served = client.get(html["url"])
    assert served.headers["content-type"].startswith("text/plain")
    assert served.headers["x-content-type-options"] == "nosniff"
    assert client.post("/api/flows/assets", content = b"plain words").status_code == 415


def test_import_export_assets_round_trip(client, monkeypatch):
    async def no_resident(request):
        return {}

    monkeypatch.setattr(flows_routes, "_resident", no_resident)
    uploaded = client.post("/api/flows/assets", content = PNG).json()["media"]
    assert uploaded["ref"] == f"sha256:{PNG_SHA}" and uploaded["kind"] == "image"
    assert client.get(f"/api/flows/assets/sha256:{PNG_SHA}").content == PNG
    assert client.get(f"/api/flows/assets/sha256:{'0' * 64}").status_code == 404
    script = b"#!/bin/sh"
    assert client.post("/api/flows/assets", content = script).status_code == 415
    blob = b'{"not": "media"}'
    assert client.post("/api/flows/assets?mime=application/json", content = blob).status_code == 415
    assert (
        client.get(f"/api/flows/assets/sha256:{hashlib.sha256(blob).hexdigest()}").status_code
        == 404
    )

    d = doc(
        [node("i", "input.image", asset = uploaded)], id = "withimg", baked = {"i": {"image": uploaded}}
    )
    client.post("/api/flows", json = {"doc": d})
    exported = client.get("/api/flows/withimg/export", params = {"mode": "baked"})
    assert exported.status_code == 200 and exported.content[:2] == b"PK"
    imported = client.post(
        "/api/flows/import", content = exported.content, params = {"name": "Copy"}
    ).json()
    assert imported["flow"]["id"] != "withimg" and imported["flow"]["name"] == "Copy"
    assert imported["report"]["baked_nodes"] == ["i"]
    assert imported["report"]["replication"] == [] and "live_note" not in imported["report"]
    world = doc(
        [node("r", "world.npc_reply"), node("i", "input.image", asset = uploaded)],
        id = "baked_world",
        baked = {"i": {"image": uploaded}},
    )
    client.post("/api/flows", json = {"doc": world})
    exported = client.get("/api/flows/baked_world/export", params = {"mode": "baked"})
    report = client.post("/api/flows/import", content = exported.content).json()["report"]
    assert "need a model" in report["live_note"]
    assert client.post("/api/flows/import", content = b"PK\x03\x04garbage").status_code == 400
    assert client.get("/api/flows/withimg/export", params = {"mode": "evil"}).status_code == 400


def test_examples_list_and_copy(client, monkeypatch):
    hello = {**_simple("hello"), "kind": "pipeline"}
    monkeypatch.setattr(flows_routes.examples, "example_ids", lambda: ["hello"])
    monkeypatch.setattr(flows_routes.examples, "example_docs", lambda: [hello])
    flows_routes._example_summaries.cache_clear()
    monkeypatch.setattr(
        flows_routes, "_example_summaries", flows_routes._example_summaries.__wrapped__
    )
    monkeypatch.setattr(
        flows_routes.examples,
        "example",
        lambda i: json.loads(json.dumps(hello)) if i == "hello" else None,
    )
    examples = client.get("/api/flows/examples").json()["examples"]
    assert [e["id"] for e in examples] == ["hello"] and examples[0]["example"] is True
    copied = client.post("/api/flows/examples/hello/copy").json()
    assert copied["id"] != "hello"
    assert copied["doc"]["meta"]["derived_from"][0]["id"] == "hello"
    assert client.post("/api/flows/examples/missing/copy").status_code == 404


def test_bundled_examples_are_listed_and_copy(client):
    listed = client.get("/api/flows/examples").json()["examples"]
    assert len(listed) == 56 and "lantern-picnic" in {e["id"] for e in listed}
    copied = client.post("/api/flows/examples/lantern-picnic/copy").json()
    assert copied["doc"]["kind"] == "story"
    needs = {e["id"]: e["needs"] for e in listed}
    assert needs["lantern-picnic"] == []
    assert needs["comic-strip"] == ["chat", "image"]
    assert needs["notebook-text-sft"] == ["training"]
    assert needs["dataset-forge"] == ["chat", "training"]


def test_invoke_entrypoint_errors(client):
    client.post("/api/flows", json = {"doc": _simple()})
    refused = client.post("/api/flows/flow1/invoke", json = {"entry": "a"})
    assert refused.status_code == 422 and "entry point" in refused.json()["detail"]


def test_asset_ids_are_content_hashes():
    assert hashlib.sha256(PNG).hexdigest() == PNG_SHA


def test_event_replay_pages_past_one_stored_page(client, monkeypatch):
    client.post("/api/flows", json = {"doc": _simple()})
    run_id = client.post("/api/flows/flow1/runs", json = {}).json()["run_id"]
    _wait_terminal(client, run_id)
    monkeypatch.setattr(flows_routes, "REPLAY_PAGE", 2)
    real = flows_routes.flows_db.events_after
    monkeypatch.setattr(
        flows_routes.flows_db, "events_after", lambda rid, after: real(rid, after, limit = 2)
    )
    replay = _events(client.post(f"/api/flows/runs/{run_id}/events", params = {"after": 0}).text)
    assert [e["seq"] for e in replay] == list(range(1, len(replay) + 1))
    assert replay[-1]["type"] == "run.completed" and len(replay) > 2


def test_deleting_a_flow_stops_its_run(client):
    async def slow(ctx):
        for _ in range(3000):
            await asyncio.sleep(0.01)
            ctx.check_cancelled()
        return {"text": "late"}

    engine_mod.ENGINE.handlers["input.text"] = slow
    client.post("/api/flows", json = {"doc": _simple()})
    run_id = client.post("/api/flows/flow1/runs", json = {}).json()["run_id"]
    assert client.delete("/api/flows/flow1", headers = {"x-test-account": "bob"}).status_code == 204
    assert client.get(f"/api/flows/runs/{run_id}").json()["run"]["status"] in ("queued", "running")
    assert client.delete("/api/flows/flow1").status_code == 204
    assert _wait_terminal(client, run_id)["run"]["status"] == "cancelled"


def test_deleting_a_flow_removes_its_run_history(client):
    client.post("/api/flows", json = {"doc": _simple()})
    run_id = client.post("/api/flows/flow1/runs", json = {}).json()["run_id"]
    _wait_terminal(client, run_id)
    assert client.get("/api/flows/flow1/runs").json()["runs"]
    assert client.delete("/api/flows/flow1").status_code == 204
    from storage import flows_db

    assert flows_db.get_run(run_id) is None
    assert flows_db.events_after(run_id, 0) == []


def test_reconnecting_to_a_run_a_restart_interrupted_ends(client):
    from storage import flows_db

    client.post("/api/flows", json = {"doc": _simple()})
    run_id = client.post("/api/flows/flow1/runs", json = {}).json()["run_id"]
    snap = _wait_terminal(client, run_id)
    engine_mod.ENGINE.runs.clear()
    token = bind_account(OWNER)
    try:
        conn = flows_db.get_connection()
        conn.execute("UPDATE flow_runs SET status = 'running' WHERE id = ?", (run_id,))
        conn.commit()
        conn.close()
        assert flows_db.reconcile_orphaned_runs() == 1
    finally:
        reset_account(token)
    events = _events(
        client.post(f"/api/flows/runs/{run_id}/events", params = {"after": snap["last_seq"]}).text
    )
    assert events and events[-1]["type"] == "snapshot"
    assert events[-1]["data"]["run"]["status"] == "interrupted"


def test_many_side_runs_keep_the_older_outputs(client):
    d = doc(
        [node("a", "input.text", value = "story"), node("b", "input.text", value = "side")],
        id = "flow1",
    )
    client.post("/api/flows", json = {"doc": d})
    _wait_terminal(client, client.post("/api/flows/flow1/runs", json = {}).json()["run_id"])
    for _ in range(22):
        run_id = client.post("/api/flows/flow1/runs", json = {"target_nodes": ["b"]}).json()["run_id"]
        _wait_terminal(client, run_id)
    engine_mod.ENGINE.runs.clear()
    latest = client.get("/api/flows/flow1/latest-outputs").json()["nodes"]
    assert latest["a"]["outputs"] == {"text": "story"}


def test_an_m4a_upload_is_audio_not_video(client):
    m4a = b"\x00\x00\x00\x20ftypM4A \x00\x00\x02\x00M4A isommp42" + b"\x00" * 64
    media = client.post("/api/flows/assets?name=memo.m4a", content = m4a).json()["media"]
    assert media["kind"] == "audio" and media["mime"] == "audio/mp4"
    assert client.get(media["url"]).content == m4a
    mp4 = b"\x00\x00\x00\x20ftypisom\x00\x00\x02\x00isomiso2" + b"\x00" * 64
    assert (
        client.post("/api/flows/assets?name=clip.mp4", content = mp4).json()["media"]["kind"]
        == "video"
    )


def test_an_aac_file_with_a_generic_mp4_brand_is_audio(client):
    pytest.importorskip("av")
    from .test_audio_dataset_decode import _m4a_bytes
    from .test_flows_media_utils import _clip

    audio = _m4a_bytes()
    assert audio[8:12] != b"M4A "
    assert (
        client.post("/api/flows/assets?name=a.m4a", content = audio).json()["media"]["kind"]
        == "audio"
    )
    for clip in (_clip(), _clip(tone_s = 0.5)):
        assert (
            client.post("/api/flows/assets?name=c.mp4", content = clip).json()["media"]["kind"]
            == "video"
        )


def test_mp4_handler_scan_stops_early_on_a_crafted_file():
    import time as _time

    from core.flows import assets as asset_store

    crafted = b"\x00\x00\x00\x20ftypisom" + b"hdlrxxxxxxxxsoun" * 2_000_000
    started = _time.perf_counter()
    asset_store.sniff_mime(crafted)
    assert _time.perf_counter() - started < 0.5


def _webm(video):
    import io
    from fractions import Fraction

    av = pytest.importorskip("av")
    import numpy as np

    buf = io.BytesIO()
    with av.open(buf, "w", format = "webm") as out:
        vs = None
        if video:
            vs = out.add_stream("libvpx", rate = 4)
            vs.width = vs.height = 32
            vs.pix_fmt = "yuv420p"
        a = out.add_stream("libopus", rate = 48000)
        a.layout = "mono"
        if vs is not None:
            for i in range(3):
                f = av.VideoFrame.from_ndarray(np.zeros((32, 32, 3), np.uint8), format = "rgb24")
                f.pts, f.time_base = i, Fraction(1, 4)
                for p in vs.encode(f):
                    out.mux(p)
            for p in vs.encode(None):
                out.mux(p)
        for k in range(200):
            fr = av.AudioFrame.from_ndarray(
                (np.random.rand(1, 960) - 0.5).astype(np.float32), format = "flt", layout = "mono"
            )
            fr.sample_rate, fr.pts, fr.time_base = 48000, k * 960, Fraction(1, 48000)
            for p in a.encode(fr):
                out.mux(p)
        for p in a.encode(None):
            out.mux(p)
    return buf.getvalue()


def test_an_audio_only_webm_recording_is_audio(client):
    try:
        audio, clip = _webm(False), _webm(True)
    except Exception as exc:  # noqa: BLE001 - a PyAV without libopus / libvpx
        pytest.skip(f"cannot encode WebM here: {exc}")
    media = client.post("/api/flows/assets?name=memo.webm", content = audio).json()["media"]
    assert media["kind"] == "audio" and media["mime"] == "audio/webm"
    assert client.get(media["url"]).content == audio
    assert (
        client.post("/api/flows/assets?name=c.webm", content = clip).json()["media"]["kind"]
        == "video"
    )


def test_glb_models_upload_and_travel_in_packages(client, monkeypatch):
    from .test_flows_glb import TRI, make_glb

    async def no_resident(request):
        return {}

    monkeypatch.setattr(flows_routes, "_resident", no_resident)
    model = make_glb(TRI)
    up = client.post("/api/flows/assets?name=boat.glb&mime=model/gltf-binary", content = model)
    assert up.status_code == 200
    media = up.json()["media"]
    assert media["kind"] == "model" and media["mime"] == "model/gltf-binary"
    assert client.get(f"/api/flows/assets/{media['ref']}").content == model
    refused = client.post(
        "/api/flows/assets", content = make_glb({**TRI, "buffers": [{"uri": "https://x.test/b"}]})
    )
    assert refused.status_code == 415 and "outside files" in refused.json()["detail"]
    prop = {"id": "m-1", "kind": "model", "x": 1.5, "y": 1.5, "asset": media}
    d = doc([node("play", "world.play", props3d = [prop])], id = "scenery")
    client.post("/api/flows", json = {"doc": d})
    exported = client.get("/api/flows/scenery/export", params = {"mode": "package"})
    assert exported.status_code == 200
    with zipfile.ZipFile(io.BytesIO(exported.content)) as zf:
        assert f"assets/{media['ref'].split(':')[1]}.glb" in zf.namelist()


def _snapshot(cache, repo, *files):
    snap = cache / ("models--" + repo.replace("/", "--")) / "snapshots" / ("a" * 40)
    snap.mkdir(parents = True, exist_ok = True)
    for name in files:
        (snap / name).parent.mkdir(parents = True, exist_ok = True)
        (snap / name).write_bytes(b"x")


def test_download_check_needs_the_weights_and_the_selected_quant(monkeypatch, tmp_path):
    import huggingface_hub.constants as hub_constants

    monkeypatch.setattr(hub_constants, "HF_HUB_CACHE", str(tmp_path))
    _snapshot(tmp_path, "org/meta-only", "config.json", "tokenizer_config.json")
    _snapshot(tmp_path, "org/other-quant-GGUF", "model-Q8_0.gguf")
    _snapshot(tmp_path, "org/split-GGUF", "Q4_K_M/model-Q4_K_M-00001-of-00002.gguf")
    _snapshot(tmp_path, "org/full", "config.json", "model.safetensors")
    _snapshot(tmp_path, "org/diffusers", "model_index.json", "transformer/diffusion.safetensors")
    _snapshot(tmp_path, "org/quant-GGUF", "model-Q8_0.gguf", "model-Q4_K_M.gguf")

    assert flows_routes._hub_cached("org/meta-only") is False
    assert flows_routes._hub_cached("org/other-quant-GGUF", "Q4_K_M") is False
    assert flows_routes._hub_cached("org/split-GGUF", "Q4_K_M") is False
    assert flows_routes._hub_cached("org/never-fetched") is False
    assert flows_routes._hub_cached("org/full") is True
    assert flows_routes._hub_cached("org/diffusers") is True
    assert flows_routes._hub_cached("org/quant-GGUF", "Q4_K_M") is True
    assert flows_routes._hub_cached("org/other-quant-GGUF") is True
    _snapshot(tmp_path, "org/split-GGUF", "Q4_K_M/model-Q4_K_M-00002-of-00002.gguf")
    assert flows_routes._hub_cached("org/split-GGUF", "Q4_K_M") is True
