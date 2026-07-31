# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for the persistent sd-server process manager (SdCppServer).

Hermetic: subprocess.Popen and the httpx client are faked, so nothing spawns a real
binary or opens a socket beyond the free-port probe."""

from __future__ import annotations

import base64
import io
import threading

import pytest
from PIL import Image

from core.inference import sd_cpp_server as srv
from core.inference.sd_cpp_args import SdCppModelFiles
from core.inference.sd_cpp_engine import SdCppCancelled
from core.inference.sd_cpp_server import SdCppServer

_FILES = SdCppModelFiles(diffusion_model = "/m/z.gguf", vae = "/m/vae.sft", llm = "/m/llm.sft")


def _png_b64(shade: int) -> str:
    buf = io.BytesIO()
    Image.new("RGB", (1, 1), (shade, shade, shade)).save(buf, format = "PNG")
    return base64.b64encode(buf.getvalue()).decode()


class _FakePopen:
    """Minimal Popen stand-in. stdout yields the scripted lines then BLOCKS until the
    process is terminated/killed/exited -- mirroring a real child that holds its pipe
    open for its lifetime (so the owner/drain thread stays alive, as in production)."""

    def __init__(
        self,
        lines = (),
        exit_code = None,
    ):
        self.pid = 4242
        self._lines = list(lines)
        self._exit = exit_code  # None == alive
        self.returncode = exit_code
        self.terminated = False
        self.killed = False
        self._done = threading.Event()
        if exit_code is not None:
            self._done.set()

    @property
    def stdout(self):
        def _gen():
            for ln in self._lines:
                yield ln
            self._done.wait()  # hold the pipe open until the process ends

        return _gen()

    def poll(self):
        return self._exit

    def terminate(self):
        self.terminated = True
        self._exit = 0
        self.returncode = 0
        self._done.set()

    def wait(self, timeout = None):
        self._done.wait(timeout)
        if self._exit is None:
            self._exit = 0
            self.returncode = 0
        return self.returncode

    def kill(self):
        self.killed = True
        self._exit = -9
        self.returncode = -9
        self._done.set()


class _Resp:
    def __init__(
        self,
        status_code,
        payload = None,
        text = "",
        bad_json = False,
    ):
        self.status_code = status_code
        self._payload = payload if payload is not None else {}
        self.text = text
        self._bad_json = bad_json

    def json(self):
        if self._bad_json:
            raise ValueError("not json")
        return self._payload


class _FakeClient:
    def __init__(
        self,
        *,
        get = None,
        post = None,
    ):
        self._get = get or (lambda url: _Resp(200, {}))
        self._post = post or (lambda url, json: _Resp(202, {"id": "job1"}))
        self.get_urls = []
        self.post_calls = []
        self.closed = False

    def get(
        self,
        url,
        timeout = None,
    ):
        self.get_urls.append(url)
        return self._get(url)

    def post(
        self,
        url,
        json = None,
        timeout = None,
    ):
        self.post_calls.append((url, json))
        return self._post(url, json)

    def close(self):
        self.closed = True


@pytest.fixture
def patched(monkeypatch):
    """Neutralise process-lifetime side effects for the manager under test."""
    monkeypatch.setattr(srv, "adopt_pid", lambda pid: None)
    monkeypatch.setattr(srv, "forget_pid", lambda pid: None)
    monkeypatch.setattr(srv, "child_popen_kwargs", lambda: {})
    monkeypatch.setattr(srv, "windows_hidden_subprocess_kwargs", lambda: {})
    return monkeypatch


def _server_with(popen, client):
    s = SdCppServer("/x/sd-server")
    s._client = client
    # Attach the fake process + port so generation tests can run without start().
    s._process = popen
    s.port = 1234
    return s


# ── start / readiness ──────────────────────────────────────────────────────────


def test_start_becomes_ready_when_capabilities_200(patched):
    popen = _FakePopen(lines = ["loading model", "listening on: http://127.0.0.1:1"])
    patched.setattr(srv.subprocess, "Popen", lambda *a, **k: popen)
    s = _server_with(
        popen, _FakeClient(get = lambda url: _Resp(200, {"model": {"path": "/m/z.gguf"}}))
    )
    s.start(_FILES, startup_timeout = 5.0)
    assert s.is_alive() is True
    assert s.port is not None


def test_start_fails_fast_when_process_exits(patched):
    # Model load failed, so the process exits before listening; start must raise with the tail.
    popen = _FakePopen(lines = ["error: bad model"], exit_code = 1)
    patched.setattr(srv.subprocess, "Popen", lambda *a, **k: popen)
    # Capabilities never answers (connection refused), so readiness relies on exit detection.
    s = _server_with(
        popen, _FakeClient(get = lambda url: (_ for _ in ()).throw(srv.httpx.ConnectError("refused")))
    )
    with pytest.raises(RuntimeError, match = "failed to become ready"):
        s.start(_FILES, startup_timeout = 2.0)


# ── generation ───────────────────────────────────────────────────────────────


def _completed_job(images_b64):
    return _Resp(
        200,
        {
            "status": "completed",
            "result": {"images": [{"index": i, "b64_json": b} for i, b in enumerate(images_b64)]},
        },
    )


def test_img_gen_returns_image_bytes_in_index_order(patched):
    popen = _FakePopen()
    s = _server_with(
        popen,
        _FakeClient(
            post = lambda url, json: _Resp(202, {"id": "jobA"}),
            # result images deliberately out of order -> manager must sort by index.
            get = lambda url: _Resp(
                200,
                {
                    "status": "completed",
                    "result": {
                        "images": [
                            {"index": 1, "b64_json": _png_b64(200)},
                            {"index": 0, "b64_json": _png_b64(50)},
                        ]
                    },
                },
            ),
        ),
    )
    blobs = s.img_gen({"prompt": "x", "batch_count": 2, "sample_params": {"sample_steps": 4}})
    assert len(blobs) == 2
    first = Image.open(io.BytesIO(blobs[0])).convert("RGB").getpixel((0, 0))
    assert first == (50, 50, 50)  # index 0 first


def test_img_gen_failed_job_raises(patched):
    popen = _FakePopen()
    s = _server_with(
        popen,
        _FakeClient(
            post = lambda url, json: _Resp(202, {"id": "jobF"}),
            get = lambda url: _Resp(
                200, {"status": "failed", "error": {"code": "x", "message": "boom"}}
            ),
        ),
    )
    with pytest.raises(RuntimeError, match = "generation failed.*boom"):
        s.img_gen({"prompt": "x"})


def test_img_gen_queue_full_raises(patched):
    popen = _FakePopen()
    s = _server_with(popen, _FakeClient(post = lambda url, json: _Resp(429, text = "busy")))
    with pytest.raises(RuntimeError, match = "queue is full"):
        s.img_gen({"prompt": "x"})


def test_img_gen_cancel_posts_cancel_and_raises(patched):
    popen = _FakePopen()
    cancel = threading.Event()
    cancel.set()  # already cancelled before the first poll
    client = _FakeClient(
        post = lambda url, json: _Resp(202, {"id": "jobC"}),
        get = lambda url: _Resp(
            200, {"status": "cancelled", "error": {"code": "cancelled", "message": "c"}}
        ),
    )
    s = _server_with(popen, client)
    with pytest.raises(SdCppCancelled):
        s.img_gen({"prompt": "x"}, cancel_event = cancel)
    assert any(url.endswith("/cancel") for url, _ in client.post_calls)


def test_img_gen_detects_server_death(patched):
    popen = _FakePopen()

    def _die_get(url):
        popen._exit = 137  # the process died between submit and poll
        return _Resp(200, {"status": "generating"})

    s = _server_with(
        popen, _FakeClient(post = lambda url, json: _Resp(202, {"id": "jobD"}), get = _die_get)
    )
    with pytest.raises(RuntimeError, match = "connection lost|process exited"):
        s.img_gen({"prompt": "x"})


# ── stdout routing + stop ──────────────────────────────────────────────────────


def test_drain_routes_lines_to_step_listener_and_tail(patched):
    s = SdCppServer("/x/sd-server")
    seen = []
    s._step_listener = seen.append
    # exit_code set so stdout ends after the scripted lines (a live fake would block).
    s._drain_stdout(_FakePopen(lines = ["sampling 1/8", "", "sampling 8/8", "done"], exit_code = 0))
    assert "sampling 1/8" in seen and "sampling 8/8" in seen
    assert "" not in seen  # blank lines skipped
    assert s._tail[-1] == "done"


def test_stop_is_idempotent_and_terminates(patched):
    popen = _FakePopen()
    patched.setattr(srv.subprocess, "Popen", lambda *a, **k: popen)
    client = _FakeClient(get = lambda url: _Resp(200, {}))
    s = _server_with(popen, client)
    s.start(_FILES, startup_timeout = 5.0)
    s.stop()
    assert popen.terminated is True
    assert s.is_alive() is False
    assert client.closed is True  # stop() releases the pooled HTTP client
    s.stop()  # second call must not raise


def test_img_gen_submit_error_raises(patched):
    popen = _FakePopen()
    s = _server_with(popen, _FakeClient(post = lambda url, json: _Resp(400, text = "bad params")))
    with pytest.raises(RuntimeError, match = "submit -> 400"):
        s.img_gen({"prompt": "x"})


def test_img_gen_malformed_submit_json_raises(patched):
    popen = _FakePopen()
    s = _server_with(popen, _FakeClient(post = lambda url, json: _Resp(202, bad_json = True)))
    with pytest.raises(RuntimeError, match = "non-JSON submit"):
        s.img_gen({"prompt": "x"})


def test_img_gen_empty_result_raises(patched):
    popen = _FakePopen()
    s = _server_with(
        popen,
        _FakeClient(
            post = lambda url, json: _Resp(202, {"id": "jobE"}),
            get = lambda url: _Resp(200, {"status": "completed", "result": {"images": []}}),
        ),
    )
    with pytest.raises(RuntimeError, match = "no images"):
        s.img_gen({"prompt": "x"})


def test_img_gen_rejected_after_stop(patched):
    popen = _FakePopen()
    patched.setattr(srv.subprocess, "Popen", lambda *a, **k: popen)
    s = _server_with(popen, _FakeClient(get = lambda url: _Resp(200, {})))
    s.start(_FILES, startup_timeout = 5.0)
    s.stop()
    with pytest.raises(RuntimeError, match = "not running"):
        s.img_gen({"prompt": "x"})


# ── cancellation + defensive parsing (review follow-ups) ───────────────────────


def test_img_gen_cancelled_before_submit_reports_cancellation(patched):
    # The server was stopped for a cancel/unload before submit; with the cancel event set this is a cancellation (409), not a "not running" 500.
    popen = _FakePopen()
    patched.setattr(srv.subprocess, "Popen", lambda *a, **k: popen)
    s = _server_with(popen, _FakeClient(get = lambda url: _Resp(200, {})))
    s.start(_FILES, startup_timeout = 5.0)
    s.stop()
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(SdCppCancelled):
        s.img_gen({"prompt": "x"}, cancel_event = cancel)


def test_img_gen_abandons_when_cancel_not_honored(patched):
    # A best-effort cancel the server ignores must not pin this call (and the generate lock) until natural completion: it raises after the grace window.
    patched.setattr(srv, "_CANCEL_GRACE_S", 0.0)
    popen = _FakePopen()
    cancel = threading.Event()
    cancel.set()
    client = _FakeClient(
        post = lambda url, json: _Resp(202, {"id": "jobG"}),
        get = lambda url: _Resp(200, {"status": "generating"}),  # never terminal
    )
    s = _server_with(popen, client)
    with pytest.raises(SdCppCancelled):
        s.img_gen({"prompt": "x"}, cancel_event = cancel, poll_interval = 0.01)
    # And the process is stopped, not left running the abandoned job: sd-server does not interrupt an in-flight job, so a server
    # that ignored the cancel would burn a core (or the GPU) to completion and hold its job slot against the next request.
    assert not s.is_alive()


def test_img_gen_non_dict_submit_json_raises(patched):
    popen = _FakePopen()
    s = _server_with(popen, _FakeClient(post = lambda url, json: _Resp(202, ["not", "a", "dict"])))
    with pytest.raises(RuntimeError, match = "unexpected submit response"):
        s.img_gen({"prompt": "x"})


def test_img_gen_non_dict_status_json_raises(patched):
    popen = _FakePopen()
    s = _server_with(
        popen,
        _FakeClient(
            post = lambda url, json: _Resp(202, {"id": "jobH"}),
            get = lambda url: _Resp(200, ["unexpected"]),
        ),
    )
    with pytest.raises(RuntimeError, match = "unexpected response type"):
        s.img_gen({"prompt": "x"}, poll_interval = 0.01)


def test_decode_images_tolerates_unexpected_shapes():
    # A misbehaving/older server can return non-dict result/images/items, so _decode_images must raise a clean "no images" rather than an AttributeError.
    for job in ({"result": ["x"]}, {"result": {"images": "nope"}}, {"result": {"images": [1, 2]}}):
        with pytest.raises(RuntimeError, match = "no images"):
            SdCppServer._decode_images(job)


def test_start_aborted_by_concurrent_stop(patched):
    # A stop() during the readiness wait must abort start() promptly, without waiting out the startup timeout, and surface as a cancellation.
    popen = _FakePopen(lines = ["loading model"])
    patched.setattr(srv.subprocess, "Popen", lambda *a, **k: popen)

    def _never_ready(url):
        raise srv.httpx.ConnectError("refused")

    s = _server_with(popen, _FakeClient(get = _never_ready))

    def _stop_soon():
        import time as _t
        _t.sleep(0.2)
        s.stop()

    threading.Thread(target = _stop_soon, daemon = True).start()
    with pytest.raises(SdCppCancelled):
        s.start(_FILES, startup_timeout = 30.0)


def test_diagnostic_tail_keeps_the_reason_not_just_the_backtrace():
    """What a Metal host produces: the abort prints its cause, then ggml_print_backtrace fills the
    buffer with stack frames. Taking the last N lines reported addresses and no cause, so the
    failure was undiagnosable from the message alone."""
    lines = [
        "loading model from flux-2-klein-4b-Q2_K.gguf",
        "ggml_metal_op_encode: error: unsupported op 'SOME_OP'",
        "/tmp/ggml/src/ggml-metal.m:1234: fatal error",
        *[f"{i}   sd-server  0x000000010311{i:04x} ggml_print_backtrace + {i}" for i in range(24)],
    ]

    tail = srv._diagnostic_tail(lines)

    assert "unsupported op 'SOME_OP'" in tail
    assert "fatal error" in tail
    # Still ends with recent context, so a failure with no marked line is not left empty.
    assert "ggml_print_backtrace" in tail


def test_diagnostic_tail_falls_back_to_the_last_lines():
    lines = [f"step {i}" for i in range(50)]
    tail = srv._diagnostic_tail(lines)
    assert "step 49" in tail
    assert "step 0" not in tail


def test_diagnostic_tail_is_bounded():
    lines = ["error: " + "x" * 500 for _ in range(20)]
    assert len(srv._diagnostic_tail(lines)) <= 1500


def test_readiness_refuses_a_port_held_by_another_process(patched):
    # _find_free_port picks an ephemeral port, closes the socket, and sd-server binds it only after loading the model, which takes
    # minutes for a big checkpoint. Another local process can take it in that window, and /v1/models is a stock OpenAI route that
    # llama.cpp's server also answers 200 on, so readiness would pass and every generation would go to an unrelated listener.
    import types

    popen = _FakePopen(lines = ["loading model"])
    patched.setattr(srv.subprocess, "Popen", lambda *a, **k: popen)
    s = _server_with(popen, _FakeClient(get = lambda url: _Resp(200, {"model": {}})))

    fake_psutil = types.SimpleNamespace(
        CONN_LISTEN = "LISTEN",
        net_connections = lambda kind = "inet": [
            types.SimpleNamespace(
                laddr = types.SimpleNamespace(port = s.port),
                status = "LISTEN",
                pid = popen.pid + 1000,  # somebody else
            )
        ],
        Process = lambda pid: types.SimpleNamespace(parent = lambda: None),
    )
    patched.setitem(__import__("sys").modules, "psutil", fake_psutil)
    assert s._port_is_ours() is False


def test_readiness_accepts_our_own_child_and_its_descendants(patched):
    import types

    popen = _FakePopen()
    s = _server_with(popen, _FakeClient(get = lambda url: _Resp(200, {})))

    def _conns(owner_pid):
        return [
            types.SimpleNamespace(
                laddr = types.SimpleNamespace(port = s.port), status = "LISTEN", pid = owner_pid
            )
        ]

    # The spawned pid itself.
    patched.setitem(
        __import__("sys").modules,
        "psutil",
        types.SimpleNamespace(
            CONN_LISTEN = "LISTEN",
            net_connections = lambda kind = "inet": _conns(popen.pid),
            Process = lambda pid: types.SimpleNamespace(parent = lambda: None),
        ),
    )
    assert s._port_is_ours() is True

    # A grandchild (wrapper script / shell) still counts as ours.
    child_pid = popen.pid + 7

    def _process(pid):
        if pid == child_pid:
            return types.SimpleNamespace(
                parent = lambda: types.SimpleNamespace(pid = popen.pid, parent = lambda: None)
            )
        return types.SimpleNamespace(parent = lambda: None)

    patched.setitem(
        __import__("sys").modules,
        "psutil",
        types.SimpleNamespace(
            CONN_LISTEN = "LISTEN",
            net_connections = lambda kind = "inet": _conns(child_pid),
            Process = _process,
        ),
    )
    assert s._port_is_ours() is True


def test_readiness_check_is_best_effort(patched):
    # No psutil, an unreadable owner pid, or a raising lookup must never fail a healthy start.
    import types

    popen = _FakePopen()
    s = _server_with(popen, _FakeClient(get = lambda url: _Resp(200, {})))

    patched.setitem(__import__("sys").modules, "psutil", None)
    assert s._port_is_ours() is True

    def _boom(kind = "inet"):
        raise PermissionError("not allowed")

    patched.setitem(
        __import__("sys").modules,
        "psutil",
        types.SimpleNamespace(CONN_LISTEN = "LISTEN", net_connections = _boom),
    )
    assert s._port_is_ours() is True

    # Owner pid not visible (common for another user's process): unknown, so keep going.
    patched.setitem(
        __import__("sys").modules,
        "psutil",
        types.SimpleNamespace(
            CONN_LISTEN = "LISTEN",
            net_connections = lambda kind = "inet": [
                types.SimpleNamespace(
                    laddr = types.SimpleNamespace(port = s.port), status = "LISTEN", pid = None
                )
            ],
        ),
    )
    assert s._port_is_ours() is True
