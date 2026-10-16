# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Needs CUDA and LLAMA_SERVER_PATH; PyTorch runs separately because Unsloth patches transformers."""

import base64
import io
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "tests" / "_shared"))
from real_accelerator import has_real_cuda  # noqa: E402

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not has_real_cuda(), reason = "needs a CUDA GPU"),
    pytest.mark.skipif(
        not os.environ.get("LLAMA_SERVER_PATH"),
        reason = "needs LLAMA_SERVER_PATH (llama.cpp >= b11443)",
    ),
    pytest.mark.skipif(
        not os.environ.get("UNSLOTH_TEST_CLEF_GGUF_CACHE"),
        reason = "needs UNSLOTH_TEST_CLEF_GGUF_CACHE holding ggml-org/Clef-Flash-GGUF",
    ),
]

TEXT = {
    "support": {
        "state": "Customer message: I was charged twice for my order last week and nobody has replied.",
        "questions": {
            "route": {
                "type": "choice",
                "instructions": "Which team should handle this?",
                "criteria": {
                    "billing": "payments and refunds",
                    "shipping": None,
                    "technical": None,
                },
            },
            "urgency": {
                "type": "score",
                "instructions": "How urgent is this?",
                "criteria": ["can wait", "this week", "today", "right now"],
            },
            "angry": {"type": "noul", "instructions": "Is the customer angry?"},
        },
    },
    "review": {
        "state": "Product review: The headphones arrived quickly and sound great, but the left ear cushion "
        "started peeling after two days. Still, for the price I'd buy them again.",
        "questions": {
            "sentiment": {
                "type": "choice",
                "instructions": "Overall sentiment of the review?",
                "criteria": {"positive": None, "neutral": None, "negative": None},
            },
            "stars": {
                "type": "score",
                "instructions": "How many stars would this reviewer give?",
                "criteria": ["1 star", "2 stars", "3 stars", "4 stars", "5 stars"],
            },
            "defect": {"type": "noul", "instructions": "Does the review report a product defect?"},
        },
    },
    "moderation": {
        "state": {
            "user": "alice",
            "message": "Can someone tell me how to reset my router password? "
            "I tried the default one on the sticker and it does not work.",
        },
        "questions": {
            "category": {
                "type": "choice",
                "instructions": "What kind of post is this?",
                "criteria": {
                    "question": "asks for help",
                    "spam": "advertising or scams",
                    "harassment": "attacks another user",
                    "announcement": None,
                },
            },
            "safe": {
                "type": "noul",
                "instructions": "Is this post safe to publish?",
                "criteria": {
                    "true": "allowed by community rules",
                    "false": "breaks community rules",
                },
            },
        },
    },
}


def _long_state(n_words = 5000, seed = 0):
    rnd = random.Random(seed)
    lines = [
        f"[{i:04d}] agent: order {rnd.randint(10000, 99999)} status checked, "
        f"carrier scan at hub {rnd.randint(1, 40)} recorded."
        for i in range(n_words // 14)
    ]
    lines.insert(
        len(lines) // 2, "[note] customer: the package arrived broken and I want a full refund now!"
    )
    return "Support log:\n" + "\n".join(lines)


TEXT["long_state"] = {
    "state": _long_state(),
    "questions": {
        "issue": {
            "type": "choice",
            "instructions": "What does the customer complain about?",
            "criteria": {
                "damaged item": None,
                "late delivery": None,
                "wrong item": None,
                "no complaint": None,
            },
        },
        "refund": {"type": "noul", "instructions": "Does the customer ask for a refund?"},
    },
}


def _red_circle():
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (256, 256), "white")
    ImageDraw.Draw(image).ellipse((48, 48, 208, 208), fill = (220, 20, 20))
    buffer = io.BytesIO()
    image.save(buffer, format = "PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


IMAGE = {
    "state": "An image was attached to a support ticket.",
    "questions": {
        "color": {
            "type": "choice",
            "instructions": "What is the color of the main shape in the image?",
            "criteria": {"red": None, "green": None, "blue": None, "yellow": None},
        },
        "shape": {
            "type": "choice",
            "instructions": "What shape is shown in the image?",
            "criteria": {"circle": None, "square": None, "triangle": None},
        },
        "has_text": {"type": "noul", "instructions": "Does the image contain written text?"},
    },
}

REFERENCE = r"""
import json, sys
import unsloth  # noqa: F401  (before transformers)
from unsloth import FastDecisionModel

requests, out = json.load(open(sys.argv[1])), sys.argv[2]
model, tokenizer = FastDecisionModel.from_pretrained("Cloudflare/clef-flash", use_gradient_checkpointing=False)
model.eval()
answers = {name: FastDecisionModel.predict(model, tokenizer, r["state"], r["questions"]) for name, r in requests.items()}
json.dump(answers, open(out, "w"), default=str)
"""


def _probabilities(answer):
    if answer["type"] == "noul":
        return {"true": float(answer["noul"])}
    return {k: float(v) for k, v in answer["probabilities"].items()}


def _top(answer):
    if answer["type"] == "noul":
        return float(answer["noul"]) > 0.5
    if answer["type"] == "choice":
        return answer["choice"]
    # The expected level: an argmax flips on a near tie (0.422 / 0.423) that both backends read the same.
    return round(float(answer["score"]))


@pytest.fixture(scope = "module")
def reference(tmp_path_factory):
    folder = tmp_path_factory.mktemp("clef_reference")
    (folder / "requests.json").write_text(json.dumps(TEXT), encoding = "utf-8")
    (folder / "reference.py").write_text(REFERENCE, encoding = "utf-8")
    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT)}
    subprocess.run(
        [
            sys.executable,
            str(folder / "reference.py"),
            str(folder / "requests.json"),
            str(folder / "out.json"),
        ],
        check = True,
        env = env,
        cwd = folder,
        timeout = 3600,
    )
    return json.loads((folder / "out.json").read_text(encoding = "utf-8"))


@pytest.fixture
def studio(tmp_path, monkeypatch):
    yield from _studio(tmp_path, monkeypatch, {"model": "clef-flash"})


def _studio(tmp_path, monkeypatch, settings):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from auth.authentication import get_current_subject
    from core.inference import gpu_arbiter
    from core.systemone import laya_runtime
    from routes import systemone
    from routes.settings import router as settings_router
    from utils.account_context import OWNER, run_as
    from utils.paths import outputs_root

    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "home"))
    for name in (
        "UNSLOTH_SYSTEMONE_MODEL",
        "UNSLOTH_SYSTEMONE_DISABLE",
        "UNSLOTH_SYSTEMONE_DEVICE",
    ):
        monkeypatch.delenv(name, raising = False)
    monkeypatch.setattr(gpu_arbiter, "_owner", None)
    monkeypatch.setattr(
        "utils.hf_cache_settings.active_hf_hub_cache",
        lambda: os.environ["UNSLOTH_TEST_CLEF_GGUF_CACHE"],
    )
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    run_as(OWNER, outputs_root).mkdir(parents = True)
    app = FastAPI()
    app.include_router(systemone.router, prefix = "/v1")
    app.include_router(settings_router, prefix = "/api/settings")
    app.dependency_overrides[get_current_subject] = lambda: "unsloth"
    client = TestClient(app)
    response = client.put(
        "/api/settings/systemone", json = {"enabled": True, "device": "gpu", **settings}
    )
    assert response.status_code == 200, response.text
    yield client
    laya_runtime.shutdown()


def _post(
    client,
    request,
    images = None,
    model = "clef-flash",
):
    body = {"model": model, "state": request["state"], "questions": request["questions"]}
    if images:
        body["images"] = images
    deadline = time.monotonic() + 900
    while True:
        response = client.post("/v1/systemone", json = body)
        loading = (
            response.status_code == 503
            and response.json()["detail"]["error_type"] == "model_loading"
        )
        if not loading or time.monotonic() > deadline:
            return response
        time.sleep(1)


def test_clef_flash_on_llama_cpp_matches_pytorch(studio, reference, monkeypatch, record_property):
    from core.systemone import laya_runtime

    rows, worst = [], 0.0
    for name, request in TEXT.items():
        started = time.monotonic()
        response = _post(studio, request)
        assert response.status_code == 200, response.text
        assert response.headers["x-unsloth-decision-backend"] == "llama.cpp"
        native = response.json()
        for question, answer in native["answers"].items():
            expected = reference[name][question]
            ours, theirs = _probabilities(answer), _probabilities(expected)
            diff = max(abs(ours[k] - theirs[k]) for k in ours)
            worst = max(worst, diff)
            rows.append(
                {
                    "request": name,
                    "question": question,
                    "type": answer["type"],
                    "same_answer": _top(answer) == _top(expected),
                    "max_abs_prob_diff": round(diff, 4),
                    "input_tokens": native["usage"]["input_tokens"],
                    "latency_s": round(time.monotonic() - started, 3),
                }
            )
    print(json.dumps(rows, indent = 1))
    record_property("parity", rows)
    assert all(row["same_answer"] for row in rows), rows
    assert worst <= 0.01, rows
    assert max(r["input_tokens"] for r in rows) > 10000

    image = _post(studio, IMAGE, images = [_red_circle()])
    assert image.status_code == 200, image.text
    answers = image.json()["answers"]
    print(json.dumps(answers, indent = 1))
    assert answers["color"]["choice"] == "red" and answers["shape"]["choice"] == "circle"
    assert answers["has_text"]["noul"] < 0.5
    status = studio.get("/api/settings/systemone").json()
    assert (status["loaded_backend"], status["input_modalities"]) == (
        "llama.cpp",
        ["text", "image"],
    )
    assert status["loaded_device"].startswith("CUDA")

    # A training run takes the GPU: the server is ended and the Decision API answers 503.
    agent = laya_runtime._agent
    monkeypatch.setattr(laya_runtime, "_training_active", lambda: True)
    busy = _post(studio, TEXT["support"])
    assert busy.status_code == 503 and "training run" in busy.json()["detail"]["message"]
    assert laya_runtime._agent is None and not agent.is_alive()


def _bare_llama_server(model_path, requests, log_path):
    """llama.cpp's own answers: the GGUF on a plain llama-server with the flags Studio passes."""
    import socket
    import urllib.request

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    binary = os.environ["LLAMA_SERVER_PATH"]
    env = {**os.environ, "LD_LIBRARY_PATH": str(Path(binary).parent)}
    command = [binary, "-m", str(model_path), "--host", "127.0.0.1", "--port", str(port)]
    command += ["--parallel", "1", "-c", "8192", "-b", "8192", "-ub", "8192", "-ngl", "-1"]
    with open(log_path, "wb") as log:
        process = subprocess.Popen(command, stdout = log, stderr = subprocess.STDOUT, env = env)
    base = f"http://127.0.0.1:{port}"
    try:
        deadline = time.monotonic() + 600
        while True:
            assert process.poll() is None, Path(log_path).read_text(errors = "replace")[-2000:]
            try:
                with urllib.request.urlopen(base + "/health", timeout = 2) as response:
                    if response.status == 200:
                        break
            except OSError:
                pass
            assert time.monotonic() < deadline
            time.sleep(0.5)
        answers = {}
        for name, request in requests.items():
            data = json.dumps({"model": "x", **request}).encode()
            post = urllib.request.Request(
                base + "/v1/systemone", data, {"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(post, timeout = 300) as response:
                answers[name] = json.loads(response.read())
        return answers
    finally:
        process.terminate()
        process.wait(30)


GGUF_ONLY = ["kev-0.8b", "laya-gguf", "julia-1"]
# The ggml_decision_sweep requests: support, review, moderation (no long state, no images).
SWEEP = {name: TEXT[name] for name in ("support", "review", "moderation")}


@pytest.mark.parametrize("name", GGUF_ONLY)
def test_a_gguf_only_model_answers_as_bare_llama_server(
    name, tmp_path, monkeypatch, record_property
):
    from core.inference import llama_cpp
    from core.systemone import catalog, laya_runtime

    # Studio's own probe gives --list-devices 30 s; a loaded CI host can take longer.
    binary = os.environ["LLAMA_SERVER_PATH"]
    listed = subprocess.run(
        [binary, "--list-devices"],
        capture_output = True,
        text = True,
        timeout = 600,
        env = llama_cpp.LlamaCppBackend._llama_server_env_for_binary(binary),
    )
    devices = llama_cpp._parse_listed_devices(listed.stdout)
    assert devices, listed.stdout + listed.stderr
    monkeypatch.setattr(
        llama_cpp.LlamaCppBackend, "_enumerated_gpu_devices", staticmethod(lambda *_: devices)
    )
    client = next(
        _studio(tmp_path, monkeypatch, {"model": name, "backend": "auto", "native_ctx": 8192})
    )
    try:
        rows = []
        for request_name, request in SWEEP.items():
            response = _post(client, request, model = name)
            assert response.status_code == 200, response.text
            assert response.headers["x-unsloth-decision-backend"] == "llama.cpp"
            rows.append((request_name, response.json()))
        status = client.get("/api/settings/systemone").json()
        assert (status["loaded_model"], status["loaded_backend"]) == (name, "llama.cpp")
        assert status["loaded_device"].startswith("CUDA")
        model_path, _ = laya_runtime._native_files(catalog.CHECKPOINTS[name], local_only = True)
        assert catalog.GGUF_COMPANIONS[name].revision in str(model_path)
    finally:
        laya_runtime.shutdown()
    bare = _bare_llama_server(model_path, SWEEP, tmp_path / "bare.log")
    worst = 0.0
    for request_name, body in rows:
        assert body["model"] == name
        expected = bare[request_name]
        assert body["usage"] == expected["usage"]
        assert set(body["answers"]) == set(expected["answers"])
        for question, answer in body["answers"].items():
            theirs = expected["answers"][question]
            assert answer.keys() == theirs.keys(), (answer, theirs)
            for key, value in answer.items():
                if isinstance(value, float):
                    worst = max(worst, abs(value - theirs[key]))
                elif key == "probabilities":
                    assert value.keys() == theirs[key].keys()
                    worst = max(worst, *(abs(v - theirs[key][k]) for k, v in value.items()))
                else:
                    assert value == theirs[key], (question, key)
    record_property("max_abs_diff", worst)
    print(json.dumps({"model": name, "max_abs_diff": worst, "studio": dict(rows)}, indent = 1))
    assert worst <= 1e-9
