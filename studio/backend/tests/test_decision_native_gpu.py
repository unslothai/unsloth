# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Stock Clef-Flash Q8_0 through Studio's Decision API on a real llama-server, against FastDecisionModel.

Needs a CUDA GPU, LLAMA_SERVER_PATH (a llama.cpp b11443 or newer build, e.g. Studio's prebuilt) and
ggml-org/Clef-Flash-GGUF at the pinned revision in UNSLOTH_TEST_CLEF_GGUF_CACHE (a hub cache dir).
The PyTorch reference (Cloudflare/clef-flash, bf16) runs in its own process: Unsloth patches
transformers process-wide.
"""

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
        "/api/settings/systemone", json = {"enabled": True, "model": "clef-flash", "device": "gpu"}
    )
    assert response.status_code == 200, response.text
    yield client
    laya_runtime.shutdown()


def _post(
    client,
    request,
    images = None,
):
    body = {"model": "clef-flash", "state": request["state"], "questions": request["questions"]}
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
