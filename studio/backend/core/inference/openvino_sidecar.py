# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""OpenAI-compatible /v1/chat/completions over OpenVINO GenAI, one model, one generation at a time.

Spawned by ``openvino_backend.OpenVinoBackend`` in a Python that has ``openvino_genai``. Kept free
of Studio imports so it runs in any such interpreter. Reasoning (``<think>...</think>``) is
returned as ``reasoning_content``, the field Studio's chat already renders.
"""

import argparse
import json
import queue
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Optional

import openvino_genai as ov_genai
import uvicorn
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict

THINK_OPEN, THINK_CLOSE = "<think>", "</think>"


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra = "ignore")

    model: Optional[str] = None
    messages: list[dict[str, Any]]
    max_tokens: Optional[int] = None
    max_completion_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    stream: bool = False
    enable_thinking: Optional[bool] = None
    chat_template_kwargs: Optional[dict[str, Any]] = None


def flatten_content(content: Any) -> str:
    """OpenAI clients send content as a string or a list of parts; the template wants a string."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            p if isinstance(p, str) else str(p.get("text") or "")
            for p in content
            if isinstance(p, (str, dict))
        )
    return str(content)


def thinking_enabled(req: ChatRequest) -> bool:
    kwargs = req.chat_template_kwargs or {}
    if "enable_thinking" in kwargs:
        return bool(kwargs["enable_thinking"])
    return True if req.enable_thinking is None else req.enable_thinking


class ThinkSplitter:
    """Splits a token stream into (reasoning, content) pieces at ``</think>``.

    With thinking on, the template leaves ``<think>`` open, so everything up to the close tag is
    reasoning. A tag split across two pieces is held back until it can be decided.
    """

    def __init__(self, thinking: bool) -> None:
        self.in_think = thinking
        self.buf = ""

    def feed(self, piece: str) -> list[tuple[str, str]]:
        self.buf += piece
        out: list[tuple[str, str]] = []
        while self.buf:
            tag = THINK_CLOSE if self.in_think else THINK_OPEN
            idx = self.buf.find(tag)
            if idx >= 0:
                if idx:
                    out.append(("reasoning" if self.in_think else "content", self.buf[:idx]))
                self.buf = self.buf[idx + len(tag) :]
                if self.in_think:
                    self.buf = self.buf.lstrip("\n")
                self.in_think = not self.in_think
                continue
            # Keep a possible partial tag at the end.
            keep = next((k for k in range(len(tag) - 1, 0, -1) if self.buf.endswith(tag[:k])), 0)
            emit = self.buf[: len(self.buf) - keep]
            if emit:
                out.append(("reasoning" if self.in_think else "content", emit))
            self.buf = self.buf[len(emit) :]
            break
        return out

    def flush(self) -> list[tuple[str, str]]:
        rest, self.buf = self.buf, ""
        return [("reasoning" if self.in_think else "content", rest)] if rest else []


def build_app(pipe, model_id: str) -> FastAPI:
    tok = pipe.get_tokenizer()
    lock = threading.Lock()
    app = FastAPI()

    def config(req: ChatRequest):
        cfg = ov_genai.GenerationConfig()
        cfg.max_new_tokens = req.max_completion_tokens or req.max_tokens or 4096
        temp = 0.6 if req.temperature is None else req.temperature
        cfg.do_sample = temp > 0
        cfg.temperature = max(temp, 1e-4)
        cfg.top_p = req.top_p or 0.95
        cfg.top_k = req.top_k or 20
        return cfg

    def prompt(req: ChatRequest, thinking: bool) -> str:
        history = [
            {"role": m.get("role", "user"), "content": flatten_content(m.get("content"))}
            for m in req.messages
        ]
        return tok.apply_chat_template(
            history, add_generation_prompt = True, extra_context = {"enable_thinking": thinking}
        )

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/v1/models")
    def models():
        return {
            "object": "list",
            "data": [{"id": model_id, "object": "model", "owned_by": "local"}],
        }

    @app.post("/v1/chat/completions")
    def chat(req: ChatRequest):
        thinking = thinking_enabled(req)
        text_prompt, cfg = prompt(req, thinking), config(req)
        rid, created = f"chatcmpl-{uuid.uuid4().hex[:24]}", int(time.time())

        def chunk(delta: dict, finish: Optional[str] = None) -> str:
            body = {
                "id": rid,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model_id,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
            }
            return f"data: {json.dumps(body)}\n\n"

        if not req.stream:
            with lock:
                res = pipe.generate(text_prompt, generation_config = cfg)
            splitter = ThinkSplitter(thinking)
            parts = splitter.feed(res.texts[0]) + splitter.flush()
            message = {
                "role": "assistant",
                "content": "".join(t for k, t in parts if k == "content"),
            }
            reasoning = "".join(t for k, t in parts if k == "reasoning")
            if reasoning:
                message["reasoning_content"] = reasoning
            return {
                "id": rid,
                "object": "chat.completion",
                "created": created,
                "model": model_id,
                "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
            }

        def events():
            pieces: "queue.Queue[Optional[str]]" = queue.Queue()

            def run():
                try:
                    with lock:
                        pipe.generate(
                            text_prompt,
                            generation_config = cfg,
                            streamer = lambda s: pieces.put(s) or ov_genai.StreamingStatus.RUNNING,
                        )
                finally:
                    pieces.put(None)

            threading.Thread(target = run, daemon = True).start()
            yield chunk({"role": "assistant"})
            splitter = ThinkSplitter(thinking)
            while (piece := pieces.get()) is not None:
                for kind, text in splitter.feed(piece):
                    yield chunk({"reasoning_content" if kind == "reasoning" else "content": text})
            for kind, text in splitter.flush():
                yield chunk({"reasoning_content" if kind == "reasoning" else "content": text})
            yield chunk({}, "stop")
            yield "data: [DONE]\n\n"

        return StreamingResponse(events(), media_type = "text/event-stream")

    return app


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required = True, help = "OpenVINO IR directory")
    ap.add_argument("--model-id", required = True)
    ap.add_argument("--port", type = int, required = True)
    ap.add_argument("--device", default = "GPU")
    ap.add_argument("--cache-gb", type = float, default = 3.0)
    args = ap.parse_args()

    sched = ov_genai.SchedulerConfig()
    sched.cache_size = max(1, int(args.cache_gb))
    sched.enable_prefix_caching = True
    sched.dynamic_split_fuse = True
    props = {
        "CACHE_DIR": str(Path.home() / ".cache" / "ov_model_cache"),
        "PERFORMANCE_HINT": "LATENCY",
        "KV_CACHE_PRECISION": "u8",
        "DYNAMIC_QUANTIZATION_GROUP_SIZE": "32",
        "scheduler_config": sched,
    }
    # VLMPipeline also loads text-only IR; LLMPipeline covers exports without the VLM layout.
    try:
        pipe = ov_genai.VLMPipeline(args.model, args.device, **props)
    except Exception:
        pipe = ov_genai.LLMPipeline(args.model, args.device, **props)
    uvicorn.run(
        build_app(pipe, args.model_id), host = "127.0.0.1", port = args.port, log_level = "warning"
    )


if __name__ == "__main__":
    main()
