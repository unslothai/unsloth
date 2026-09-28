# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""OpenAI-compatible /v1/chat/completions over OpenVINO GenAI, one model, one generation at a time.

Spawned by ``openvino_backend.OpenVinoBackend`` in a Python that has ``openvino_genai``. Kept free
of Studio imports so it runs in any such interpreter. Reasoning (``<think>...</think>``) is
returned as ``reasoning_content``, the field Studio's chat already renders. ``tools`` go into the
chat template and ``<tool_call>`` blocks come back as OpenAI ``tool_calls``.
"""

import argparse
import json
import queue
import re
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Optional

import openvino_genai as ov_genai
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from starlette.concurrency import iterate_in_threadpool
from pydantic import BaseModel, ConfigDict

THINK_OPEN, THINK_CLOSE = "<think>", "</think>"
# ponytail: fixed hold-back window for models that reason despite enable_thinking=false; longer
# stray reasoning leaks into content (without the tag).
_STRAY_THINK_WINDOW = 4000
TOOL_OPEN = "<tool_call>"
_FUNC_RE = re.compile(r"<function=([^>\s]+)>(.*?)(?:</function>|$)", re.DOTALL)
_PARAM_RE = re.compile(
    r"<parameter=([^>\s]+)>\n?(.*?)\n?(?:</parameter>|(?=<parameter=)|$)", re.DOTALL
)


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
    tools: Optional[list[dict[str, Any]]] = None
    tool_choice: Optional[Any] = None


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
        # With thinking off the template closes the think block, but some finetunes reason anyway
        # and end with a bare ``</think>``; the start is held back until that can be told apart.
        self.held: Optional[str] = None if thinking else ""

    def feed(self, piece: str) -> list[tuple[str, str]]:
        if self.held is None:
            return self._split(piece)
        self.held += piece
        idx = self.held.find(THINK_CLOSE)
        if idx >= 0:
            before, _, reasoning = self.held[:idx].rpartition(THINK_OPEN)
            rest = self.held[idx + len(THINK_CLOSE) :].lstrip("\n")
            self.held = None
            out = [("content", before)] if before.strip() else []
            if reasoning.strip():
                out.append(("reasoning", reasoning.strip()))
            return out + self._split(rest)
        if len(self.held) > _STRAY_THINK_WINDOW:
            rest, self.held = self.held, None
            return self._split(rest)
        return []

    def _split(self, piece: str) -> list[tuple[str, str]]:
        self.buf += piece
        out: list[tuple[str, str]] = []
        while self.buf:
            # Outside a block a bare </think> is stray reasoning's end: dropped, not shown.
            tags = (THINK_CLOSE,) if self.in_think else (THINK_OPEN, THINK_CLOSE)
            idx, tag = min(((self.buf.find(t), t) for t in tags if t in self.buf), default = (-1, ""))
            if idx >= 0:
                if idx:
                    out.append(("reasoning" if self.in_think else "content", self.buf[:idx]))
                self.buf = self.buf[idx + len(tag) :]
                if tag == THINK_CLOSE:
                    self.buf = self.buf.lstrip("\n")
                if tag == THINK_OPEN or self.in_think:
                    self.in_think = not self.in_think
                continue
            # Keep a possible partial tag at the end.
            keep = max(
                (k for t in tags for k in range(len(t) - 1, 0, -1) if self.buf.endswith(t[:k])),
                default = 0,
            )
            emit = self.buf[: len(self.buf) - keep]
            if emit:
                out.append(("reasoning" if self.in_think else "content", emit))
            self.buf = self.buf[len(emit) :]
            break
        return out

    def flush(self) -> list[tuple[str, str]]:
        if self.held is not None:
            held, self.held = self.held, None
            return self._split(held) + self.flush()
        rest, self.buf = self.buf, ""
        return [("reasoning" if self.in_think else "content", rest)] if rest else []


def _param_value(raw: str, schema: dict) -> Any:
    if schema.get("type") == "string":
        return raw
    try:
        return json.loads(raw)
    except ValueError:
        return raw


def parse_tool_calls(text: str, tools: list[dict]) -> list[dict]:
    """``<tool_call>`` blocks (Qwen XML ``<function=..>`` or JSON body) as OpenAI tool_calls."""
    props = {
        t["function"]["name"]: (t["function"].get("parameters") or {}).get("properties") or {}
        for t in tools
        if isinstance(t, dict) and isinstance(t.get("function"), dict) and "name" in t["function"]
    }
    calls = []
    for block in text.split(TOOL_OPEN)[1:]:
        body = block.split("</tool_call>")[0].strip()
        if m := _FUNC_RE.search(body):
            name, schema = m.group(1), props.get(m.group(1), {})
            args = {
                k: _param_value(v, schema.get(k) or {}) for k, v in _PARAM_RE.findall(m.group(2))
            }
        else:
            try:
                obj = json.loads(body)
                name, args = obj["name"], obj.get("arguments") or {}
            except (ValueError, KeyError, TypeError):
                continue
        calls.append(
            {
                "index": len(calls),
                "id": f"call_{uuid.uuid4().hex[:24]}",
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": args if isinstance(args, str) else json.dumps(args),
                },
            }
        )
    return calls


class ToolSplitter:
    """Passes content through until ``<tool_call>``, then keeps the rest to parse at the end."""

    def __init__(self, active: bool) -> None:
        self.active = active
        self.buf = ""
        self.captured: Optional[str] = None

    def feed(self, text: str) -> str:
        if not self.active:
            return text
        if self.captured is not None:
            self.captured += text
            return ""
        self.buf += text
        idx = self.buf.find(TOOL_OPEN)
        if idx >= 0:
            out, self.captured, self.buf = self.buf[:idx], self.buf[idx:], ""
            return out.rstrip()
        keep = next(
            (k for k in range(len(TOOL_OPEN) - 1, 0, -1) if self.buf.endswith(TOOL_OPEN[:k])), 0
        )
        out, self.buf = self.buf[: len(self.buf) - keep], self.buf[len(self.buf) - keep :]
        return out

    def finish(self, tools: list[dict]) -> tuple[str, list[dict]]:
        """Leftover content and the parsed calls; unparseable markup is returned as content."""
        rest, self.buf = self.buf, ""
        if self.captured is None:
            return rest, []
        calls = parse_tool_calls(self.captured, tools)
        return ("" if calls else self.captured), calls


def history_message(m: dict) -> dict:
    """An OpenAI message as the chat template wants it: text content, tool arguments as a dict."""
    out = {k: v for k, v in m.items() if k in ("role", "name", "tool_call_id")}
    out.setdefault("role", "user")
    out["content"] = flatten_content(m.get("content"))
    if m.get("tool_calls"):
        calls = []
        for c in m["tool_calls"]:
            fn = dict(c.get("function") or {})
            if isinstance(fn.get("arguments"), str):
                try:
                    fn["arguments"] = json.loads(fn["arguments"] or "{}")
                except ValueError:
                    fn["arguments"] = {}
            calls.append({**c, "function": fn})
        out["tool_calls"] = calls
    return out


def chat_history(messages: list[dict]) -> list[dict]:
    """Template-ready history. Qwen-style templates allow one system message, first; agents such
    as opencode send several, so they are joined into one."""
    history = [history_message(m) for m in messages]
    system = [m["content"] for m in history if m["role"] == "system"]
    rest = [m for m in history if m["role"] != "system"]
    return ([{"role": "system", "content": "\n\n".join(system)}] if system else []) + rest


def finish_reason(
    res: Any,
    calls: list,
    out_of_room: bool = False,
) -> str:
    """``out_of_room``: the token budget or the context is used up, which OpenVINO can report as a
    plain stop (it generates nothing when the prompt fills the context)."""
    if calls:
        return "tool_calls"
    reasons = getattr(res, "finish_reasons", None) or []
    hit = reasons and reasons[0] == ov_genai.GenerationFinishReason.LENGTH
    return "length" if hit or out_of_room else "stop"


def _text_config(model_dir: str) -> dict:
    try:
        cfg = json.loads((Path(model_dir) / "config.json").read_text())
    except (OSError, ValueError):
        return {}
    return cfg.get("text_config") or cfg


def model_context(model_dir: str, cache_gb: Optional[float] = None) -> Optional[int]:
    """Tokens a request may use: the model's ``max_position_embeddings``, capped by what the KV
    cache holds. OpenVINO silently generates nothing for a request the cache cannot fit."""
    cfg = _text_config(model_dir)
    context = cfg.get("max_position_embeddings")
    try:
        # Hybrid models (Qwen3-Next style) keep KV only in their full-attention layers.
        types_ = cfg.get("layer_types") or []
        layers = sum("full" in str(t) for t in types_) or cfg["num_hidden_layers"]
        heads = cfg.get("num_key_value_heads") or cfg["num_attention_heads"]
        head_dim = cfg.get("head_dim") or cfg["hidden_size"] // cfg["num_attention_heads"]
    except (KeyError, TypeError, ZeroDivisionError):
        return context
    if not cache_gb:
        return context
    # ponytail: f16 K+V per token and a 15% margin, measured on Arc B60 (3 GB held ~140k tokens
    # of Ornith 35B; with prefix caching the linear-attention checkpoints take the rest). Use
    # OpenVINO's own figure once it exposes one: openvino.genai#4545.
    capacity = int(cache_gb * 2**30 / (layers * 2 * heads * head_dim * 2) * 0.85)
    return min(context, capacity) if context else capacity


def generation_error(exc: Exception) -> tuple[int, str]:
    """Status and message for a failed generate(). OpenVINO drops a request the cache cannot hold
    and raises about it; that is the caller's prompt being too long, so a 400."""
    message = str(exc)
    if "did not fit in the available cache budget" in message:
        return 400, (
            "The conversation does not fit in the model's KV cache. Shorten it or the tool list."
        )
    return 500, f"Generation failed: {message[:500]}"


def build_app(
    pipe,
    model_id: str,
    context: Optional[int] = None,
) -> FastAPI:
    tok = pipe.get_tokenizer()
    lock = threading.Lock()
    app = FastAPI()

    def config(req: ChatRequest, prompt_tokens: int):
        cfg = ov_genai.GenerationConfig()
        wanted = req.max_completion_tokens or req.max_tokens or 4096
        # Never ask for more than the context has left; the reply then ends with "length".
        cfg.max_new_tokens = min(wanted, context - prompt_tokens) if context else wanted
        temp = 0.6 if req.temperature is None else req.temperature
        cfg.do_sample = temp > 0
        cfg.temperature = max(temp, 1e-4)
        cfg.top_p = req.top_p or 0.95
        cfg.top_k = req.top_k or 20
        return cfg

    def prompt(req: ChatRequest, thinking: bool) -> str:
        return tok.apply_chat_template(
            chat_history(req.messages),
            add_generation_prompt = True,
            tools = tools_for(req) or None,
            extra_context = {"enable_thinking": thinking},
        )

    def tools_for(req: ChatRequest) -> list[dict]:
        return [] if req.tool_choice == "none" else (req.tools or [])

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
        if not req.messages:
            raise HTTPException(400, "'messages' must not be empty.")
        try:
            text_prompt = prompt(req, thinking)
        except Exception as exc:  # the template's raise_exception, e.g. a misplaced system message
            raise HTTPException(400, f"The chat template rejected the messages: {exc}") from exc
        prompt_tokens = tok.encode(text_prompt, add_special_tokens = False).input_ids.shape[-1]
        if context and prompt_tokens >= context:
            raise HTTPException(
                400,
                f"This model's maximum context length is {context} tokens (model limit or KV "
                f"cache size), but the messages use {prompt_tokens}. Shorten the conversation "
                "or the tool list.",
            )
        cfg = config(req, prompt_tokens)

        def finish(res: Any, calls: list) -> str:
            done = usage(res)["completion_tokens"]
            full = context is not None and prompt_tokens + done >= context
            return finish_reason(res, calls, done >= cfg.max_new_tokens or full)

        def usage(res: Any) -> dict:
            metrics = getattr(res, "perf_metrics", None)
            done = metrics.get_num_generated_tokens() if metrics is not None else 0
            return {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": done,
                "total_tokens": prompt_tokens + done,
            }

        rid, created = f"chatcmpl-{uuid.uuid4().hex[:24]}", int(time.time())

        def chunk(
            delta: dict,
            finish: Optional[str] = None,
            **extra,
        ) -> str:
            body = {
                "id": rid,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model_id,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
                **extra,
            }
            return f"data: {json.dumps(body)}\n\n"

        if not req.stream:
            try:
                with lock:
                    res = pipe.generate(text_prompt, generation_config = cfg)
            except Exception as exc:
                raise HTTPException(*generation_error(exc)) from exc
            splitter = ThinkSplitter(thinking)
            parts = splitter.feed(res.texts[0]) + splitter.flush()
            tool_splitter = ToolSplitter(bool(tools_for(req)))
            content = tool_splitter.feed("".join(t for k, t in parts if k == "content"))
            rest, calls = tool_splitter.finish(tools_for(req))
            message = {"role": "assistant", "content": (content + rest) or None}
            if calls:
                message["tool_calls"] = [
                    {k: v for k, v in c.items() if k != "index"} for c in calls
                ]
            reasoning = "".join(t for k, t in parts if k == "reasoning")
            if reasoning:
                message["reasoning_content"] = reasoning
            return {
                "id": rid,
                "object": "chat.completion",
                "created": created,
                "model": model_id,
                "choices": [
                    {
                        "index": 0,
                        "message": message,
                        "finish_reason": finish(res, calls),
                    }
                ],
                "usage": usage(res),
            }

        # Set when the client goes away, so the model stops instead of finishing an unread reply.
        cancelled = threading.Event()

        def streamer(piece: str):
            pieces.put(piece)
            if cancelled.is_set():
                return ov_genai.StreamingStatus.CANCEL
            return ov_genai.StreamingStatus.RUNNING

        pieces: "queue.Queue[Optional[str]]" = queue.Queue()

        def events():
            result: dict = {}

            def run():
                try:
                    with lock:
                        result["res"] = pipe.generate(
                            text_prompt,
                            generation_config = cfg,
                            streamer = streamer,
                        )
                except Exception as exc:  # re-raised to the client below, not lost in the thread
                    result["error"] = exc
                finally:
                    pieces.put(None)

            threading.Thread(target = run, daemon = True).start()
            yield chunk({"role": "assistant"})
            splitter = ThinkSplitter(thinking)
            tool_splitter = ToolSplitter(bool(tools_for(req)))

            def deltas(parts):
                for kind, text in parts:
                    if kind == "reasoning":
                        yield chunk({"reasoning_content": text})
                    elif text := tool_splitter.feed(text):
                        yield chunk({"content": text})

            while (piece := pieces.get()) is not None:
                yield from deltas(splitter.feed(piece))
            if "error" in result:
                status, message = generation_error(result["error"])
                error = {"message": message, "type": "server_error", "code": status}
                yield f"data: {json.dumps({'error': error})}\n\n"
                yield "data: [DONE]\n\n"
                return
            yield from deltas(splitter.flush())
            rest, calls = tool_splitter.finish(tools_for(req))
            if rest:
                yield chunk({"content": rest})
            if calls:
                yield chunk({"tool_calls": calls})
            yield chunk({}, finish(result.get("res"), calls), usage = usage(result.get("res")))
            yield "data: [DONE]\n\n"

        async def guarded():
            try:
                async for event in iterate_in_threadpool(events()):
                    yield event
            finally:
                cancelled.set()

        return StreamingResponse(guarded(), media_type = "text/event-stream")

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
        build_app(pipe, args.model_id, model_context(args.model, sched.cache_size)),
        host = "127.0.0.1",
        port = args.port,
        log_level = "warning",
    )


if __name__ == "__main__":
    main()
