# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

# TRL's AsyncGRPOTrainer generates through a vLLM server over HTTP. With fast_inference=True the model already owns
# an in-process vLLM engine sharing its base weights, so serve that engine on localhost instead: the rollout process
# is unchanged, no weights are copied, and only the LoRA adapter moves on each sync.

__all__ = [
    "UnslothAsyncVLLMServer",
]

import json
import logging
import threading
import uuid
from concurrent.futures import Future
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

# Adapter ids of their own, so they never collide with the ids Unsloth's GRPOTrainer hands out.
_LORA_ID_BASE = 1_000_000
# Every request is served by the newest adapter (see `_lora_for`), so any number of named versions is servable.
_SERVABLE_LORAS = 1 << 20


def _sampled_logprob(logprobs, position, token_id):
    entry = logprobs[position]
    value = entry[token_id] if token_id in entry else next(iter(entry.values()))
    return float(getattr(value, "logprob", value))


class _Engine:
    """Owns the vLLM engine on one thread; every engine call happens there."""

    def __init__(
        self,
        llm,
        load_lora,
        sampling_params = None,
    ):
        if sampling_params is None:
            from vllm import SamplingParams as sampling_params
        self.sampling_params = sampling_params
        self.llm = llm
        self.engine = llm.llm_engine
        self.load_lora = load_lora
        self.defaults = llm.get_default_sampling_params()
        self.cv = threading.Condition()
        self.pending = []
        self.commands = []
        self.active = {}
        self.paused = False
        self.stopping = False
        self.loras = {}
        self.newest = None
        self.lora_holds = {}
        self.next_lora_id = _LORA_ID_BASE
        self.thread = threading.Thread(
            target = self._run, name = "unsloth-async-grpo-vllm", daemon = True
        )
        self.thread.start()

    # Called from HTTP threads.
    def submit(self, prompt_ids, params, model):
        future = Future()
        with self.cv:
            if self.stopping:
                raise RuntimeError("Unsloth: the in-process vLLM server is stopped.")
            self.pending.append((prompt_ids, params, model, future))
            self.cv.notify()
        return future

    def call(self, fn):
        future = Future()
        with self.cv:
            if self.stopping:
                raise RuntimeError("Unsloth: the in-process vLLM server is stopped.")
            self.commands.append((fn, future))
            self.cv.notify()
        return future.result()

    def stop(self):
        with self.cv:
            self.stopping = True
            self.cv.notify()
        if threading.current_thread() is not self.thread:
            self.thread.join(timeout = 60)

    # Engine thread only.
    def _run(self):
        while True:
            with self.cv:
                while not (
                    self.stopping
                    or self.commands
                    or (not self.paused and (self.pending or self.active))
                ):
                    self.cv.wait()
                if self.stopping:
                    break
                commands, self.commands = self.commands, []
                pending = []
                if not self.paused:
                    pending, self.pending = self.pending, []
            for fn, future in commands:
                try:
                    future.set_result(fn())
                except BaseException as error:
                    future.set_exception(error)
            for item in pending:
                self._admit(*item)
            if self.paused or not self.active:
                continue
            try:
                outputs = self.engine.step()
            except BaseException as error:
                self._fail_active(error)
                continue
            for output in outputs:
                if output.finished and output.request_id in self.active:
                    future, lora_id = self.active.pop(output.request_id)
                    self._release(lora_id)
                    future.set_result(output)
        self._fail_active(RuntimeError("Unsloth: the in-process vLLM server stopped."))
        with self.cv:
            pending, self.pending = self.pending, []
        for item in pending:
            item[-1].set_exception(RuntimeError("Unsloth: the in-process vLLM server stopped."))

    def _admit(self, prompt_ids, params, model, future):
        request_id = uuid.uuid4().hex
        lora = self._lora_for(model)
        try:
            sampling = self.sampling_params(**{**self._default_kwargs(), **params})
            self.engine.add_request(
                request_id, {"prompt_token_ids": prompt_ids}, sampling, lora_request = lora
            )
        except BaseException as error:
            future.set_exception(error)
            return
        lora_id = lora.lora_int_id if lora is not None else None
        if lora_id is not None:
            self.hold_lora(lora_id)
        self.active[request_id] = (future, lora_id)

    def _default_kwargs(self):
        # The generation config's sampling defaults, as `vllm serve` applies them to fields a request leaves unset.
        names = ("temperature", "top_p", "top_k", "min_p", "repetition_penalty")
        return {
            name: getattr(self.defaults, name) for name in names if hasattr(self.defaults, name)
        }

    def _lora_for(self, model):
        # Serving the newest adapter keeps one adapter active (Unsloth engines hold max_loras=1) and is never staler
        # than the version the rollout named; the sampled logprobs are the serving policy's own.
        if self.newest is None:
            return None
        return self.loras[self.newest]

    def hold_lora(self, lora_id):
        self.lora_holds.setdefault(lora_id, [False, 0])[1] += 1

    def _release(self, lora_id):
        if lora_id is None or lora_id not in self.lora_holds:
            return
        state = self.lora_holds[lora_id]
        state[1] -= 1
        self._maybe_remove(lora_id)

    def _maybe_remove(self, lora_id):
        state = self.lora_holds.get(lora_id)
        if state is None or not state[0] or state[1] > 0:
            return
        del self.lora_holds[lora_id]
        try:
            self.engine.remove_lora(lora_id)
        except Exception as error:
            logger.info(f"Unsloth: could not remove LoRA {lora_id} from vLLM: {error}")

    def _fail_active(self, error):
        active, self.active = self.active, {}
        if active:
            try:
                self.engine.abort_request(list(active))
            except Exception:
                pass
        for future, lora_id in active.values():
            if not future.done():
                future.set_exception(error)
            if lora_id is not None and lora_id in self.lora_holds:
                self.lora_holds[lora_id][1] -= 1
                self._maybe_remove(lora_id)

    # Commands (run on the engine thread through `call`).
    def add_adapter(self, name, path):
        lora_id = self.next_lora_id
        self.next_lora_id += 1
        lora = self.load_lora(path, lora_request_id = lora_id)
        # Load now, while the trainer waits on this sync, not lazily on the first request.
        self.engine.add_lora(lora)
        self.loras[name] = lora
        self.newest = name
        self.lora_holds.setdefault(lora.lora_int_id, [False, 0])
        return lora.lora_int_id

    def remove_adapter(self, name):
        lora = self.loras.pop(name, None)
        if lora is None:
            return False
        if self.newest == name:
            self.newest = next(reversed(self.loras), None)
        state = self.lora_holds.setdefault(lora.lora_int_id, [False, 0])
        state[0] = True
        self._maybe_remove(lora.lora_int_id)
        return True

    def set_paused(self, paused):
        self.paused = paused
        return True

    def abort_all(self):
        self._fail_active(RuntimeError("Unsloth: generation aborted when training ended."))
        return True


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "UnslothAsyncVLLM"

    def log_message(self, *args):
        pass

    def _send(self, code, payload):
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _body(self):
        length = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(length) or b"{}") if length else {}

    def do_GET(self):
        owner = self.server.owner
        path = urlparse(self.path).path
        if path == "/health":
            return self._send(200, {"status": "ok"})
        if path == "/v1/models":
            return self._send(
                200,
                {
                    "object": "list",
                    "data": [
                        {
                            "id": owner.model_name,
                            "object": "model",
                            "max_model_len": owner.max_model_len,
                        }
                    ],
                },
            )
        if path == "/server_info":
            return self._send(200, {"vllm_config": owner.server_info()})
        if path == "/get_world_size":
            return self._send(200, {"world_size": 1})
        return self._send(404, {"error": f"unknown path {path}"})

    def do_POST(self):
        owner = self.server.owner
        path = urlparse(self.path).path
        try:
            body = self._body()
            if path == "/v1/completions":
                return self._send(200, owner.complete(body))
            if path == "/v1/load_lora_adapter":
                owner.engine.call(
                    lambda: owner.engine.add_adapter(body["lora_name"], body["lora_path"])
                )
                return self._send(200, {"status": "ok"})
            if path == "/v1/unload_lora_adapter":
                owner.engine.call(lambda: owner.engine.remove_adapter(body.get("lora_name")))
                return self._send(200, {"status": "ok"})
            if path == "/pause":
                owner.engine.call(lambda: owner.engine.set_paused(True))
                return self._send(200, {"status": "ok"})
            if path == "/resume":
                owner.engine.call(lambda: owner.engine.set_paused(False))
                return self._send(200, {"status": "ok"})
        except Exception as error:
            return self._send(400, {"error": f"{type(error).__name__}: {error}"})
        return self._send(404, {"error": f"unknown path {path}"})


class UnslothAsyncVLLMServer:
    """Serves `model.vllm_engine` to TRL's AsyncGRPOTrainer on 127.0.0.1."""

    def __init__(
        self,
        model,
        model_name = None,
        sampling_params = None,
    ):
        llm = model.vllm_engine
        self.vllm_config = llm.llm_engine.vllm_config
        self.max_model_len = self.vllm_config.model_config.max_model_len
        self.model_name = model_name or getattr(
            getattr(model, "config", None), "_name_or_path", "model"
        )
        load_lora = getattr(model, "load_lora", None)
        if load_lora is None:
            raise ValueError(
                "Unsloth: AsyncGRPOTrainer with fast_inference=True needs a LoRA model from get_peft_model."
            )
        _keep_awake(llm)
        self.engine = _Engine(llm, load_lora, sampling_params)
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        self.httpd.daemon_threads = True
        self.httpd.owner = self
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        self._thread = threading.Thread(
            target = self.httpd.serve_forever,
            name = "unsloth-async-grpo-http",
            daemon = True,
        )
        self._thread.start()
        self.stopped = False

    def server_info(self):
        lora_config = self.vllm_config.lora_config
        return {
            "model_config": {"dtype": str(self.vllm_config.model_config.dtype)},
            "lora_config": None
            if lora_config is None
            else {
                "max_lora_rank": lora_config.max_lora_rank,
                "max_loras": _SERVABLE_LORAS,
            },
            "parallel_config": {"data_parallel_size": 1},
        }

    def complete(self, body):
        prompt = body["prompt"]
        if not (isinstance(prompt, list) and all(isinstance(x, int) for x in prompt)):
            raise ValueError("Unsloth: the in-process vLLM server takes token-id prompts only.")
        params = {"n": 1, "max_tokens": body.get("max_tokens"), "detokenize": False}
        for name in ("temperature", "top_p", "top_k", "min_p", "repetition_penalty", "seed"):
            if body.get(name) is not None:
                params[name] = body[name]
        want_logprobs = body.get("logprobs") is not None
        if want_logprobs:
            params["logprobs"] = int(body["logprobs"])
        output = self.engine.submit(prompt, params, body.get("model")).result()
        completion = output.outputs[0]
        token_ids = list(completion.token_ids)
        choice = {
            "index": 0,
            "text": "",
            "token_ids": token_ids,
            "finish_reason": completion.finish_reason,
            "logprobs": None,
        }
        if want_logprobs:
            choice["logprobs"] = {
                "token_logprobs": [
                    _sampled_logprob(completion.logprobs, i, t) for i, t in enumerate(token_ids)
                ]
            }
        return {
            "id": output.request_id,
            "object": "text_completion",
            "model": body.get("model"),
            "choices": [choice],
        }

    def abort_all(self):
        if not self.stopped:
            self.engine.call(self.engine.abort_all)

    def stop(self):
        if self.stopped:
            return
        self.stopped = True
        self.engine.stop()
        self.httpd.shutdown()
        self.httpd.server_close()


def _keep_awake(llm):
    # Async generation overlaps training, so a standby (sleep mode) engine has to stay awake for the whole run.
    try:
        if llm.llm_engine.is_sleeping():
            llm.wake_up()
    except Exception as error:
        logger.info(f"Unsloth: could not check vLLM sleep state: {error}")
