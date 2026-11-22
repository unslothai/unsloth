# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

# TRL's AsyncGRPOTrainer generates through a vLLM server over HTTP. With fast_inference=True the model already owns
# an in-process vLLM engine sharing its base weights, so serve that engine on localhost instead: the rollout process
# is unchanged, no weights are copied, and only the LoRA adapter moves on each sync.

__all__ = [
    "UnslothAsyncVLLMServer",
]

import hmac
import itertools
import json
import logging
import os
import secrets
import threading
import time
import uuid
import weakref
from concurrent.futures import Future
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

# Adapter ids of their own, so they never collide with the ids Unsloth's GRPOTrainer hands out.
_LORA_ID_BASE = 1_000_000
# Process-wide: one engine outlives each trainer's server, and vLLM ignores a re-added id it already holds.
_LORA_IDS = itertools.count(_LORA_ID_BASE)
# Every request is served by the newest adapter (see `_lora_for`), so any number of named versions is servable.
_SERVABLE_LORAS = 1 << 20
_BATCH_WAIT_S = 0.0005


def _inplace_lora():
    # One slot overwritten per sync: with max_loras=1 a new id per sync stalls new requests behind stragglers.
    return os.environ.get("UNSLOTH_ASYNC_GRPO_INPLACE_LORA", "1") != "0"


def _tensor_sync():
    # Kill switch: 0 hands each synced adapter to vLLM through TRL's safetensors files and a load from disk.
    return os.environ.get("UNSLOTH_ASYNC_GRPO_TENSOR_SYNC", "1") != "0"


def _tensor_request_builder():
    # Zoo's load_lora(load_tensors = True) body without the merged-rollout branch, which folds the adapter into
    # the shared base around a generate call and cannot serve a continuously stepping engine.
    try:
        from unsloth_zoo.vllm_utils import _build_lora_request
    except Exception:
        return None
    return _build_lora_request


def _host_snapshot(tensors, dtype = None):
    """Copies `tensors` to host memory with one device-to-host copy per dtype, so the version keeps its values
    after the optimizer steps the live weights. Returns name -> view of a flat host buffer."""
    import torch

    groups = {}
    for name, tensor in tensors.items():
        tensor = tensor.detach()
        if dtype is not None and tensor.is_floating_point():
            tensor = tensor.to(dtype)
        groups.setdefault((tensor.device, tensor.dtype), []).append((name, tensor))
    out = {}
    for (device, _), items in groups.items():
        flat = torch.cat([tensor.reshape(-1) for _, tensor in items])
        flat = flat.to("cpu") if device.type != "cpu" else flat.clone()
        offset = 0
        for name, tensor in items:
            size = tensor.numel()
            out[name] = flat[offset : offset + size].view(tensor.shape)
            offset += size
    return {name: out[name] for name in tensors}


def _pause_on_sync():
    # Kill switch: 1 stops decoding from TRL's /pause to its /resume, as `vllm serve` does.
    return os.environ.get("UNSLOTH_ASYNC_GRPO_PAUSE_ON_SYNC", "0") == "1"


def _logprob(value):
    # Clamp -inf like vLLM's OpenAI server (plain JSON).
    return max(float(getattr(value, "logprob", value)), -9999.0)


def _sampled_logprob(logprobs, position, token_id):
    # vLLM always records the sampled token; any other entry is another token's logprob and would corrupt the
    # importance ratio TRL builds from these, so refuse instead of substituting it.
    entry = logprobs[position]
    if token_id not in entry:
        raise ValueError(
            f"Unsloth: vLLM returned no logprob for sampled token {token_id} at position {position}."
        )
    return _logprob(entry[token_id])


def _top_logprobs(entry, num_top_logprobs):
    # Keyed "token_id:<id>" as TRL's `_by_token_id` parses.
    return {
        f"token_id:{token_id}": _logprob(value)
        for i, (token_id, value) in enumerate(entry.items())
        if i <= num_top_logprobs
    }


def _scheduler(llm_engine):
    # In-process V1 engine (Unsloth sets VLLM_ENABLE_V1_MULTIPROCESSING=0): LLMEngine -> InprocClient -> EngineCore.
    core = getattr(llm_engine, "engine_core", None)
    core = getattr(core, "engine_core", core)
    return getattr(core, "scheduler", None)


def _track_preemptions(scheduler):
    # vllm:num_preemptions_total needs log_stats (off in-process): bank per-request counts on free so the total
    # stays monotonic. Once per scheduler, since the engine outlives each server.
    if scheduler is None or hasattr(scheduler, "_unsloth_preemptions_freed"):
        return
    free_blocks = getattr(scheduler, "_free_blocks", None)
    if free_blocks is None or not isinstance(getattr(scheduler, "requests", None), dict):
        return

    def _free_blocks(request, *args, **kwargs):
        result = free_blocks(request, *args, **kwargs)
        scheduler._unsloth_preemptions_freed += getattr(request, "num_preemptions", 0) or 0
        return result

    scheduler._unsloth_preemptions_freed = 0
    scheduler._free_blocks = _free_blocks


def _preemptions(scheduler):
    freed = getattr(scheduler, "_unsloth_preemptions_freed", None)
    if freed is None:
        return None
    live = sum(
        getattr(request, "num_preemptions", 0) or 0 for request in scheduler.requests.values()
    )
    return freed + live


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
        self.pause_on_sync = _pause_on_sync()
        self.syncing = False
        self.stopping = False
        self.loras = {}
        self.newest = None
        self.lora_holds = {}
        self.inplace = _inplace_lora()
        self.tensor_sync = _tensor_sync()
        # (path, LoRARequest) snapshotted by `stage` for the next load of that path.
        self.staged = None
        self.slot_id = None
        self.slot_version = 0
        self.aborted = None
        self.stats = {
            "requests": 0,
            "tokens": 0,
            "steps": 0,
            "step_s": 0.0,
            "max_batch": 0,
            "syncs": 0,
            "sync_steps": 0,
            "tensor_syncs": 0,
        }
        try:
            _track_preemptions(_scheduler(self.engine))
        except Exception as error:
            logger.info(f"Unsloth: vLLM preemptions will not be reported: {error}")
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
            if self.aborted is not None:
                # Requests taken from the queue alongside an abort never reach the engine.
                for item in pending:
                    item[-1].set_exception(self.aborted)
                pending, self.aborted = [], None
            for item in pending:
                self._admit(*item)
            if self.paused or not self.active:
                continue
            started = time.perf_counter()
            batch = len(self.active)
            try:
                outputs = self.engine.step()
            except BaseException as error:
                self._fail_active(error)
                continue
            with self.cv:
                # Releases the GIL so the HTTP thread can hand over requests that arrived during the step.
                if not (self.pending or self.commands or self.stopping):
                    self.cv.wait(_BATCH_WAIT_S)
            stats = self.stats
            stats["steps"] += 1
            stats["step_s"] += time.perf_counter() - started
            stats["max_batch"] = max(stats["max_batch"], batch)
            if self.syncing:
                stats["sync_steps"] += 1
            for output in outputs:
                if output.finished and output.request_id in self.active:
                    future, lora_id = self.active.pop(output.request_id)
                    self._release(lora_id)
                    stats["requests"] += 1
                    stats["tokens"] += len(output.outputs[0].token_ids)
                    future.set_result(output)
        self._fail_active(RuntimeError("Unsloth: the in-process vLLM server stopped."))
        with self.cv:
            pending, self.pending = self.pending, []
        for item in pending:
            item[-1].set_exception(RuntimeError("Unsloth: the in-process vLLM server stopped."))
        for name in list(self.loras):
            self.remove_adapter(name)

    def _admit(self, prompt_ids, params, model, future):
        request_id = uuid.uuid4().hex
        lora = self._lora_for(model)
        prompt = {"prompt_token_ids": prompt_ids}
        if lora is not None and lora.lora_int_id == self.slot_id:
            # Prefix cache keys on adapter name; salt by version so old-weight KV is never reused.
            prompt["cache_salt"] = f"unsloth-lora-{self.slot_id}-v{self.slot_version}"
        try:
            sampling = self.sampling_params(**{**self._default_kwargs(), **params})
            self.engine.add_request(request_id, prompt, sampling, lora_request = lora)
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

    # Called from the trainer thread while TRL saves the adapter; the engine keeps stepping.
    def stage_id(self):
        if not self.inplace:
            return next(_LORA_IDS)
        if self.slot_id is None:
            self.slot_id = next(_LORA_IDS)
        return self.slot_id

    def stage(self, path, lora):
        self.staged = (path, lora)

    def _take_staged(self, path):
        staged, self.staged = self.staged, None
        if staged is None or staged[0] != path:
            return None
        self.stats["tensor_syncs"] += 1
        return staged[1]

    def _load(self, path, lora_id):
        lora = self._take_staged(path)
        if lora is not None and lora.lora_int_id == lora_id:
            return lora
        return self.load_lora(path, lora_request_id = lora_id)

    # Commands (run on the engine thread through `call`).
    def add_adapter(self, name, path):
        if self.inplace:
            return self._overwrite_slot(name, path)
        staged = self.staged
        lora_id = (
            staged[1].lora_int_id if staged is not None and staged[0] == path else next(_LORA_IDS)
        )
        lora = self._load(path, lora_id)
        # Load now, while the trainer waits on this sync, not lazily on the first request.
        self.engine.add_lora(lora)
        self.loras[name] = lora
        self.newest = name
        self.lora_holds.setdefault(lora.lora_int_id, [False, 0])
        return lora.lora_int_id

    def _reload_slot(self, lora, previous):
        # Manager skips held ids, so drop and re-add between engine steps (no step sees it empty).
        loaded = self.slot_id in self.lora_holds
        if loaded:
            self.engine.remove_lora(self.slot_id)
        try:
            self.engine.add_lora(lora)
        except BaseException:
            if loaded and previous is not None:
                # Requests in flight still name the slot: put the previous version back.
                self.engine.add_lora(previous)
            raise
        self.slot_version += 1

    def _overwrite_slot(self, name, path):
        if self.slot_id is None:
            self.slot_id = next(_LORA_IDS)
        lora = self._load(path, self.slot_id)
        self._reload_slot(lora, self.loras.get(self.newest) if self.newest is not None else None)
        self.loras[name] = lora
        self.newest = name
        self.lora_holds.setdefault(self.slot_id, [False, 0])[0] = False
        return self.slot_id

    def remove_adapter(self, name):
        lora = self.loras.pop(name, None)
        if lora is None:
            return False
        was_newest = self.newest == name
        if was_newest:
            self.newest = next(reversed(self.loras), None)
        if lora.lora_int_id == self.slot_id and self.loras:
            # One slot serves every name; restore the newest remaining version.
            if was_newest:
                self._reload_slot(self.loras[self.newest], lora)
            return True
        state = self.lora_holds.setdefault(lora.lora_int_id, [False, 0])
        state[0] = True
        self._maybe_remove(lora.lora_int_id)
        return True

    def load(self):
        scheduler = _scheduler(self.engine)
        waiting = len(self.pending)
        usage = running = preemptions = None
        if scheduler is not None:
            try:
                usage = float(scheduler.kv_cache_manager.usage)
                running = len(scheduler.running)
                waiting += len(scheduler.waiting) + len(getattr(scheduler, "skipped_waiting", ()))
            except Exception:
                usage = running = None
            try:
                preemptions = _preemptions(scheduler)
            except Exception:
                preemptions = None
        if running is None:
            running = len(self.active)
        return {
            "usage": usage,
            "running": running,
            "waiting": waiting,
            "preemptions": preemptions,
            "stats": dict(self.stats),
        }

    def set_paused(self, paused):
        self.paused = paused
        return True

    def begin_sync(self):
        # TRL pauses before saving the adapter and resumes after loading it. The load is already an engine-thread
        # command between steps with rollback (`_reload_slot`), so decoding goes on through the save and barrier on
        # the previous adapter. TRL stamps a group's version at dispatch and bumps it only after the load, so those
        # tokens carry the version that sampled them.
        self.syncing = True
        self.stats["syncs"] += 1
        if self.pause_on_sync:
            self.paused = True
        return True

    def end_sync(self):
        self.syncing = False
        self.paused = False
        return True

    def abort_all(self):
        error = RuntimeError("Unsloth: generation aborted when training ended.")
        with self.cv:
            pending, self.pending = self.pending, []
        for item in pending:
            item[-1].set_exception(error)
        self.aborted = error
        self._fail_active(error)
        return True


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "UnslothAsyncVLLM"

    def log_message(self, *args):
        pass

    def _send(self, code, payload):
        body = json.dumps(payload).encode()
        try:
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            # The rollout process exits with requests in flight when training ends.
            self.close_connection = True

    def _route(self):
        # Any local process can reach 127.0.0.1, so the URL carries a per-run secret: "/<token>/<endpoint>".
        _, _, rest = urlparse(self.path).path.partition("/")
        token, slash, path = rest.partition("/")
        if not slash or not hmac.compare_digest(token.encode(), self.server.owner._token.encode()):
            return None
        return "/" + path

    def do_GET(self):
        path = self._route()
        if path is None:
            return self._send(404, {"error": "not found"})
        if path == "/metrics":
            return self._send_text(*self.server.owner.metrics())
        self._send(*self.server.owner.get(path))

    def _send_text(self, code, text):
        body = text.encode()
        try:
            self.send_response(code)
            self.send_header("Content-Type", "text/plain; version=0.0.4")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            self.close_connection = True

    def do_POST(self):
        owner = self.server.owner
        length = int(self.headers.get("Content-Length") or 0)
        path = self._route()
        if path is None:
            self.close_connection = True
            return self._send(404, {"error": "not found"})
        # TRL sends JSON or no body; a browser can POST text/plain cross-origin without a preflight.
        content_type = self.headers.get("Content-Type")
        if content_type and content_type.split(";")[0].strip().lower() != "application/json":
            self.close_connection = True
            return self._send(415, {"error": "Content-Type must be application/json"})
        try:
            body = json.loads(self.rfile.read(length) or b"{}") if length else {}
        except ValueError as error:
            return self._send(400, {"error": f"bad JSON: {error}"})
        if path == "/v1/completions":
            try:
                output = owner.start(body).result()
                return self._send(200, owner.finish(output, body.get("logprobs")))
            except Exception as error:
                return self._send(400, {"error": f"{type(error).__name__}: {error}"})
        self._send(*owner.post(path, body))


class _Server(ThreadingHTTPServer):
    # Backlog of 5 is too small: each in-flight completion is its own connection, reset on overflow.
    request_queue_size = 1024
    daemon_threads = True


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
        self.max_num_seqs = self.vllm_config.scheduler_config.max_num_seqs
        self.model_name = model_name or getattr(
            getattr(model, "config", None), "_name_or_path", "model"
        )
        load_lora = getattr(model, "load_lora", None)
        if load_lora is None:
            raise ValueError(
                "Unsloth: AsyncGRPOTrainer with fast_inference=True needs a LoRA model from get_peft_model."
            )
        _keep_awake(llm)
        try:
            self._model = weakref.ref(model)
        except TypeError:
            self._model = lambda: None
        self._tensor_dir = None
        self.engine = _Engine(llm, load_lora, sampling_params)
        self.stopped = False
        self._token = secrets.token_urlsafe(32)
        self.url = f"http://127.0.0.1:{self._serve()}/{self._token}"

    def _serve(self):
        httpd = _Server(("127.0.0.1", 0), _Handler)
        httpd.owner = self
        self._thread = threading.Thread(
            target = httpd.serve_forever, name = "unsloth-async-grpo-http", daemon = True
        )
        self._thread.start()

        def shutdown():
            httpd.shutdown()
            httpd.server_close()

        self._shutdown = shutdown
        return httpd.server_address[1]

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

    def get(self, path):
        if path == "/health":
            return 200, {"status": "ok"}
        if path == "/v1/models":
            model = {"id": self.model_name, "object": "model", "max_model_len": self.max_model_len}
            return 200, {"object": "list", "data": [model]}
        if path == "/server_info":
            return 200, {"vllm_config": self.server_info()}
        if path == "/get_world_size":
            return 200, {"world_size": 1}
        return 404, {"error": f"unknown path {path}"}

    def kv_cache_size_tokens(self):
        cache_config = getattr(self.vllm_config, "cache_config", None)
        tokens = getattr(cache_config, "kv_cache_size_tokens", None)
        if tokens is None:
            blocks = getattr(cache_config, "num_gpu_blocks", None)
            block_size = getattr(cache_config, "block_size", None)
            if blocks and block_size:
                tokens = blocks * block_size
        return tokens

    def metrics(self):
        # Prometheus lines TRL >= 1.17 parses for its KV cap.
        try:
            load = self.engine.call(self.engine.load)
        except Exception as error:
            return 503, f"# {type(error).__name__}: {error}\n"
        name = str(self.model_name).replace("\\", "").replace('"', "")
        labels = f'engine="0",model_name="{name}"'
        lines = []
        tokens = self.kv_cache_size_tokens()
        if tokens:
            lines.append(
                f'vllm:cache_config_info{{{labels},kv_cache_size_tokens="{int(tokens)}"}} 1.0'
            )
        if load["usage"] is not None:
            lines.append(f"vllm:kv_cache_usage_perc{{{labels}}} {load['usage']}")
        lines.append(f"vllm:num_requests_running{{{labels}}} {float(load['running'])}")
        lines.append(f"vllm:num_requests_waiting{{{labels}}} {float(load['waiting'])}")
        if load["preemptions"] is not None:
            lines.append(f"vllm:num_preemptions_total{{{labels}}} {float(load['preemptions'])}")
        # Host-side engine counters (no device sync); TRL ignores them.
        stats = load["stats"]
        for name, key in (
            ("requests_finished_total", "requests"),
            ("generation_tokens_total", "tokens"),
            ("engine_steps_total", "steps"),
            ("engine_step_seconds_total", "step_s"),
            ("engine_max_batch", "max_batch"),
            ("syncs_total", "syncs"),
            ("sync_steps_total", "sync_steps"),
        ):
            lines.append(f"unsloth:async_grpo_{name}{{{labels}}} {float(stats[key])}")
        return 200, "\n".join(lines) + "\n"

    def post(self, path, body):
        engine = self.engine
        commands = {
            "/v1/load_lora_adapter": lambda: engine.add_adapter(
                body["lora_name"], body["lora_path"]
            ),
            "/v1/unload_lora_adapter": lambda: engine.remove_adapter(body.get("lora_name")),
            "/pause": engine.begin_sync,
            "/resume": engine.end_sync,
        }
        if path not in commands:
            return 404, {"error": f"unknown path {path}"}
        try:
            engine.call(commands[path])
        except Exception as error:
            return 400, {"error": f"{type(error).__name__}: {error}"}
        return 200, {"status": "ok"}

    def start(self, body):
        prompt = body["prompt"]
        if not (isinstance(prompt, list) and all(isinstance(x, int) for x in prompt)):
            raise ValueError("Unsloth: the in-process vLLM server takes token-id prompts only.")
        params = {"n": 1, "max_tokens": body.get("max_tokens"), "detokenize": False}
        for name in ("temperature", "top_p", "top_k", "min_p", "repetition_penalty", "seed"):
            if body.get(name) is not None:
                params[name] = body[name]
        if body.get("logprobs") is not None:
            params["logprobs"] = int(body["logprobs"])
            self._allow_logprobs(params["logprobs"])
        return self.engine.submit(prompt, params, body.get("model"))

    def _allow_logprobs(self, num_logprobs):
        # Engine is built with max_logprobs=0; TRL >= 1.17 asks for top-k (checked only at add_request).
        model_config = getattr(self.vllm_config, "model_config", None)
        limit = getattr(model_config, "max_logprobs", None)
        if limit is not None and limit != -1 and num_logprobs > limit:
            model_config.max_logprobs = num_logprobs

    @staticmethod
    def finish(output, num_top_logprobs = None):
        completion = output.outputs[0]
        token_ids = list(completion.token_ids)
        logprobs = None
        if completion.logprobs is not None:
            logprobs = {
                "token_logprobs": [
                    _sampled_logprob(completion.logprobs, i, t) for i, t in enumerate(token_ids)
                ]
            }
            if num_top_logprobs is not None:
                # TRL >= 1.17 reads them for score centering (`num_sampler_logprobs`).
                logprobs["top_logprobs"] = [
                    _top_logprobs(completion.logprobs[i], int(num_top_logprobs))
                    for i in range(len(token_ids))
                ]
        choice = {
            "index": 0,
            "text": "",
            "token_ids": token_ids,
            "finish_reason": completion.finish_reason,
            "logprobs": logprobs,
        }
        return {"id": output.request_id, "object": "text_completion", "choices": [choice]}

    def stage_adapter(self, model, adapter_name, dest_dir):
        """Snapshot the adapter TRL is about to save to `dest_dir` for its next `/v1/load_lora_adapter`, so the
        sync skips the safetensors round trip (PEFT's save also probes the Hub for the base config each time).
        False: nothing staged, TRL saves the files as usual."""
        engine = self.engine
        if self.stopped or not engine.tensor_sync or model is not self._model():
            return False
        if adapter_name != "default" or set(getattr(model, "peft_config", None) or ()) != {
            "default"
        }:
            return False
        build = _tensor_request_builder()
        if build is None:
            return False
        if self._tensor_dir is None:
            # Holds only adapter_config.json, written once; dot-prefixed so TRL's adapter namespace ignores it.
            self._tensor_dir = os.path.join(os.path.dirname(dest_dir), ".unsloth-live-adapter")
        import torch

        dtype = getattr(getattr(self.vllm_config, "lora_config", None), "lora_dtype", None)
        try:
            lora = build(model, self._tensor_dir, True, engine.stage_id())
            if lora is None or not getattr(lora, "lora_tensors", None):
                return False
            # What a load from TRL's files gives vLLM: the values at save time, cast to the LoRA dtype.
            lora.lora_tensors = _host_snapshot(
                lora.lora_tensors, dtype if isinstance(dtype, torch.dtype) else None
            )
        except Exception as error:
            logger.info(f"Unsloth: syncing the adapter through files instead of tensors: {error}")
            engine.tensor_sync = False
            return False
        engine.stage(dest_dir, lora)
        return True

    def abort_all(self):
        if not self.stopped:
            self.engine.call(self.engine.abort_all)

    def stop(self):
        if self.stopped:
            return
        self.stopped = True
        self.engine.stop()
        self._shutdown()


def _keep_awake(llm):
    # Async generation overlaps training, so a standby (sleep mode) engine has to stay awake for the whole run.
    try:
        if llm.llm_engine.is_sleeping():
            llm.wake_up()
    except Exception as error:
        logger.info(f"Unsloth: could not check vLLM sleep state: {error}")
