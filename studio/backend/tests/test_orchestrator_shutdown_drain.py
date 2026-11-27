# SPDX-License-Identifier: AGPL-3.0-only

import ast
import logging
import multiprocessing as mp
from pathlib import Path

from core.inference import orchestrator as orch_mod
from core.inference.orchestrator import InferenceOrchestrator


def _queued_generate_worker(cmd_queue, resp_queue, cancel_event, drain_event, ready, ran):
    worker_path = Path(orch_mod.__file__).with_name("worker.py")
    tree = ast.parse(worker_path.read_text(encoding = "utf-8"))
    entrypoint = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "run_inference_process"
    )
    command_loop = next(node for node in reversed(entrypoint.body) if isinstance(node, ast.While))
    branch = next(
        node
        for node in ast.walk(command_loop)
        if isinstance(node, ast.If) and ast.unparse(node.test) == "cmd_type == 'generate'"
    )
    skip = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_drain_skip_generate"
    )
    generate = ast.parse(
        "def generate(cmd, resp_queue, drain_event, cancel_event, backend):\n    pass"
    ).body[0]
    generate.body = [
        ast.For(
            target = ast.Name(id = "_once", ctx = ast.Store()),
            iter = ast.Tuple(elts = [ast.Constant(value = None)], ctx = ast.Load()),
            body = branch.body,
            orelse = [],
        )
    ]

    def dispatch(backend, cmd, replies, cancel):
        ran.set()
        cancel.wait(30)

    namespace = {
        "Any": object,
        "logger": logging.getLogger(__name__),
        "_send_response": lambda replies, response: replies.put(response),
        "_dispatch_generate": dispatch,
    }
    module = ast.fix_missing_locations(ast.Module(body = [skip, generate], type_ignores = []))
    exec(compile(module, str(worker_path), "exec"), namespace)
    ready.set()
    if not cancel_event.wait(30):
        raise RuntimeError("Shutdown never signalled cancellation")
    while True:
        cmd = cmd_queue.get(timeout = 30)
        if cmd["type"] == "shutdown":
            return
        namespace["generate"](cmd, resp_queue, drain_event, cancel_event, None)


def test_shutdown_skips_queued_generate_before_it_can_clear_cancel(monkeypatch):
    import utils.process_lifetime as lifetime

    ctx = mp.get_context("spawn")
    o = InferenceOrchestrator()
    o._cmd_queue, o._resp_queue = ctx.Queue(), ctx.Queue()
    o._cancel_event, o._drain_event = ctx.Event(), ctx.Event()
    ready, ran = ctx.Event(), ctx.Event()
    proc = ctx.Process(
        target = _queued_generate_worker,
        args = (o._cmd_queue, o._resp_queue, o._cancel_event, o._drain_event, ready, ran),
        daemon = True,
    )
    o._proc = proc
    forced = []

    def terminate(pid, timeout = 5):
        assert pid == proc.pid
        forced.append(pid)
        proc.terminate()
        proc.join(timeout)

    monkeypatch.setattr(lifetime, "terminate_pid", terminate)
    proc.start()
    try:
        o._cmd_queue.put({"type": "generate", "request_id": "queued"})
        assert ready.wait(30)
        assert o._shutdown_subprocess(timeout = 0.3) is True
        assert not forced, "a queued generate cleared cancellation and forced worker termination"
        assert not ran.is_set()
        assert proc.exitcode == 0
    finally:
        if proc.is_alive():
            proc.terminate()
            proc.join(5)
        o._proc = None
        proc.close()
