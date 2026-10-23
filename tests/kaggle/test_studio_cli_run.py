# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Residency is judged per process, since a shared device total cannot attribute co-tenant frees."""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PAYLOAD = ROOT / "tests" / "kaggle" / "studio_gpu" / "run_studio_gpu.py"
SRC = PAYLOAD.read_text(encoding = "utf-8")


def _func(name: str) -> ast.FunctionDef:
    for cls in ast.walk(ast.parse(SRC)):
        if not isinstance(cls, ast.ClassDef):
            continue
        for node in cls.body:
            if isinstance(node, ast.FunctionDef) and node.name == name:
                return node
    raise AssertionError(f"no method named {name!r}")


def _body(name: str = "assert_cli_run") -> str:
    return ast.get_source_segment(SRC, _func(name)) or ""


def test_the_assertion_exists_and_is_driven_from_the_run():
    assert _body()
    assert "self.assert_cli_run()" in _body("execute")


def test_it_runs_after_the_ui_phase_has_stopped_the_server():
    """The CLI launch must follow the UI phase, because two backends on one studio home is unsupported."""
    body = _body("execute")
    ui_at = body.index("self.assert_chat_ui()")
    cli_at = body.index("self.assert_cli_run()")
    assert ui_at < cli_at, "the CLI launch must come after the UI phase stops the server"


def test_it_uses_its_own_port():
    """The first server's port may still be in TIME_WAIT, and a bind failure
    there would read as a broken CLI."""
    assert "port = self.args.port + 1" in _body()


def test_no_public_url_is_ever_opened_from_ci():
    """--secure implies a Cloudflare quick tunnel, which publishes this server
    to the internet from a CI kernel. --no-cloudflare is explicit rather than
    relying on the default, because a default is a thing that changes."""
    body = "".join(_body().split())  # formatter-proof; see the cloudflare guards
    assert '"--no-cloudflare",' in body
    assert '"--secure"' not in body


def test_the_key_comes_from_the_marker_the_cli_itself_prints():
    body = "".join(_body().split())
    assert '"--start-api-key-marker",' in body
    assert '"UNSLOTH_START_API_KEY:"intext' in body  # whitespace-stripped


def test_the_key_is_registered_as_a_secret_before_anything_reads_the_log():
    """The log is packed into the evidence bundle. `redacted()` is what keeps
    the key out of it, and it can only redact a secret it has been told about,
    so the registration has to happen at the moment the key is parsed."""
    src = _body()
    parse_at = src.index('text.split("UNSLOTH_START_API_KEY:"')
    add_at = src.index("self.secrets.add(api_key)")
    assert parse_at < add_at, "the key must be registered where it is parsed"
    assert '"unsloth_run.log",' in _body("emit_evidence")


def test_gpu_use_is_measured_and_an_unmeasurable_reading_is_a_failure():
    """ "nvidia-smi did not answer" and "the model was on the GPU" are opposite
    outcomes; treating the first as a pass is the exact shape this directory
    has been caught by before."""
    body = _body()
    assert "baseline = nvidia_used_mib()" in body
    assert "settled = nvidia_used_mib()" in body
    # The verdict lives in cli_run_gpu_failure and is driven by the rules at the end of this file.
    assert "cli_run_gpu_failure(" in body
    assert "failures.append(failure)" in body
    verdict = _verdict()
    assert verdict(None, None, None, None)[0], "an unmeasurable reading passed"


def test_a_corrupted_key_must_be_refused():
    """Without this, a server ignoring the Authorization header entirely
    satisfies the claim that the minted key authenticated."""
    func = _func("assert_cli_run")
    guarded = [
        n
        for n in ast.walk(func)
        if isinstance(n, ast.If)
        and "bad_key_status" not in ast.unparse(n.test)
        and "code < 400" in ast.unparse(n.test)
    ]
    assert guarded, "nothing refuses a corrupted key"


def test_the_child_is_always_torn_down():
    """A `unsloth run` left alive holds a card and a port for the rest of the
    session, and the kernel's next phase reads that as its own failure."""
    func = _func("assert_cli_run")
    tries = [n for n in ast.walk(func) if isinstance(n, ast.Try) and n.finalbody]
    assert tries, "the teardown must be in a finally, or a raised assertion leaks the server"
    finals = "\n".join(ast.unparse(n) for t in tries for n in t.finalbody)
    assert "proc.terminate()" in finals
    assert "proc.kill()" in finals, "terminate alone leaves a hung server running"


def test_the_vram_sample_comes_AFTER_a_served_completion():
    """A completion proves the weights are resident, so VRAM is sampled after it, not while starting."""
    func = _func("assert_cli_run")
    src = ast.get_source_segment(SRC, func) or ""
    sample_at = src.index('detail["vram_after_mib"]')
    completion_at = src.index('detail["completion_status"]')
    assert completion_at < sample_at, (
        "VRAM is sampled before a completion has been served, so a slow load "
        "reads as a CPU fallback"
    )


def _verdict():
    """The real function, loaded by path rather than reimplemented."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("_studio_payload_cli", PAYLOAD)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.cli_run_gpu_failure


def test_a_co_tenant_freeing_memory_does_not_read_as_a_CPU_fallback():
    """A co-tenant freeing memory must not read as CPU fallback while this launch's pid holds the VRAM."""
    failure, detail = _verdict()({}, {6841: 2628}, 2816.0, 2634.0)
    assert failure is None, failure
    assert detail["process_vram_mib"] == 2628
    # The device delta is still recorded as evidence, just not the verdict.
    assert detail["vram_delta_mib"] == -182.0


def test_a_co_tenant_ALREADY_on_the_card_cannot_satisfy_the_claim():
    """Counting every process would pass on a card a training leg is using and
    a server that never left the CPU. Only pids that APPEARED count."""
    failure, detail = _verdict()({99: 12000}, {99: 12000}, 100.0, 100.0)
    assert failure and "served from the CPU" in failure
    assert detail["process_vram_mib"] == 0


def test_a_real_cpu_fallback_still_fails():
    """The case the assertion exists for: the launch answered, and no process
    of its own ever appeared on the GPU."""
    failure, _ = _verdict()({99: 12000}, {99: 12000, 4242: 3}, 100.0, 101.0)
    assert failure and "served from the CPU" in failure


def test_a_unified_memory_part_that_cannot_attribute_is_judged_on_the_device_delta():
    """On unified-memory parts with no per-process attribution, judge the run on the device VRAM delta."""
    failure, detail = _verdict()(None, None, 132.0, 560.0)
    assert failure is None, failure
    assert detail["vram_delta_mib"] == 428.0
    failure, detail = _verdict()(None, None, 132.0, 272.0)
    assert failure and "served from the CPU" in failure
    assert detail["vram_delta_mib"] == 140.0


def test_the_device_delta_is_the_fallback_only_when_processes_are_unreadable():
    """An nvidia-smi that answers a total but cannot enumerate apps still gets a
    verdict rather than a silent pass."""
    verdict = _verdict()
    assert verdict(None, None, 100.0, 4000.0)[0] is None
    failure, _ = verdict(None, None, 100.0, 110.0)
    assert failure and "could not enumerate processes" in failure
    assert verdict(None, None, None, None)[0] == (
        "nvidia-smi did not answer, so GPU use is unmeasured"
    )


def test_the_before_sample_is_taken_before_the_launch():
    """An `apps_before` read after the server started would contain the server,
    so nothing would ever have `appeared` and every run would fail."""
    body = _body()
    # One listing call, so the attributed mapping and the listed pids describe the same moment.
    assert body.index("_listing_before = nvidia_compute_apps_listing()") < body.index(
        "subprocess.Popen"
    )
    assert body.index("apps_before = attributed_apps(_listing_before)") < body.index(
        "subprocess.Popen"
    )
