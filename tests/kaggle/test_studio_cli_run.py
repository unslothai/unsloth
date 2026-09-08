# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""`unsloth run`: the headless model server, and what a banner does not prove.

`unsloth run` is a different launch from `unsloth studio`, and nothing in CI
covered it. It starts the backend, waits for health, mints an API key
IN-PROCESS, and only then loads the model over HTTP. Any of those four steps
can fail while the command still prints a banner, so the rules here are about
what came back rather than what was printed.

The two that carry the most weight, because each closes a hole the other
leaves:

**GPU residency across the launch.** A GGUF server that fell back to the CPU
serves text perfectly well, so "it answered" is not evidence it reached the
card. The measurement is PER-PROCESS rather than a device total: under
--studio-concurrent a training leg shares the card, and on kernel
unsloth-probe-full-concurrent-417238 the device delta read -182.0 MiB while
the same report showed this launch holding 2628 MiB. A shared counter cannot
attribute. `cli_run_gpu_failure` is a pure function precisely so the rules
below can DRIVE it with those numbers rather than describe it.

**A corrupted key must be REFUSED.** Without it, a server that ignores the
Authorization header entirely satisfies "the minted key authenticated".
"""

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
    """Not a preference. `unsloth run` starts a SECOND backend against the same
    studio home, and two backends sharing one home's state is a configuration
    nobody ships. It also makes the VRAM delta meaningless: a card still
    holding the first server's model cannot show this launch's growth."""
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
    # And the log must actually be in the bundle, or nothing redacts it.
    assert '"unsloth_run.log",' in _body("emit_evidence")


def test_gpu_use_is_measured_and_an_unmeasurable_reading_is_a_failure():
    """ "nvidia-smi did not answer" and "the model was on the GPU" are opposite
    outcomes; treating the first as a pass is the exact shape this directory
    has been caught by before."""
    body = _body()
    assert "baseline = nvidia_used_mib()" in body
    assert "settled = nvidia_used_mib()" in body
    # The verdict itself moved into `cli_run_gpu_failure` when the device delta
    # stopped being a valid ruler under --studio-concurrent, so it is DRIVEN
    # rather than grepped -- see the rules at the end of this file. What stays
    # here is that the assertion consults it and reports what it says.
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
    """`unsloth run` prints its API key while it is still starting.

    Sampling there read 0.0 MiB of growth on kernel
    unsloth-probe-studio-full2-815a0c, on a launch whose own log says
    `Starting llama-server: ... -ngl -1 --fit off` -- Studio asking for every
    layer on the card. A completion that came back is the cheap proof the
    weights are resident, so the ruler has to go after it or the check measures
    a race.
    """
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
    """The exact numbers from unsloth-probe-full-concurrent-417238.

    Device VRAM 2816 -> 2634, a delta of -182.0, while `nvidia-smi` shows this
    launch's own pid holding 2628 MiB. Under the old device-delta rule that was
    a failure saying `unsloth run` served from the CPU; the model was on the
    card the whole time and a training leg on the same card freed memory inside
    the window. This is the regression guard for that reading.
    """
    failure, detail = _verdict()({}, {6841: 2628}, 2816.0, 2634.0)
    assert failure is None, failure
    assert detail["process_vram_mib"] == 2628
    # The device delta is still RECORDED -- it is evidence, it is just not the
    # verdict -- and it is still the number that misled.
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
    assert body.index("apps_before = nvidia_compute_apps()") < body.index("subprocess.Popen")
