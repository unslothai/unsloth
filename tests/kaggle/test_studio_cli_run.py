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

**VRAM growth across the launch.** A GGUF server that fell back to the CPU
serves text perfectly well, so "it answered" is not evidence it reached the
card. The delta is only this launch's because the assertion runs LAST, after
the chat-UI phase has stopped the server and emptied the card.

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
    assert "if baseline is None or settled is None:" in body
    assert "delta < 200.0" in body


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
    sample_at = src.index("detail[\"vram_after_mib\"]")
    completion_at = src.index("detail[\"completion_status\"]")
    assert completion_at < sample_at, (
        "VRAM is sampled before a completion has been served, so a slow load "
        "reads as a CPU fallback"
    )
