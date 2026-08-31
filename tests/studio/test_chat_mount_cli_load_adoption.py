# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The chat mount must end up on the model the server is loading, not the one it is replacing.

Runs the real ``refresh`` -> ``syncInferenceStatusToStore`` -> ``waitForServerModel`` sequence from
use-chat-model-runtime.ts under node with the module boundary stubbed, so these assert behaviour
rather than source text. During a replacement the backend reports the outgoing model as
``active_model`` and the incoming one in ``loading``; adopting the outgoing one sets the checkpoint
the mount observer needs empty, and the observer then exits without ever seeing the new model land.
"""

import json
import os
import re
import shutil
import subprocess
import tempfile
import textwrap
from pathlib import Path

import pytest

WORKDIR = Path(__file__).resolve().parents[2]


def _source_path(relative_path: str) -> Path:
    direct = WORKDIR / relative_path
    if direct.exists():
        return direct
    return WORKDIR / "unsloth_repo" / relative_path


HOOK = _source_path("studio/frontend/src/features/chat/hooks/use-chat-model-runtime.ts")
TEMP = WORKDIR / "temp" / "chat_mount_cli_load_adoption"
OUTGOING = "org/outgoing-A-GGUF"
INCOMING = "org/incoming-B-GGUF"

# Every module-boundary name the sliced region uses. A missing one is a bare ReferenceError that
# the sync's own catch turns into a "Failed to refresh models" toast, which would read as a
# wrong-checkpoint assertion instead of naming the real cause.
PREAMBLE = """
export const EVENTS: any[] = [];

type Scenario = {
  /** What /status answers, as a function of ms since the scenario was set. */
  status: (elapsedMs: number) => any;
  models: any[];
};
let SCENARIO: Scenario = { status: () => ({}), models: [] };
let scenarioStart = 0;
export function setScenario(next: Scenario): void {
  SCENARIO = next;
  scenarioStart = Date.now();
}

let STATE: Record<string, any> = {};
export function setStoreState(next: Record<string, any>): void {
  STATE = { ...next, ...ACTIONS };
}
export function storeState(): Record<string, any> {
  return STATE;
}

const ACTIONS = {
  setModels(models: any[]) {
    STATE.models = models;
    EVENTS.push({ kind: "setModels", ids: models.map((model: any) => model.id) });
  },
  setLoras(loras: any[]) {
    STATE.loras = loras;
  },
  setModelsError(message: string | null) {
    STATE.modelsError = message;
    if (message) EVENTS.push({ kind: "modelsError", message });
  },
  setCheckpoint(checkpoint: string, ggufVariant?: string | null) {
    STATE.params = { ...STATE.params, checkpoint };
    STATE.activeGgufVariant = ggufVariant ?? null;
    EVENTS.push({ kind: "setCheckpoint", checkpoint });
  },
  clearCheckpoint() {
    STATE.params = { ...STATE.params, checkpoint: "" };
    EVENTS.push({ kind: "clearCheckpoint" });
  },
};

export const useChatRuntimeStore = {
  getState: () => STATE,
  setState: (patch: Record<string, any>) => {
    STATE = { ...STATE, ...patch };
    EVENTS.push({ kind: "setState", patch });
  },
};

export async function listModels(): Promise<any> {
  return { models: SCENARIO.models };
}
export async function listLoras(): Promise<any> {
  return { loras: [] };
}
export async function getInferenceStatus(): Promise<any> {
  const status = SCENARIO.status(Date.now() - scenarioStart);
  EVENTS.push({ kind: "status", active_model: status.active_model, loading: status.loading });
  return status;
}
export async function loadOpenAIAutoSwitchSettings(): Promise<any> {
  return { idleUnloadActive: false };
}

export const toChatModelRow = (model: any) => model;
export const toLoraSummary = (lora: any) => lora;
export const isExternalModelId = (id: unknown) =>
  typeof id === "string" && id.startsWith("external:");
export const isSpeechOnlyStatus = (status: any) => Boolean(status?.speechOnly);
export const isIdleUnloadedStatus = (status: any, _armed: boolean) =>
  Boolean(status?.idleUnloaded);
export const resolveInferenceCheckpointId = (status: any) =>
  status?.model_identifier ?? status?.active_model ?? null;

export function applyActiveModelStatusToStore(status: any, _options: any): void {
  STATE.residentCheckpoint = resolveInferenceCheckpointId(status);
  EVENTS.push({ kind: "applyActiveModel", checkpoint: STATE.residentCheckpoint });
}
export function syncModelCapabilities(_checkpoint: string, _status: any): void {}
export function refreshContextUsage(_options: any): void {}

// The observer's adoption, stubbed at the module boundary it really crosses. It keeps the three
// refusals the real one makes, because "a pick already landed" is exactly what a concurrent
// refresh produces here: a stub that adopted anyway would hide the bug these scenarios exist for.
export async function tryAdoptServerActiveModel(options: any): Promise<boolean> {
  if (STATE.params.checkpoint) return true;
  const status = options?.status;
  if (!status?.active_model) return false;
  if ((status.loading?.length ?? 0) > 0) return false;
  if (isSpeechOnlyStatus(status)) return false;
  const checkpoint = resolveInferenceCheckpointId(status);
  if (!checkpoint) return false;
  ACTIONS.setCheckpoint(checkpoint, status.gguf_variant ?? null);
  STATE.residentCheckpoint = checkpoint;
  EVENTS.push({ kind: "adopt", checkpoint });
  return true;
}

export const toast = {
  info: (title: string, options?: any) => EVENTS.push({ kind: "toast.info", title, options }),
  error: (title: string, options?: any) => EVENTS.push({ kind: "toast.error", title, options }),
};
"""


def _require_node() -> None:
    if shutil.which("node") is None:
        pytest.skip("node not available")
    if not HOOK.exists():
        pytest.skip("studio chat sources not present")
    try:
        result = subprocess.run(
            ["node", "--experimental-strip-types", "--version"],
            capture_output = True,
            text = True,
            # A cold Windows runner is slow to start node; an impatient probe would fail the gate.
            timeout = 60,
        )
    except (OSError, subprocess.SubprocessError):
        pytest.skip("node could not be started")
    if result.returncode != 0:
        pytest.skip("node --experimental-strip-types not available")


def _between(source: str, start: str, end: str) -> str:
    first = source.find(start)
    assert first != -1, f"could not locate {start!r} in {HOOK.name}"
    last = source.find(end, first)
    assert last != -1, f"could not locate {end!r} after {start!r} in {HOOK.name}"
    return source[first:last]


def _build_harness(run_dir: Path) -> None:
    """Slice the mount path verbatim out of the hook."""
    source = HOOK.read_text(encoding = "utf-8")
    poll = _between(source, "const CLI_LOAD_POLL_IDLE_MS", "function parseTrailingEpoch(")
    assert "async function waitForServerModel(" in poll
    sync = _between(
        source,
        "// Prevent older concurrent status reads",
        "/**\n * Reconcile the UI after the SERVER unloaded",
    )
    assert "async function syncInferenceStatusToStore(" in sync
    # The sequencing IS the bug, so refresh has to run for real rather than be re-typed here.
    # useCallback and its dependency array are React plumbing with no behaviour to preserve.
    match = re.search(
        r"const refresh = useCallback\(\n"
        r"    async \(options\?: \{(?P<params>.*?)\}\) => \{(?P<body>.*?)\n"
        r"    \},\n    \[\],\n  \);",
        source,
        re.S,
    )
    assert match, "could not locate the refresh callback in use-chat-model-runtime.ts"
    refresh = (
        f"export async function refresh(options?: {{{match.group('params')}}}) "
        f"{{{match.group('body')}\n}}\n"
    )
    assert "syncInferenceStatusToStore(options)" in refresh
    (run_dir / "harness.ts").write_text(
        "// @ts-nocheck\n" + PREAMBLE + "\n" + poll + "\n" + sync + "\n" + refresh,
        encoding = "utf-8",
    )


def _run(script_body: str) -> dict:
    _require_node()
    TEMP.mkdir(parents = True, exist_ok = True)
    # Its own directory per invocation: a shared file lets one runner read another's rewrite.
    run_dir = Path(tempfile.mkdtemp(prefix = "run", dir = TEMP))
    _build_harness(run_dir)
    script = (
        textwrap.dedent(
            """
        // @ts-nocheck
        import {
          EVENTS,
          refresh,
          setScenario,
          setStoreState,
          storeState,
        } from "./harness.ts";

        const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

        function emptyStore(over = {}) {
          return {
            params: { checkpoint: "" },
            modelLoading: false,
            residentCheckpoint: undefined,
            activeGgufVariant: null,
            contextUsage: null,
            activeThreadId: null,
            ggufContextLength: null,
            models: [],
            loras: [],
            ...over,
          };
        }

        const REPLACING = {
          active_model: "%(outgoing)s",
          model_identifier: "%(outgoing)s",
          is_gguf: true,
          gguf_variant: "Q4_K_M",
          loading: ["%(incoming)s"],
        };
        const SETTLED_ON_INCOMING = {
          active_model: "%(incoming)s",
          model_identifier: "%(incoming)s",
          is_gguf: true,
          gguf_variant: "Q4_K_M",
          loading: [],
        };
        /** Mid-replacement until `ms`, then the incoming model is resident. */
        const settlesAfter = (ms) => (elapsed) =>
          elapsed < ms ? REPLACING : SETTLED_ON_INCOMING;
        """
            % {"outgoing": OUTGOING, "incoming": INCOMING}
        )
        + textwrap.dedent(script_body)
        + textwrap.dedent(
            """
        console.log(
          JSON.stringify({
            checkpoint: storeState().params.checkpoint,
            residentCheckpoint: storeState().residentCheckpoint ?? null,
            models: (storeState().models ?? []).map((model) => model.id),
            events: EVENTS,
          }),
        );
        """
        )
    )
    (run_dir / "run.mts").write_text(script, encoding = "utf-8")
    completed = subprocess.run(
        ["node", "--experimental-strip-types", "--no-warnings", "run.mts"],
        cwd = str(run_dir),
        capture_output = True,
        text = True,
        # Explicit: text alone decodes with the Windows ANSI code page, which mangles the
        # non-ASCII toast copy node emits as UTF-8.
        encoding = "utf-8",
        timeout = 120,
        env = dict(os.environ, NODE_NO_WARNINGS = "1"),
    )
    assert completed.returncode == 0, f"stderr: {completed.stderr}\nstdout: {completed.stdout}"
    last = [line for line in completed.stdout.strip().splitlines() if line.strip()][-1]
    return json.loads(last)


MOUNT = """
        setStoreState(emptyStore());
        const mount = refresh({
          includeLoras: true,
          signal: new AbortController().signal,
          waitForServerModel: true,
        });
"""


def test_mount_adopts_the_incoming_cli_model_not_the_outgoing_one():
    """The reported case: A resident, a CLI-requested B in flight, and no local pick to protect."""
    out = _run(
        """
        setScenario({
          models: [{ id: "%(outgoing)s" }, { id: "%(incoming)s" }],
          status: settlesAfter(700),
        });
        """
        % {"outgoing": OUTGOING, "incoming": INCOMING}
        + MOUNT
        + "        await mount;\n"
    )
    assert out["checkpoint"] == INCOMING
    assert out["residentCheckpoint"] == INCOMING


def test_a_refresh_landing_mid_wait_does_not_end_it_on_the_outgoing_model():
    """ChatPage fires a second inventory refresh 1.2s after mount, and the lifecycle bus and a
    tab regaining focus fire more. None asks to wait, so each would adopt the outgoing model and
    leave the observer with the non-empty checkpoint its loop refuses to poll on."""
    out = _run(
        """
        setScenario({
          models: [{ id: "%(outgoing)s" }, { id: "%(incoming)s" }],
          status: settlesAfter(1500),
        });
        """
        % {"outgoing": OUTGOING, "incoming": INCOMING}
        + MOUNT
        + """
        await sleep(600);
        await refresh({ includeLoras: false });
        await mount;
        """
    )
    assert out["checkpoint"] == INCOMING
    assert out["residentCheckpoint"] == INCOMING


def test_mount_publishes_the_model_list_while_it_waits():
    """Waiting must not cost the selector its inventory: the picker stays usable during the load."""
    out = _run(
        """
        setScenario({
          models: [{ id: "%(outgoing)s" }, { id: "%(incoming)s" }],
          status: settlesAfter(700),
        });
        """
        % {"outgoing": OUTGOING, "incoming": INCOMING}
        + MOUNT
        + "        await mount;\n"
    )
    kinds = [event["kind"] for event in out["events"]]
    assert out["models"] == [OUTGOING, INCOMING]
    assert kinds.index("setModels") < kinds.index("setCheckpoint")


def test_mount_still_adopts_a_settled_model_with_no_load_in_flight():
    """Nothing loading is the ordinary mount, and it must still hydrate from the first read."""
    out = _run(
        """
        setScenario({
          models: [{ id: "%(incoming)s" }],
          status: settlesAfter(0),
        });
        """
        % {"incoming": INCOMING}
        + MOUNT
        + "        await mount;\n"
    )
    assert out["checkpoint"] == INCOMING
    # One read: the sync's own. A settled status must not cost a second round trip.
    assert [event["kind"] for event in out["events"]].count("status") == 1


def test_a_refresh_that_is_not_the_mount_waiter_still_publishes_residency():
    """The regression guard for the fix that removed the blanket loading bail-out.

    A refresh with no waiter behind it (the lifecycle bus, the deferred inventory read) owns
    residency itself, so a load in flight must not stop it publishing what is resident.
    """
    out = _run(
        """
        setScenario({
          models: [{ id: "%(outgoing)s" }],
          status: () => REPLACING,
        });
        setStoreState(emptyStore());
        await refresh({ includeLoras: false });
        """
        % {"outgoing": OUTGOING}
    )
    assert out["checkpoint"] == OUTGOING
    assert out["residentCheckpoint"] == OUTGOING
