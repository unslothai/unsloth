# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The chat mount must end up on the model the server is loading, not the one it is replacing.

Runs the real ``refresh`` -> ``syncInferenceStatusToStore`` -> ``waitForServerModel`` sequence from
use-chat-model-runtime.ts under node with the module boundary stubbed, so these assert behaviour
rather than source text. A replacement names the outgoing model ``active_model`` and the incoming
one ``loading``; adopting the outgoing one sets the checkpoint the observer needs empty.
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
CHAT_PAGE = _source_path("studio/frontend/src/features/chat/chat-page.tsx")
# Inlined for real, not stubbed: the gate is shared with the send-path poll, so a stub would
# only prove this file agrees with itself.
WAIT_GATE = _source_path("studio/frontend/src/features/chat/lib/server-model-wait.ts")
TEMP = WORKDIR / "temp" / "chat_mount_cli_load_adoption"
OUTGOING = "org/outgoing-A-GGUF"
INCOMING = "org/incoming-B-GGUF"

# Every module-boundary name the sliced region uses. A missing one is a ReferenceError the
# sync's catch turns into a toast, so it reads as a wrong-checkpoint failure instead.
PREAMBLE = """
export const EVENTS: any[] = [];

type Scenario = {
  status: (elapsedMs: number) => any;
  models: any[];
  /** Per-call listModels() latency, last entry repeating: lets a later refresh answer first. */
  listModelsDelays?: number[];
  /** 1-based /status reads that only end on abort (or 5s). Indexed, so their neighbours answer. */
  hangStatusReads?: number[];
  listLorasFails?: boolean;
};
let SCENARIO: Scenario = { status: () => ({}), models: [] };
let scenarioStart = 0;
let listModelsCalls = 0;
export function setScenario(next: Scenario): void {
  SCENARIO = next;
  scenarioStart = Date.now();
  listModelsCalls = 0;
  statusReads = 0;
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
  const delays = SCENARIO.listModelsDelays ?? [];
  const delay = delays[Math.min(listModelsCalls, delays.length - 1)] ?? 0;
  listModelsCalls += 1;
  if (delay > 0) await new Promise((resolve) => setTimeout(resolve, delay));
  return { models: SCENARIO.models };
}
export async function listLoras(): Promise<any> {
  if (SCENARIO.listLorasFails) throw new Error("lora scan failed");
  return { loras: [] };
}
let statusReads = 0;
export async function getInferenceStatus(signal?: AbortSignal): Promise<any> {
  statusReads += 1;
  if ((SCENARIO.hangStatusReads ?? []).includes(statusReads)) {
    EVENTS.push({ kind: "status.hang", signalled: Boolean(signal) });
    // Bounded so an unfixed source fails on the assertion below rather than hanging.
    await new Promise((resolve, reject) => {
      const timer = setTimeout(resolve, 5000);
      signal?.addEventListener("abort", () => {
        clearTimeout(timer);
        reject(new Error("aborted"));
      });
    });
  }
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

// Stubbed at the module boundary it really crosses, but keeping the three refusals: "a pick
// already landed" is what a concurrent refresh produces, and adopting anyway would hide the bug.
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

// The two ponyfills server-model-wait.ts imports. Its own per-read cap is 30s, which no test
// should wait out, so the timeout here is short and the real constant is asserted separately.
export const TEST_POLL_TIMEOUT_MS = 1200;
export function disposableTimeoutSignal(_ms: number) {
  const controller = new AbortController();
  const timer = setTimeout(
    () => controller.abort(new DOMException("The operation timed out.", "TimeoutError")),
    TEST_POLL_TIMEOUT_MS,
  );
  return { signal: controller.signal, dispose: () => clearTimeout(timer) };
}
export function pollSignal(parent: AbortSignal, _ms: number) {
  const timeout = disposableTimeoutSignal(_ms);
  const controller = new AbortController();
  const onAbort = (reason: unknown) => {
    if (!controller.signal.aborted) controller.abort(reason);
  };
  const a = () => onAbort(parent.reason);
  const b = () => onAbort(timeout.signal.reason);
  parent.addEventListener("abort", a, { once: true });
  timeout.signal.addEventListener("abort", b, { once: true });
  if (parent.aborted) a();
  return {
    signal: controller.signal,
    dispose: () => {
      parent.removeEventListener("abort", a);
      timeout.signal.removeEventListener("abort", b);
      timeout.dispose();
    },
  };
}
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
    # The real gate, with its imports dropped: the preamble supplies those two ponyfills.
    gate = "\n".join(
        line
        for line in WAIT_GATE.read_text(encoding = "utf-8").splitlines()
        if not line.startswith(
            (
                "import ",
                "  disposableTimeoutSignal,",
                "  pollSignal,",
                "  type PollSignal,",
                '} from "@/features/hub',
            )
        )
    ).replace(": PollSignal", "")
    assert "export function beginServerModelWait(" in gate
    poll = _between(source, "const CLI_LOAD_POLL_IDLE_MS", "function parseTrailingEpoch(")
    assert "async function waitForServerModel(" in poll
    sync = _between(
        source,
        "let syncGeneration = 0;",
        "export async function resyncInferenceStatusAfterServerModelChange(",
    )
    assert "async function syncInferenceStatusToStore(" in sync
    # The sequencing IS the bug, so refresh runs for real rather than being re-typed here.
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
    # The handoff owner has to come along, or the wait is registered nowhere.
    assert "async function refreshAndWaitForServerModel(" in sync
    (run_dir / "harness.ts").write_text(
        "// @ts-nocheck\n" + PREAMBLE + "\n" + gate + "\n" + poll + "\n" + sync + "\n" + refresh,
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
          beginServerModelWait,
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
        const IDLE = { active_model: null, model_identifier: null, loading: [] };
        const STARTING = {
          active_model: null,
          model_identifier: null,
          is_gguf: true,
          loading: ["%(incoming)s"],
        };
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


# The ChatPage mount effect, argument for argument (chat-page.tsx).
MOUNT = """
        setStoreState(emptyStore());
        const mount = refresh({
          includeLoras: false,
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
    """ChatPage refreshes again 1.2s after mount, the lifecycle bus and a returning tab more.
    None asks to wait, so each would adopt the outgoing model and leave the observer nothing."""
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


def test_mount_waits_for_a_cli_load_that_starts_after_it():
    """The case the PR was opened for: the UI opens before `studio run -m` reaches the loader,
    so the first read is a bare idle server and the load only shows up a poll or two later."""
    out = _run(
        """
        setScenario({
          models: [{ id: "%(incoming)s" }],
          status: (elapsed) =>
            elapsed < 600 ? IDLE : elapsed < 1600 ? STARTING : SETTLED_ON_INCOMING,
        });
        """
        % {"incoming": INCOMING}
        + MOUNT
        + "        await mount;\n"
    )
    assert out["checkpoint"] == INCOMING
    assert out["residentCheckpoint"] == INCOMING


def test_a_refresh_that_answers_before_the_mount_sync_does_not_end_the_wait():
    """Registered before the sync is issued, not after it returns: a refresh issued second can
    answer first, and inside that window it saw no wait outstanding and adopted the outgoing one."""
    out = _run(
        """
        setScenario({
          models: [{ id: "%(outgoing)s" }, { id: "%(incoming)s" }],
          status: settlesAfter(1500),
          listModelsDelays: [400, 0],
        });
        """
        % {"outgoing": OUTGOING, "incoming": INCOMING}
        + MOUNT
        + """
        await sleep(50);
        await refresh({ includeLoras: false });
        await mount;
        """
    )
    assert out["checkpoint"] == INCOMING
    assert out["residentCheckpoint"] == INCOMING


def test_aborting_the_mount_releases_the_wait_even_on_a_stalled_status_read():
    """Deferring residency only holds if the wait can end: checked around the request rather
    than passed into it, an unmount leaves the poll parked and every later refresh behind it."""
    out = _run(
        """
        setScenario({
          models: [{ id: "%(outgoing)s" }],
          status: () => REPLACING,
          // The observer's first poll: the reads on either side of it still answer.
          hangStatusReads: [2],
        });
        setStoreState(emptyStore());
        const controller = new AbortController();
        const mount = refresh({
          includeLoras: false,
          signal: controller.signal,
          waitForServerModel: true,
        });
        await sleep(300);
        controller.abort();
        await sleep(50);
        await refresh({ includeLoras: false });
        await mount;
        """
        % {"outgoing": OUTGOING}
    )
    assert out["checkpoint"] == OUTGOING
    assert out["residentCheckpoint"] == OUTGOING


def test_aborting_the_mount_releases_the_wait_even_on_an_unsignalled_read():
    """And down on the abort itself, not on the way out: listModels and listLoras take no
    AbortSignal, so waiting for the sync to return leaves the gate up past its own page."""
    out = _run(
        """
        setScenario({
          models: [{ id: "%(outgoing)s" }],
          status: () => REPLACING,
          listModelsDelays: [2500, 0],
        });
        setStoreState(emptyStore());
        const controller = new AbortController();
        const mount = refresh({
          includeLoras: false,
          signal: controller.signal,
          waitForServerModel: true,
        });
        await sleep(300);
        controller.abort();
        await sleep(50);
        await refresh({ includeLoras: false });
        await mount;
        """
        % {"outgoing": OUTGOING}
    )
    assert out["checkpoint"] == OUTGOING
    assert out["residentCheckpoint"] == OUTGOING


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


def test_a_failing_lora_scan_takes_the_model_list_with_it():
    """Why the mount must not ask for LoRAs: the three reads share one Promise.all and the LoRA
    handler rethrows, so a rejection skips setModels and the picker offers nothing at all."""
    out = _run(
        """
        setScenario({
          models: [{ id: "%(outgoing)s" }],
          status: () => SETTLED_ON_INCOMING,
          listLorasFails: true,
        });
        setStoreState(emptyStore());
        await refresh({ includeLoras: true });
        """
        % {"outgoing": OUTGOING}
    )
    assert out["models"] == []
    assert [event["kind"] for event in out["events"]].count("modelsError") == 1


def test_the_mount_refresh_does_not_wait_on_the_lora_inventory():
    """So the mount reads models and status only. The deferred inventory refresh 1.2s later
    owns the LoRA list, and its failing is survivable: the picker already has its models."""
    source = CHAT_PAGE.read_text(encoding = "utf-8")
    effect = _between(source, "if (getTrainingCompareHandoff()) return;", "}, 1200);")
    assert re.search(r"void refresh\(\{\s*includeLoras: false,", effect), effect
    assert "waitForServerModel: !useChatRuntimeStore.getState().params.checkpoint" in effect
    assert "refreshDeferredModelInventories();" in effect


def test_a_wait_from_the_send_path_also_stops_a_refresh_publishing_the_outgoing_model():
    """The gate is shared, so the send-path poll gets the same protection as the mount one.

    That poll's stopEarly reads params.checkpoint to decide the user picked something. A
    refresh hydrating the outgoing model mid-replacement would look exactly like a pick and
    hand it to the send, which then talks to the model the server is replacing.
    """
    out = _run(
        """
        setScenario({
          models: [{ id: "%(outgoing)s" }],
          status: () => REPLACING,
        });
        setStoreState(emptyStore());
        const release = beginServerModelWait();
        await refresh({ includeLoras: false });
        const during = storeState().params.checkpoint;
        release();
        await refresh({ includeLoras: false });
        EVENTS.push({ kind: "duringWait", checkpoint: during });
        """
        % {"outgoing": OUTGOING}
    )
    during = [e for e in out["events"] if e["kind"] == "duringWait"][0]["checkpoint"]
    assert during == "", "a refresh published the outgoing model while a wait was outstanding"
    # And the gate is a gate, not a mute: once the wait is over the same refresh publishes.
    assert out["checkpoint"] == OUTGOING


def test_a_stalled_status_read_does_not_park_the_poll_on_one_request():
    """Each read is capped, so the loop's own deadline is real.

    fetch has no timeout, so before this the advertised cap bounded nothing: one half-open
    read parked the poll indefinitely, holding the shared gate and the send's lease with it.
    """
    out = _run(
        """
        setScenario({
          models: [{ id: "%(incoming)s" }],
          // Reads 2 and 3 are the observer's, and neither ever answers.
          hangStatusReads: [2, 3],
          status: (elapsed) => (elapsed < 4000 ? STARTING : SETTLED_ON_INCOMING),
        });
        """
        % {"incoming": INCOMING}
        + MOUNT
        + "        await mount;\n"
    )
    assert out["checkpoint"] == INCOMING
    # Two capped reads plus the ones on either side: the loop kept going instead of parking.
    assert [event["kind"] for event in out["events"]].count("status.hang") == 2
    assert [event["kind"] for event in out["events"]].count("status") >= 3


def test_the_shipped_per_read_cap_is_the_one_the_harness_stands_in_for():
    """The harness shortens the cap so no test waits it out; this pins what really ships."""
    gate = WAIT_GATE.read_text(encoding = "utf-8")
    assert "export const STATUS_POLL_TIMEOUT_MS = 30_000;" in gate
    assert re.search(r"return parent\s*\?\s*pollSignal\(parent, STATUS_POLL_TIMEOUT_MS\)", gate)
    adapter = _source_path("studio/frontend/src/features/chat/api/chat-adapter.ts").read_text(
        encoding = "utf-8"
    )
    assert "statusPollSignal(options?.abortSignal)" in adapter
    assert "beginServerModelWait(options?.abortSignal)" in adapter


def test_a_refresh_that_is_not_the_mount_waiter_still_publishes_residency():
    """The guard for the commit that removed the blanket loading bail-out: a refresh with no
    waiter behind it owns residency itself, so a load in flight must not stop it publishing."""
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
