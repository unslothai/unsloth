// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Source assertions: a .tsx barrel in the store's import graph blocks a bare node test.

import assert from "node:assert/strict";
import test from "node:test";

import { readText } from "./helpers/kit.ts";

const store = readText("../src/features/chat/stores/chat-runtime-store.ts");
const provider = readText("../src/features/chat/runtime-provider.tsx");
const composer = readText("../src/components/assistant-ui/thread.tsx");

function slice(source: string, from: string, to: string): string {
  const start = source.indexOf(from);
  assert.ok(start !== -1, `not found: ${from}`);
  const end = source.indexOf(to, start + from.length);
  assert.ok(end !== -1, `not found: ${to}`);
  return source.slice(start, end);
}

test("the read waits for this chat's own write before it can be believed", () => {
  // A GET that overtakes an in-flight PATCH returns the pre-edit snapshot.
  const sync = slice(provider, "const sync = () => {", "// The read did not answer");
  assert.match(sync, /awaitThreadScopedSettingsWrite\(activeThreadId\)/);
  // initialize() resolves before its POST lands, so a first read can overtake row creation.
  assert.match(sync, /awaitStoredChatThreadWrites\(activeThreadId\)/);
  assert.ok(
    sync.indexOf("awaitThreadScopedSettingsWrite") <
      sync.indexOf("getStoredChatThreadReadResult"),
    "the read is not sequenced after the write",
  );
});

/** sync()'s body without comments, so prose mentions of a call do not skew the order checks. */
function syncCode(): string {
  return slice(provider, "const sync = () => {", "// The read did not answer")
    .replace(/\/\*[\s\S]*?\*\//g, "")
    .replace(/^[ \t]*\/\/.*$/gm, "");
}

test("both waits sit inside the attempt's deadline, not in front of it", () => {
  // Neither wait is bounded on its own, so both must sit inside the deadline.
  const sync = syncCode();
  const race = sync.indexOf("Promise.race");
  assert.ok(race !== -1, "the per-attempt deadline is gone");

  for (const wait of ["awaitThreadScopedSettingsWrite", "awaitStoredChatThreadWrites"]) {
    assert.ok(
      sync.indexOf(wait) > race,
      `${wait}() is awaited before the deadline opens, so its time is unbounded`,
    );
  }
});

test("the pairing wait still outlasts the worst case read chain", () => {
  // Read from source so this cannot drift from THREAD_PAIRING_WAIT_MS's stated arithmetic.
  const constant = (source: string, name: string): number => {
    const match = source.match(new RegExp(`${name}\\s*=\\s*([0-9_]+)`));
    assert.ok(match, `${name} not found`);
    return Number(match[1].replace(/_/g, ""));
  };
  const store = readText("../src/features/chat/stores/chat-runtime-store.ts");

  const attempts = constant(provider, "THREAD_READ_RETRIES") + 1;
  const worstCase =
    attempts * constant(provider, "THREAD_READ_TIMEOUT_MS") +
    (attempts - 1) * constant(provider, "THREAD_READ_RETRY_MS");

  assert.ok(
    worstCase < constant(store, "THREAD_PAIRING_WAIT_MS"),
    `the read chain can take ${worstCase}ms, at or past the gate's give-up, so a slow ` +
      "read refuses the user's send instead of falling back to the installation defaults",
  );
});

test("running out of read retries ends the pairing rather than parking sends forever", () => {
  // The pending flag gates the composer, so a pairing with no retry left blocks every send.
  const retry = slice(provider, "const retryThreadRead = () => {", "\n    sync();");
  const exhausted = slice(retry, "if (retriesLeft <= 0) {", "return;");
  assert.match(exhausted, /applyThreadScopedSettings\(null, null\)/);
  assert.match(exhausted, /releaseHeldThreadScopedEdits\(\)/);
  assert.match(exhausted, /toast\.error/);
  const reopen = retry.indexOf("beginThreadScopedPairing");
  assert.ok(
    reopen > retry.indexOf("if (retriesLeft <= 0) {"),
    "pairing is re-opened before the exhaustion check, which leaves it open forever",
  );
});

test("a failed backend read is retried, not treated as a chat with no snapshot", () => {
  // cacheable:false means the GET failed and Dexie answered; do not release held edits.
  const fallback = slice(provider, "if (thread && !cacheable) {", "}");
  assert.match(fallback, /retryThreadRead\(\)/);
});

test("a chat with no row releases its held edits on the first answer", () => {
  const missing = slice(provider, "if (!thread) {", "unpaired = true;");
  assert.match(missing, /releaseHeldThreadScopedEdits\(\)/);
});

test("the defaults are captured when pairing begins, not reconstructed later", () => {
  // On the first pairing there is no earlier capture, so the default is sampled up front.
  const begin = slice(store, "export function beginThreadScopedPairing", "\n}");
  assert.match(begin, /pairingWindowDefaults =/);
  assert.match(begin, /readThreadScopedSettings\(/);
  const capture = slice(
    store,
    "if (threadScopedSettingsThreadId === null) {",
    "globalThreadScopedDefaults = captured",
  );
  assert.match(capture, /pairingWindowDefaults \?\?\s*globalThreadScopedDefaults/);
});

test("dropping to the defaults keeps holding for a chat still awaiting its read", () => {
  // Releasing held edits here would move every snapshot-less chat's default.
  const guard = slice(
    store,
    "      } else if (",
    "releaseHeldThreadScopedEdits();",
  );
  assert.match(guard, /threadId !== null \|\|/);
  assert.match(guard, /pendingPairingThreadId !== state\.activeThreadId/);
});

test("an edit whose chat was never read is sent as a merge, not a replacement", () => {
  // The store shows installation defaults, so a full snapshot would erase untouched stored keys.
  const commit = slice(
    store,
    "export function commitHeldThreadScopedEditsToTheirThread",
    "\n}",
  );
  assert.match(commit, /heldThreadScopedChanges\(held\)/);
  assert.match(commit, /sendThreadScopedSettingsBeacon\(threadId, changes, true\)/);
  const merge = slice(store, "async function mergeThreadScopedSettingsIntoRow", "\n}");
  assert.match(merge, /settingsPatch: changes/);
});

test("hydration does not overwrite a field whose edit is still held", () => {
  // A held edit advances no mutation version, so it must be protected explicitly.
  const loop = slice(store, "for (const key of SCALAR_SETTING_KEYS) {", "return nextState;");
  assert.match(loop, /if \(isHeldThreadScopedField\(key\)\) \{/);
  const held = slice(loop, "if (isHeldThreadScopedField(key)) {", "\n    }");
  assert.match(held, /continue;/);
  assert.doesNotMatch(
    held,
    /\(nextState as Record<ScalarSettingKey, unknown>\)\[key\] = value;/,
    "the server's value is applied over the held edit",
  );
});

test("every snapshot write is ordered against the others", () => {
  // Aborting a fetch does not stop a started handler, so writes carry a server-ordered seq.
  assert.match(store, /function nextThreadSettingsSeq\(\): number \{/);
  const write = slice(store, "function writeThreadScopedSettings", "\n}");
  assert.match(write, /const settingsSeq = nextThreadSettingsSeq\(\);/);
  assert.match(write, /\{ settings, settingsSeq, settingsWriter/);
  const beacon = slice(store, "function sendThreadScopedSettingsBeacon", "\n}");
  assert.match(beacon, /settingsSeq: nextThreadSettingsSeq\(\)/);
  assert.match(beacon, /takeThreadSettingsWriteTicket\(threadId\)/);
  assert.match(beacon, /threadSettingsWriteAborts\.get\(threadId\)\?\.abort\(\)/);
});

test("a normal flush stays resendable until it lands", () => {
  // visibilitychange(hidden) flushes and clears the snapshot before pagehide fires.
  const flush = slice(store, "function flushThreadScopedSettingsWrite", "\n}");
  assert.match(flush, /trackUnsettledThreadSettingsWrite\(threadId, snapshot\)/);
  const track = slice(store, "function trackUnsettledThreadSettingsWrite", "\n}");
  assert.match(track, /unsettledThreadSettingsWrites\.set\(threadId, entry\)/);
  const terminal = slice(store, "function flushSettingsOnPageHidden", "\n}");
  assert.match(terminal, /commitHeldThreadScopedEditsToTheirThread\(true\)/);
  assert.match(terminal, /beaconUnsettledThreadSettingsWrites\(sentNewest\)/);
});

test("a capability clamp never overwrites the preference the chat stored", () => {
  const build = slice(store, "function buildThreadScopedSnapshot", "\n}");
  for (const key of [
    "toolsEnabled",
    "codeToolsEnabled",
    "imageToolsEnabled",
    "webFetchToolsEnabled",
  ]) {
    assert.ok(
      store.includes(`"${key}"`) && build.includes("CLAMPED_PILL_KEYS"),
      `${key} is not covered by the clamp preservation`,
    );
  }
  assert.match(build, /modelLoaded &&\s*!capable\[key\] &&/);
});

test("write ordering is per writer, never one browser's counter against another's", () => {
  // Comparing seqs across clients would refuse the lagging one's edits while answering 200.
  assert.match(store, /const threadSettingsWriter = crypto\.randomUUID\(\);/);
  const next = slice(store, "function nextThreadSettingsSeq", "\n}");
  assert.doesNotMatch(next, /Date\.now\(\)/, "the seq is a clock again");
  for (const site of [
    slice(store, "function sendThreadScopedSettingsBeacon", "\n}"),
    slice(store, "function writeThreadScopedSettings", "\n}"),
    slice(store, "async function mergeThreadScopedSettingsIntoRow", "\n}"),
  ]) {
    assert.match(site, /settingsWriter/);
  }
});

test("a tab-close write that could not be confirmed is replayed next session", () => {
  // A chat whose row is still being created answers 404 to the beacon.
  const beacon = slice(store, "function sendThreadScopedSettingsBeacon", "\n}");
  assert.match(beacon, /rememberThreadSettingsForReplay\(threadId, body\)/);
  assert.match(store, /export function replayUnconfirmedThreadSettings/);
  assert.match(store, /replayUnconfirmedThreadSettings\(\);/);
});

test("a model that forces thinking on does not erase a chat's stored preference", () => {
  const build = slice(store, "function buildThreadScopedSnapshot", "\n}");
  assert.match(build, /activeThreadScopedSettings\?\.reasoningEnabled === false/);
  assert.match(build, /reasoningAlwaysOn/);
});

test("compare mode drops the thread-scoped state rather than keeping the last chat's", () => {
  const disabled = slice(
    provider,
    "// Compare panes share one composer",
    "return;",
  );
  assert.match(disabled, /applyThreadScopedSettings\(null, null\)/);
});

test("a retry does not resample the defaults over the edit it is holding", () => {
  // Retry re-pairs with the edit still in the store, so sample once, not per attempt.
  const begin = slice(store, "export function beginThreadScopedPairing", "\n}");
  assert.match(begin, /if \(pairingWindowDefaultsThreadId !== threadId\) \{/);
  assert.match(
    slice(store, "applyThreadScopedSettings: (threadId, settings) =>", "} else if ("),
    /pairingWindowDefaultsThreadId = null;/,
  );
});

test("a pending pin answers the override rather than the null it has stored", () => {
  // A chat being pinned has no snapshot yet, so the override must come from the held edit.
  const override = slice(store, "export function threadScopedOverride", "\n}");
  assert.match(override, /threadSettingsWriteSnapshot\[key\] !== undefined/);
});

test("the prompt queue waits for this chat's settings too, not just a direct send", () => {
  const submit = slice(
    composer,
    "const handleSubmit = useCallback(",
    "startHydratedPromptQueue(",
  );
  assert.match(submit, /threadScopedSettingsPending && !overlay/);
});

test("the defaults sample is never taken from the outgoing chat's values", () => {
  // On A -> B the store still holds A's pills, so B's pairing must not sample them.
  const begin = slice(store, "export function beginThreadScopedPairing", "\n}");
  assert.match(
    begin,
    /threadScopedSettingsThreadId === null\s*\?\s*readThreadScopedSettings\(/,
  );
  assert.match(begin, /:\s*globalThreadScopedDefaults;/);
});

test("the read that gates sends cannot hang forever", () => {
  // The GET has no timeout, so each attempt needs its own deadline.
  const sync = slice(provider, "const sync = () => {", "// The read did not answer");
  assert.match(sync, /Promise\.race\(\[/);
  assert.match(sync, /THREAD_READ_TIMEOUT_MS/);
});

test("every run waits for the chat's settings, not just the composer", () => {
  // Reload, Continue and send-from-edit bypass handleSubmit, so the wait is in the adapter.
  const adapter = readText("../src/features/chat/api/chat-adapter.ts");
  const run = slice(adapter, "await useChatRuntimeStore.getState().hydratePersistedSettings();", "let runtime =");
  assert.match(run, /await awaitThreadScopedPairing\(runThreadId\)/);
});

test("a replay entry survives a failed replay", () => {
  // authFetch resolves for 404 and 5xx, so the replay must check the status.
  const replay = slice(store, "export function replayUnconfirmedThreadSettings", "\n}");
  assert.match(replay, /if \(res\.ok\) forgetReplayedThreadSettings\(threadId, body\)/);
  assert.doesNotMatch(
    replay,
    /localStorage\.removeItem\(THREAD_SETTINGS_REPLAY_KEY\);\n    if \(!raw\)/,
    "the whole batch is still dropped before it is known to have landed",
  );
});

test("a debounce-fired write is resendable on a terminal event too", () => {
  const schedule = slice(store, "function scheduleThreadScopedSettingsWrite", "\n}");
  assert.match(schedule, /trackUnsettledThreadSettingsWrite\(pendingThreadId, pendingSnapshot\)/);
});

test("forking settles a held edit, not just the debounce", () => {
  const composerSrc = composer;
  assert.match(composerSrc, /await settleThreadScopedSettingsForCopy\(remoteId\)/);
  const settle = slice(store, "export async function settleThreadScopedSettingsForCopy", "\n}");
  assert.match(settle, /commitHeldThreadScopedEditsToTheirThread\(\)/);
  const await_ = slice(store, "export async function awaitThreadScopedSettingsWrite", "\n}");
  assert.doesNotMatch(await_, /commitHeldThreadScopedEditsToTheirThread/);
});

test("the run's wait is bound to the run's own chat", () => {
  // Gates are per chat; one shared promise released A's run on B's pairing.
  const wait = slice(store, "export function awaitThreadScopedPairing", "\n}");
  assert.match(wait, /threadId: string \| null \| undefined/);
  assert.match(wait, /pairingSettledByThreadId\.get\(threadId\)/);
  const adapter = readText("../src/features/chat/api/chat-adapter.ts");
  assert.match(adapter, /await awaitThreadScopedPairing\(runThreadId\)/);
});

test("two unsettled writes for one chat do not cancel each other's tracking", () => {
  // Ordinary edits share a null snapshot, so tracking must be by identity, not value.
  const track = slice(store, "function trackUnsettledThreadSettingsWrite", "\n}");
  assert.match(track, /const entry: UnsettledThreadSettingsWrite = \{ snapshot \}/);
  assert.match(track, /unsettledThreadSettingsWrites\.get\(threadId\) === entry/);
});

test("a terminal event does not send a stale snapshot after the newest one", () => {
  // Each beacon takes a higher seq, so an older unsettled snapshot must not be re-sent.
  const terminal = slice(store, "function flushSettingsOnPageHidden", "\n}");
  assert.match(terminal, /const sentNewest = new Set<string>\(\)/);
  assert.match(terminal, /beaconUnsettledThreadSettingsWrites\(sentNewest\)/);
  const beacon = slice(store, "function beaconUnsettledThreadSettingsWrites", "\n}");
  assert.match(beacon, /if \(alreadySent\.has\(threadId\)\) continue;/);
});

test("last session's replay is ordered before this session's writes", () => {
  // The replay has the previous session's writer id, so it must settle before new edits.
  assert.match(store, /let threadSettingsReplaySettled: Promise<void>/);
  const write = slice(store, "function writeThreadScopedSettings", "\n}");
  assert.match(write, /\.then\(\(\) => threadSettingsReplaySettled\)/);
});

test("a retry keeps the defaults snapshot it already took", () => {
  const commit = slice(
    store,
    "export function commitHeldThreadScopedEditsToTheirThread",
    "\n}",
  );
  assert.doesNotMatch(commit, /pairingWindowDefaultsThreadId = null;/);
});

test("an explicit clear beats a capability preservation", () => {
  // Enabling Search clears Deep Research on purpose; do not restore the stored true.
  const build = slice(store, "function buildThreadScopedSnapshot", "\n}");
  assert.match(build, /!explicitlyEditedThreadFields\.has\("deepResearchEnabled"\)/);
  const capture = slice(store, "function captureThreadScopedEdit", "\n}");
  assert.match(capture, /explicitlyEditedThreadFields\.add\(field\)/);
});

test("the thread read that gates sends aborts when it times out", () => {
  // `bounded` is the 30s write timeout; each read attempt needs its own deadline.
  const sync = slice(provider, "const sync = () => {", "// The read did not answer");
  assert.match(sync, /timeoutMs: THREAD_READ_TIMEOUT_MS/);
  assert.match(sync, /signal: read\.signal/);
});

test("a read nobody is waiting for any more is cancelled", () => {
  const effect = slice(provider, "const reads = new Set<AbortController>();", "\n  }, [activeThreadId");
  assert.match(effect, /abortReads\(\);/);
  const api = readText("../src/features/chat/api/chat-api.ts");
  const get = slice(api, "export async function getChatThread", "\n}");
  assert.match(get, /options\.timeoutMs !== undefined/);
  assert.match(get, /combineAbortSignals\(\[timeout\.signal, options\.signal\]\)/);
});

test("the ensure step in front of a settings write is bounded too", () => {
  // It runs before the write, outside the caller's signal and write timeout.
  const storage = readText("../src/features/chat/utils/chat-history-storage.ts");
  const update = slice(storage, "export async function updateStoredChatThread", "\n}");
  assert.match(update, /ensureStoredChatThread\(threadId, undefined, \{/);
  assert.match(update, /bounded: true/);
  assert.match(update, /signal: options\.signal/);
});

test("a tab-close snapshot is replayed even if global settings fail to hydrate", () => {
  const hydrate = slice(store, "hydratePersistedSettings: async () => {", "beginModelLoading");
  const catchArm = slice(hydrate, "} catch {", "settingsHydrationPromise = null;");
  assert.match(catchArm, /replayUnconfirmedThreadSettings\(\);/);
  const replay = slice(store, "export function replayUnconfirmedThreadSettings", "\n}");
  assert.match(replay, /if \(threadSettingsReplayStarted\) return;/);
});

test("a default hydration had to skip is not restored from the pre-hydration copy", () => {
  assert.match(store, /const hydratedDefaultsByHeldField = new Map<string, unknown>\(\);/);
  const loop = slice(store, "for (const key of SCALAR_SETTING_KEYS) {", "return nextState;");
  assert.match(loop, /hydratedDefaultsByHeldField\.set\(key, value\)/);
  const restore = slice(store, "const beforeWindow = (pairingWindowDefaults ??", "globalThreadScopedDefaults = captured");
  assert.match(restore, /hydratedDefaultsByHeldField\.has\(field\)/);
  assert.ok(
    restore.indexOf("hydratedDefaultsByHeldField.has(field)") <
      restore.indexOf("field in beforeWindow"),
    "the pre-window copy still wins over the server's value",
  );
  const release = slice(store, "export function releaseHeldThreadScopedEdits", "\n}");
  assert.match(release, /hydratedDefaultsByHeldField\.delete\(edit\.field\)/);
});

test("one chat's pairing ending does not release another chat's run", () => {
  const close = slice(store, "function closeThreadScopedPairingGate", "\n}");
  assert.match(close, /pairingSettledByThreadId\.get\(threadId\)\?\.resolve\(\)/);
  assert.doesNotMatch(
    close,
    /for \(const \{ resolve \} of pairingSettledByThreadId\.values\(\)\) resolve\(\)/,
    "every gate is still released at once",
  );
  const commit = slice(
    store,
    "export function commitHeldThreadScopedEditsToTheirThread",
    "\n}",
  );
  assert.match(commit, /closeThreadScopedPairingGate\(null\)/);
});

test("a run cannot wait on a gate that will never open", () => {
  const wait = slice(store, "export function awaitThreadScopedPairing", "\n}");
  assert.match(wait, /Promise\.race\(\[/);
  assert.match(wait, /THREAD_PAIRING_WAIT_MS/);
});

test("a write that lands clears the replay entry it would otherwise be reverted by", () => {
  const write = slice(store, "function writeThreadScopedSettings", "\n}");
  assert.match(write, /forgetReplayedThreadSettings\(threadId\)/);
});

test("a failed write stays tracked for the terminal beacon", () => {
  const track = slice(store, "function trackUnsettledThreadSettingsWrite", "\n}");
  assert.match(track, /\.then\(\(landed\) =>/);
  assert.match(track, /if \(landed &&/);
});

test("the replay cannot block the session's writes forever", () => {
  const replay = slice(store, "export function replayUnconfirmedThreadSettings", "\n}");
  assert.match(replay, /THREAD_SETTINGS_REPLAY_TIMEOUT_MS/);
  assert.match(replay, /signal: timeout\.signal/);
});

test("a fork stops when the chat's settings could not be saved", () => {
  const merge = slice(store, "async function mergeThreadScopedSettingsIntoRow", "\n}");
  assert.match(merge, /throw error;/);
  assert.match(composer, /Could not fork this chat/);
});

test("an unsaved chat's edit reaches the installation defaults without a round trip", () => {
  // A chat is unsaved only until its first send, so key on the pending-new-thread id,
  // not the permanent `__LOCALID_` prefix.
  const effect = slice(
    provider,
    "const { applyThreadScopedSettings } = useChatRuntimeStore.getState();",
    "if (!enabled) {",
  );
  assert.match(effect, /activeThreadId === pendingNewThreadId/);
  assert.doesNotMatch(effect, /isAssistantLocalThreadId/);
  assert.match(effect, /applyThreadScopedSettings\(null, null\)/);
});

test("the pairing effect tracks the runtime's pending new thread", () => {
  assert.match(
    provider,
    /const pendingNewThreadId = useAuiState\(\(\{ threads \}\) => threads\.newThreadId\)/,
  );
  const deps = slice(provider, "}, [activeThreadId, enabled,", ");");
  assert.match(deps, /pendingNewThreadId/);
});

test("a run whose pairing never settled is refused, not run on another chat's settings", () => {
  const wait = slice(store, "export function awaitThreadScopedPairing", "\n}");
  assert.match(wait, /Promise<boolean>/);
  assert.match(wait, /resolve\(false\)/);
  const adapter = readText("../src/features/chat/api/chat-adapter.ts");
  assert.match(adapter, /if \(!\(await awaitThreadScopedPairing\(runThreadId\)\)\) \{/);
  assert.match(adapter, /the message was not sent/);
});

test("the pairing wait outlasts the read it is waiting for", () => {
  // Must stay longer than the read's own budget, or a slow read fails the send.
  const waitMs = /THREAD_PAIRING_WAIT_MS = ([\d_]+)/.exec(store);
  assert.ok(waitMs, "no pairing wait constant");
  const pairing = Number(waitMs[1].replace(/_/g, ""));
  const readMs = /THREAD_READ_TIMEOUT_MS = ([\d_]+)/.exec(provider);
  const retries = /THREAD_READ_RETRIES = (\d+)/.exec(provider);
  const gap = /THREAD_READ_RETRY_MS = ([\d_]+)/.exec(provider);
  assert.ok(readMs && retries && gap, "no read budget constants");
  const budget =
    (Number(retries[1]) + 1) * Number(readMs[1].replace(/_/g, "")) +
    Number(retries[1]) * Number(gap[1].replace(/_/g, ""));
  assert.ok(
    pairing > budget,
    `pairing wait ${pairing}ms does not outlast the read budget ${budget}ms`,
  );
});

test("a fork does not copy a row whose edit failed to reach it", () => {
  // The replacement write resolves false on failure; awaiting the chain alone cannot tell.
  const await_ = slice(store, "export async function awaitThreadScopedSettingsWrite", "\n}");
  assert.match(await_, /Promise<boolean>/);
  assert.match(await_, /landed !== false/);
  const settle = slice(store, "export async function settleThreadScopedSettingsForCopy", "\n}");
  assert.match(settle, /if \(!\(await awaitThreadScopedSettingsWrite\(threadId\)\)\) \{/);
  assert.match(settle, /throw new Error/);
});

test("a replay only clears the body it actually sent", () => {
  const replay = slice(store, "export function replayUnconfirmedThreadSettings", "\n}");
  assert.match(replay, /forgetReplayedThreadSettings\(threadId, body\)/);
  const forget = slice(store, "function forgetReplayedThreadSettings", "\n}");
  assert.match(forget, /JSON\.stringify\(pending\[threadId\]\) !== JSON\.stringify\(expected\)/);
});

test("a provider constraint does not rewrite what the chat stored", () => {
  // Kimi's builtin search cannot run with thinking; that provider-forced value is not saved.
  assert.match(store, /const constraintSuppressedThreadFields = new Set<string>\(\);/);
  const reasoning = slice(store, "setReasoningEnabled: (reasoningEnabled, options)", "\n    }),");
  assert.match(reasoning, /noteConstraintSuppressedThreadField\("reasoningEnabled"\)/);
  const tools = slice(store, "setToolsEnabled: (toolsEnabled, options)", "\n    }),");
  assert.match(tools, /noteConstraintSuppressedThreadField\("toolsEnabled"\)/);
  const build = slice(store, "function buildThreadScopedSnapshot", "\nconst THREAD_SETTINGS_REPLAY_KEY");
  assert.match(build, /keepsStoredValueUnderConstraint\("reasoningEnabled", threadId, settings\)/);
  assert.match(build, /keepsStoredValueUnderConstraint\("toolsEnabled", threadId, settings\)/);
  const keeps = slice(store, "function keepsStoredValueUnderConstraint", "\n}");
  assert.match(keeps, /!explicitlyEditedThreadFields\.has\(key\)/);
  const capture = slice(store, "function captureThreadScopedEdit", "\n}");
  assert.match(capture, /constraintSuppressedThreadFields\.delete\(field\)/);
  assert.match(store, /constraintSuppressedThreadFields\.clear\(\);/);
});

// Sampling params live under `params`; a direct field read is undefined and stores nothing.
test("the sampling params are read and applied through params", () => {
  assert.match(
    store,
    /return isThreadScopedParamKey\(key\)\s*\?\s*state\.params\[key\]\s*:\s*\(state as Record<string, unknown>\)\[key\];/,
  );
  assert.equal(
    store.match(/readThreadScopedValue\(state, key\)/g)?.length,
    3,
    "the snapshot, the held edit and the sameness check",
  );
  assert.match(store, /paramsPatch\[key\] = value;/);
  assert.match(
    store,
    /if \(hasKeys\(paramsPatch\)\) \{\s*nextState\.params = \{ \.\.\.state\.params, \.\.\.paramsPatch \};/,
  );
});

// A model's recommendation is not a user choice and must not pin the chat.
test("only a user edit to a sampling param lands on the chat", () => {
  const drop = slice(store, "function withoutCapturedThreadEdits", "\n}");
  assert.match(
    drop,
    /isThreadScopedParamKey\(key\) &&\s*!fromModelDefaults &&[\s\S]{0,300}?captureThreadScopedEdit\(\s*key\b/,
  );
  const setParams = slice(store, "setParams: (params, options)", "\n  setCustomPresets:");
  assert.match(setParams, /persistParamEdit\(\s*sharedParams,/);
  assert.match(setParams, /getParamsByModelAfterEdit\([\s\S]{0,200}?sharedParams,/);
  assert.doesNotMatch(
    setParams,
    /getParamsByModelAfterEdit\([\s\S]{0,200}?changedParams,/,
    "the chat's edit is remembered against the model and leaks to new chats on it",
  );

  const runtime = readText("../src/features/chat/hooks/use-chat-model-runtime.ts");
  const status = readText("../src/features/chat/lib/apply-inference-status-to-store.ts");
  for (const source of [runtime, status]) {
    assert.match(
      source,
      /mergeBackendRecommendedInference\([\s\S]{0,1200}?fromModelDefaults: true/,
    );
  }
});

test("an edit held through the pairing window keeps its sampling value", () => {
  const changes = slice(store, "function heldThreadScopedChanges", "\n}");
  assert.match(changes, /readThreadScopedValue\(\s*live,\s*edit\.field as ThreadScopedSettingKey,\s*\)/);
  assert.doesNotMatch(
    changes,
    /edited\[edit\.field\] = live\[edit\.field\]/,
    "reads the field directly, which is undefined for every sampling key",
  );
});

// fromModelDefaults only changed persistence; the recommendation still hit the live params.
test("a model's recommendation does not overwrite the chat's sampling", () => {
  const setParams = slice(store, "setParams: (params, options)", "\n  setCustomPresets:");
  assert.match(
    setParams,
    /const effective = replayed\s*\?\s*restoreThreadScopedParams\(nextParams\)\s*:\s*nextParams;/,
  );
  assert.match(setParams, /const replayed = checkpointChanged \|\| fromModelDefaults;/);
  assert.match(setParams, /params: effective,/);
  assert.doesNotMatch(setParams, /params: nextParams,/);

  const restore = slice(store, "function restoreThreadScopedParams", "\n}");
  assert.match(restore, /const held = [\s\S]{0,120}?threadScopedOverride\(key\)/);
  assert.match(restore, /if \(held === undefined/);
  // ?? and never ||: 0, "" and -1 are deliberate values.
  assert.doesNotMatch(restore, /\|\| threadScopedOverride\(key\)/);
});

// Think mode must apply its params even when the chat pins sampling.
test("toggling Think applies its params even in a chat that pins sampling", () => {
  const qwen = readText("../src/features/chat/utils/qwen-params.ts");
  assert.match(qwen, /store\.setParams\(\{ \.\.\.store\.params, \.\.\.params \}\);/);
  assert.doesNotMatch(
    qwen,
    /fromModelDefaults/,
    "the toggle is treated as a model default, so a pinned chat never changes mode params",
  );
  const runtime = readText("../src/features/chat/hooks/use-chat-model-runtime.ts");
  const post = slice(runtime, "store.setParams({ ...store.params, ...p }", "\n              }");
  assert.match(post, /fromModelDefaults: true/);
});

// Diff against the model's values, or a pinned key hides the model default from installation.
test("a chat pinning a param does not withhold the model's default from the rest", () => {
  const setParams = slice(store, "setParams: (params, options)", "\n  setCustomPresets:");
  assert.match(
    setParams,
    /getChangedInferenceParams\(\s*nextParams,\s*state\.params,\s*!fromModelDefaults,(\s*[^)]*,)?\s*\)/,
  );
  assert.doesNotMatch(
    setParams,
    /getChangedInferenceParams\(\s*effective,/,
    "the restored object decides what is persisted, so pinned keys are withheld",
  );
  // Called once: it bumps mutation versions, so a second diff double-counts.
  assert.equal(setParams.match(/getChangedInferenceParams\(/g)?.length, 1);
  assert.match(setParams, /params: effective,/);
});

// applyThreadScopedSettings falls back to an in-memory copy that must track written defaults.
test("the in-memory defaults follow the model defaults that were just written", () => {
  const setParams = slice(store, "setParams: (params, options)", "\n  setCustomPresets:");
  assert.match(setParams, /noteThreadScopedDefaults\(sharedParams\);/);
  const note = slice(store, "function noteThreadScopedDefaults", "\n}");
  assert.match(note, /if \(!isThreadScopedParamKey\(key\)\) continue;/);
  assert.match(note, /if \(globalThreadScopedDefaults === null\) continue;/);
  // A default published inside the pairing window must be recorded for held-field restore.
  assert.match(
    note,
    /if \(isHeldThreadScopedField\(key\)\) \{\s*hydratedDefaultsByHeldField\.set\(key, value\);/,
  );
  // Not ??: a cleared seed is stored as null and must not fall through to the installation pin.
  assert.match(
    store,
    /firstSetThreadScopedValue\(\s*stored\?\.\[key\],\s*globalThreadScopedDefaults\?\.\[key\],/,
  );
});

// setCheckpoint replays remembered params outside setParams, so it must restore the chat's.
test("switching model in a chat keeps the chat's sampling, not the model's", () => {
  const set = slice(store, "setCheckpoint: (modelId, ggufVariant, options)", "\n  setActiveThreadId:");
  assert.match(
    set,
    /const restoredParams = checkpointChanged\s*\?\s*restoreThreadScopedParams\(nextParams\)\s*:\s*nextParams;/,
  );
  assert.match(set, /params: restoredParams,/);
  assert.match(set, /getReplayStatePatch\(state, nextParams, outgoing, baseParams\)/);
});

// The outgoing model's snapshot is written first, so it must exclude the open chat's values.
test("the model being left does not remember the open chat's values", () => {
  const remember = slice(store, "function rememberOutgoingModel", "\n}");
  assert.match(
    remember,
    /pickRememberedParams\(\s*withoutActiveThreadParams\(state, outgoing\),\s*\)/,
  );
  assert.doesNotMatch(
    remember,
    /pickRememberedParams\(outgoing\)/,
    "the chat's sampling and prompt are stored as the model's own",
  );
  const strip = slice(store, "function withoutActiveThreadParams", "\n}");
  assert.match(
    strip,
    /if \(held === undefined && threadScopedOverride\(key\) === undefined\) continue;/,
  );
  assert.match(
    strip,
    /firstSetThreadScopedValue\(\s*remembered\?\.\[key\],\s*globalThreadScopedDefaults\?\.\[key\],/,
  );
  assert.match(
    strip,
    /held !== undefined \? pairingWindowDefaults\?\.\[key\] : undefined/,
  );
  assert.match(
    strip,
    /if \(threadScopedSettingsThreadId === null && pendingPairingThreadId === null\)/,
  );
});
