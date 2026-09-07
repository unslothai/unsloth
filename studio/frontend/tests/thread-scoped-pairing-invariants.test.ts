// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The pairing window is the gap between opening a saved chat and its snapshot arriving.
// Everything in it is racy by nature: the composer is live, the installation defaults are
// on screen, and three async paths (global hydration, the thread read, the debounced
// write) touch the same module state. Each invariant below is one that has actually been
// broken during review, and breaking any of them again either leaks one chat's settings
// into another or loses an edit the user watched happen.
//
// Source assertions rather than a driven store: a .tsx barrel sits in the store's import
// graph, so it cannot be loaded in a bare node test. The sibling store tests do the same.

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

function read(path: string): string {
  return readFileSync(new URL(path, import.meta.url), "utf8");
}

const store = read("../src/features/chat/stores/chat-runtime-store.ts");
const provider = read("../src/features/chat/runtime-provider.tsx");
const composer = read("../src/components/assistant-ui/thread.tsx");

function slice(source: string, from: string, to: string): string {
  const start = source.indexOf(from);
  assert.ok(start !== -1, `not found: ${from}`);
  const end = source.indexOf(to, start + from.length);
  assert.ok(end !== -1, `not found: ${to}`);
  return source.slice(start, end);
}

test("the read waits for this chat's own write before it can be believed", () => {
  // A chat edited, left and re-entered has its PATCH in flight; a GET that overtakes it
  // returns the pre-edit snapshot, which is then applied over what the user set.
  const sync = slice(provider, "const sync = () => {", "// The read did not answer");
  assert.match(sync, /awaitThreadScopedSettingsWrite\(activeThreadId\)/);
  // And the row's own creation: initialize() resolves as soon as the id is minted and leaves
  // the POST tracked, so a first send's read can overtake it, see no row, and release this
  // chat's held edits into the installation defaults.
  assert.match(sync, /awaitStoredChatThreadWrites\(activeThreadId\)/);
  // and the read only happens after it, not alongside
  assert.ok(
    sync.indexOf("awaitThreadScopedSettingsWrite") <
      sync.indexOf("getStoredChatThreadReadResult"),
    "the read is not sequenced after the write",
  );
});

/** sync()'s body with comments removed, for the two order assertions below. These call
 * sites are heavily commented, and a prose mention of a call ahead of the call itself
 * would otherwise read as the call being in the wrong place. */
function syncCode(): string {
  return slice(provider, "const sync = () => {", "// The read did not answer")
    .replace(/\/\*[\s\S]*?\*\//g, "")
    .replace(/^[ \t]*\/\/.*$/gm, "");
}

test("both waits sit inside the attempt's deadline, not in front of it", () => {
  // Neither wait is bounded on its own, so in front of the deadline their time goes
  // uncounted and a stalled write ends in a refused send. The reasoning is at the call site.
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
  // The arithmetic THREAD_PAIRING_WAIT_MS's own comment claims. Read from source so the
  // two cannot drift apart: whichever constant moves, this is what catches it.
  const constant = (source: string, name: string): number => {
    const match = source.match(new RegExp(`${name}\\s*=\\s*([0-9_]+)`));
    assert.ok(match, `${name} not found`);
    return Number(match[1].replace(/_/g, ""));
  };
  const store = read("../src/features/chat/stores/chat-runtime-store.ts");

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
  // threadScopedSettingsPending gates the composer, so a pairing left open with no retry
  // left holds every send behind a wait that nothing can resolve.
  const retry = slice(provider, "const retryThreadRead = () => {", "\n    sync();");
  const exhausted = slice(retry, "if (retriesLeft <= 0) {", "return;");
  assert.match(exhausted, /applyThreadScopedSettings\(null, null\)/);
  assert.match(exhausted, /releaseHeldThreadScopedEdits\(\)/);
  // and it says so, rather than silently running on settings the chat did not choose
  assert.match(exhausted, /toast\.error/);
  // the pairing is only re-opened while tries remain
  const reopen = retry.indexOf("beginThreadScopedPairing");
  assert.ok(
    reopen > retry.indexOf("if (retriesLeft <= 0) {"),
    "pairing is re-opened before the exhaustion check, which leaves it open forever",
  );
});

test("a failed backend read is retried, not treated as a chat with no snapshot", () => {
  // cacheable:false means the GET failed and Dexie answered; releasing here would send
  // the chat's edits to the installation defaults and leave its real snapshot unread.
  const fallback = slice(provider, "if (thread && !cacheable) {", "}");
  assert.match(fallback, /retryThreadRead\(\)/);
});

test("a chat with no row releases its held edits on the first answer", () => {
  // Waiting for a second missing read meant an unsaved chat's click was written to a row
  // that does not exist, or attached to the chat once it was saved.
  const missing = slice(provider, "if (!thread) {", "unpaired = true;");
  assert.match(missing, /releaseHeldThreadScopedEdits\(\)/);
});

test("the defaults are captured when pairing begins, not reconstructed later", () => {
  // An edit during the window overwrites the store, and on the session's first pairing
  // there is no earlier capture to fall back on: the value has to be taken up front.
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
  // Those edits are that chat's. Releasing them here moves every snapshot-less chat's
  // default, which is the leak the whole hold-and-commit path exists to prevent.
  const guard = slice(
    store,
    "      } else if (",
    "releaseHeldThreadScopedEdits();",
  );
  assert.match(guard, /threadId !== null \|\|/);
  assert.match(guard, /pendingPairingThreadId !== state\.activeThreadId/);
});

test("an edit whose chat was never read is sent as a merge, not a replacement", () => {
  // The store is showing the installation defaults, so a full snapshot built from them
  // erases everything the chat had stored that the user did not touch.
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
  // A held edit advances no mutation version, so the server's value would replace it and
  // then be persisted to the thread as if the user had chosen it.
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
  // Aborting a fetch does not stop a handler the server has already started, so the
  // ordering is enforced server-side and each write has to carry its stamp.
  assert.match(store, /function nextThreadSettingsSeq\(\): number \{/);
  const write = slice(store, "function writeThreadScopedSettings", "\n}");
  assert.match(write, /const settingsSeq = nextThreadSettingsSeq\(\);/);
  assert.match(write, /\{ settings, settingsSeq, settingsWriter/);
  const beacon = slice(store, "function sendThreadScopedSettingsBeacon", "\n}");
  assert.match(beacon, /settingsSeq: nextThreadSettingsSeq\(\)/);
  // the beacon also stands down anything queued and cancels anything already out
  assert.match(beacon, /takeThreadSettingsWriteTicket\(threadId\)/);
  assert.match(beacon, /threadSettingsWriteAborts\.get\(threadId\)\?\.abort\(\)/);
});

test("a normal flush stays resendable until it lands", () => {
  // visibilitychange(hidden) then pagehide: the first flushes normally and clears the
  // pending snapshot, so the terminal event would find nothing to beacon.
  const flush = slice(store, "function flushThreadScopedSettingsWrite", "\n}");
  assert.match(flush, /trackUnsettledThreadSettingsWrite\(threadId, snapshot\)/);
  const track = slice(store, "function trackUnsettledThreadSettingsWrite", "\n}");
  assert.match(track, /unsettledThreadSettingsWrites\.set\(threadId, entry\)/);
  const terminal = slice(store, "function flushSettingsOnPageHidden", "\n}");
  assert.match(terminal, /commitHeldThreadScopedEditsToTheirThread\(true\)/);
  assert.match(terminal, /beaconUnsettledThreadSettingsWrites\(sentNewest\)/);
});

test("a capability clamp never overwrites the preference the chat stored", () => {
  // The clamp is the model's, not the user's. All four pills chat-page clamps go through
  // the same preservation, under each pill's own capability rule.
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
  // Comparing unrelated clients means whichever is behind has every edit refused while
  // the server still answers 200, so the user is told it saved and it did not.
  assert.match(store, /const threadSettingsWriter = crypto\.randomUUID\(\);/);
  const next = slice(store, "function nextThreadSettingsSeq", "\n}");
  assert.doesNotMatch(next, /Date\.now\(\)/, "the seq is a clock again");
  // every write says who it came from, or the server cannot scope the comparison
  for (const site of [
    slice(store, "function sendThreadScopedSettingsBeacon", "\n}"),
    slice(store, "function writeThreadScopedSettings", "\n}"),
    slice(store, "async function mergeThreadScopedSettingsIntoRow", "\n}"),
  ]) {
    assert.match(site, /settingsWriter/);
  }
});

test("a tab-close write that could not be confirmed is replayed next session", () => {
  // A chat whose row is still being created answers 404, and the creation that follows
  // knows nothing about the edit.
  const beacon = slice(store, "function sendThreadScopedSettingsBeacon", "\n}");
  assert.match(beacon, /rememberThreadSettingsForReplay\(threadId, body\)/);
  assert.match(store, /export function replayUnconfirmedThreadSettings/);
  // and something actually calls it on the way back in
  assert.match(store, /replayUnconfirmedThreadSettings\(\);/);
});

test("a model that forces thinking on does not erase a chat's stored preference", () => {
  const build = slice(store, "function buildThreadScopedSnapshot", "\n}");
  assert.match(build, /activeThreadScopedSettings\?\.reasoningEnabled === false/);
  assert.match(build, /reasoningAlwaysOn/);
});

test("compare mode drops the thread-scoped state rather than keeping the last chat's", () => {
  // One composer, two threads: no single snapshot applies, and leaving the module
  // pointing at the last single chat lets a model load read its pills back.
  const disabled = slice(
    provider,
    "// Compare panes share one composer",
    "return;",
  );
  assert.match(disabled, /applyThreadScopedSettings\(null, null\)/);
});

test("a retry does not resample the defaults over the edit it is holding", () => {
  // retryThreadRead commits and re-pairs with the edited value still in the store, so
  // sampling per attempt would take that edit for the installation default.
  const begin = slice(store, "export function beginThreadScopedPairing", "\n}");
  assert.match(begin, /if \(pairingWindowDefaultsThreadId !== threadId\) \{/);
  // and the sample is let go once the window closes, or the next visit reuses it
  assert.match(
    slice(store, "applyThreadScopedSettings: (threadId, settings) =>", "} else if ("),
    /pairingWindowDefaultsThreadId = null;/,
  );
});

test("a pending pin answers the override rather than the null it has stored", () => {
  // A chat being pinned has no snapshot yet, so falling through reads null and a model
  // load puts the global values back over the edit the queued write will persist.
  const override = slice(store, "export function threadScopedOverride", "\n}");
  assert.match(override, /threadSettingsWriteSnapshot\[key\] !== undefined/);
});

test("the prompt queue waits for this chat's settings too, not just a direct send", () => {
  // handleSubmit reaches the queue branch first, and the queued prompt is snapshotted
  // from whatever the store shows at that moment.
  const submit = slice(
    composer,
    "const handleSubmit = useCallback(",
    "startHydratedPromptQueue(",
  );
  assert.match(submit, /threadScopedSettingsPending && !overlay/);
});

test("the defaults sample is never taken from the outgoing chat's values", () => {
  // Switching A -> B, the store still holds A's pills when B's pairing opens, so
  // sampling it would make A's choices the default every snapshot-less chat follows.
  const begin = slice(store, "export function beginThreadScopedPairing", "\n}");
  assert.match(
    begin,
    /threadScopedSettingsThreadId === null\s*\?\s*readThreadScopedSettings\(/,
  );
  assert.match(begin, /:\s*globalThreadScopedDefaults;/);
});

test("the read that gates sends cannot hang forever", () => {
  // The underlying GET has no timeout, and neither the catch nor the retry budget can
  // run for a promise that never settles, so every send stays parked.
  const sync = slice(provider, "const sync = () => {", "// The read did not answer");
  assert.match(sync, /Promise\.race\(\[/);
  assert.match(sync, /THREAD_READ_TIMEOUT_MS/);
});

test("every run waits for the chat's settings, not just the composer", () => {
  // Reload, Continue and send-from-edit never touch handleSubmit; they all reach the
  // adapter, so the wait belongs there.
  const adapter = read("../src/features/chat/api/chat-adapter.ts");
  const run = slice(adapter, "await useChatRuntimeStore.getState().hydratePersistedSettings();", "let runtime =");
  assert.match(run, /await awaitThreadScopedPairing\(runThreadId\)/);
});

test("a replay entry survives a failed replay", () => {
  // authFetch resolves for 404 and 5xx, and the missing-row case this exists for is
  // exactly the one that 404s.
  const replay = slice(store, "export function replayUnconfirmedThreadSettings", "\n}");
  assert.match(replay, /if \(res\.ok\) forgetReplayedThreadSettings\(threadId, body\)/);
  assert.doesNotMatch(
    replay,
    /localStorage\.removeItem\(THREAD_SETTINGS_REPLAY_KEY\);\n    if \(!raw\)/,
    "the whole batch is still dropped before it is known to have landed",
  );
});

test("a debounce-fired write is resendable on a terminal event too", () => {
  // Once the timer fires there is no pending debounce left to find, so the in-flight
  // request is the only copy and teardown cancels it.
  const schedule = slice(store, "function scheduleThreadScopedSettingsWrite", "\n}");
  assert.match(schedule, /trackUnsettledThreadSettingsWrite\(pendingThreadId, pendingSnapshot\)/);
});

test("forking settles a held edit, not just the debounce", () => {
  const composerSrc = composer;
  assert.match(composerSrc, /await settleThreadScopedSettingsForCopy\(remoteId\)/);
  const settle = slice(store, "export async function settleThreadScopedSettingsForCopy", "\n}");
  assert.match(settle, /commitHeldThreadScopedEditsToTheirThread\(\)/);
  // and the pairing's own wait must NOT commit, or sync closes the window it just opened
  const await_ = slice(store, "export async function awaitThreadScopedSettingsWrite", "\n}");
  assert.doesNotMatch(await_, /commitHeldThreadScopedEditsToTheirThread/);
});

test("the run's wait is bound to the run's own chat", () => {
  // One promise for all chats meant a run for A was released by B's pairing ending, and
  // then read B's settings for A's run.
  const wait = slice(store, "export function awaitThreadScopedPairing", "\n}");
  assert.match(wait, /threadId: string \| null \| undefined/);
  assert.match(wait, /pairingSettledByThreadId\.get\(threadId\)/);
  const adapter = read("../src/features/chat/api/chat-adapter.ts");
  assert.match(adapter, /await awaitThreadScopedPairing\(runThreadId\)/);
});

test("two unsettled writes for one chat do not cancel each other's tracking", () => {
  // Both ordinary edits carry a null snapshot, so comparing by value let the first
  // request's settle delete the second's entry.
  const track = slice(store, "function trackUnsettledThreadSettingsWrite", "\n}");
  assert.match(track, /const entry: UnsettledThreadSettingsWrite = \{ snapshot \}/);
  assert.match(track, /unsettledThreadSettingsWrites\.get\(threadId\) === entry/);
});

test("a terminal event does not send a stale snapshot after the newest one", () => {
  // Each beacon takes a higher seq than the last, so re-sending an older unsettled
  // snapshot for a chat already flushed would make the stale one win.
  const terminal = slice(store, "function flushSettingsOnPageHidden", "\n}");
  assert.match(terminal, /const sentNewest = new Set<string>\(\)/);
  assert.match(terminal, /beaconUnsettledThreadSettingsWrites\(sentNewest\)/);
  const beacon = slice(store, "function beaconUnsettledThreadSettingsWrites", "\n}");
  assert.match(beacon, /if \(alreadySent\.has\(threadId\)\) continue;/);
});

test("last session's replay is ordered before this session's writes", () => {
  // The replay carries the previous session's writer id, so nothing on the server
  // orders it against an edit made now; on a slow link it can land second and revert it.
  assert.match(store, /let threadSettingsReplaySettled: Promise<void>/);
  const write = slice(store, "function writeThreadScopedSettings", "\n}");
  assert.match(write, /\.then\(\(\) => threadSettingsReplaySettled\)/);
});

test("a retry keeps the defaults snapshot it already took", () => {
  // retryThreadRead commits and re-pairs the same chat, so clearing the id here would
  // let the retry resample the store, which by then holds the held edit.
  const commit = slice(
    store,
    "export function commitHeldThreadScopedEditsToTheirThread",
    "\n}",
  );
  assert.doesNotMatch(commit, /pairingWindowDefaultsThreadId = null;/);
});

test("an explicit clear beats a capability preservation", () => {
  // Enabling Search clears Deep Research deliberately; restoring the stored true would
  // bring it back, alongside Search, once the chat is on a local model again.
  const build = slice(store, "function buildThreadScopedSnapshot", "\n}");
  assert.match(build, /!explicitlyEditedThreadFields\.has\("deepResearchEnabled"\)/);
  const capture = slice(store, "function captureThreadScopedEdit", "\n}");
  assert.match(capture, /explicitlyEditedThreadFields\.add\(field\)/);
});

test("the thread read that gates sends aborts when it times out", () => {
  // `bounded` alone is the 30s WRITE timeout, so the losing side of the race below stayed
  // open while the retry opened the next one. Each attempt carries its own deadline.
  const sync = slice(provider, "const sync = () => {", "// The read did not answer");
  assert.match(sync, /timeoutMs: THREAD_READ_TIMEOUT_MS/);
  assert.match(sync, /signal: read\.signal/);
});

test("a read nobody is waiting for any more is cancelled", () => {
  const effect = slice(provider, "const reads = new Set<AbortController>();", "\n  }, [activeThreadId");
  assert.match(effect, /abortReads\(\);/);
  const api = read("../src/features/chat/api/chat-api.ts");
  const get = slice(api, "export async function getChatThread", "\n}");
  assert.match(get, /options\.timeoutMs !== undefined/);
  assert.match(get, /combineAbortSignals\(\[timeout\.signal, options\.signal\]\)/);
});

test("the ensure step in front of a settings write is bounded too", () => {
  // It runs BEFORE the write, so neither the caller's signal nor the write timeout
  // reaches it, and a stall there leaves the whole per-thread chain pending.
  const storage = read("../src/features/chat/utils/chat-history-storage.ts");
  const update = slice(storage, "export async function updateStoredChatThread", "\n}");
  assert.match(update, /ensureStoredChatThread\(threadId, undefined, \{/);
  assert.match(update, /bounded: true/);
  assert.match(update, /signal: options\.signal/);
});

test("a tab-close snapshot is replayed even if global settings fail to hydrate", () => {
  const hydrate = slice(store, "hydratePersistedSettings: async () => {", "beginModelLoading");
  const catchArm = slice(hydrate, "} catch {", "settingsHydrationPromise = null;");
  assert.match(catchArm, /replayUnconfirmedThreadSettings\(\);/);
  // and it must not then run twice
  const replay = slice(store, "export function replayUnconfirmedThreadSettings", "\n}");
  assert.match(replay, /if \(threadSettingsReplayStarted\) return;/);
});

test("a default hydration had to skip is not restored from the pre-hydration copy", () => {
  // The held edit belongs to the chat; when its window closes the installation default
  // goes back to a value, and the server's is the authoritative one.
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
  // Holding the promises per chat is pointless if every settle resolves all of them,
  // which is what the first attempt at this did.
  const close = slice(store, "function closeThreadScopedPairingGate", "\n}");
  assert.match(close, /pairingSettledByThreadId\.get\(threadId\)\?\.resolve\(\)/);
  assert.doesNotMatch(
    close,
    /for \(const \{ resolve \} of pairingSettledByThreadId\.values\(\)\) resolve\(\)/,
    "every gate is still released at once",
  );
  // leaving mid-read must NOT open the gate: that chat's snapshot never arrived
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
  // The replay carries the previous session's writer id, so the server will apply it
  // whenever it arrives, including over an edit made since.
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
  // Holding edits behind a read certain to 404 is what stopped a pill clicked on a fresh
  // /chat from reaching localStorage at once, which playwright_chat_ui asserts. A chat is
  // unsaved only until its first send, so the test is the runtime's pending-new-thread id:
  // the `__LOCALID_` prefix stays for good and gated every app-created chat out of its own
  // settings (#8686).
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
  // The wait only runs out for a chat left mid-read, whose gate is held shut on purpose;
  // by then the store describes whatever chat the user moved to.
  const wait = slice(store, "export function awaitThreadScopedPairing", "\n}");
  assert.match(wait, /Promise<boolean>/);
  assert.match(wait, /resolve\(false\)/);
  const adapter = read("../src/features/chat/api/chat-adapter.ts");
  assert.match(adapter, /if \(!\(await awaitThreadScopedPairing\(runThreadId\)\)\) \{/);
  assert.match(adapter, /the message was not sent/);
});

test("the pairing wait outlasts the read it is waiting for", () => {
  // Shorter than the read's own budget and an ordinary slow read would fail the send.
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
  // The replacement write reports failure by resolving false, so awaiting the chain
  // alone cannot tell a saved edit from a lost one.
  const await_ = slice(store, "export async function awaitThreadScopedSettingsWrite", "\n}");
  assert.match(await_, /Promise<boolean>/);
  assert.match(await_, /landed !== false/);
  const settle = slice(store, "export async function settleThreadScopedSettingsForCopy", "\n}");
  assert.match(settle, /if \(!\(await awaitThreadScopedSettingsWrite\(threadId\)\)\) \{/);
  assert.match(settle, /throw new Error/);
});

test("a replay only clears the body it actually sent", () => {
  // A terminal event in this session can store a newer body for the same thread while
  // an older replay is still out; that newer one is unconfirmed by definition.
  const replay = slice(store, "export function replayUnconfirmedThreadSettings", "\n}");
  assert.match(replay, /forgetReplayedThreadSettings\(threadId, body\)/);
  const forget = slice(store, "function forgetReplayedThreadSettings", "\n}");
  assert.match(forget, /JSON\.stringify\(pending\[threadId\]\) !== JSON\.stringify\(expected\)/);
});

test("a provider constraint does not rewrite what the chat stored", () => {
  // Kimi's builtin search may not run with thinking, so the composer moves the other
  // pill with { persist: false } -- which bypasses the capture path on purpose. That
  // value is the provider's, not the user's, and the next snapshot must not save it.
  assert.match(store, /const constraintSuppressedThreadFields = new Set<string>\(\);/);
  const reasoning = slice(store, "setReasoningEnabled: (reasoningEnabled, options)", "\n    }),");
  assert.match(reasoning, /noteConstraintSuppressedThreadField\("reasoningEnabled"\)/);
  const tools = slice(store, "setToolsEnabled: (toolsEnabled, options)", "\n    }),");
  assert.match(tools, /noteConstraintSuppressedThreadField\("toolsEnabled"\)/);
  const build = slice(store, "function buildThreadScopedSnapshot", "\nconst THREAD_SETTINGS_REPLAY_KEY");
  assert.match(build, /keepsStoredValueUnderConstraint\("reasoningEnabled", threadId, settings\)/);
  assert.match(build, /keepsStoredValueUnderConstraint\("toolsEnabled", threadId, settings\)/);
  // but a choice the user makes themselves still wins, and the flag is the open chat's
  const keeps = slice(store, "function keepsStoredValueUnderConstraint", "\n}");
  assert.match(keeps, /!explicitlyEditedThreadFields\.has\(key\)/);
  const capture = slice(store, "function captureThreadScopedEdit", "\n}");
  assert.match(capture, /constraintSuppressedThreadFields\.delete\(field\)/);
  assert.match(store, /constraintSuppressedThreadFields\.clear\(\);/);
});

// The sampling params live under `params`, so the generic key loops have to be told where
// to look: reading the field directly gets undefined and stores nothing, which looks
// exactly like the feature working until you reopen.
test("the sampling params are read and applied through params", () => {
  assert.match(
    store,
    /return isThreadScopedParamKey\(key\)\s*\?\s*state\.params\[key\]\s*:\s*\(state as Record<string, unknown>\)\[key\];/,
  );
  // Read through the same helper on both paths, including the held-edit branch.
  assert.equal(
    store.match(/readThreadScopedValue\(state, key\)/g)?.length,
    3,
    "the snapshot, the held edit and the sameness check",
  );
  // One object, so they are gathered and set together rather than as fields.
  assert.match(store, /paramsPatch\[key\] = value;/);
  assert.match(
    store,
    /if \(hasKeys\(paramsPatch\)\) \{\s*nextState\.params = \{ \.\.\.state\.params, \.\.\.paramsPatch \};/,
  );
});

// A model's own recommendation is not a choice the user made in this chat. Storing it
// would pin every chat to whatever model happened to load while it was open.
test("only a user edit to a sampling param lands on the chat", () => {
  const drop = slice(store, "function withoutCapturedThreadEdits", "\n}");
  assert.match(
    drop,
    // The capture may carry the edited value as a further argument; what this pins is
    // that it happens for a sampling key, and only when the value is not the model's.
    /isThreadScopedParamKey\(key\) &&\s*!fromModelDefaults &&[\s\S]{0,300}?captureThreadScopedEdit\(\s*key\b/,
  );
  // What the chat takes reaches neither the installation defaults nor this model's memory.
  const setParams = slice(store, "setParams: (params, options)", "\n  setCustomPresets:");
  assert.match(setParams, /persistParamEdit\(\s*sharedParams,/);
  assert.match(setParams, /getParamsByModelAfterEdit\([\s\S]{0,200}?sharedParams,/);
  assert.doesNotMatch(
    setParams,
    /getParamsByModelAfterEdit\([\s\S]{0,200}?changedParams,/,
    "the chat's edit is remembered against the model and leaks to new chats on it",
  );

  // Both paths that apply a model's own params say so.
  const runtime = read("../src/features/chat/hooks/use-chat-model-runtime.ts");
  const status = read("../src/features/chat/lib/apply-inference-status-to-store.ts");
  for (const source of [runtime, status]) {
    assert.match(
      source,
      /mergeBackendRecommendedInference\([\s\S]{0,1200}?fromModelDefaults: true/,
    );
  }
});

// The store holds the edit under `params`, so the flush that runs when the user leaves
// mid-read must look there: a direct field read is undefined, the sanitizer drops it, and
// the edit the user watched happen is gone with no error.
test("an edit held through the pairing window keeps its sampling value", () => {
  const changes = slice(store, "function heldThreadScopedChanges", "\n}");
  assert.match(changes, /readThreadScopedValue\(\s*live,\s*edit\.field as ThreadScopedSettingKey,\s*\)/);
  assert.doesNotMatch(
    changes,
    /edited\[edit\.field\] = live\[edit\.field\]/,
    "reads the field directly, which is undefined for every sampling key",
  );
});

// fromModelDefaults only ever changed where the value was persisted; the recommendation
// still landed in the live params, so the chat ran the model's sampling and the next
// unrelated edit snapshotted that over what the chat had stored.
test("a model's recommendation does not overwrite the chat's sampling", () => {
  const setParams = slice(store, "setParams: (params, options)", "\n  setCustomPresets:");
  // Over the replay, not the raw params: a chat outranks the model's memory and defaults.
  assert.match(
    setParams,
    /const effective = replayed\s*\?\s*restoreThreadScopedParams\(nextParams\)\s*:\s*nextParams;/,
  );
  assert.match(setParams, /const replayed = checkpointChanged \|\| fromModelDefaults;/);
  // and the restored object is the one that reaches the store
  assert.match(setParams, /params: effective,/);
  assert.doesNotMatch(setParams, /params: nextParams,/);

  const restore = slice(store, "function restoreThreadScopedParams", "\n}");
  // An edit still waiting on the chat's read answers first, but the stored snapshot is
  // what a paired chat restores from, so the override has to be consulted either way.
  assert.match(restore, /const held = [\s\S]{0,120}?threadScopedOverride\(key\)/);
  assert.match(restore, /if \(held === undefined/);
  // ?? and never ||: 0, "" and -1 are values the user sets on purpose.
  assert.doesNotMatch(restore, /\|\| threadScopedOverride\(key\)/);
});

// A pinned chat stores every sampling key, so restoring them all would mean the mode the
// user just asked for arrives with the previous mode's temperature and top-p. The
// load-time path applies the same table unasked, so it stays marked.
test("toggling Think applies its params even in a chat that pins sampling", () => {
  const qwen = read("../src/features/chat/utils/qwen-params.ts");
  assert.match(qwen, /store\.setParams\(\{ \.\.\.store\.params, \.\.\.params \}\);/);
  assert.doesNotMatch(
    qwen,
    /fromModelDefaults/,
    "the toggle is treated as a model default, so a pinned chat never changes mode params",
  );
  // The post-load application of the same table stays marked.
  const runtime = read("../src/features/chat/hooks/use-chat-model-runtime.ts");
  const post = slice(runtime, "store.setParams({ ...store.params, ...p }", "\n              }");
  assert.match(post, /fromModelDefaults: true/);
});

// Restoring the chat's value makes it equal on both sides of the diff, so diffing the
// restored object drops the key and the model's recommendation never reaches the
// installation defaults, leaving every new chat on whatever model loaded before it.
test("a chat pinning a param does not withhold the model's default from the rest", () => {
  const setParams = slice(store, "setParams: (params, options)", "\n  setCustomPresets:");
  assert.match(
    setParams,
    /getChangedInferenceParams\(\s*nextParams,\s*state\.params,\s*!fromModelDefaults,\s*\)/,
  );
  assert.doesNotMatch(
    setParams,
    /getChangedInferenceParams\(\s*effective,/,
    "the restored object decides what is persisted, so pinned keys are withheld",
  );
  // Called once: it bumps the mutation versions, so a second diff double-counts.
  assert.equal(setParams.match(/getChangedInferenceParams\(/g)?.length, 1);
  // The live store still gets the restored object; only persistence uses the model's.
  assert.match(setParams, /params: effective,/);
});

// The write above changes the installation default, but applyThreadScopedSettings falls
// back to an in-memory copy taken on the way into a chat. Left stale, a chat opened after
// a model load runs the sampling of whichever model was loaded before it.
test("the in-memory defaults follow the model defaults that were just written", () => {
  const setParams = slice(store, "setParams: (params, options)", "\n  setCustomPresets:");
  assert.match(setParams, /noteThreadScopedDefaults\(sharedParams\);/);
  const note = slice(store, "function noteThreadScopedDefaults", "\n}");
  assert.match(note, /if \(!isThreadScopedParamKey\(key\)\) continue;/);
  // Only ever updated, never created: with no chat open there is nothing to hold.
  assert.match(note, /if \(globalThreadScopedDefaults === null\) continue;/);
  // A held field is restored from the pre-window sample when the pairing closes, so a
  // default published inside the window has to be recorded or this session stays behind.
  assert.match(
    note,
    /if \(isHeldThreadScopedField\(key\)\) \{\s*hydratedDefaultsByHeldField\.set\(key, value\);/,
  );
  // and it is the fallback apply() actually reads. Not ??: a cleared seed is stored as
  // null, and ?? would read that as a missing key and hand back the installation pin.
  assert.match(
    store,
    /firstSetThreadScopedValue\(\s*stored\?\.\[key\],\s*globalThreadScopedDefaults\?\.\[key\],/,
  );
});

// setCheckpoint replays the destination model's remembered params without going through
// setParams. An external switch has no load after it to correct the result, so the chat
// silently adopts that model's prompt and sampling and keeps them.
test("switching model in a chat keeps the chat's sampling, not the model's", () => {
  const set = slice(store, "setCheckpoint: (modelId, ggufVariant, options)", "\n  setActiveThreadId:");
  assert.match(
    set,
    /const restoredParams = checkpointChanged\s*\?\s*restoreThreadScopedParams\(nextParams\)\s*:\s*nextParams;/,
  );
  assert.match(set, /params: restoredParams,/);
  // Persistence still reads the unrestored object, as in setParams: the model's own
  // values reach the defaults even though the chat keeps running on its own.
  assert.match(set, /getReplayStatePatch\(state, nextParams, outgoing, baseParams\)/);
});

// rememberOutgoingModel snapshots the live params, which inside a chat are that chat's.
// Filtering the incoming edit cannot undo it: the outgoing snapshot is written first,
// and on a model with no entry it is persisted whole.
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
  // Only keys this chat owns, and what the model already knew beats the installation copy,
  // so leaving a chat does not flatten a preference set outside one. A chat whose read is
  // still out owns its keys too: the edit is in the held list rather than in a snapshot.
  assert.match(
    strip,
    /if \(held === undefined && threadScopedOverride\(key\) === undefined\) continue;/,
  );
  assert.match(
    strip,
    /firstSetThreadScopedValue\(\s*remembered\?\.\[key\],\s*globalThreadScopedDefaults\?\.\[key\],/,
  );
  // and for a held key neither of those may exist yet, so the sample taken when the
  // window opened is the pre-edit value.
  assert.match(
    strip,
    /held !== undefined \? pairingWindowDefaults\?\.\[key\] : undefined/,
  );
  // With no chat open and none awaiting its read there is nothing of a chat's here.
  assert.match(
    strip,
    /if \(threadScopedSettingsThreadId === null && pendingPairingThreadId === null\)/,
  );
});
