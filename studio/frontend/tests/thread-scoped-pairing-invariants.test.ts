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
  // and the read only happens after it, not alongside
  assert.ok(
    sync.indexOf("awaitThreadScopedSettingsWrite") <
      sync.indexOf("getStoredChatThreadReadResult"),
    "the read is not sequenced after the write",
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
  assert.match(loop, /if \(isHeldThreadScopedField\(key\)\) \{\s*continue;/);
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
  const sync = slice(provider, "const sync = () => {", "// The read did not answer");
  assert.match(sync, /getStoredChatThreadReadResult\(activeThreadId, \{ bounded: true \}\)/);
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
  // assistant-ui gives an unsaved thread a `__LOCALID_` id (RemoteThreadListThreadList
  // RuntimeCore), which no row can exist for, so its read can only 404. Holding edits
  // behind that certain-to-fail read is what stopped a pill clicked on a fresh /chat
  // from reaching localStorage straight away, which playwright_chat_ui asserts.
  const effect = slice(
    provider,
    "const { applyThreadScopedSettings } = useChatRuntimeStore.getState();",
    "if (!enabled) {",
  );
  assert.match(effect, /isAssistantLocalThreadId\(activeThreadId\)/);
  assert.match(effect, /applyThreadScopedSettings\(null, null\)/);
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
