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
  assert.match(
    sync,
    /awaitThreadScopedSettingsWrite\(activeThreadId\)\s*\.then\(\(\) => getStoredChatThreadReadResult\(activeThreadId\)\)/,
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
  assert.match(begin, /pairingWindowDefaults = readThreadScopedSettings\(/);
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
  assert.match(flush, /unsettledThreadSettingsWrites\.set\(threadId, snapshot\)/);
  const terminal = slice(store, "function flushSettingsOnPageHidden", "\n}");
  assert.match(terminal, /commitHeldThreadScopedEditsToTheirThread\(true\)/);
  assert.match(terminal, /beaconUnsettledThreadSettingsWrites\(\)/);
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
