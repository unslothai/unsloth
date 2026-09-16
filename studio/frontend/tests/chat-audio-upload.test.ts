// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc } from "./helpers/kit.ts";

import {
  appendChatAudioTranscript,
  chatAudioUploadFenceMatches,
  chatAudioUploadFileError,
  completeChatAudioUpload,
} from "../src/features/chat/utils/chat-audio-upload.ts";

const hookSource = readSrc("features/chat/hooks/use-chat-audio-upload.ts");
const sharedComposerSource = readSrc("features/chat/shared-composer.tsx");
const threadSource = readSrc("components/assistant-ui/thread.tsx");
const dialogSource = readSrc("features/chat/components/chat-audio-upload.tsx");

test("audio upload rejects empty and oversized files", () => {
  assert.equal(chatAudioUploadFileError({ size: 0 }), "empty");
  assert.equal(chatAudioUploadFileError({ size: 25 * 1024 * 1024 }), null);
  assert.equal(
    chatAudioUploadFileError({ size: 25 * 1024 * 1024 + 1 }),
    "too-large",
  );
});

test("audio upload accepts only its current owner, generation and auth session", () => {
  const started = { generation: 3, owner: "thread-a", authSessionEpoch: 7 };
  assert.equal(chatAudioUploadFenceMatches(started, { ...started }), true);
  assert.equal(
    chatAudioUploadFenceMatches(started, { ...started, generation: 4 }),
    false,
  );
  assert.equal(
    chatAudioUploadFenceMatches(started, { ...started, owner: "thread-b" }),
    false,
  );
  assert.equal(
    chatAudioUploadFenceMatches(started, { ...started, authSessionEpoch: 8 }),
    false,
  );
});

test("audio transcript appends once without replacing a draft typed meanwhile", () => {
  assert.equal(appendChatAudioTranscript("", " hello "), "hello");
  assert.equal(appendChatAudioTranscript("draft", " hello "), "draft hello");
  assert.equal(appendChatAudioTranscript("draft  ", "hello"), "draft hello");
  assert.equal(appendChatAudioTranscript("draft", "   "), "draft");
});

test("an ignored abort cannot commit a late transcript", async () => {
  let resolveTranscript: (value: string) => void = () => {};
  const transcript = new Promise<string>((resolve) => {
    resolveTranscript = resolve;
  });
  const controller = new AbortController();
  const fence = { generation: 1, owner: "thread-a", authSessionEpoch: 2 };
  let commits = 0;
  const completion = completeChatAudioUpload({
    started: fence,
    current: () => fence,
    signal: controller.signal,
    blocked: () => false,
    transcribe: () => transcript,
    commit: () => {
      commits += 1;
    },
  });
  controller.abort();
  resolveTranscript("late words");
  assert.equal(await completion, "stale");
  assert.equal(commits, 0);
});

test("an owner change before transcription resolves cannot commit", async () => {
  let resolveTranscript: (value: string) => void = () => {};
  const transcript = new Promise<string>((resolve) => {
    resolveTranscript = resolve;
  });
  const started = { generation: 1, owner: "thread-a", authSessionEpoch: 2 };
  let current = started;
  let commits = 0;
  const completion = completeChatAudioUpload({
    started,
    current: () => current,
    signal: new AbortController().signal,
    blocked: () => false,
    transcribe: () => transcript,
    commit: () => {
      commits += 1;
    },
  });
  current = { ...started, owner: "thread-b" };
  resolveTranscript("late words");
  assert.equal(await completion, "stale");
  assert.equal(commits, 0);
});

test("an A to B to A generation change still rejects the first result", async () => {
  const started = { generation: 1, owner: "thread-a", authSessionEpoch: 2 };
  let current = started;
  let commits = 0;
  const completion = completeChatAudioUpload({
    started,
    current: () => current,
    signal: new AbortController().signal,
    blocked: () => false,
    transcribe: async () => {
      current = { ...started, generation: 3 };
      return "late words";
    },
    commit: () => {
      commits += 1;
    },
  });
  assert.equal(await completion, "stale");
  assert.equal(commits, 0);
});

test("a current result commits exactly once and empty audio commits nothing", async () => {
  const fence = { generation: 1, owner: "thread-a", authSessionEpoch: 2 };
  const committed: string[] = [];
  const run = (value: string) =>
    completeChatAudioUpload({
      started: fence,
      current: () => fence,
      signal: new AbortController().signal,
      blocked: () => false,
      transcribe: async () => value,
      commit: (result) => committed.push(result),
    });
  assert.equal(await run(" hello "), "committed");
  assert.deepEqual(committed, ["hello"]);
  assert.equal(await run("   "), "empty");
  assert.deepEqual(committed, ["hello"]);
});

test("the main composer has one Dictate entry and does not gate it on draft text", () => {
  const callStart = threadSource.indexOf("<ComposerRightControls");
  assert.notEqual(callStart, -1);
  const call = threadSource.slice(
    callStart,
    threadSource.indexOf("/>", callStart),
  );
  assert.match(call, /dictationDisabled=\{dictationEntryDisabled\}/);
  assert.match(threadSource, /const dictationEntryDisabled = !chatActive;/);
  const entryGate = threadSource.slice(
    threadSource.indexOf("const dictationEntryDisabled"),
    threadSource.indexOf(
      "const audioUpload",
      threadSource.indexOf("const dictationEntryDisabled"),
    ),
  );
  assert.doesNotMatch(
    entryGate,
    /disabled|isComposing|hasPendingAttachments|hasSendableContent/,
  );
  const controlsStart = threadSource.indexOf("const ComposerRightControls:");
  const controls = threadSource.slice(controlsStart);
  assert.equal((controls.match(/tooltip="Dictate"/g) ?? []).length, 1);
  assert.doesNotMatch(controls, /Upload01Icon|DialogTrigger/);
});

test("a later disabled transition cancels the active upload", () => {
  assert.match(
    hookSource,
    /useLayoutEffect\(\(\) => \{\s*if \(!disabled\) return;\s*invalidateRefs\(\);[\s\S]*?queueMicrotask\([\s\S]*?setBusy\(false\);/,
  );
});

test("owner changes fence stale completions before passive effects", () => {
  assert.match(
    hookSource,
    /useLayoutEffect\(\(\) => \{\s*const ownerChanged = ownerRef\.current !== owner;[\s\S]*?if \(ownerChanged\) clearOperation\(\);/,
  );
});

test("an open dialog observes a missing model becoming ready", () => {
  assert.match(
    hookSource,
    /const refreshReadiness = useCallback\(\s*async \(silent = false\) => \{[\s\S]*?if \(!silent\) setReadiness\(\{ state: "checking", model: targetModel \}\);/,
  );
  const pollStart = hookSource.indexOf(
    '(readiness.state !== "missing" && readiness.state !== "downloading")',
  );
  assert.notEqual(pollStart, -1);
  const poll = hookSource.slice(
    pollStart,
    hookSource.indexOf("  }, [dialogOpen", pollStart),
  );
  assert.match(
    poll,
    /setInterval\(\(\) => void refreshReadiness\(true\), 1500\)/,
  );
});

test("Main cancels an upload only when a queue or parked send is accepted", () => {
  const queueStart = threadSource.indexOf(
    "const startHydratedPromptQueue = useCallback",
  );
  const queue = threadSource.slice(
    queueStart,
    threadSource.indexOf("const queuePastedTextPrompt", queueStart),
  );
  const accepted = queue.indexOf("cancelAudioUpload();");
  assert.ok(accepted > queue.indexOf(".then((target) =>"));
  assert.ok(accepted < queue.indexOf("startPromptQueue(", accepted));
  assert.ok(queue.indexOf("onAborted?.()", accepted) > accepted);

  const releaseStart = threadSource.indexOf(
    "// Fire the parked send once indexing clears",
  );
  const release = threadSource.slice(
    releaseStart,
    threadSource.indexOf("// Drop any queued send", releaseStart),
  );
  const send = release.lastIndexOf("sendReservedComposer();");
  const cancel = release.lastIndexOf("cancelAudioUpload();", send);
  assert.ok(cancel >= 0 && cancel < send);
  assert.ok(cancel > release.indexOf("if (isResearchActive)"));
});

test("Compare routes button and shortcut through the same Dictate entry", () => {
  const wrapperStart = sharedComposerSource.indexOf(
    "const startDictation = useCallback",
  );
  assert.notEqual(wrapperStart, -1);
  const wrapper = sharedComposerSource.slice(
    wrapperStart,
    sharedComposerSource.indexOf("  }, [", wrapperStart),
  );
  assert.match(wrapper, /if \(audioUpload\.busy \|\| !chatActive\) return;/);
  assert.match(wrapper, /currentDictationEntryMode\(\) === "recording-file"/);
  assert.match(wrapper, /audioUpload\.openDialog\(\)/);
  assert.match(wrapper, /startDictationSession\(\)/);
  assert.match(sharedComposerSource, /onClick=\{startDictation\}/);

  const shortcutStart = sharedComposerSource.indexOf(
    'useShortcut(\n    "startDictation"',
  );
  const shortcut = sharedComposerSource.slice(
    shortcutStart,
    sharedComposerSource.indexOf("\n  );", shortcutStart),
  );
  assert.match(shortcut, /startDictation\(\)/);
});

test("the controlled dialog has separate Android recorder and saved-file inputs", () => {
  assert.doesNotMatch(dialogSource, /DialogTrigger|Upload01Icon/);
  assert.match(dialogSource, /accept="audio\/\*"\s*capture="user"/);
  assert.match(dialogSource, /accept=\{AUDIO_PICKER_ACCEPT\}/);
  const chooseInput = dialogSource.slice(
    dialogSource.indexOf("ref={chooseInputRef}"),
  );
  assert.doesNotMatch(
    chooseInput.slice(0, chooseInput.indexOf("/>")),
    /capture=/,
  );
  assert.match(dialogSource, /platform === "android"/);
  assert.match(dialogSource, /platform === "ios"/);
});

test("picker launch snapshots before the native input opens", () => {
  assert.match(
    dialogSource,
    /if \(!input \|\| !audioUpload\.snapshotForPicker\(\)\) return;\s*input\.click\(\);/,
  );
  assert.match(
    hookSource,
    /pickerSnapshotRef\.current = snapshot;\s*return snapshot;/,
  );
  assert.match(
    hookSource,
    /const snapshot = pickerSnapshotRef\.current;\s*pickerSnapshotRef\.current = null;/,
  );
});

test("Compare automatic queue takeovers invalidate the previous draft upload", () => {
  assert.match(
    sharedComposerSource,
    /audioUpload\.cancel\(\);\s*setText\(next\);\s*setTimeout\(\(\) => \{ sendRef\.current\?\.\(\); \}, 100\);/,
  );
  const runListStart = sharedComposerSource.indexOf("onRunList={(items) => {");
  const runList = sharedComposerSource.slice(
    runListStart,
    sharedComposerSource.indexOf("        }}", runListStart),
  );
  const incompleteModelGuard = runList.indexOf(
    "if (hasCompareHandles && !isGeneralizedCompare)",
  );
  const cancel = runList.indexOf("audioUpload.cancel()");
  const replace = runList.indexOf("setText(filtered[0])");
  assert.ok(incompleteModelGuard >= 0 && incompleteModelGuard < cancel);
  assert.ok(cancel >= 0 && cancel < replace);
});

test("Compare cancels uploads only after a send reaches an accepted effect", () => {
  const sendStart = sharedComposerSource.indexOf("async function send() {");
  const send = sharedComposerSource.slice(
    sendStart,
    sharedComposerSource.indexOf("sendRef.current = send", sendStart),
  );
  const cancellations = [...send.matchAll(/audioUpload\.cancel\(\)/g)];
  assert.equal(cancellations.length, 2);
  for (const cancellation of cancellations) {
    assert.match(
      send.slice(cancellation.index, cancellation.index + 100),
      /audioUpload\.cancel\(\);\s*clearSubmittedDraft\(\);/,
    );
  }
  assert.doesNotMatch(
    send.slice(0, send.indexOf("if (isGeneralizedCompare)")),
    /audioUpload\.cancel\(\)/,
  );
});
