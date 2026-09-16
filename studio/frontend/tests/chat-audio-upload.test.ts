// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc } from "./helpers/kit.ts";

import {
  ChatAudioReadinessRefreshQueue,
  appendChatAudioTranscript,
  chatAudioUploadFenceMatches,
  chatAudioUploadFileError,
  completeChatAudioUpload,
} from "../src/features/chat/utils/chat-audio-upload.ts";

const hookSource = readSrc("features/chat/hooks/use-chat-audio-upload.ts");
const sharedComposerSource = readSrc("features/chat/shared-composer.tsx");
const threadSource = readSrc("components/assistant-ui/thread.tsx");
const dialogSource = readSrc("features/chat/components/chat-audio-upload.tsx");
const modelAdapterSource = readSrc(
  "features/chat/adapters/studio-model-dictation-adapter.ts",
);

test("audio upload rejects empty and oversized files", () => {
  assert.equal(chatAudioUploadFileError({ size: 0 }), "empty");
  assert.equal(chatAudioUploadFileError({ size: 25 * 1024 * 1024 }), null);
  assert.equal(
    chatAudioUploadFileError({ size: 25 * 1024 * 1024 + 1 }),
    "too-large",
  );
});

test("audio upload classifies ambiguous 3GP files and refuses video", () => {
  const selectFileStart = hookSource.indexOf("const selectFile = useCallback(");
  const selectFile = hookSource.slice(
    selectFileStart,
    hookSource.indexOf("  const retry = useCallback", selectFileStart),
  );
  const classified = selectFile.indexOf(
    "const classifiedFile = await classifiedAttachmentFile(file);",
  );
  const refused = selectFile.indexOf("if (isVideoFile(classifiedFile))");
  const transcribed = selectFile.indexOf(
    "void runTranscription(classifiedFile, snapshot);",
  );
  assert.ok(classified >= 0 && classified < refused && refused < transcribed);
  assert.match(selectFile, /audioUploadVideoUnsupported[\s\S]*?return;/);
  const staleClassification = selectFile.slice(classified, refused);
  assert.match(staleClassification, /chatAudioUploadFenceMatches/);
  assert.doesNotMatch(staleClassification, /clearOperation\(\)/);
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
    /const refreshReadiness = useCallback\(\s*async \(silent = false\) => \{[\s\S]*?if \(!silent\) \{[\s\S]*?state: targetModel \? "checking" : "error"/,
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

test("readiness refreshes stay serialized across invalidation and reopen", async () => {
  const queue = new ChatAudioReadinessRefreshQueue();
  let resolveFirst = () => {};
  let resolveSecond = () => {};
  const firstGate = new Promise<void>((resolve) => {
    resolveFirst = resolve;
  });
  const secondGate = new Promise<void>((resolve) => {
    resolveSecond = resolve;
  });
  let calls = 0;
  let firstGeneration = -1;
  const committed: string[] = [];

  const first = queue.run(false, async (generation) => {
    firstGeneration = generation;
    calls += 1;
    await firstGate;
    if (queue.isCurrent(generation)) committed.push("first");
  });
  assert.equal(calls, 1);
  assert.equal(await queue.run(true, async () => {}), "skipped");

  queue.invalidate();
  const reopened = queue.run(false, async (generation) => {
    calls += 1;
    await secondGate;
    if (queue.isCurrent(generation)) committed.push("reopened");
  });
  assert.equal(calls, 1);
  assert.equal(queue.isCurrent(firstGeneration), false);
  assert.equal(await queue.run(true, async () => {}), "skipped");

  resolveFirst();
  assert.equal(await first, "ran");
  assert.deepEqual(committed, []);
  await Promise.resolve();
  assert.equal(calls, 2);
  assert.equal(await queue.run(true, async () => {}), "skipped");

  resolveSecond();
  assert.equal(await reopened, "ran");
  assert.equal(calls, 2);
  assert.deepEqual(committed, ["reopened"]);
});

test("a newer explicit readiness refresh supersedes one still in flight", async () => {
  const queue = new ChatAudioReadinessRefreshQueue();
  let release = () => {};
  const gate = new Promise<void>((resolve) => {
    release = resolve;
  });
  let firstGeneration = -1;
  const committed: string[] = [];

  const first = queue.run(false, async (generation) => {
    firstGeneration = generation;
    await gate;
    if (queue.isCurrent(generation)) committed.push("old-model");
  });
  const replacement = queue.run(false, async (generation) => {
    if (queue.isCurrent(generation)) committed.push("new-model");
  });

  assert.equal(queue.isCurrent(firstGeneration), false);
  release();
  assert.equal(await first, "ran");
  assert.equal(await replacement, "ran");
  assert.deepEqual(committed, ["new-model"]);
});

test("invalidation aborts a hung refresh before a reopened one starts", async () => {
  const queue = new ChatAudioReadinessRefreshQueue();
  let calls = 0;
  let firstAborted = false;

  const first = queue.run(false, async (_generation, signal) => {
    calls += 1;
    await new Promise<void>((resolve) => {
      signal.addEventListener(
        "abort",
        () => {
          firstAborted = true;
          resolve();
        },
        { once: true },
      );
    });
  });
  queue.invalidate();
  const reopened = queue.run(false, async () => {
    calls += 1;
  });

  assert.equal(await first, "ran");
  assert.equal(firstAborted, true);
  assert.equal(await reopened, "ran");
  assert.equal(calls, 2);
});

test("readiness status passes the queue's abort signal to authFetch", () => {
  assert.match(
    modelAdapterSource,
    /export async function fetchSttStatus\([\s\S]*?signal\?: AbortSignal[\s\S]*?authFetch\([\s\S]*?\{ signal \},[\s\S]*?withAbort\(request, signal\)/,
  );
  assert.match(
    hookSource,
    /queue\.run\(silent, async \(attempt, signal\)[\s\S]*?fetchSttStatus\(undefined, targetModel, signal\)/,
  );
});

test("a hung readiness refresh times out and releases the queue", async () => {
  const queue = new ChatAudioReadinessRefreshQueue(5);
  let timedOut = false;

  const first = queue.run(false, async (_generation, signal) => {
    await new Promise<void>((resolve) => {
      signal.addEventListener(
        "abort",
        () => {
          timedOut = true;
          resolve();
        },
        { once: true },
      );
    });
  });

  assert.equal(await first, "ran");
  assert.equal(timedOut, true);
  assert.equal(await queue.run(false, async () => {}), "ran");
});

test("Main cancels an upload only after a queue or normal send is accepted", () => {
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

  const sendStart = threadSource.indexOf(
    "const sendReservedComposer = useCallback",
  );
  const send = threadSource.slice(
    sendStart,
    threadSource.indexOf("const interceptSend", sendStart),
  );
  const refused = send.indexOf("if (!reservationToken)");
  const cancel = send.indexOf("cancelAudioUpload();", refused);
  assert.ok(refused >= 0 && cancel > refused);
  assert.ok(cancel < send.indexOf("aui.composer().send();", cancel));

  const submitStart = threadSource.indexOf("const handleSubmit = useCallback");
  const submit = threadSource.slice(
    submitStart,
    threadSource.indexOf("const stopQueue", submitStart),
  );
  assert.equal(submit.includes("audioUpload.cancel();"), false);
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

test("Compare keeps its current draft synchronous for transcript appends", () => {
  assert.match(
    sharedComposerSource,
    /const setCurrentText = useCallback\([\s\S]*?typeof value === "function" \? value\(textRef\.current\) : value;[\s\S]*?textRef\.current = next;\s*setText\(next\);/,
  );
  assert.match(
    sharedComposerSource,
    /const writeAudioUploadDraft = useCallback\(\s*\(value: string\) => setCurrentText\(value\)/,
  );
  assert.match(
    sharedComposerSource,
    /onChange=\{\(e\) => \{[\s\S]*?setCurrentText\(e\.target\.value\);/,
  );
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

test("Compare queue setup preserves an upload until a queued send is accepted", () => {
  const advanceQueueStart = sharedComposerSource.indexOf(
    "function advanceQueue() {",
  );
  const advanceQueue = sharedComposerSource.slice(
    advanceQueueStart,
    sharedComposerSource.indexOf("\n  }", advanceQueueStart) + 4,
  );
  assert.match(
    advanceQueue,
    /setCurrentText\(next\);\s*setTimeout\(\(\) => \{ sendRef\.current\?\.\(\); \}, 100\);/,
  );
  assert.doesNotMatch(advanceQueue, /audioUpload\.cancel\(\)/);

  const runListStart = sharedComposerSource.indexOf("onRunList={(items) => {");
  const runList = sharedComposerSource.slice(
    runListStart,
    sharedComposerSource.indexOf("        }}", runListStart),
  );
  const incompleteModelGuard = runList.indexOf(
    "if (hasCompareHandles && !isGeneralizedCompare)",
  );
  const replace = runList.indexOf("setCurrentText(filtered[0])");
  assert.ok(incompleteModelGuard >= 0 && incompleteModelGuard < replace);
  assert.doesNotMatch(runList, /audioUpload\.cancel\(\)/);
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
