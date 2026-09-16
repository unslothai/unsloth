import assert from "node:assert/strict";
import test from "node:test";
import ts from "typescript";
import { readSrc } from "./helpers/kit.ts";
import { composerSubmitIntent } from "../src/features/chat/utils/composer-preferences.ts";

const text = readSrc("components/assistant-ui/thread.tsx");
const source = ts.createSourceFile(
  "thread.tsx",
  text,
  ts.ScriptTarget.Latest,
  true,
  ts.ScriptKind.TSX,
);
function lift(predicate: (node: ts.Node) => boolean) {
  const matches: ts.Node[] = [];
  function visit(node: ts.Node) {
    if (predicate(node)) matches.push(node);
    ts.forEachChild(node, visit);
  }
  visit(source);
  assert.equal(matches.length, 1);
  return ts.transpileModule(`return (${matches[0].getText(source)});`, {
    compilerOptions: {
      target: ts.ScriptTarget.ES2022,
      module: ts.ModuleKind.None,
    },
  }).outputText;
}
const keyCallback = lift(
  (n) =>
    ts.isArrowFunction(n) &&
    n
      .getText(source)
      .startsWith("(e: KeyboardEvent<HTMLTextAreaElement>) =>") &&
    n.getText(source).includes("composerSubmitIntent"),
);
const releaseCallback = lift(
  (n) =>
    ts.isArrowFunction(n) &&
    n.getText(source).startsWith("() =>") &&
    n
      .getText(source)
      .includes("const behavior = pendingFollowUpBehaviorRef.current;"),
);
const reservedSendCallback = lift(
  (n) =>
    ts.isArrowFunction(n) &&
    n.getText(source).startsWith("(...alsoGuard: string[]) =>") &&
    n.getText(source).includes("reservePreStreamRun"),
);
function createCallback(
  code: string,
  deps: Record<string, unknown>,
): () => void {
  return new Function(...Object.keys(deps), code)(...Object.values(deps));
}

test("the shipped main composer key handler protects IME and mention selection", () => {
  let submits = 0;
  let prevented = 0;
  const composingRef = { current: false };
  const skipEnterRef = { current: false };
  const deps = {
    composerSubmitIntent,
    sendShortcut: "mod-enter",
    submitOnEnter: true,
    skipEnterRef,
    composingRef,
    justSentRef: undefined,
    refreshStuckTimer: () => undefined,
    setCompositionState: (value: boolean) => {
      composingRef.current = value;
    },
    onSubmitKey: () => {
      submits += 1;
    },
  };
  const onKey = createCallback(keyCallback, deps) as unknown as (
    event: Record<string, unknown>,
  ) => void;
  const event = {
    key: "Enter",
    metaKey: true,
    ctrlKey: false,
    shiftKey: false,
    altKey: false,
    keyCode: 13,
    nativeEvent: { isComposing: false },
    preventDefault: () => {
      prevented += 1;
    },
  };
  onKey({ ...event, nativeEvent: { isComposing: true } });
  onKey(event); // Candidate-confirming Enter can arrive with native isComposing=false.
  assert.equal(submits, 0);
  composingRef.current = false;
  skipEnterRef.current = true;
  onKey(event);
  assert.equal(submits, 0);
  skipEnterRef.current = false;
  onKey({ ...event, metaKey: false });
  assert.equal(submits, 0);
  onKey(event);
  assert.equal(submits, 1);
  assert.ok(prevented > 0);
});

for (const active of ["runtime", "pre-stream", "queue", "idle"]) {
  test(`parked send retains the submitted follow-up choice with ${active} state at release`, () => {
    const calls: unknown[] = [];
    const pendingFollowUpBehaviorRef = { current: "steer" };
    const pendingSendRef = { current: true };
    const deps = {
      pendingSend: true,
      pendingSendRef,
      pendingFollowUpBehaviorRef,
      indexingActive: false,
      threadScopedSettingsPending: false,
      hasMaterializingImageAttachments: false,
      hasMaterializingAudioAttachments: false,
      hasMaterializingVideoAttachments: false,
      aui: {
        composer: () => ({
          getState: () => ({ text: "Follow up", attachments: [] }),
        }),
        thread: () => ({
          getState: () => ({ isRunning: active === "runtime" }),
        }),
      },
      setPendingSend: () => undefined,
      dismissWaitToast: () => undefined,
      hasPreStreamRunReservation: () => active === "pre-stream",
      preStreamThreadIds: ["own-chat"],
      isResearchActive: false,
      findPromptQueueEntry: () => active === "queue",
      usePromptQueueUI: { getState: () => ({}) },
      disableQueue: false,
      canQueueCurrentPrompt: true,
      queueComposerText: (wait: boolean, behavior: string) =>
        calls.push([wait, behavior]),
      canQueuePastedTextPrompt: false,
      overlay: false,
      hasAttachments: false,
      hasPendingAudio: false,
      clearStoredDraft: () => calls.push("clear"),
      sendReservedComposer: () => calls.push("send"),
    };
    const release = createCallback(releaseCallback, deps);
    release();
    release(); // An old render cannot release a cancelled/consumed send twice.
    assert.deepEqual(
      calls,
      active === "idle"
        ? ["clear", "send"]
        : [[active !== "queue", "steer"]],
    );
    assert.equal(pendingSendRef.current, false);
  });
}

test("main cancels an audio upload only after a normal send reservation succeeds", () => {
  const run = (reservationToken: symbol | null) => {
    const calls: string[] = [];
    const deps = {
      aui: {
        threads: () => ({ __internal_getAssistantRuntime: () => undefined }),
        composer: () => ({
          getState: () => ({ text: "Draft" }),
          send: () => calls.push("send"),
        }),
      },
      reservePreStreamRun: () => reservationToken,
      preStreamThreadIds: ["chat"],
      parseExternalModelId: () => null,
      useChatRuntimeStore: {
        getState: () => ({
          params: { checkpoint: "local/model" },
          incognito: false,
          activeGgufVariant: null,
        }),
      },
      preStreamRunReservationRef: { current: null },
      toast: { error: () => calls.push("refused") },
      cancelAudioUpload: () => calls.push("cancel-upload"),
      claimThreadCreation: () => calls.push("claim"),
      projectScope: null,
      armJustSent: () => calls.push("arm"),
      releasePreStreamRunReservation: () => true,
      notifyPromptQueueRunFailed: () => undefined,
      referenceThreadId: "chat",
    };
    const send = createCallback(reservedSendCallback, deps);
    send();
    return calls;
  };

  assert.deepEqual(run(null), ["refused"]);
  assert.deepEqual(run(Symbol("reservation")), [
    "cancel-upload",
    "claim",
    "send",
    "arm",
  ]);
});

test("main, edit and comparison composers use the setting and expose settings access", () => {
  assert.match(text, /submitMode="none"/);
  assert.match(
    text,
    /submitMode=\{sendShortcut === "mod-enter" \? "ctrlEnter" : "enter"\}/,
  );
  assert.match(
    text,
    /onQueueClick=\{\(\) => formRef.current\?\.requestSubmit\(\)\}/,
  );
  assert.match(text, /scrollTarget: "chat-composer"/);
  const compare = readSrc("features/chat/shared-composer.tsx");
  assert.match(compare, /composerSubmitIntent\(e, sendShortcut\)/);
  assert.match(compare, /scrollTarget: "chat-composer"/);
  const page = readSrc("features/chat/chat-page.tsx");
  assert.match(page, /showContextWindowUsage &&\s*view.mode === "single"/);
});
