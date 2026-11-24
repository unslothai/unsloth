import assert from "node:assert/strict";
import test, { after } from "node:test";
import {
  BaseComposerRuntimeCore,
  ThreadComposerRuntimeImpl,
} from "@assistant-ui/core/internal";
import ts from "@typescript/typescript6";
import {
  ComposerAttachmentPreparationError,
  observeComposerSend,
} from "../src/components/assistant-ui/composer-send-lifecycle.ts";
import {
  composerKeyEventForImeSubmit,
  composerSubmitIntent,
  imeKeydownBlocksComposerSubmit,
  parkedSendSteer,
} from "../src/features/chat/utils/composer-preferences.ts";
import {
  closeTapMessageChannels,
  composerStoreAccessor,
} from "./helpers/composer-store-accessor.ts";
import { readSrc } from "./helpers/kit.ts";

after(() => closeTapMessageChannels());

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
    n.getText(source).startsWith("(options: SendReservedComposerOptions") &&
    n.getText(source).includes("reservePreStreamRun") &&
    n.getText(source).includes("const failSend"),
);
const steerParkedSendCallback = lift(
  (n) =>
    ts.isArrowFunction(n) &&
    n.getText(source).startsWith("(state: Omit<ParkedSendSteerState"),
);
function createCallback(
  code: string,
  deps: Record<string, unknown>,
): () => void {
  return new Function(...Object.keys(deps), code)(...Object.values(deps));
}

class RealComposerCore extends BaseComposerRuntimeCore {
  readonly path = { ref: "thread.composer", threadId: "thread-1" };
  readonly sent: Array<{
    metadata?: { custom?: Record<string, unknown> };
  }> = [];

  protected getAttachmentAdapter(): undefined {
    return undefined;
  }

  protected getDictationAdapter(): undefined {
    return undefined;
  }

  get canCancel(): boolean {
    return false;
  }

  protected handleSend(message: {
    metadata?: { custom?: Record<string, unknown> };
  }): void {
    this.sent.push(message);
  }

  protected handleCancel(): void {}
}

function realAccessorFor(core: RealComposerCore) {
  const runtime = new ThreadComposerRuntimeImpl({
    path: core.path,
    getState: () => core,
    subscribe: (callback: () => void) => core.subscribe(callback),
  } as never);
  const fixture = composerStoreAccessor(runtime);
  return { accessor: fixture.accessor, dispose: fixture.dispose, runtime };
}

test("the shipped main composer key handler protects IME and mention selection", () => {
  let submits = 0;
  let prevented = 0;
  const composingRef = { current: false };
  const imeSessionOpenRef = { current: false };
  const skipEnterRef = { current: false };
  const deps = {
    composerSubmitIntent,
    composerKeyEventForImeSubmit,
    imeKeydownBlocksComposerSubmit,
    sendShortcut: "mod-enter",
    submitOnEnter: true,
    skipEnterRef,
    composingRef,
    imeSessionOpenRef,
    compositionEndedAtRef: { current: -Infinity },
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

test("idle IME Enter sends, the WebKit candidate-confirming Enter does not (#12137)", () => {
  let submits = 0;
  const composingRef = { current: false };
  const imeSessionOpenRef = { current: false };
  const compositionEndedAtRef = { current: -Infinity };
  const deps = {
    composerSubmitIntent,
    composerKeyEventForImeSubmit,
    imeKeydownBlocksComposerSubmit,
    sendShortcut: "enter",
    submitOnEnter: true,
    skipEnterRef: { current: false },
    composingRef,
    imeSessionOpenRef,
    compositionEndedAtRef,
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
  const imeEnter = (timeStamp: number) => ({
    key: "Enter",
    metaKey: false,
    ctrlKey: false,
    shiftKey: false,
    altKey: false,
    keyCode: 229,
    timeStamp,
    nativeEvent: { isComposing: false },
    preventDefault: () => undefined,
  });
  imeSessionOpenRef.current = true;
  onKey({ ...imeEnter(1000), nativeEvent: { isComposing: true } });
  assert.equal(submits, 0);
  imeSessionOpenRef.current = false;
  composingRef.current = false;
  compositionEndedAtRef.current = 1995;
  onKey(imeEnter(2000));
  assert.equal(submits, 0);
  assert.equal(composingRef.current, true);
  onKey(imeEnter(2100));
  assert.equal(submits, 1);
  assert.equal(composingRef.current, false);
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
      threadIsRunning: false,
      promptQueueThreadIds: ["own-chat"],
      attachmentsAreQueueableText: false,
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
      canQueueTextAttachmentsPrompt: false,
      overlay: false,
      hasAttachments: false,
      hasPendingAudio: false,
      clearStoredDraft: () => calls.push("clear"),
      sendReservedComposer: () => calls.push("send"),
    };
    const release = createCallback(releaseCallback, deps);
    release();
    release(); // stale renders cannot release a consumed or cancelled send twice.
    assert.deepEqual(
      calls,
      active === "idle" ? ["clear", "send"] : [[active !== "queue", "steer"]],
    );
    assert.equal(pendingSendRef.current, false);
  });
}

for (const [active, behavior] of ["runtime", "pre-stream", "queue"].flatMap(
  (a) => [
    [a, "queue"],
    [a, "steer"],
  ],
)) {
  test(`a ${behavior} attachment stays parked through ${active} state and sends once (#9210)`, () => {
    const calls: string[] = [];
    const state = { active: true };
    const pendingSendRef = { current: true };
    const pendingFollowUpBehaviorRef = { current: behavior };
    const deps = {
      pendingSend: true,
      pendingSendRef,
      pendingFollowUpBehaviorRef,
      indexingActive: false,
      threadScopedSettingsPending: false,
      threadIsRunning: false,
      promptQueueThreadIds: ["own-chat"],
      attachmentsAreQueueableText: false,
      hasMaterializingImageAttachments: false,
      hasMaterializingAudioAttachments: false,
      hasMaterializingVideoAttachments: false,
      aui: {
        composer: () => ({
          getState: () => ({ text: "", attachments: [{ id: "image-1" }] }),
        }),
        thread: () => ({
          getState: () => ({
            isRunning: state.active && active === "runtime",
          }),
          cancelRun: () => calls.push("cancel-run") && (state.active = false),
        }),
      },
      setPendingSend: () => undefined,
      dismissWaitToast: () => calls.push("dismiss"),
      hasPreStreamRunReservation: () => state.active && active === "pre-stream",
      preStreamThreadIds: ["own-chat"],
      isResearchActive: false,
      findPromptQueueEntry: () => state.active && active === "queue",
      hasPendingQueueStart: () => false,
      usePromptQueueUI: { getState: () => ({}) },
      disableQueue: false,
      canQueueCurrentPrompt: false,
      queueComposerText: () => calls.push("queue-text"),
      canQueueTextAttachmentsPrompt: false,
      queueTextAttachmentsPrompt: () => calls.push("queue-paste"),
      overlay: false,
      hasAttachments: true,
      hasPendingAudio: false,
      clearStoredDraft: () => calls.push("clear"),
      sendReservedComposer: () => calls.push("send"),
      toast: { error: () => calls.push("error") },
    };
    const steerParkedSend = createCallback(steerParkedSendCallback, {
      ...deps,
      parkedSendSteer,
      cancelPreStreamRunForThreadIds: () => calls.push("cancel-pre-stream"),
    });
    const release = createCallback(releaseCallback, {
      ...deps,
      steerParkedSend,
    });
    release();
    // queued prompts stay ahead of the image.
    const cancels =
      behavior === "queue" || active === "queue"
        ? []
        : active === "runtime"
          ? ["cancel-pre-stream", "cancel-run"]
          : ["cancel-pre-stream"];
    assert.deepEqual(calls, cancels);
    assert.equal(pendingSendRef.current, true);
    state.active = false;
    release();
    release();
    assert.deepEqual(calls, [...cancels, "dismiss", "clear", "send"]);
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
      isPreStreamRunReservationActive: () => false,
      ComposerAttachmentPreparationError,
      composerSendObserverRef: { current: null },
      observeComposerSend: () => {
        calls.push("observe");
        return {
          kind: "observing",
          attempt: {
            token: null,
            stop: () => calls.push("stop"),
            isActive: () => true,
            isDetached: () => false,
          },
        };
      },
    };
    const send = createCallback(reservedSendCallback, deps);
    send();
    return calls;
  };

  assert.deepEqual(run(null), ["refused"]);
  assert.deepEqual(run(Symbol("reservation")), [
    "cancel-upload",
    "claim",
    "observe",
    "send",
    "arm",
  ]);
});

test("a reported attachment failure releases the send exactly once", () => {
  const calls: string[] = [];
  let onFailed: ((error: unknown) => void) | null = null;
  let stopCalls = 0;
  const deps = {
    aui: {
      threads: () => ({ __internal_getAssistantRuntime: () => undefined }),
      composer: () => ({
        getState: () => ({ text: "Draft", quote: undefined }),
        send: () => calls.push("send"),
      }),
    },
    reservePreStreamRun: () => Symbol("reservation"),
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
    toast: { error: () => calls.push("toast") },
    cancelAudioUpload: () => undefined,
    claimThreadCreation: () => undefined,
    projectScope: null,
    armJustSent: () => undefined,
    releasePreStreamRunReservation: () => {
      calls.push("release");
      return true;
    },
    notifyPromptQueueRunFailed: () => calls.push("queue-failed"),
    referenceThreadId: "chat",
    isPreStreamRunReservationActive: () => false,
    ComposerAttachmentPreparationError,
    composerSendObserverRef: { current: null },
    observeComposerSend: (
      _composer: unknown,
      _threads: unknown,
      _token: unknown,
      callbacks: { onFailed: (error: unknown) => void },
    ) => {
      onFailed = callbacks.onFailed;
      return {
        kind: "observing",
        attempt: {
          token: null,
          stop: () => {
            stopCalls += 1;
          },
          isActive: () => true,
          isDetached: () => false,
        },
      };
    },
  };
  const send = createCallback(reservedSendCallback, deps);
  send();
  assert.deepEqual(calls, ["send"]);
  const fail = onFailed as ((error: unknown) => void) | null;
  assert.ok(fail);
  fail(new Error("upload failed"));
  fail(new Error("second report"));
  assert.deepEqual(calls, ["send", "release", "queue-failed", "toast"]);
  assert.equal(stopCalls, 1);
});

test("a quoted attachment failure releases quote and pre-stream reservations once", () => {
  const calls: string[] = [];
  let onFailed: ((error: unknown) => void) | null = null;
  const composerState: { text: string; quote: unknown } = {
    text: "Draft",
    quote: undefined,
  };
  const deps = {
    aui: {
      threads: () => ({ __internal_getAssistantRuntime: () => undefined }),
      composer: () => ({
        getState: () => composerState,
        setQuote: (quote: unknown) => {
          composerState.quote = quote;
        },
        send: () => {
          composerState.quote = undefined;
          calls.push("send");
        },
        __internal_getRuntime: () => ({
          getState: () => ({ quote: undefined, isEmpty: false }),
        }),
      }),
    },
    reservePreStreamRun: () => Symbol("reservation"),
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
    toast: { error: () => calls.push("toast") },
    cancelAudioUpload: () => undefined,
    claimThreadCreation: () => undefined,
    projectScope: null,
    armJustSent: () => undefined,
    releasePreStreamRunReservation: () => {
      calls.push("release-pre-stream");
      return true;
    },
    notifyPromptQueueRunFailed: () => calls.push("queue-failed"),
    referenceThreadId: "chat",
    quoteThreadId: "thread-1",
    reserveQuoteDispatch: () => ({
      kind: "reserved",
      dispatch: {
        contextId: "ctx-1",
        quoteIds: ["q1"],
        envelope: { quotes: [] },
        origin: "composer",
      },
    }),
    quoteInfoForDispatch: () => ({
      messageId: "__unsloth_quote_context_v1__",
      text: "ctx-1",
    }),
    parseQuoteContext: () => ({
      kind: "valid",
      envelope: { contextId: "ctx-1" },
    }),
    markQuoteDispatchSendStarted: () => calls.push("send-started"),
    markQuoteDispatchSendFailed: () => calls.push("send-failed"),
    releaseQuoteDispatch: () => calls.push("release-quote"),
    isPreStreamRunReservationActive: () => false,
    ComposerAttachmentPreparationError,
    composerSendObserverRef: { current: null },
    observeComposerSend: (
      _composer: unknown,
      _threads: unknown,
      _token: unknown,
      callbacks: { onFailed: (error: unknown) => void },
    ) => {
      onFailed = callbacks.onFailed;
      return {
        kind: "observing",
        attempt: {
          token: null,
          stop: () => calls.push("stop-observer"),
          isActive: () => true,
          isDetached: () => false,
        },
      };
    },
  };
  const send = createCallback(
    reservedSendCallback,
    deps,
  ) as unknown as (options?: { attachQuotes?: boolean }) => void;
  send({ attachQuotes: true });
  assert.deepEqual(calls, ["send-started", "send"]);
  const fail = onFailed as ((error: unknown) => void) | null;
  assert.ok(fail);
  fail(new Error("upload failed"));
  fail(new Error("second report"));
  assert.deepEqual(calls, [
    "send-started",
    "send",
    "stop-observer",
    "send-failed",
    "release-quote",
    "release-pre-stream",
    "queue-failed",
    "toast",
  ]);
});

test("a busy registration leaves the prior observer and its reservation untouched", () => {
  const calls: string[] = [];
  const reservation = Symbol("reservation");
  const previous = {
    token: reservation,
    stop: () => calls.push("stop-previous"),
    isActive: () => true,
  };
  const composerSendObserverRef = { current: previous };
  const preStreamRunReservationRef = { current: null as symbol | null };
  let observeCalls = 0;
  let releaseCalls = 0;
  const deps = {
    aui: {
      threads: () => ({ __internal_getAssistantRuntime: () => undefined }),
      composer: () => ({
        getState: () => ({ text: "Draft", quote: undefined, isEmpty: false }),
        send: () => calls.push("send"),
      }),
    },
    reservePreStreamRun: () => reservation,
    preStreamThreadIds: ["chat"],
    parseExternalModelId: () => null,
    useChatRuntimeStore: {
      getState: () => ({
        params: { checkpoint: "local/model" },
        incognito: false,
        activeGgufVariant: null,
      }),
    },
    preStreamRunReservationRef,
    toast: { error: () => calls.push("toast") },
    cancelAudioUpload: () => undefined,
    claimThreadCreation: () => undefined,
    projectScope: null,
    armJustSent: () => undefined,
    releasePreStreamRunReservation: () => {
      releaseCalls += 1;
      return true;
    },
    notifyPromptQueueRunFailed: () => calls.push("queue-failed"),
    referenceThreadId: "chat",
    isPreStreamRunReservationActive: (token: symbol | null) =>
      token === reservation,
    ComposerAttachmentPreparationError,
    composerSendObserverRef,
    observeComposerSend: () => {
      observeCalls += 1;
      return { kind: "busy", activeToken: reservation };
    },
  };
  const send = createCallback(reservedSendCallback, deps);
  send();
  assert.equal(observeCalls, 1, "the rejected send does not register twice");
  assert.equal(releaseCalls, 0, "the active attempt keeps its reservation");
  assert.equal(composerSendObserverRef.current, previous);
  assert.equal(preStreamRunReservationRef.current, reservation);
  assert.deepEqual(calls, ["toast"]);
});

test("a stale observer is reclaimed without releasing the new reservation", () => {
  const calls: string[] = [];
  const reservation = Symbol("new-reservation");
  const previous = {
    token: Symbol("old-reservation"),
    stop: () => calls.push("stop-previous"),
    isActive: () => true,
  };
  let onFailed: ((error: unknown) => void) | null = null;
  let releaseCalls = 0;
  const deps = {
    aui: {
      threads: () => ({ __internal_getAssistantRuntime: () => undefined }),
      composer: () => ({
        getState: () => ({ text: "Draft", quote: undefined, isEmpty: false }),
        send: () => calls.push("send"),
      }),
    },
    reservePreStreamRun: () => reservation,
    preStreamThreadIds: ["chat"],
    parseExternalModelId: () => null,
    useChatRuntimeStore: {
      getState: () => ({
        params: { checkpoint: "local/model" },
        incognito: false,
        activeGgufVariant: null,
      }),
    },
    preStreamRunReservationRef: { current: null as symbol | null },
    toast: { error: () => calls.push("toast") },
    cancelAudioUpload: () => undefined,
    claimThreadCreation: () => undefined,
    projectScope: null,
    armJustSent: () => undefined,
    releasePreStreamRunReservation: () => {
      releaseCalls += 1;
      return true;
    },
    notifyPromptQueueRunFailed: () => calls.push("queue-failed"),
    referenceThreadId: "chat",
    isPreStreamRunReservationActive: () => false,
    ComposerAttachmentPreparationError,
    composerSendObserverRef: { current: previous },
    observeComposerSend: (
      _composer: unknown,
      _threads: unknown,
      _token: unknown,
      callbacks: { onFailed: (error: unknown) => void },
    ) => {
      onFailed = callbacks.onFailed;
      return {
        kind: "observing",
        attempt: {
          token: reservation,
          stop: () => calls.push("stop-current"),
          isActive: () => true,
          isDetached: () => false,
        },
      };
    },
  };
  const send = createCallback(reservedSendCallback, deps);
  send();
  assert.deepEqual(calls, ["stop-previous", "send"]);
  const fail = onFailed as ((error: unknown) => void) | null;
  assert.ok(fail);
  fail(new Error("upload failed"));
  fail(new Error("second report"));
  assert.deepEqual(calls, [
    "stop-previous",
    "send",
    "stop-current",
    "queue-failed",
    "toast",
  ]);
  assert.equal(
    releaseCalls,
    1,
    "only the current attempt's reservation is released",
  );
});

test("an unavailable observer fails before sending", () => {
  const calls: unknown[] = [];
  const deps = {
    aui: {
      threads: () => ({ __internal_getAssistantRuntime: () => undefined }),
      composer: () => ({
        getState: () => ({
          text: "Draft",
          quote: undefined,
          isEmpty: false,
        }),
        __internal_getRuntime: () => undefined,
        send: () => calls.push("send"),
      }),
    },
    reservePreStreamRun: () => Symbol("reservation"),
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
    toast: {
      error: (title: string) => calls.push(["toast", title]),
    },
    cancelAudioUpload: () => undefined,
    claimThreadCreation: () => undefined,
    projectScope: null,
    armJustSent: () => calls.push("arm"),
    releasePreStreamRunReservation: () => {
      calls.push("release");
      return true;
    },
    notifyPromptQueueRunFailed: () => calls.push("queue-failed"),
    referenceThreadId: "chat",
    isPreStreamRunReservationActive: () => false,
    ComposerAttachmentPreparationError,
    composerSendObserverRef: { current: null },
    observeComposerSend,
  };
  const send = createCallback(reservedSendCallback, deps);
  send();

  assert.deepEqual(calls, [
    "release",
    "queue-failed",
    ["toast", "Could not send message"],
  ]);
});

test("main, edit and comparison composers use the setting and expose settings access", () => {
  assert.match(text, /submitMode="none"/);
  assert.match(
    text,
    /effectiveSendShortcut\(sendShortcut, editMultiline \? "\\n" : ""\) === "mod-enter"\s*\? "ctrlEnter"\s*: "enter"/,
  );
  assert.match(
    text,
    /onQueueClick=\{\(\) => formRef.current\?\.requestSubmit\(\)\}/,
  );
  assert.match(text, /scrollTarget: "chat-composer"/);
  const compare = readSrc("features/chat/shared-composer.tsx");
  assert.match(
    compare,
    /composerSubmitIntent\(\s*imeKey \? composerKeyEventForImeSubmit\(e\) : e,\s*sendShortcut,\s*text,?\s*\)/,
  );
  assert.match(compare, /scrollTarget: "chat-composer"/);
  const page = readSrc("features/chat/chat-page.tsx");
  assert.match(
    page,
    /showContextWindowUsage &&[\s\S]{0,100}view\.mode === "project" && activeThreadId != null/,
  );
});

test("a real quoted send is not misclassified by the production ignored-send check", async () => {
  const core = new RealComposerCore();
  core.setText("x");
  core.setQuote({
    messageId: "__unsloth_quote_context_v1__",
    text: "unsloth-quote-context:v1\n{}",
  });
  const fixture = realAccessorFor(core);
  await new Promise<void>((resolve) => setTimeout(resolve, 0));
  await new Promise<void>((resolve) => setTimeout(resolve, 10));

  let released = 0;
  let toasts = 0;
  const deps = {
    aui: {
      threads: () => ({ __internal_getAssistantRuntime: () => undefined }),
      composer: () => fixture.accessor,
    },
    reservePreStreamRun: () => Symbol("reservation"),
    preStreamThreadIds: ["thread-1"],
    parseExternalModelId: () => null,
    useChatRuntimeStore: {
      getState: () => ({
        params: { checkpoint: "local/model" },
        incognito: false,
        activeGgufVariant: null,
      }),
    },
    preStreamRunReservationRef: { current: null },
    toast: {
      error: () => {
        toasts += 1;
      },
    },
    cancelAudioUpload: () => undefined,
    claimThreadCreation: () => undefined,
    projectScope: null,
    armJustSent: () => undefined,
    releasePreStreamRunReservation: () => true,
    notifyPromptQueueRunFailed: () => undefined,
    referenceThreadId: "thread-1",
    quoteThreadId: "thread-1",
    reserveQuoteDispatch: () => ({
      kind: "reserved",
      dispatch: {
        contextId: "ctx-1",
        quoteIds: ["q1"],
        envelope: { quotes: [] },
        origin: "composer",
      },
    }),
    getQuoteDispatch: () => null,
    quoteInfoForDispatch: () => ({
      messageId: "__unsloth_quote_context_v1__",
      text: "unsloth-quote-context:v1\n{}",
    }),
    parseQuoteContext: () => ({
      kind: "valid",
      envelope: { contextId: "ctx-1" },
    }),
    markQuoteDispatchSendStarted: () => undefined,
    markQuoteDispatchSendSettled: () => undefined,
    markQuoteDispatchSendFailed: () => undefined,
    releaseQuoteDispatch: () => {
      released += 1;
    },
    isPreStreamRunReservationActive: () => false,
    ComposerAttachmentPreparationError,
    composerSendObserverRef: { current: null },
    observeComposerSend,
  };
  const send = createCallback(
    reservedSendCallback,
    deps,
  ) as unknown as (options?: { attachQuotes?: boolean }) => void;
  send({ attachQuotes: true });

  assert.equal(toasts, 0, "the real quoted send must not fail");
  assert.equal(released, 0, "the quote dispatch must not be released");

  await Promise.resolve();
  await Promise.resolve();
  await new Promise<void>((resolve) => setTimeout(resolve, 0));

  assert.equal(core.sent.length, 1);
  assert.deepEqual(core.sent[0]?.metadata?.custom?.quote, {
    messageId: "__unsloth_quote_context_v1__",
    text: "unsloth-quote-context:v1\n{}",
  });
  assert.equal(toasts, 0);
  assert.equal(released, 0);

  const observer = (
    deps.composerSendObserverRef as {
      current: { stop: () => void } | null;
    }
  ).current;
  observer?.stop();
  fixture.dispose();
});

test("a detached attachment failure releases reservations without a toast", () => {
  const calls: string[] = [];
  let onFailed: ((error: unknown) => void) | null = null;
  const composerState: { text: string; quote: unknown } = {
    text: "Draft",
    quote: undefined,
  };
  const deps = {
    aui: {
      threads: () => ({ __internal_getAssistantRuntime: () => undefined }),
      composer: () => ({
        getState: () => composerState,
        setQuote: (quote: unknown) => {
          composerState.quote = quote;
        },
        send: () => {
          composerState.quote = undefined;
          calls.push("send");
        },
        __internal_getRuntime: () => ({
          getState: () => ({ quote: undefined, isEmpty: false }),
        }),
      }),
    },
    reservePreStreamRun: () => Symbol("reservation"),
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
    toast: { error: () => calls.push("toast") },
    cancelAudioUpload: () => undefined,
    claimThreadCreation: () => undefined,
    projectScope: null,
    armJustSent: () => undefined,
    releasePreStreamRunReservation: () => {
      calls.push("release-pre-stream");
      return true;
    },
    notifyPromptQueueRunFailed: () => calls.push("queue-failed"),
    referenceThreadId: "chat",
    quoteThreadId: "thread-1",
    reserveQuoteDispatch: () => ({
      kind: "reserved",
      dispatch: {
        contextId: "ctx-1",
        quoteIds: ["q1"],
        envelope: { quotes: [] },
        origin: "composer",
      },
    }),
    quoteInfoForDispatch: () => ({
      messageId: "__unsloth_quote_context_v1__",
      text: "ctx-1",
    }),
    parseQuoteContext: () => ({
      kind: "valid",
      envelope: { contextId: "ctx-1" },
    }),
    markQuoteDispatchSendStarted: () => calls.push("send-started"),
    markQuoteDispatchSendFailed: () => calls.push("send-failed"),
    releaseQuoteDispatch: () => calls.push("release-quote"),
    isPreStreamRunReservationActive: () => false,
    ComposerAttachmentPreparationError,
    composerSendObserverRef: { current: null },
    observeComposerSend: (
      _composer: unknown,
      _threads: unknown,
      _token: unknown,
      callbacks: { onFailed: (error: unknown) => void },
    ) => {
      onFailed = callbacks.onFailed;
      return {
        kind: "observing",
        attempt: {
          token: null,
          stop: () => calls.push("stop-observer"),
          isActive: () => true,
          detach: () => undefined,
          isDetached: () => true,
        },
      };
    },
  };
  const send = createCallback(
    reservedSendCallback,
    deps,
  ) as unknown as (options?: { attachQuotes?: boolean }) => void;
  send({ attachQuotes: true });
  assert.deepEqual(calls, ["send-started", "send"]);

  const fail = onFailed as ((error: unknown) => void) | null;
  assert.ok(fail);
  fail(new Error("upload failed"));
  assert.deepEqual(calls, [
    "send-started",
    "send",
    "stop-observer",
    "send-failed",
    "release-quote",
    "release-pre-stream",
    "queue-failed",
  ]);
});
