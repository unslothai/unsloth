// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import {
  loadWithStubs,
  stubJsxRuntime,
  type StubElement,
} from "./helpers/module-stubs.ts";

const ID = "__LOCALID_attachment";
const SAVED_ID = "saved-target";
const OUTGOING_ID = "__LOCALID_outgoing";
const flush = () => new Promise<void>((resolve) => setImmediate(resolve));
type Scope = { type: "thread"; threadId: string };

// @/features/chat re-exports the class from @/features/chat/api/chat-api, so the
// bar and the scope materializer it calls must share one stub or a tombstone
// would fail the instanceof check in one and not the other.
class ChatThreadDeletedErrorStub extends Error {}

const isAssistantLocalThreadId = (id: string | null | undefined) =>
  typeof id === "string" && id.startsWith("__LOCALID_");

function harness(
  options: {
    propId?: string | null;
    initialized?: boolean;
    missing?: boolean;
    temporary?: boolean;
    persist?: Promise<void>;
    itemId?: string;
    storedIds?: string[];
  } = {},
) {
  let initialized = options.initialized ?? false;
  let incognito = false;
  let initializeCalls = 0;
  let nativePending = false;
  let cursor = 0;
  const slots: unknown[] = [];
  const effects: Array<() => void> = [];
  const uploads: Scope[] = [];
  const errors: string[] = [];
  const adopted: string[] = [];
  const itemId = options.itemId ?? ID;
  const storedIds = new Set(options.storedIds ?? []);
  const state = {
    ragEnabled: true,
    ragSource: { type: "thread" },
    activeProjectId: null,
    projectAttachmentTarget: "thread",
    projectAttachmentTargetByThread: {},
    adoptPendingProjectAttachmentTarget: (id: string) => adopted.push(id),
    clearPendingProjectAttachmentTarget() {},
  };
  const store = Object.assign(
    (select: (s: typeof state) => unknown) => select(state),
    {
      getState: () => state,
    },
  );
  const item = {
    getState: () => ({
      id: itemId,
      remoteId: initialized ? itemId : undefined,
      status: initialized ? "regular" : "new",
    }),
    initialize: async () => {
      initializeCalls++;
      await Promise.resolve();
      initialized = true;
      incognito = options.temporary ?? false;
      return { remoteId: itemId };
    },
  };
  const nativeState = {
    pendingAttachments: {},
    takeAttachments: () => {
      nativePending = false;
      return [
        {
          path: { token: "native-docx", sizeBytes: 20, modifiedMs: 1 },
          displayLabel: "report.docx",
        },
      ];
    },
  };
  const nativeStore = Object.assign(() => nativePending, {
    getState: () => nativeState,
  });
  const { ThreadDocumentsBar } = loadWithStubs<{
    ThreadDocumentsBar: (props: { threadId: string | null }) => StubElement;
  }>(
    new URL(
      "../src/features/rag/components/thread-documents-bar.tsx",
      import.meta.url,
    ),
    {
      react: {
        useRef(value: unknown) {
          const index = cursor++;
          slots[index] ??= { current: value };
          return slots[index];
        },
        useState(value: unknown) {
          const index = cursor++;
          if (!(index in slots)) slots[index] = value;
          return [
            slots[index],
            (next: unknown) => {
              slots[index] = next;
            },
          ];
        },
        useCallback: (fn: unknown) => fn,
        useEffect(effect: () => void, deps: unknown[]) {
          const index = cursor++;
          const previous = slots[index] as unknown[] | undefined;
          if (previous && deps.every((dep, i) => Object.is(dep, previous[i])))
            return;
          slots[index] = deps;
          effects.push(effect);
        },
      },
      "react/jsx-runtime": stubJsxRuntime(),
      "@hugeicons/react": {},
      "@hugeicons/core-free-icons": {},
      "@/lib/tick-icon": {},
      "@assistant-ui/react": { useAui: () => ({ threadListItem: () => item }) },
      "@/lib/utils": { cn: () => "" },
      "@/features/chat/stores/chat-runtime-store": {
        useChatRuntimeStore: store,
        readPendingAttachmentTargetClaim: () => null,
      },
      "@/features/chat": {
        chatHistoryClearBoundary: { capture: () => 0 },
        ChatThreadDeletedError: ChatThreadDeletedErrorStub,
        isThreadIncognito: () => incognito,
        getStoredChatThread: async () => undefined,
        ensureStoredChatThread: async (threadId: string) => {
          if (storedIds.has(threadId)) return { id: threadId };
          if (!initialized || options.missing) return undefined;
          await options.persist;
          return threadId === itemId ? { id: itemId } : undefined;
        },
      },
      "@/features/chat/api/chat-api": {
        ChatThreadDeletedError: ChatThreadDeletedErrorStub,
      },
      "@/features/chat/utils/thread-ids": { isAssistantLocalThreadId },
      "@/features/chat/utils/chat-thread-tombstones": {
        isChatThreadDeleted: () => false,
      },
      // The real materializer: the bar delegates to it, so stubbing it away
      // would leave these tests asserting against code the bar never runs.
      "../utils/materialize-thread-scope": loadWithStubs<{
        materializeThreadScope: (m: unknown) => Promise<string>;
      }>(
        new URL(
          "../src/features/rag/utils/materialize-thread-scope.ts",
          import.meta.url,
        ),
        {
          "@/features/chat/api/chat-api": {
            ChatThreadDeletedError: ChatThreadDeletedErrorStub,
          },
          "@/features/chat/utils/thread-ids": { isAssistantLocalThreadId },
        },
      ),
      "@/features/native-intents": {
        useNativeAttachmentTargetKey: () => ID,
        useNativeIntentStore: nativeStore,
      },
      "@/lib/toast": {
        toast: { error: (message: string) => errors.push(message) },
      },
      "@/components/ui/dropdown-menu": {},
      "@/components/ui/alert-dialog": {},
      "../api/rag-api": {},
      "../api/rag-availability": {
        useRagAvailabilityStore: (
          select: (s: { isUnavailable: () => boolean }) => unknown,
        ) => select({ isUnavailable: () => false }),
      },
      "../types/rag": { RAG_UPLOAD_ACCEPT: ".docx" },
      "./document-status-chip": {},
      "./use-rag-documents": {
        useRagDocuments: () => ({
          documents: [],
          uploading: false,
          hasIndexing: false,
          loading: false,
          upload: async (_files: unknown, resolve: () => Promise<Scope>) => {
            try {
              uploads.push(await resolve());
            } catch (error) {
              errors.push((error as Error).message);
            }
          },
        }),
      },
    },
  );
  let tree: StubElement;
  function render() {
    cursor = 0;
    tree = ThreadDocumentsBar({
      threadId: options.propId === undefined ? ID : options.propId,
    });
    effects.splice(0).forEach((effect) => effect());
  }
  function pick() {
    const input = (tree.props.children as StubElement[]).find(
      (child) => child?.type === "input",
    )!;
    const onChange = input.props.onChange as (event: unknown) => void;
    onChange({
      target: {
        files: [new File(["document"], "report.docx")],
        value: "report.docx",
      },
    });
  }
  return {
    render,
    pick,
    uploads,
    errors,
    adopted,
    drop() {
      nativePending = true;
      render();
    },
    get initializeCalls() {
      return initializeCalls;
    },
  };
}

for (const entry of ["picker", "native drop"] as const) {
  test(`${entry} initializes an empty chat even though it already has a local ID`, async () => {
    const app = harness();
    app.render();
    await flush();
    app.render(); // resolve the project lookup before native drops can drain
    if (entry === "picker") app.pick();
    else app.drop();
    await flush();
    assert.deepEqual(app.errors, []);
    assert.equal(app.initializeCalls, 1);
    assert.deepEqual(app.uploads, [{ type: "thread", threadId: ID }]);
    assert.deepEqual(app.adopted, [ID]);
  });
}

test("attachments still initialize a chat before its ID reaches the bar", async () => {
  const app = harness({ propId: null });
  app.render();
  app.pick();
  await flush();
  assert.deepEqual(app.errors, []);
  assert.equal(app.initializeCalls, 1);
  assert.equal(app.uploads[0]?.threadId, ID);
});

test("a saved chat with a local ID is reused without initialization", async () => {
  const app = harness({ initialized: true });
  app.render();
  app.pick();
  await flush();
  assert.deepEqual(app.errors, []);
  assert.equal(app.initializeCalls, 0);
  assert.equal(app.uploads[0]?.threadId, ID);
});

test("navigation does not initialize the outgoing new item before a saved target switch lands", async () => {
  const app = harness({
    propId: SAVED_ID,
    itemId: OUTGOING_ID,
    storedIds: [SAVED_ID],
  });
  app.render();
  app.pick();
  await flush();
  assert.deepEqual(app.errors, []);
  assert.equal(app.initializeCalls, 0);
  assert.deepEqual(app.uploads, [{ type: "thread", threadId: SAVED_ID }]);
});

test("a missing initialized chat is still rejected instead of recreated", async () => {
  const app = harness({ initialized: true, missing: true });
  app.render();
  app.pick();
  await flush();
  assert.deepEqual(app.errors, [`Thread ${ID} was not persisted`]);
  assert.equal(app.initializeCalls, 0);
  assert.equal(app.uploads.length, 0);
});

test("overlapping attachments initialize once and wait for persistence", async () => {
  let persist!: () => void;
  const app = harness({
    persist: new Promise<void>((resolve) => {
      persist = resolve;
    }),
  });
  app.render();
  app.pick();
  app.pick();
  await flush();
  assert.equal(app.uploads.length, 0);
  assert.equal(app.initializeCalls, 1);
  persist();
  await flush();
  assert.deepEqual(app.errors, []);
  assert.equal(app.uploads.length, 2);
  assert.ok(app.uploads.every((scope) => scope.threadId === ID));
});

test("initialization tags a temporary chat before the persistence check", async () => {
  const app = harness({ temporary: true, missing: true });
  app.render();
  app.pick();
  await flush();
  assert.deepEqual(app.errors, []);
  assert.equal(app.initializeCalls, 1);
  assert.equal(app.uploads[0]?.threadId, ID);
});
