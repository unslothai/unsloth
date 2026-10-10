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

function harness(
  options: {
    propId?: string | null;
    initialized?: boolean;
    missing?: boolean;
    temporary?: boolean;
    persist?: Promise<void>;
    itemId?: string;
    storedIds?: string[];
    ragSource?: { type: "thread" } | { type: "kb"; kbId: string };
  } = {},
) {
  let initialized = options.initialized ?? false;
  let incognito = false;
  let initializeCalls = 0;
  let nativePending = false;
  let cursor = 0;
  const slots: unknown[] = [];
  const effects: Array<() => void> = [];
  const cleanups: Array<() => void> = [];
  const uploads: Scope[] = [];
  const errors: string[] = [];
  const toasts: Array<{
    title: string;
    data: { duration?: number; action?: { label: string; onClick: () => void } };
  }> = [];
  const dismissed: number[] = [];
  const adopted: string[] = [];
  const itemId = options.itemId ?? ID;
  const storedIds = new Set(options.storedIds ?? []);
  const state = {
    ragEnabled: true,
    ragSource: options.ragSource ?? { type: "thread" },
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
          path: {
            token: "native-docx",
            sizeBytes: 20,
            modifiedMs: 1,
            expiresAtMs: Date.now() + 900_000,
          },
          displayLabel: "report.docx",
        },
      ];
    },
  };
  let pickedFiles: ((files: File[]) => void) | null = null;
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
      "lucide-react": {},
      "@/lib/tick-icon": {},
      "@/lib/api-base": { isTauri: false },
      "@/lib/open-file-picker": {
        openFilePicker: (_accept: string, onFiles: (files: File[]) => void) => {
          pickedFiles = onFiles;
        },
      },
      "@/components/assistant-ui/attachment": {},
      "@/components/ui/spinner": {},
      "./preview-store": { useDocumentPreviewStore: () => undefined },
      "@/lib/chevron-icons": {},
      "@assistant-ui/react": { useAui: () => ({ threadListItem: () => item }) },
      "@/lib/utils": { cn: () => "" },
      "@/features/chat/stores/chat-runtime-store": {
        useChatRuntimeStore: store,
        readPendingAttachmentTargetClaim: () => null,
      },
      "@/features/chat/hooks/use-rag-tool-disabled": {
        useRagToolDisabled: () => false,
      },
      "@/features/chat": {
        chatHistoryClearBoundary: { capture: () => 0 },
        ChatThreadDeletedError: class extends Error {},
        isThreadIncognito: () => incognito,
        isPastedTextFile: () => false,
        annotationsOfFile: () => null,
        getStoredChatThread: async () => undefined,
        ensureStoredChatThread: async (threadId: string) => {
          if (storedIds.has(threadId)) return { id: threadId };
          if (!initialized || options.missing) return undefined;
          await options.persist;
          return threadId === itemId ? { id: itemId } : undefined;
        },
      },
      "@/features/native-intents": {
        useNativeAttachmentTargetKey: () => ID,
        useNativeIntentStore: nativeStore,
      },
      "@/lib/toast": {
        toast: Object.assign(
          (title: string, data: (typeof toasts)[number]["data"]) =>
            toasts.push({ title, data }) - 1,
          {
            error: (message: string) => errors.push(message),
            dismiss: (id: number) => dismissed.push(id),
          },
        ),
      },
      "@/components/ui/dropdown-menu": {},
      "@/components/ui/alert-dialog": {},
      "../api/rag-api": {
        listKnowledgeBases: async () => [{ id: "kb-1", name: "Product docs" }],
        subscribeKnowledgeBasesChanged: () => () => {},
      },
      "../api/rag-availability": {
        useRagAvailabilityStore: (
          select: (s: { isUnavailable: () => boolean }) => unknown,
        ) => select({ isUnavailable: () => false }),
      },
      "../types/rag": { CHAT_FILES_ACCEPT: ".docx" },
      "./document-status-chip": {},
      "./knowledge-base-dialog": { KnowledgeBaseDialog: "KnowledgeBaseDialog" },
      "./staged-source": { EXPIRY_GRACE_MS: 30_000 },
      "./use-rag-documents": {
        uploadItemFromIntent: (intent: {
          path: { token: string };
          displayLabel: string;
        }) => ({
          kind: "native",
          token: intent.path.token,
          name: intent.displayLabel,
        }),
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
    effects.splice(0).forEach((effect) => {
      const cleanup = (effect as () => unknown)();
      if (typeof cleanup === "function") cleanups.push(cleanup as () => void);
    });
  }
  function pick() {
    const addCard = findComponent(tree, "AddFilesCard")!;
    (addCard.props.onClick as () => void)();
    pickedFiles!([new File(["document"], "report.docx")]);
  }
  return {
    render,
    pick,
    uploads,
    errors,
    toasts,
    dismissed,
    adopted,
    unmount() {
      cleanups.splice(0).forEach((cleanup) => cleanup());
    },
    setRagSource(source: { type: "thread" } | { type: "kb"; kbId: string }) {
      state.ragSource = source;
    },
    get tree() {
      return tree;
    },
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

function findElement(node: unknown, type: string): StubElement | undefined {
  if (Array.isArray(node)) {
    for (const child of node) {
      const hit = findElement(child, type);
      if (hit) return hit;
    }
    return undefined;
  }
  if (!node || typeof node !== "object" || !("props" in node)) return undefined;
  const element = node as StubElement;
  if (element.type === type) return element;
  return findElement(element.props.children, type);
}

function findComponent(node: unknown, name: string): StubElement | undefined {
  if (Array.isArray(node)) {
    for (const child of node) {
      const hit = findComponent(child, name);
      if (hit) return hit;
    }
    return undefined;
  }
  if (!node || typeof node !== "object" || !("props" in node)) return undefined;
  const element = node as StubElement;
  if (typeof element.type === "function" && element.type.name === name) return element;
  return findComponent(element.props.children, name);
}

/** The KB panel's "Manage files" button. */
function manageFiles(tree: StubElement): StubElement {
  const panel = findComponent(tree, "ChatFilesPanel")!;
  return panel.props.headerControls as StubElement;
}

function kbDialog(tree: StubElement): StubElement | undefined {
  return findElement(tree, "KnowledgeBaseDialog");
}

test("a native drop into a knowledge base chat offers to add the files to that knowledge base", async () => {
  const app = harness({ ragSource: { type: "kb", kbId: "kb-1" } });
  app.render();
  await flush();
  app.render();
  app.drop();
  await flush();
  // Nothing goes to the thread: a thread upload would index somewhere this chat never reads.
  assert.deepEqual(app.uploads, []);
  assert.equal(app.initializeCalls, 0);
  assert.deepEqual(app.errors, []);
  assert.equal(kbDialog(app.tree)!.props.open, false);

  // Names the file and the knowledge base, and stays while the dropped paths are readable.
  assert.equal(app.toasts.length, 1);
  const [{ title, data }] = app.toasts;
  assert.equal(title, 'Add "report.docx" to "Product docs"?');
  assert.equal(data.action?.label, "Add");
  assert.ok(data.duration! > 60_000, `duration ${data.duration}`);
  data.action!.onClick();
  app.render();
  const dialog = kbDialog(app.tree)!;
  assert.equal(dialog.props.open, true);
  assert.deepEqual(dialog.props.focus, {
    kbId: "kb-1",
    uploads: [{ kind: "native", token: "native-docx", name: "report.docx" }],
  });

  (dialog.props.onOpenChange as (open: boolean) => void)(false);
  app.render();
  assert.equal(kbDialog(app.tree)!.props.open, false);
});

test("the drop's Add still opens its knowledge base after the chat's source moves", async () => {
  const app = harness({ ragSource: { type: "kb", kbId: "kb-1" } });
  app.render();
  await flush();
  app.render();
  app.drop();
  await flush();
  app.setRagSource({ type: "thread" });
  app.render();
  app.toasts[0].data.action!.onClick();
  app.render();
  const dialog = kbDialog(app.tree)!;
  assert.equal(dialog.props.open, true);
  assert.deepEqual(dialog.props.focus, {
    kbId: "kb-1",
    uploads: [{ kind: "native", token: "native-docx", name: "report.docx" }],
  });
  assert.deepEqual(app.uploads, []);
});

test("the knowledge base panel opens that knowledge base without uploading anything", () => {
  const app = harness({ ragSource: { type: "kb", kbId: "kb-1" } });
  app.render();
  (manageFiles(app.tree).props.onClick as () => void)();
  app.render();
  const dialog = kbDialog(app.tree)!;
  assert.equal(dialog.props.open, true);
  assert.deepEqual(dialog.props.focus, { kbId: "kb-1" });
});

test("an open dialog keeps its place when deleting the active knowledge base moves the source", () => {
  // React keeps an instance only at the same type and position. Deleting the active KB
  // inside the dialog switches the chat to its own files, and a dialog that moved in the
  // tree would remount, replaying its animation and dropping its state.
  const app = harness({ ragSource: { type: "kb", kbId: "kb-1" } });
  app.render();
  (manageFiles(app.tree).props.onClick as () => void)();
  app.render();
  const place = (tree: StubElement) => ({
    root: tree.type,
    first: (tree.props.children as StubElement[])[0]?.type,
  });
  const before = place(app.tree);
  app.setRagSource({ type: "thread" });
  app.render();
  assert.deepEqual(place(app.tree), before);
  assert.deepEqual(before, {
    root: Symbol.for("Fragment"),
    first: "KnowledgeBaseDialog",
  });
  assert.equal(kbDialog(app.tree)!.props.open, true);
});

test("leaving the chat dismisses a drop's offer, whose Add could no longer open anything", async () => {
  const app = harness({ ragSource: { type: "kb", kbId: "kb-1" } });
  app.render();
  await flush();
  app.render();
  app.drop();
  await flush();
  assert.equal(app.toasts.length, 1);
  assert.deepEqual(app.dismissed, []);
  app.unmount();
  assert.deepEqual(app.dismissed, [0]);
});
