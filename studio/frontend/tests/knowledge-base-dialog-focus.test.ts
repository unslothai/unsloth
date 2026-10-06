// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import {
  loadWithStubs,
  stubJsxRuntime,
  type StubElement,
} from "./helpers/module-stubs.ts";

// Runs the real KnowledgeBaseDialog and its documents view on a small hook host. Each
// component instance keeps its own hook slots, and a StrictMode mount runs its effects,
// their cleanups, and the effects again on the same state and refs, as React 19 does.

type Effect = {
  deps?: unknown[];
  cleanup?: (() => void) | void;
  run?: () => (() => void) | void;
};

class Host {
  slots: unknown[] = [];
  cursor = 0;
  queued: number[] = [];

  begin() {
    this.cursor = 0;
    this.queued = [];
  }

  flushEffects() {
    for (const index of this.queued.splice(0)) {
      const effect = this.slots[index] as Effect;
      effect.cleanup?.();
      effect.cleanup = effect.run?.();
    }
  }

  strictRemount() {
    for (const slot of this.slots) {
      const effect = slot as Effect | undefined;
      if (effect && "run" in effect) effect.cleanup?.();
    }
    for (const slot of this.slots) {
      const effect = slot as Effect | undefined;
      if (effect && "run" in effect) effect.cleanup = effect.run?.();
    }
  }

  unmount() {
    for (const slot of this.slots) {
      const effect = slot as Effect | undefined;
      if (effect && "run" in effect) effect.cleanup?.();
    }
  }
}

const sameDeps = (a?: unknown[], b?: unknown[]) =>
  Boolean(a && b && a.length === b.length && a.every((v, i) => Object.is(v, b[i])));

let current: Host;
const react = {
  useState(initial: unknown) {
    const host = current;
    const index = host.cursor++;
    if (!(index in host.slots)) host.slots[index] = initial;
    return [
      host.slots[index],
      (next: unknown) => {
        host.slots[index] =
          typeof next === "function"
            ? (next as (prev: unknown) => unknown)(host.slots[index])
            : next;
      },
    ];
  },
  useRef(initial: unknown) {
    const index = current.cursor++;
    current.slots[index] ??= { current: initial };
    return current.slots[index];
  },
  useCallback(fn: unknown, deps: unknown[]) {
    const index = current.cursor++;
    const prev = current.slots[index] as { fn: unknown; deps: unknown[] } | undefined;
    if (prev && sameDeps(prev.deps, deps)) return prev.fn;
    current.slots[index] = { fn, deps };
    return fn;
  },
  useEffect(run: () => (() => void) | void, deps?: unknown[]) {
    const index = current.cursor++;
    const prev = current.slots[index] as Effect | undefined;
    if (prev && sameDeps(prev.deps, deps)) return;
    current.slots[index] = { ...prev, deps, run };
    current.queued.push(index);
  },
};

// The documents view defers its upload with window.setTimeout.
globalThis.window = { setTimeout, clearTimeout } as unknown as Window &
  typeof globalThis;

const KB = { id: "kb-1", name: "Product docs", documentCount: 0 };
const ITEM = { kind: "native", token: "tok-notes", name: "notes.md" };

function harness(rows: Array<typeof KB> = [KB]) {
  const state = { uploading: false };
  const uploads: unknown[][] = [];
  const abandoned: unknown[][] = [];
  const errors: string[] = [];
  const { KnowledgeBaseDialog } = loadWithStubs<{
    KnowledgeBaseDialog: (props: Record<string, unknown>) => StubElement;
  }>(
    new URL(
      "../src/features/rag/components/knowledge-base-dialog.tsx",
      import.meta.url,
    ),
    {
      react,
      "react/jsx-runtime": stubJsxRuntime(),
      "@hugeicons/core-free-icons": {},
      "@hugeicons/react": {},
      "lucide-react": {},
      "@/components/ui/alert-dialog": {},
      "@/components/ui/button": { Button: "Button" },
      "@/components/ui/dialog": {
        Dialog: "Dialog",
        DialogContent: "DialogContent",
        DialogDescription: "DialogDescription",
        DialogHeader: "DialogHeader",
        DialogTitle: "DialogTitle",
      },
      "@/components/ui/input": {},
      "@/components/ui/label": {},
      "@/components/ui/spinner": {},
      "@/components/ui/textarea": {},
      "@/lib/toast": {
        toast: { error: (message: string) => errors.push(message) },
      },
      "@/lib/utils": { cn: () => "" },
      "../api/rag-api": {
        listKnowledgeBases: async () => rows,
        listKnowledgeBaseDocuments: async () => [],
      },
      "../api/rag-availability": {
        useRagAvailabilityStore: (
          select: (s: {
            isUnavailable: () => boolean;
            unavailableReason: () => string | null;
          }) => unknown,
        ) =>
          select({ isUnavailable: () => false, unavailableReason: () => null }),
      },
      "../types/rag": { isLinkedFolderManaged: () => false },
      "./document-status-chip": {},
      "./linked-folders-manager": {},
      "./source-drop-policy": { RAG_SOURCE_UPLOAD_ACCEPT: "" },
      "./use-rag-documents": {
        // Like the real hook: a fresh `upload` every render (its scope argument is a new
        // object each time), and an unmount cleanup that abandons uploads already running.
        useRagDocuments: () => {
          const generation = react.useRef(0) as { current: number };
          react.useEffect(
            () => () => {
              generation.current += 1;
            },
            [],
          );
          return {
            documents: [],
            loading: false,
            uploading: state.uploading,
            refresh: async () => {},
            upload: async (items: unknown[]) => {
              const started = generation.current;
              await Promise.resolve();
              (started === generation.current ? uploads : abandoned).push(items);
            },
            remove: async () => {},
          };
        },
      },
      "./use-source-drop": {
        useSourceDrop: () => ({
          dragging: false,
          dropProps: {},
          nativeDropTarget: () => {},
        }),
      },
    },
  );

  const dialog = new Host();
  let documents: Host | null = null;
  let documentsMounts = 0;
  let tree: StubElement;
  let props: Record<string, unknown> = {};

  function find(
    node: unknown,
    match: (element: StubElement) => boolean,
  ): StubElement | undefined {
    if (Array.isArray(node)) {
      for (const child of node) {
        const hit = find(child, match);
        if (hit) return hit;
      }
      return undefined;
    }
    if (!node || typeof node !== "object" || !("props" in node)) return undefined;
    const element = node as StubElement;
    if (match(element)) return element;
    return find(element.props.children, match);
  }

  // Radix renders DialogContent only while the dialog is open, so a closed dialog has no
  // documents view mounted, whatever its view state says.
  const documentsView = () =>
    props.open
      ? find(
          tree,
          (e) =>
            typeof e.type === "function" && e.type.name === "KnowledgeBaseDocuments",
        )
      : undefined;

  // One render pass: the dialog, then its documents view if it shows one. A view that
  // just appeared mounts the way StrictMode mounts it.
  function render() {
    current = dialog;
    dialog.begin();
    tree = KnowledgeBaseDialog(props);
    dialog.flushEffects();
    const view = documentsView();
    if (!view) {
      documents?.unmount();
      documents = null;
      return;
    }
    const mounting = documents === null;
    if (mounting) documentsMounts += 1;
    documents ??= new Host();
    current = documents;
    documents.begin();
    (view.type as (p: Record<string, unknown>) => StubElement)(view.props);
    documents.flushEffects();
    if (mounting) documents.strictRemount();
  }

  async function settle() {
    for (let i = 0; i < 4; i++) {
      render();
      await new Promise<void>((resolve) => setTimeout(resolve, 0));
    }
  }

  return {
    uploads,
    abandoned,
    errors,
    async open(focus: Record<string, unknown> | null) {
      props = { open: true, onOpenChange() {}, focus };
      await settle();
    },
    async close() {
      props = { ...props, open: false, focus: null };
      await settle();
    },
    settle,
    get title() {
      return find(tree, (e) => e.type === "DialogTitle")?.props.children;
    },
    get documentsView() {
      return documentsView();
    },
    get documentsMounts() {
      return documentsMounts;
    },
    set uploading(value: boolean) {
      state.uploading = value;
    },
    openRow() {
      const row = find(tree, (e) => e.props.title === "Open to add or remove documents");
      (row!.props.onClick as () => void)();
    },
    back() {
      (documentsView()!.props.onBack as () => void)();
    },
  };
}

test("files handed to a knowledge base upload once, on the StrictMode mount that stays", async () => {
  const app = harness();
  await app.open({ kbId: KB.id, uploads: [ITEM] });
  assert.equal(app.title, KB.name);
  assert.deepEqual(app.uploads, [[ITEM]]);
  assert.deepEqual(app.abandoned, []);

  // Later renders hand the view a new `upload` each time; the batch must not go again.
  await app.settle();
  assert.deepEqual(app.uploads, [[ITEM]]);
});

test("going back and opening the knowledge base again does not upload the files twice", async () => {
  const app = harness();
  await app.open({ kbId: KB.id, uploads: [ITEM] });
  app.back();
  await app.settle();
  assert.equal(app.documentsView, undefined);
  app.openRow();
  await app.settle();
  assert.equal(app.title, KB.name);
  assert.deepEqual(app.uploads, [[ITEM]]);
});

test("reopening the dialog on the same knowledge base without files uploads nothing more", async () => {
  const app = harness();
  await app.open({ kbId: KB.id, uploads: [ITEM] });
  await app.close();
  assert.equal(app.documentsView, undefined);
  await app.open({ kbId: KB.id });
  assert.equal(app.title, KB.name);
  // Closing unmounted the view, so this is a fresh mount. The host lets that mount's timer
  // fire before the open effect's reset renders, which is stricter than React's ordering.
  assert.equal(app.documentsMounts, 2);
  assert.deepEqual(app.uploads, [[ITEM]]);
});

test("a missing knowledge base says so only when files were handed over", async () => {
  const withFiles = harness([]);
  await withFiles.open({ kbId: KB.id, uploads: [ITEM] });
  assert.equal(withFiles.title, "Knowledge bases");
  assert.deepEqual(withFiles.errors, ["Knowledge base not found"]);
  assert.deepEqual(withFiles.uploads, []);

  const fromChip = harness([]);
  await fromChip.open({ kbId: KB.id });
  assert.equal(fromChip.title, "Knowledge bases");
  assert.deepEqual(fromChip.errors, []);
});

test("files handed to a missing knowledge base go to the one opened next, once", async () => {
  const app = harness();
  await app.open({ kbId: "kb-deleted", uploads: [ITEM] });
  assert.equal(app.title, "Knowledge bases");
  assert.deepEqual(app.errors, ["Knowledge base not found"]);
  assert.deepEqual(app.uploads, []);

  app.openRow();
  await app.settle();
  assert.equal(app.title, KB.name);
  assert.deepEqual(app.uploads, [[ITEM]]);

  app.back();
  await app.settle();
  app.openRow();
  await app.settle();
  assert.deepEqual(app.uploads, [[ITEM]]);
});

test("a handoff that arrives while another waits joins it instead of replacing it", async () => {
  const app = harness();
  await app.open({ kbId: "kb-deleted", uploads: [ITEM] });
  assert.deepEqual(app.uploads, []);
  // The chat's next drop is handed over while the first batch is still waiting here.
  const LATER = { kind: "native", token: "tok-faq", name: "faq.md" };
  await app.open({ kbId: KB.id, uploads: [LATER] });
  assert.equal(app.title, KB.name);
  assert.deepEqual(app.uploads, [[ITEM, LATER]]);

  // Once uploading, the batch is no longer waiting, so a later handoff goes alone.
  const LAST = { kind: "native", token: "tok-last", name: "last.md" };
  await app.open({ kbId: KB.id, uploads: [LAST] });
  assert.deepEqual(app.uploads, [[ITEM, LATER], [LAST]]);
});

test("a handoff into the knowledge base on screen waits for its running upload without remounting", async () => {
  const app = harness();
  await app.open({ kbId: KB.id, uploads: [ITEM] });
  assert.deepEqual(app.uploads, [[ITEM]]);
  app.uploading = true;
  // A second drop's "Add" while the first batch is still uploading in this view.
  const LATER = { kind: "native", token: "tok-faq", name: "faq.md" };
  await app.open({ kbId: KB.id, uploads: [LATER] });
  assert.equal(app.documentsMounts, 1);
  assert.deepEqual(app.uploads, [[ITEM]]);
  app.uploading = false;
  await app.settle();
  assert.deepEqual(app.uploads, [[ITEM], [LATER]]);
  assert.deepEqual(app.abandoned, []);
});
