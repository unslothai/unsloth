// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import { loadWithStubs, stubJsxRuntime } from "./helpers/module-stubs.ts";

// The source menu refetches on mount, on open and after every mutation, so its fetches
// overlap. Its fallback moves a chat off a deleted KB, which only works if an answer
// from before the delete cannot land after the one from after it.

type Rows = Array<{ id: string; name: string }>;

function harness() {
  const slots: unknown[] = [];
  let cursor = 0;
  const effects: Array<() => unknown> = [];
  const pending: Array<(rows: Rows) => void> = [];
  const listeners: Array<() => void> = [];
  const sources: unknown[] = [];
  const state = {
    ragEnabled: true,
    ragSource: { type: "kb", kbId: "kb-1" } as unknown,
    setRagEnabled() {},
    setRagSource(source: unknown) {
      state.ragSource = source;
      sources.push(source);
    },
  };
  const { KnowledgeBaseComposerButton } = loadWithStubs<{
    KnowledgeBaseComposerButton: () => unknown;
  }>(
    new URL(
      "../src/features/rag/components/knowledge-base-composer-button.tsx",
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
        useCallback(fn: unknown) {
          const index = cursor++;
          slots[index] ??= fn;
          return slots[index];
        },
        useEffect(effect: () => unknown, deps: unknown[]) {
          const index = cursor++;
          const previous = slots[index] as unknown[] | undefined;
          if (previous && deps.every((dep, i) => Object.is(dep, previous[i])))
            return;
          slots[index] = deps;
          effects.push(effect);
        },
        // Focus-only; nothing to run here.
        useLayoutEffect() {
          cursor++;
        },
      },
      "react/jsx-runtime": stubJsxRuntime(),
      "lucide-react": {},
      "@/lib/tick-icon": {},
      "@hugeicons/react": {},
      "@hugeicons/core-free-icons": {},
      "@/components/ui/dropdown-menu": {},
      "@/features/chat/hooks/use-rag-tool-disabled": {
        useRagToolDisabled: () => false,
      },
      "@/features/chat/stores/chat-runtime-store": {
        useChatRuntimeStore: (select: (s: typeof state) => unknown) =>
          select(state),
      },
      "../api/rag-api": {
        listKnowledgeBases: () =>
          new Promise<Rows>((resolve) => pending.push(resolve)),
        subscribeKnowledgeBasesChanged: (onChanged: () => void) => {
          listeners.push(onChanged);
          return () => {};
        },
      },
      "./knowledge-base-dialog": { KnowledgeBaseDialog: "KnowledgeBaseDialog" },
      "./embedding-model-menu-picker": {
        EmbeddingModelMenuChip: "EmbeddingModelMenuChip",
        EmbeddingModelMenuList: "EmbeddingModelMenuList",
      },
      "@/features/auth": { useIsAccountOwner: () => true },
    },
  );
  function render() {
    cursor = 0;
    KnowledgeBaseComposerButton();
    effects.splice(0).forEach((effect) => effect());
  }
  return { render, pending, listeners, sources };
}

const flush = () => new Promise<void>((resolve) => setImmediate(resolve));

test("a list fetched before a delete cannot land after the one fetched after it", async () => {
  const app = harness();
  app.render();
  assert.equal(app.pending.length, 1);
  // The active KB is deleted while the mount fetch is still out.
  app.listeners.forEach((onChanged) => onChanged());
  assert.equal(app.pending.length, 2);

  // Both answers arrive before the next render, so React batches them and the fallback
  // sees only whichever landed last.
  app.pending[1]([]);
  app.pending[0]([{ id: "kb-1", name: "Product docs" }]);
  await flush();
  app.render();
  await flush();
  app.render();

  assert.deepEqual(app.sources, [{ type: "thread" }]);
});
