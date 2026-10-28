// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { validateChatSearch } from "@/features/chat";
import { createRoute, redirect } from "@tanstack/react-router";
import { useChatRuntimeStore } from "@/features/chat/stores/chat-runtime-store";
import { queryDrafts } from "@/features/chat/query-draft-context";
import { requireAuth } from "../auth-guards";
import { Route as rootRoute } from "./__root";

export const Route = createRoute({
  getParentRoute: () => rootRoute,
  path: "/chat",
  staticData: { title: "Chat" },
  beforeLoad: async ({ search, preload, abortController }) => {
    await requireAuth(search.q);
    if (preload || abortController.signal.aborted) return;
    if (!search.q) {
      if (search.new?.startsWith("query-") && !queryDrafts.has(search.new)) {
        const { loadQueryDraft } = await import(
          "@/features/chat/utils/query-draft"
        );
        if (abortController.signal.aborted) return;
        const draft = loadQueryDraft(search.new);
        if (draft) queryDrafts.set(search.new, draft);
      }
      return;
    }
    const { createQueryDraft } = await import(
      "@/features/chat/utils/query-draft"
    );
    if (abortController.signal.aborted) return;
    const store = useChatRuntimeStore.getState();
    const draft = createQueryDraft(search.q, store.incognito);
    const nonce = draft.nonce;
    queryDrafts.set(nonce, draft);
    store.setActiveThreadId(null);
    store.setActiveProjectId(null);
    // Resolve before mounting the composer: an old shared paste must not begin restoring into it.
    throw redirect({ to: "/chat", search: { new: nonce }, replace: true });
  },
  validateSearch: (search: Record<string, unknown>) =>
    validateChatSearch(search),
  component: () => null,
});
