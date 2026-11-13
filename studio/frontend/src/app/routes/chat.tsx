// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { validateChatSearch } from "@/features/chat/chat-search";
import { seedQueryDraft } from "@/features/chat/query-draft";
import { useChatRuntimeStore } from "@/features/chat/stores/chat-runtime-store";
import { createRoute, redirect } from "@tanstack/react-router";
import { requireAuth } from "../auth-guards";
import { Route as rootRoute } from "./__root";

export const Route = createRoute({
  getParentRoute: () => rootRoute,
  path: "/chat",
  staticData: { titleKey: "routes.chat" },
  beforeLoad: async ({ search, preload, abortController }) => {
    await requireAuth(search.q);
    if (!search.q || preload || abortController.signal.aborted) return;
    const store = useChatRuntimeStore.getState();
    const nonce = seedQueryDraft(search.q, store.incognito);
    store.setActiveThreadId(null);
    store.setActiveProjectId(null);
    // Dropping `q` from the URL means a reload or Back never seeds it again.
    throw redirect({ to: "/chat", search: { new: nonce }, replace: true });
  },
  validateSearch: (search: Record<string, unknown>) => validateChatSearch(search),
  component: () => null,
});
