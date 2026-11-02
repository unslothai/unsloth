// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { getPostAuthRoute } from "@/features/auth";
import { chatQueryParam } from "@/features/chat";
import { createRoute, redirect } from "@tanstack/react-router";
import { requireAuth } from "../auth-guards";
import { Route as rootRoute } from "./__root";

export const Route = createRoute({
  getParentRoute: () => rootRoute,
  path: "/",
  beforeLoad: async ({ location }) => {
    // `/?q=...` is the shorter search-engine URL; carry the query on to the chat it opens.
    const q = chatQueryParam(
      (location.search as Record<string, unknown> | undefined)?.q,
    );
    await requireAuth(q);
    const to = getPostAuthRoute();
    if (to === "/chat" && q) {
      throw redirect({ to, search: { q } });
    }
    throw redirect({ to });
  },
  component: () => null,
});
