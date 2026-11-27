// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { createRoute, redirect } from "@tanstack/react-router";
import { getPostAuthRoute } from "@/features/auth";
import { queryParam } from "@/features/chat/chat-search";
import { requireAuth } from "../auth-guards";
import { Route as rootRoute } from "./__root";

export const Route = createRoute({
  getParentRoute: () => rootRoute,
  path: "/",
  beforeLoad: async ({ location }) => {
    // `/?q=...` is the shorter search-engine URL.
    const q = queryParam(location.search);
    await requireAuth(q);
    const to = getPostAuthRoute();
    throw redirect(to === "/chat" ? { to, search: { q } } : { to });
  },
  component: () => null,
});
