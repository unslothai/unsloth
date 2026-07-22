// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { validateChatSearch } from "@/features/chat";
import { createRoute } from "@tanstack/react-router";
import { requireAuth } from "../auth-guards";
import { Route as rootRoute } from "./__root";

// RootLayout renders ChatPage persistently (so it survives leaving the tab); this
// route only owns the URL, auth gate, and search validation.
export const Route = createRoute({
  getParentRoute: () => rootRoute,
  path: "/chat",
  staticData: { title: "Chat" },
  beforeLoad: () => requireAuth(),
  validateSearch: (search: Record<string, unknown>) => validateChatSearch(search),
  component: () => null,
});
