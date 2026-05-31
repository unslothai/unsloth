// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { createRoute, redirect } from "@tanstack/react-router";
import { getPostAuthRoute } from "@/features/auth";
import { useSettingsDialogStore } from "@/features/settings";
import { requireAuth } from "../auth-guards";
import { Route as rootRoute } from "./__root";

// /knowledge-bases deep-links to the settings modal's Knowledge Bases
// tab: open it, then redirect home. Mirrors /settings.
export const Route = createRoute({
  getParentRoute: () => rootRoute,
  path: "/knowledge-bases",
  staticData: { title: "Knowledge Bases" },
  beforeLoad: async () => {
    await requireAuth();
    useSettingsDialogStore.getState().openDialog("knowledge-bases");
    throw redirect({ to: getPostAuthRoute() });
  },
  component: () => null,
});
