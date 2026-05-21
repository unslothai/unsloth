// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { createRoute, redirect } from "@tanstack/react-router";
import { getPostAuthRoute } from "@/features/auth";
import { useSettingsDialogStore } from "@/features/settings";
import { requireAuth } from "../auth-guards";
import { Route as rootRoute } from "./__root";

// /settings is a deep link to the modal. Open it, then redirect home.
export const Route = createRoute({
  getParentRoute: () => rootRoute,
  path: "/settings",
  beforeLoad: async () => {
    await requireAuth();
    useSettingsDialogStore.getState().openDialog();
    throw redirect({ to: getPostAuthRoute() });
  },
  component: () => null,
});
