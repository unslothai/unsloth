// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { createRoute, redirect } from "@tanstack/react-router";
import { getPostAuthRoute } from "@/features/auth";
import { useSettingsDialogStore } from "@/features/settings";
import { requireAuth } from "../auth-guards";
import { Route as rootRoute } from "./__root";

// /settings is a deep link to the modal. Open it, then redirect home.
// `staticData.title` is for the rare case where `beforeLoad` returns
// without throwing (e.g. future refactor); the live "Settings" tab
// title while the dialog is visible is driven by `useSettingsDialogStore`
// in `__root.tsx`, since the redirect means /settings never matches.
export const Route = createRoute({
  getParentRoute: () => rootRoute,
  path: "/settings",
  staticData: { title: "Settings" },
  beforeLoad: async () => {
    await requireAuth();
    useSettingsDialogStore.getState().openDialog();
    throw redirect({ to: getPostAuthRoute() });
  },
  component: () => null,
});
