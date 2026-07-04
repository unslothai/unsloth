// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { createRoute } from "@tanstack/react-router";
import { requireAuth } from "../auth-guards";
import { Route as rootRoute } from "./__root";

// RootLayout renders VideoPage persistently (so an in-flight generation is not cancelled
// when leaving the tab); this route only owns the URL + auth gate.
export const Route = createRoute({
  getParentRoute: () => rootRoute,
  path: "/video",
  staticData: { title: "Video" },
  beforeLoad: () => requireAuth(),
  component: () => null,
});
