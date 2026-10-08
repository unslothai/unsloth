// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { createRoute } from "@tanstack/react-router";
import { validateAudioSearch } from "../../features/audio/route-search.ts";
import { requireAuth } from "../auth-guards";
import { Route as rootRoute } from "./__root";

// RootLayout renders AudioPage persistently (so an in-flight generation is not cancelled when leaving the tab); this route only owns the URL + auth gate.
export const Route = createRoute({
  getParentRoute: () => rootRoute,
  path: "/audio",
  staticData: { title: "Audio" },
  validateSearch: validateAudioSearch,
  beforeLoad: () => requireAuth(),
  component: () => null,
});
