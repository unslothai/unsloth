// SPDX-License-Identifier: AGPL-3.0-only - See /studio/LICENSE.AGPL-3.0
// Copyright © 2025 Unsloth AI

import { createRoute } from "@tanstack/react-router";
import { lazy } from "react";
import { requireGuest } from "../auth-guards";
import { Route as rootRoute } from "./__root";

const SignupPage = lazy(() =>
  import("@/features/auth").then((m) => ({
    default: m.SignupPage,
  })),
);

export const Route = createRoute({
  getParentRoute: () => rootRoute,
  path: "/signup",
  beforeLoad: () => requireGuest(),
  component: SignupPage,
});
