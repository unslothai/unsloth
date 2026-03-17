// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { createRoute } from "@tanstack/react-router";
import { lazy } from "react";
import { requirePasswordChangeFlow } from "../auth-guards";
import { Route as rootRoute } from "./__root";

const ChangePasswordPage = lazy(() =>
  import("@/features/auth").then((m) => ({
    default: m.ChangePasswordPage,
  })),
);

export const Route = createRoute({
  getParentRoute: () => rootRoute,
  path: "/change-password",
  beforeLoad: () => requirePasswordChangeFlow(),
  component: ChangePasswordPage,
});
