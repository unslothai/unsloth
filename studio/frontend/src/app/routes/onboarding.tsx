// SPDX-License-Identifier: AGPL-3.0-only - See /studio/LICENSE.AGPL-3.0
// Copyright © 2025 Unsloth AI

import { createRoute } from "@tanstack/react-router";
import { lazy } from "react";
import { requireAuth } from "../auth-guards";
import { Route as rootRoute } from "./__root";

const WizardLayout = lazy(() =>
  import("@/features/onboarding/components/wizard-layout").then((m) => ({
    default: m.WizardLayout,
  })),
);

export const Route = createRoute({
  getParentRoute: () => rootRoute,
  path: "/onboarding",
  beforeLoad: () => requireAuth(),
  component: WizardLayout,
});
