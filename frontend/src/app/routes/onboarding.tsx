import { createRoute } from "@tanstack/react-router";
import { lazy } from "react";
import { Route as rootRoute } from "./__root";

const WizardLayout = lazy(() =>
  import("@/features/onboarding/components/wizard-layout").then((m) => ({
    default: m.WizardLayout,
  })),
);

export const Route = createRoute({
  getParentRoute: () => rootRoute,
  path: "/onboarding",
  component: WizardLayout,
});
