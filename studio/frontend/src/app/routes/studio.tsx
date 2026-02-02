import { createRoute } from "@tanstack/react-router";
import { lazy } from "react";
import { Route as rootRoute } from "./__root";

const StudioPage = lazy(() =>
  import("@/features/studio/studio-page").then((m) => ({
    default: m.StudioPage,
  })),
);

export const Route = createRoute({
  getParentRoute: () => rootRoute,
  path: "/studio",
  component: StudioPage,
});
