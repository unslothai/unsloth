import { createRoute } from "@tanstack/react-router";
import { lazy } from "react";
import { Route as rootRoute } from "./__root";

const CanvasLabPage = lazy(() =>
  import("@/features/canvas-lab").then((m) => ({
    default: m.CanvasLabPage,
  })),
);

export const Route = createRoute({
  getParentRoute: () => rootRoute,
  path: "/canvas-lab",
  component: CanvasLabPage,
});
