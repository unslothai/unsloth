import { createRoute } from "@tanstack/react-router";
import { lazy } from "react";
import { Route as rootRoute } from "./__root";

const ExportPage = lazy(() =>
  import("@/features/export/export-page").then((m) => ({
    default: m.ExportPage,
  })),
);

export const Route = createRoute({
  getParentRoute: () => rootRoute,
  path: "/export",
  component: ExportPage,
});
