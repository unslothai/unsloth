


import { createRoute, lazyRouteComponent } from "@tanstack/react-router";
import { requireAuth } from "../auth-guards";
import { Route as rootRoute } from "./__root";

const ApiMonitorPage = lazyRouteComponent(
  () => import("@/features/api-monitor"),
  "ApiMonitorPage",
);

export const Route = createRoute({
  getParentRoute: () => rootRoute,
  // Not "/api": the backend owns that prefix and its SPA fallback 404s those paths.
  path: "/api-monitor",
  staticData: { title: "API" },
  beforeLoad: () => requireAuth(),
  component: ApiMonitorPage,
});
