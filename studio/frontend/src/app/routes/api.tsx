


import { FEATURE_API_MONITOR } from "@/config/disabled-features";
import {
  createRoute,
  lazyRouteComponent,
  redirect,
} from "@tanstack/react-router";
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
  beforeLoad: () => {
    if (!FEATURE_API_MONITOR) {
      throw redirect({ to: "/chat" });
    }
    return requireAuth();
  },
  component: ApiMonitorPage,
});
