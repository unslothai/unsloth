import { FEATURE_AGENTS_NAV } from "@/config/disabled-features";
import {
  createRoute,
  lazyRouteComponent,
  redirect,
} from "@tanstack/react-router";
import { requireAuth } from "../auth-guards";
import { Route as rootRoute } from "./__root";

const AgentsPage = lazyRouteComponent(
  () => import("@/features/agents/agents-page"),
  "AgentsPage",
);

export const Route = createRoute({
  getParentRoute: () => rootRoute,
  path: "/agents",
  staticData: { title: "Agents" },
  beforeLoad: () => {
    if (!FEATURE_AGENTS_NAV) throw redirect({ to: "/chat" });
    return requireAuth();
  },
  component: AgentsPage,
});
