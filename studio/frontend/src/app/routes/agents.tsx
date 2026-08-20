import { getProductCapability } from "@/config/platform-capabilities";
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
    if (!getProductCapability("agents").available) {
      throw redirect({ to: "/chat" });
    }
    return requireAuth();
  },
  component: AgentsPage,
});
