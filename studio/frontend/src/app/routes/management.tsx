import { ManagementPage } from "@/features/management/management-page";
import { FEATURE_MANAGEMENT_NAV } from "@/config/disabled-features";
import { createRoute, redirect } from "@tanstack/react-router";
import { requireAuth } from "../auth-guards";
import { Route as rootRoute } from "./__root";

export const Route = createRoute({
  getParentRoute: () => rootRoute,
  path: "/management",
  staticData: { title: "Yönetim" },
  beforeLoad: () => {
    if (!FEATURE_MANAGEMENT_NAV) throw redirect({ to: "/chat" });
    return requireAuth();
  },
  component: ManagementPage,
});
