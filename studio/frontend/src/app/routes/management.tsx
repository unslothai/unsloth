import { ManagementPage } from "@/features/management/management-page";
import { getProductCapability } from "@/config/platform-capabilities";
import { createRoute, redirect } from "@tanstack/react-router";
import { requireAuth } from "../auth-guards";
import { Route as rootRoute } from "./__root";

export const Route = createRoute({
  getParentRoute: () => rootRoute,
  path: "/management",
  staticData: { title: "Yönetim" },
  beforeLoad: () => {
    if (!getProductCapability("management").available) {
      throw redirect({ to: "/chat" });
    }
    return requireAuth();
  },
  component: ManagementPage,
});
