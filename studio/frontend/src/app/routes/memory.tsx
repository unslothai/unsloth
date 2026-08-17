import { FEATURE_MEMORY_NAV } from "@/config/disabled-features";
import { isPlatformMemoryEnabled } from "@/integrations/platform-backend";
import { MemoryPage } from "@/features/memory/memory-page";
import { createRoute, redirect } from "@tanstack/react-router";
import { requireAuth } from "../auth-guards";
import { Route as rootRoute } from "./__root";

export const Route = createRoute({
  getParentRoute: () => rootRoute,
  path: "/memory",
  staticData: { title: "Hafıza" },
  beforeLoad: () => {
    if (!FEATURE_MEMORY_NAV || !isPlatformMemoryEnabled()) {
      throw redirect({ to: "/chat" });
    }
    return requireAuth();
  },
  component: MemoryPage,
});
