import { FEATURE_SEARCH_NAV } from "@/config/disabled-features";
import { isPlatformSearchEnabled } from "@/integrations/platform-backend";
import { SearchPage } from "@/features/search/search-page";
import { createRoute, redirect } from "@tanstack/react-router";
import { requireAuth } from "../auth-guards";
import { Route as rootRoute } from "./__root";

export const Route = createRoute({
  getParentRoute: () => rootRoute,
  path: "/search",
  staticData: { title: "Arama" },
  beforeLoad: () => {
    if (!FEATURE_SEARCH_NAV || !isPlatformSearchEnabled()) {
      throw redirect({ to: "/chat" });
    }
    return requireAuth();
  },
  component: SearchPage,
});
