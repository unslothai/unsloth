


import { createRoute, lazyRouteComponent } from "@tanstack/react-router";
import { requireAuth } from "../auth-guards";
import { Route as rootRoute } from "./__root";

const StudioPage = lazyRouteComponent(
  () => import("@/features/studio/studio-page"),
  "StudioPage",
);

export const Route = createRoute({
  getParentRoute: () => rootRoute,
  path: "/studio",
  staticData: { titleKey: "studio.routeTitle" },
  beforeLoad: () => requireAuth(),
  component: StudioPage,
});
