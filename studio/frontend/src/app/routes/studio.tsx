import { FEATURE_TRAIN } from "@/config/disabled-features";
import {
  createRoute,
  lazyRouteComponent,
  redirect,
} from "@tanstack/react-router";
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
  beforeLoad: () => {
    // Train is switched off (see config/disabled-features): keep the route so
    // deep links resolve, but send them to chat instead of loading the page.
    if (!FEATURE_TRAIN) throw redirect({ to: "/chat" });
    return requireAuth();
  },
  component: StudioPage,
});
