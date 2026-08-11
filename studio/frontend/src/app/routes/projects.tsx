


import { FEATURE_PROJECTS } from "@/config/disabled-features";
import {
  createRoute,
  lazyRouteComponent,
  redirect,
} from "@tanstack/react-router";
import { requireAuth } from "../auth-guards";
import { Route as rootRoute } from "./__root";

const ProjectsPage = lazyRouteComponent(
  () => import("@/features/chat/projects-page"),
  "ProjectsPage",
);

export const Route = createRoute({
  getParentRoute: () => rootRoute,
  path: "/projects",
  staticData: { title: "Projects" },
  beforeLoad: () => {
    if (!FEATURE_PROJECTS) {
      throw redirect({ to: "/chat" });
    }
    return requireAuth();
  },
  component: ProjectsPage,
});
