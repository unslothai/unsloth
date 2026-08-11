


import { FEATURE_RECIPES } from "@/config/disabled-features";
import {
  createRoute,
  lazyRouteComponent,
  redirect,
} from "@tanstack/react-router";
import { requireAuth } from "../auth-guards";
import { Route as rootRoute } from "./__root";

const DataRecipesPage = lazyRouteComponent(
  () => import("@/features/data-recipes"),
  "DataRecipesPage",
);

export const Route = createRoute({
  getParentRoute: () => rootRoute,
  path: "/data-recipes",
  staticData: { title: "Data Recipes" },
  beforeLoad: () => {
    if (!FEATURE_RECIPES) {
      throw redirect({ to: "/chat" });
    }
    return requireAuth();
  },
  component: DataRecipesPage,
});
