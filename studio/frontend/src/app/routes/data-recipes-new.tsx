import { createRoute } from "@tanstack/react-router";
import { lazy } from "react";
import { Route as rootRoute } from "./__root";

const RecipeStudioPage = lazy(() =>
  import("@/features/recipe-studio").then((m) => ({
    default: m.RecipeStudioPage,
  })),
);

export const Route = createRoute({
  getParentRoute: () => rootRoute,
  path: "/data-recipes/new",
  component: RecipeStudioPage,
});
