import { createRouter } from "@tanstack/react-router";
import { Route as rootRoute } from "./routes/__root";
import { Route as dataRecipesRoute } from "./routes/data-recipes";
import { Route as dataRecipeRoute } from "./routes/data-recipes.$recipeId";
import { Route as chatRoute } from "./routes/chat";
import { Route as exportRoute } from "./routes/export";
import { Route as gridTestRoute } from "./routes/grid-test";
import { Route as homeRoute } from "./routes/home";
import { Route as loginRoute } from "./routes/login";
import { Route as onboardingRoute } from "./routes/onboarding";
import { Route as signupRoute } from "./routes/signup";
import { Route as studioRoute } from "./routes/studio";

const routeTree = rootRoute.addChildren([
  homeRoute,
  onboardingRoute,
  loginRoute,
  signupRoute,
  gridTestRoute,
  studioRoute,
  chatRoute,
  exportRoute,
  dataRecipesRoute,
  dataRecipeRoute,
]);

export const router = createRouter({ routeTree });

declare module "@tanstack/react-router" {
  interface Register {
    router: typeof router;
  }
}
