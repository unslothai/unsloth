import { createRouter } from "@tanstack/react-router";
import { Route as rootRoute } from "./routes/__root";
import { Route as chatRoute } from "./routes/chat";
import { Route as gridTestRoute } from "./routes/grid-test";
import { Route as homeRoute } from "./routes/home";
import { Route as onboardingRoute } from "./routes/onboarding";
import { Route as studioRoute } from "./routes/studio";

const routeTree = rootRoute.addChildren([
  homeRoute,
  onboardingRoute,
  gridTestRoute,
  studioRoute,
  chatRoute,
]);

export const router = createRouter({ routeTree });

declare module "@tanstack/react-router" {
  interface Register {
    router: typeof router;
  }
}
