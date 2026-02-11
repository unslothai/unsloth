import { ComponentExample } from "@/components/component-example";
import { createRoute } from "@tanstack/react-router";
import { requireAuth } from "../auth-guards";
import { Route as rootRoute } from "./__root";

export const Route = createRoute({
  getParentRoute: () => rootRoute,
  path: "/",
  beforeLoad: () => requireAuth(),
  component: HomePage,
});

function HomePage() {
  return <ComponentExample />;
}
