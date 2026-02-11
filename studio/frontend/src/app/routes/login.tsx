import { createRoute } from "@tanstack/react-router";
import { lazy } from "react";
import { Route as rootRoute } from "./__root";

const LoginPage = lazy(() =>
  import("@/features/auth").then((m) => ({ default: m.LoginPage })),
);

export const Route = createRoute({
  getParentRoute: () => rootRoute,
  path: "/login",
  component: LoginPage,
});
