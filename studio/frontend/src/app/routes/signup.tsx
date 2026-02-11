import { createRoute } from "@tanstack/react-router";
import { lazy } from "react";
import { Route as rootRoute } from "./__root";

const SignupPage = lazy(() =>
  import("@/features/auth").then((m) => ({
    default: m.SignupPage,
  })),
);

export const Route = createRoute({
  getParentRoute: () => rootRoute,
  path: "/signup",
  component: SignupPage,
});
