import { ConnectorOAuthCallbackPage } from "@/features/files/connector-oauth-callback-page";
import { createRoute } from "@tanstack/react-router";
import { Route as rootRoute } from "./__root";

export const Route = createRoute({
  getParentRoute: () => rootRoute,
  path: "/connector-oauth/$source/callback",
  staticData: { title: "Rag Platform", isAuthFlow: true },
  component: ConnectorOAuthCallbackPage,
});
