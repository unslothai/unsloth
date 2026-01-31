import { createRoute } from "@tanstack/react-router";
import { lazy } from "react";
import { Route as rootRoute } from "./__root";

const ChatPage = lazy(() =>
  import("@/features/chat/chat-page").then((m) => ({ default: m.ChatPage })),
);

export const Route = createRoute({
  getParentRoute: () => rootRoute,
  path: "/chat",
  component: ChatPage,
});
