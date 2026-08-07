import { FEATURE_IMAGES } from "@/config/disabled-features";
import { createRoute, redirect } from "@tanstack/react-router";
import { requireAuth } from "../auth-guards";
import { Route as rootRoute } from "./__root";

// RootLayout renders ImagesPage persistently (so an in-flight batch is not cancelled when leaving the tab); this route only owns the URL + auth gate.
export const Route = createRoute({
  getParentRoute: () => rootRoute,
  path: "/images",
  staticData: { title: "Images" },
  // A diffusion pick made from the chat picker arrives here as ?model= (+ ?quant=), which the page loads and then clears.
  validateSearch: (
    search: Record<string, unknown>,
  ): { model?: string; quant?: string } => ({
    ...(typeof search.model === "string" ? { model: search.model } : {}),
    ...(typeof search.quant === "string" ? { quant: search.quant } : {}),
  }),
  beforeLoad: () => {
    // Images is switched off (see config/disabled-features): keep the route so
    // deep links resolve, but send them to chat instead of an empty shell.
    if (!FEATURE_IMAGES) throw redirect({ to: "/chat" });
    return requireAuth();
  },
  component: () => null,
});
