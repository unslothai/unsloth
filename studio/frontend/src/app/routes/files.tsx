import { getProductCapability } from "@/config/platform-capabilities";
import { FilesPage } from "@/features/files/files-page";
import { createRoute, redirect } from "@tanstack/react-router";
import { requireAuth } from "../auth-guards";
import { Route as rootRoute } from "./__root";

export const Route = createRoute({
  getParentRoute: () => rootRoute,
  path: "/files",
  staticData: { title: "Dosyalar" },
  beforeLoad: () => {
    if (!getProductCapability("files").available) {
      throw redirect({ to: "/chat" });
    }
    return requireAuth();
  },
  component: FilesPage,
});
