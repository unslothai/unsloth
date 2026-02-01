import { Navbar } from "@/components/navbar";
import {
  Outlet,
  createRootRoute,
  useRouterState,
} from "@tanstack/react-router";
import { Suspense } from "react";
import { AppProvider } from "../provider";

export const Route = createRootRoute({
  component: RootLayout,
});

const HIDDEN_NAVBAR_ROUTES = ["/onboarding"];

function RootLayout() {
  const pathname = useRouterState({ select: (s) => s.location.pathname });
  const hideNavbar = HIDDEN_NAVBAR_ROUTES.includes(pathname);

  return (
    <AppProvider>
      {!hideNavbar && <Navbar />}
      <Suspense fallback={null}>
        <Outlet />
      </Suspense>
    </AppProvider>
  );
}
