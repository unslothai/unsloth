// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { AppSidebar } from "@/components/app-sidebar";
import { Navbar } from "@/components/navbar";
import { SidebarInset, SidebarProvider } from "@/components/ui/sidebar";
import { usePlatformStore } from "@/config/env";
import { useSidebarPin } from "@/hooks/use-sidebar-pin";
import {
  Outlet,
  createRootRoute,
  redirect,
  useRouterState,
} from "@tanstack/react-router";
import { AppProvider } from "../provider";

const CHAT_ONLY_ALLOWED = new Set(["/", "/chat", "/login", "/signup", "/change-password"]);

function isChatOnlyAllowed(pathname: string): boolean {
  if (CHAT_ONLY_ALLOWED.has(pathname)) return true;
  if (pathname === "/data-recipes" || pathname.startsWith("/data-recipes/")) return true;
  return false;
}

export const Route = createRootRoute({
  beforeLoad: ({ location }) => {
    const chatOnly = usePlatformStore.getState().isChatOnly();
    if (chatOnly && !isChatOnlyAllowed(location.pathname)) {
      throw redirect({ to: "/chat" });
    }
  },
  component: RootLayout,
});

const HIDDEN_NAVBAR_ROUTES = ["/onboarding", "/login", "/change-password"];

function RootLayout() {
  const pathname = useRouterState({ select: (s) => s.location.pathname });
  const hideNavbar = HIDDEN_NAVBAR_ROUTES.includes(pathname);
  const { pinned, setPinned, togglePinned, hovered, setHovered } =
    useSidebarPin();

  return (
    <AppProvider>
      {hideNavbar ? (
        <main className="flex-1">
          <Outlet />
        </main>
      ) : (
        <SidebarProvider
          pinned={pinned}
          setPinned={setPinned}
          togglePinned={togglePinned}
          hovered={hovered}
          setHovered={setHovered}
          className="!min-h-0 h-dvh overflow-hidden"
        >
          <AppSidebar />
          <SidebarInset className="overflow-hidden">
            <Navbar />
            <div className={`flex min-h-0 min-w-0 flex-1 basis-0 flex-col overflow-hidden ${pathname.startsWith("/chat") ? "" : "pt-14 md:pt-0"}`}>
              <Outlet />
            </div>
          </SidebarInset>
        </SidebarProvider>
      )}
    </AppProvider>
  );
}
