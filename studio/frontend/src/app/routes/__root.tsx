// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { AppSidebar } from "@/components/app-sidebar";
import { Navbar } from "@/components/navbar";
import { fetchDeviceType, usePlatformStore } from "@/config/env";
import { SidebarInset, SidebarProvider } from "@/components/ui/sidebar";
import { SettingsDialog, useSettingsDialogStore } from "@/features/settings";
import { useTrainingUnloadGuard } from "@/features/training/hooks/use-training-unload-guard";
import { useSidebarPin } from "@/hooks/use-sidebar-pin";
import {
  Outlet,
  createRootRoute,
  redirect,
  useMatches,
  useRouterState,
} from "@tanstack/react-router";
import { AnimatePresence, motion } from "motion/react";
import { Suspense, useEffect, useLayoutEffect, type ReactNode } from "react";
import { AppProvider } from "../provider";

// Augment TanStack Router's empty `StaticDataRouteOption` interface so
// `createRoute({ staticData: { title: "..." } })` is typed at every leaf
// route and the layout below can read `match.staticData.title` without
// an `as { staticData?: { title?: string } }` escape hatch.
declare module "@tanstack/react-router" {
  interface StaticDataRouteOption {
    title?: string;
  }
}

// Fallback while a lazy route bundle (Train/Recipes/Export) loads.
// /chat is synchronous and never hits this.
const RouteFallback: ReactNode = (
  <div className="flex h-full min-h-0 flex-1 items-center justify-center text-muted-foreground text-sm">
    Loading...
  </div>
);

const CHAT_ONLY_ALLOWED = new Set([
  "/",
  "/chat",
  "/login",
  "/signup",
  "/change-password",
]);

function isChatOnlyAllowed(pathname: string): boolean {
  if (CHAT_ONLY_ALLOWED.has(pathname)) return true;
  if (pathname === "/data-recipes" || pathname.startsWith("/data-recipes/")) return true;
  return false;
}

export const Route = createRootRoute({
  beforeLoad: async ({ location }) => {
    // Ensure platform info is fetched before checking chat-only guard.
    // fetchDeviceType caches after first call, so subsequent navigations are instant.
    await fetchDeviceType();
    const chatOnly = usePlatformStore.getState().isChatOnly();
    if (chatOnly && !isChatOnlyAllowed(location.pathname)) {
      throw redirect({ to: "/chat" });
    }
  },
  component: RootLayout,
});

const HIDDEN_NAVBAR_ROUTES = ["/onboarding", "/login", "/change-password"];

// Title shown when no matched route declares its own `staticData.title`
// (e.g. the bare `/` redirect or an unknown path).
const DEFAULT_DOCUMENT_TITLE = "Unsloth Studio";

function RootLayout() {
  const pathname = useRouterState({ select: (s) => s.location.pathname });
  const hideNavbar = HIDDEN_NAVBAR_ROUTES.includes(pathname);
  const isChatRoute = pathname.startsWith("/chat");
  const { pinned, setPinned, togglePinned } = useSidebarPin();

  useTrainingUnloadGuard();

  // Read the deepest matched route's `staticData.title` instead of carrying
  // a centralized pathname -> title map in this file. Each route declares
  // its own title in its `createRoute({ staticData: { title: "..." } })`
  // so the layout stays decoupled from individual route names.
  const matchedTitle = useMatches({
    select: (matches) => {
      for (let i = matches.length - 1; i >= 0; i--) {
        const title = matches[i].staticData.title;
        if (title) return title;
      }
      return null;
    },
  });

  // `/settings` is a modal deep link: its route throws `redirect` in
  // `beforeLoad`, so by the time `useMatches` resolves we're on the
  // post-auth route (usually `/chat`). Prefer "Settings" while the
  // dialog is open so the tab title matches what the user sees.
  const settingsDialogOpen = useSettingsDialogStore((s) => s.open);
  const documentTitle = settingsDialogOpen ? "Settings" : matchedTitle;

  // `useLayoutEffect` so the tab title is updated synchronously before the
  // browser paints the new route. Using `useEffect` here let the previous
  // route's title flash for one frame during in-app navigation.
  useLayoutEffect(() => {
    document.title = documentTitle
      ? `${documentTitle} - ${DEFAULT_DOCUMENT_TITLE}`
      : DEFAULT_DOCUMENT_TITLE;
  }, [documentTitle]);

  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.defaultPrevented) return;
      if ((e.metaKey || e.ctrlKey) && e.key === ",") {
        e.preventDefault();
        useSettingsDialogStore.getState().openDialog();
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, []);

  return (
    <AppProvider>
      <SettingsDialog />
      {hideNavbar ? (
        <main className="flex-1">
          <Suspense fallback={RouteFallback}>
            <Outlet />
          </Suspense>
        </main>
      ) : (
        <SidebarProvider
          pinned={pinned}
          setPinned={setPinned}
          togglePinned={togglePinned}
          className="!min-h-0 h-[calc(100dvh-var(--studio-titlebar-height,0px))] overflow-hidden"
        >
          <AppSidebar />
          <SidebarInset className={isChatRoute ? "overflow-hidden" : "overflow-y-auto"}>
            <Navbar />
            <div
              className={`flex min-h-0 min-w-0 flex-1 basis-0 flex-col ${isChatRoute ? "overflow-hidden" : "overflow-visible"} ${isChatRoute ? "" : "pt-14 md:pt-0"}`}
            >
              <AnimatePresence initial={false} mode="wait">
                <motion.div
                  key={pathname}
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                  transition={{ duration: 0.15 }}
                  className={`flex min-h-0 min-w-0 flex-1 basis-0 flex-col ${isChatRoute ? "overflow-hidden" : "overflow-visible"}`}
                >
                  <Suspense fallback={RouteFallback}>
                    <Outlet />
                  </Suspense>
                </motion.div>
              </AnimatePresence>
            </div>
          </SidebarInset>
        </SidebarProvider>
      )}
    </AppProvider>
  );
}
