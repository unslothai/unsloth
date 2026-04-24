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
  useRouterState,
} from "@tanstack/react-router";
import { AnimatePresence, motion } from "motion/react";
import { Suspense, useEffect } from "react";
import { AppProvider } from "../provider";

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

function RootLayout() {
  const pathname = useRouterState({ select: (s) => s.location.pathname });
  const hideNavbar = HIDDEN_NAVBAR_ROUTES.includes(pathname);
  const isChatRoute = pathname.startsWith("/chat");
  const { pinned, setPinned, togglePinned } = useSidebarPin();

  useTrainingUnloadGuard();

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
          <Suspense fallback={null}>
            <Outlet />
          </Suspense>
        </main>
      ) : (
        <SidebarProvider
          pinned={pinned}
          setPinned={setPinned}
          togglePinned={togglePinned}
          className="!min-h-0 h-dvh overflow-hidden"
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
                  <Suspense fallback={null}>
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
