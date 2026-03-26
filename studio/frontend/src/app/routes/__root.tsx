// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Navbar } from "@/components/navbar";
import { fetchDeviceType, usePlatformStore } from "@/config/env";
import {
  Outlet,
  createRootRoute,
  redirect,
  useRouterState,
} from "@tanstack/react-router";
import { motion } from "motion/react";
import { AppProvider } from "../provider";

const CHAT_ONLY_ALLOWED = new Set(["/", "/chat", "/login", "/signup", "/change-password"]);

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

  return (
    <AppProvider>
      {!hideNavbar && <Navbar />}
      <motion.main
        key={pathname}
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 0.12, ease: [0.165, 0.84, 0.44, 1] }}
        className="flex-1"
      >
        <Outlet />
      </motion.main>
    </AppProvider>
  );
}
