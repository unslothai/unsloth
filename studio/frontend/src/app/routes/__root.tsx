// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Navbar } from "@/components/navbar";
import { VersionFooter } from "@/components/version-footer";
import { usePlatformStore } from "@/config/env";
import {
  Outlet,
  createRootRoute,
  redirect,
  useRouterState,
} from "@tanstack/react-router";
import { AnimatePresence, motion } from "motion/react";
import { Suspense } from "react";
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
const HIDDEN_FOOTER_ROUTES = ["/onboarding", "/login", "/change-password"];

function RootLayout() {
  const pathname = useRouterState({ select: (s) => s.location.pathname });
  const hideNavbar = HIDDEN_NAVBAR_ROUTES.includes(pathname);
  const hideFooter = HIDDEN_FOOTER_ROUTES.includes(pathname);

  return (
    <AppProvider>
      {!hideNavbar && <Navbar />}
      <AnimatePresence initial={false}>
        <motion.div
          key={pathname}
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.15 }}
          className="flex-1"
        >
          <Suspense fallback={null}>
            <Outlet />
          </Suspense>
        </motion.div>
      </AnimatePresence>
      {!hideFooter && <VersionFooter />}
    </AppProvider>
  );
}
