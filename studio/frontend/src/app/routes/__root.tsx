// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { AppSidebar } from "@/components/app-sidebar";
import { Navbar } from "@/components/navbar";
import { fetchDeviceType, usePlatformStore } from "@/config/env";
import { SidebarInset, SidebarProvider } from "@/components/ui/sidebar";
import {
  SettingsDialog,
  useSettingsDialogStore,
} from "@/features/settings";
import {
  ChatPage,
  clearNewChatDraft,
  useChatRuntimeStore,
  type ChatSearch,
} from "@/features/chat";
import { RemoteCodeConsentDialog } from "@/features/security";
import { TransformersUpgradeDialog } from "@/features/transformers-upgrade";
import { useTrainingUnloadGuard } from "@/features/training";
import { useExportRuntimeLifecycle } from "@/features/export";
import { hasAuthToken } from "@/features/auth";
import { usePersonalizationSync } from "@/features/profile";
import { useSidebarPin } from "@/hooks/use-sidebar-pin";
import { useT, type TranslationKey } from "@/i18n";
import {
  Outlet,
  createRootRoute,
  redirect,
  useMatches,
  useNavigate,
  useRouterState,
} from "@tanstack/react-router";
import { AnimatePresence, motion } from "motion/react";
import {
  Suspense,
  useEffect,
  useLayoutEffect,
  useMemo,
  useState,
} from "react";
import { AppProvider } from "../provider";

declare module "@tanstack/react-router" {
  interface StaticDataRouteOption {
    title?: string;
    titleKey?: TranslationKey;
    isAuthFlow?: boolean;
  }
}

function RouteFallback() {
  const t = useT();

  return (
    <div className="flex h-full min-h-0 flex-1 items-center justify-center text-muted-foreground text-sm">
      {t("common.loading")}
    </div>
  );
}

function PersonalizationSyncMount() {
  usePersonalizationSync(hasAuthToken());
  return null;
}

const CHAT_ONLY_ALLOWED = new Set([
  "/",
  "/chat",
  "/projects",
  "/hub",
  "/login",
  "/signup",
  "/change-password",
  // Export stays reachable on chat-only hosts so the page can show its own grayed-out reason
  // instead of a silent redirect; it self-gates via export capability, so nothing runs.
  "/export",
]);

function isChatOnlyAllowed(pathname: string): boolean {
  if (CHAT_ONLY_ALLOWED.has(pathname)) return true;
  if (pathname === "/data-recipes" || pathname.startsWith("/data-recipes/")) return true;
  return false;
}

export const Route = createRootRoute({
  beforeLoad: async ({ location }) => {
    // Fetch platform info before the chat-only guard. fetchDeviceType caches,
    // so later navigations are instant.
    await fetchDeviceType();
    const chatOnly = usePlatformStore.getState().isChatOnly();
    if (chatOnly && !isChatOnlyAllowed(location.pathname)) {
      throw redirect({ to: "/chat" });
    }
  },
  component: RootLayout,
});

const HIDDEN_NAVBAR_ROUTES = ["/onboarding", "/login", "/change-password"];

// Fallback when no matched route declares a `staticData.title`.
const DEFAULT_DOCUMENT_TITLE = "Unsloth Studio";

function RootLayout() {
  const t = useT();
  const pathname = useRouterState({ select: (s) => s.location.pathname });
  const hideNavbar = HIDDEN_NAVBAR_ROUTES.includes(pathname);
  const isAuthFlowRoute = useMatches({
    select: (matches) => matches.some((match) => match.staticData.isAuthFlow),
  });
  // Exact match: a prefix would treat /chatty as chat, hiding its not-found UI.
  const isChatRoute = pathname === "/chat";
  const { pinned, setPinned, togglePinned } = useSidebarPin();
  const navigate = useNavigate();

  // ChatPage is mounted persistently below (not via the /chat route) so an in-flight
  // generation survives leaving the tab: it mounts lazily on first /chat visit, then
  // stays mounted, its search frozen to the last /chat value while off-route.
  const rawSearch = useRouterState({ select: (s) => s.location.search }) as
    | Record<string, unknown>
    | undefined;
  const rawThread =
    typeof rawSearch?.thread === "string" ? rawSearch.thread : undefined;
  const rawCompare =
    typeof rawSearch?.compare === "string" ? rawSearch.compare : undefined;
  const rawNew = typeof rawSearch?.new === "string" ? rawSearch.new : undefined;
  const rawProject =
    typeof rawSearch?.project === "string" ? rawSearch.project : undefined;
  const liveChatSearch = useMemo<ChatSearch>(
    () => ({
      thread: rawThread,
      compare: rawCompare,
      new: rawNew,
      project: rawProject,
    }),
    [rawThread, rawCompare, rawNew, rawProject],
  );
  // Freeze the last /chat search and latch "mounted" via render-phase setState
  // (React's "adjust state during render" pattern), avoiding effects/refs.
  const [frozenChatSearch, setFrozenChatSearch] =
    useState<ChatSearch>(liveChatSearch);
  const [chatMounted, setChatMounted] = useState(isChatRoute);
  if (isChatRoute && frozenChatSearch !== liveChatSearch) {
    setFrozenChatSearch(liveChatSearch);
  }
  if (isChatRoute && !chatMounted) {
    setChatMounted(true);
  }
  const chatSearch = isChatRoute ? liveChatSearch : frozenChatSearch;
  const shouldMountChat = isChatRoute || chatMounted;

  useTrainingUnloadGuard();
  // Global export driver: streams worker logs and tracks status from any route
  // so an export keeps running and stays visible while training / chatting.
  useExportRuntimeLifecycle();

  const matchedTitle = useMatches({
    select: (matches) => {
      for (let i = matches.length - 1; i >= 0; i--) {
        const { title, titleKey } = matches[i].staticData;
        if (titleKey) return t(titleKey);
        if (title) return title;
      }
      return null;
    },
  });

  const settingsDialogOpen = useSettingsDialogStore((s) => s.open);
  const documentTitle =
    settingsDialogOpen && !isAuthFlowRoute ? t("settings.title") : matchedTitle;

  useLayoutEffect(() => {
    document.title = documentTitle
      ? `${documentTitle} - ${DEFAULT_DOCUMENT_TITLE}`
      : DEFAULT_DOCUMENT_TITLE;
  }, [documentTitle]);

  useEffect(() => {
    if (isAuthFlowRoute) {
      useSettingsDialogStore.getState().closeDialog();
    }
    const handler = (e: KeyboardEvent) => {
      if (e.defaultPrevented) return;
      if ((e.metaKey || e.ctrlKey) && e.key === ",") {
        if (isAuthFlowRoute) return;
        e.preventDefault();
        useSettingsDialogStore.getState().openDialog();
        return;
      }
      // Cmd/Ctrl+Shift+O opens a new chat.
      if ((e.metaKey || e.ctrlKey) && e.shiftKey && e.code === "KeyO") {
        e.preventDefault();
        clearNewChatDraft(); // fresh chat starts empty, no bleed from the last one
        const chatRuntime = useChatRuntimeStore.getState();
        chatRuntime.setActiveThreadId(null);
        chatRuntime.setActiveProjectId(null);
        chatRuntime.setIncognito(false);
        void navigate({
          to: "/chat",
          search: { new: crypto.randomUUID() },
        });
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [isAuthFlowRoute, navigate]);

  useEffect(() => {
    if (isChatRoute) return;
    const chatRuntime = useChatRuntimeStore.getState();
    // A URL-less chat's provider is keyed off the active thread id; clearing it
    // mid-generation would remount and cancel the stream. Only reset when idle.
    const anyRunning = Object.values(chatRuntime.runningByThreadId).some(
      Boolean,
    );
    if (anyRunning) return;
    chatRuntime.setActiveProjectId(null);
    chatRuntime.setActiveThreadId(null);
    chatRuntime.setIncognito(false);
  }, [isChatRoute]);

  return (
    <AppProvider>
      <PersonalizationSyncMount />
      {!isAuthFlowRoute && <SettingsDialog />}
      <RemoteCodeConsentDialog />
      <TransformersUpgradeDialog />
      {hideNavbar ? (
        <main className="flex-1 pt-[var(--studio-hidden-route-top-inset,0px)] [--studio-titlebar-height:var(--studio-hidden-route-top-inset,0px)]">
          <Suspense fallback={<RouteFallback />}>
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
              className={`relative flex min-h-0 min-w-0 flex-1 basis-0 flex-col ${isChatRoute ? "overflow-hidden" : "overflow-visible"} ${isChatRoute ? "" : "pt-14 md:pt-[var(--studio-non-chat-content-top-inset,var(--studio-content-top-inset,0px))] md:[--studio-titlebar-height:var(--studio-non-chat-content-top-inset,var(--studio-content-top-inset,0px))]"}`}
            >
              {/* Stays mounted across navigation so an in-flight generation is
                  not cancelled when leaving /chat; hidden (not unmounted) off-route.
                  `active` lets ChatPage close its body-portaled surfaces (model
                  selector, settings sheet, tour) so they don't bleed over other tabs. */}
              {shouldMountChat && (
                <div
                  className={
                    isChatRoute
                      ? "flex min-h-0 min-w-0 flex-1 basis-0 flex-col overflow-hidden"
                      : "hidden"
                  }
                  inert={!isChatRoute || undefined}
                >
                  <ChatPage search={chatSearch} active={isChatRoute} />
                </div>
              )}
              {/* Use mode="popLayout" instead of "wait" to prevent UI freezes when
                  switching from heavy pages (like Export with many checkpoints).
                  "popLayout" allows the new route to mount immediately while the
                  old one animates out, avoiding blocking on expensive exit renders.
                  See issue #5850. */}
              {!isChatRoute && (
                <AnimatePresence initial={false} mode="popLayout">
                  <motion.div
                    key={pathname}
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                    transition={{ duration: 0.15 }}
                    className="flex min-h-0 min-w-0 flex-1 basis-0 flex-col overflow-visible"
                  >
                    <Suspense fallback={<RouteFallback />}>
                      <Outlet />
                    </Suspense>
                  </motion.div>
                </AnimatePresence>
              )}
            </div>
          </SidebarInset>
        </SidebarProvider>
      )}
    </AppProvider>
  );
}
