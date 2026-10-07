// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { ImageViewer } from "@/components/image-viewer";
import { useAppShellReadySignal } from "@/components/app-readiness";
import { AppSidebar } from "@/components/app-sidebar";
import { CommandPalette } from "@/components/command-palette";
import { Navbar } from "@/components/navbar";
import { SidebarEdgeTrigger } from "@/components/sidebar-edge-trigger";
import { SidebarInset, SidebarProvider } from "@/components/ui/sidebar";
import { fetchDeviceType, usePlatformStore } from "@/config/env";
import { videoNavHint } from "@/config/hardware-verdict";
import { ApiMonitorOverlay } from "@/features/api-monitor/api-monitor-overlay";
import {
  AUTH_SESSION_CLEARED_EVENT,
  AUTH_SESSION_STORED_EVENT,
  hasAuthToken,
  hasSettledAuthSession,
  useIsAccountOwner,
} from "@/features/auth";
import {
  ChatPage,
  type ChatSearch,
  clearNewChatDraft,
  hydrateModelDisclaimerPreference,
  openFolderAsProject,
  startLlamaCppAutoReload,
  StopRunningChatsDialog,
  useOpeningFolder,
  useChatRuntimeStore,
} from "@/features/chat";
import { useExportRuntimeLifecycle } from "@/features/export";
import { FIND_SCOPE_ATTRIBUTE, FindInPage } from "@/features/find-in-page";
import { HfTokenWarningDialog } from "@/features/hf-auth";
import { InterfaceZoom, zoomInterfaceFromMenu } from "@/features/interface-zoom";
import { bootstrapPersistedCredentials } from "@/features/credentials/bootstrap";
import { SharedRunConfigLinkHandler } from "@/features/model-picker";
import { backfillModelOverrides } from "@/features/model-picker/api/migrate-model-overrides";
import { hydratePins } from "@/lib/pins-mirror";
import { usePersonalizationSync } from "@/features/profile";
import { RemoteCodeConsentDialog } from "@/features/security";
import {
  SETTINGS_TABS,
  SettingsDialogMount,
  settingsTabVisible,
  triggerShortcut,
  useHubSourceNotice,
  useSettingsDialogStore,
  useShortcut,
  useShortcutAvailable,
} from "@/features/settings";
import { useLowDiskNotice } from "@/features/settings/hooks/use-low-disk-notice";
import { useTrainingUnloadGuard } from "@/features/training";
import { TransformersUpgradeDialog } from "@/features/transformers-upgrade";
import { LlmCompressorConsentDialog } from "@/features/export/components/llm-compressor-consent-dialog";
import { useNativePathLeasesSupported } from "@/features/native-intents";
import { useRagAvailabilityStore } from "@/features/rag";
import { useIsMobileShell } from "@/hooks/use-mobile";
import { useSidebarPin } from "@/hooks/use-sidebar-pin";
import { useTypeToActivate } from "@/hooks/use-type-to-activate";
import { type TranslationKey, useT } from "@/i18n";
import { isTauri } from "@/lib/api-base";
import { createNavigationNonce } from "@/lib/navigation-nonce";
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
  lazy,

  type ReactNode,
  Suspense,
  useEffect,
  useLayoutEffect,
  useMemo,
  useRef,

  useState,
} from "react";
import { AppProvider } from "../provider";
import { useDesktopShellReady } from "../desktop-shell-ready";
import { type HelpAction, helpActionAvailable, runHelpAction } from "@/components/help-actions";
import type { SettingsMenuAction } from "../app-menu-chords";
import { useAppMenuActions } from "../use-app-menu-actions";

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

// Retires the reload shell inside the route's Suspense, so a lazy page keeps the shell up.
function InitialReadyPage({
  children,
}: {
  children: (signalReady: () => void) => ReactNode;
}) {
  return children(useAppShellReadySignal());
}

function ReloadSnapshotReady() {
  const signalReady = useAppShellReadySignal();
  useLayoutEffect(() => {
    signalReady();
  }, [signalReady]);
  return null;
}

// reload-snapshot.js runs outside React; mirror privacy state so Temporary Chat is never serialized.
function ReloadSnapshotPrivacy() {
  const incognito = useChatRuntimeStore((state) => state.incognito);

  useLayoutEffect(() => {
    document.documentElement.toggleAttribute(
      "data-reload-snapshot-private",
      incognito,
    );
    return () => {
      document.documentElement.removeAttribute(
        "data-reload-snapshot-private",
      );
    };
  }, [incognito]);

  return null;
}

function RouteBoundary({
  children,
  readyWhenCommitted = true,
}: {
  children: ReactNode;
  readyWhenCommitted?: boolean;
}) {
  return (
    <Suspense fallback={<RouteFallback />}>
      {readyWhenCommitted && <ReloadSnapshotReady />}
      {children}
    </Suspense>
  );
}

// Mounted persistently below so an in-flight batch survives leaving the tab; lazy on first visit.
const ImagesPage = lazy(() =>
  import("@/features/images").then((m) => ({ default: m.ImagesPage })),
);

const VideoPage = lazy(() =>
  import("@/features/video").then((m) => ({ default: m.VideoPage })),
);

const AudioPage = lazy(() =>
  import("@/features/audio").then((m) => ({ default: m.AudioPage })),
);

// Enabled once the session settles: a read refused mid password change would never retry until reload.
function PersonalizationSyncMount() {
  useRouterState({ select: (s) => s.location.pathname });
  usePersonalizationSync(hasSettledAuthSession());
  return null;
}

// A full disk is not a training problem, so the warning cannot live on the
// training route: it belongs to whichever route the user happens to be on when
// space runs out. Mounted here it subscribes once for the session, and stays
// subscribed across navigation, instead of coming and going with /studio.
function LowDiskNoticeMount() {
  useLowDiskNotice();
  return null;
}

function HubSourceNoticeMount() {
  useHubSourceNotice();
  return null;
}

// Models page and picker read chat settings too, so hydration cannot wait for ChatPage.
function ChatSettingsHydrationMount() {
  const hydratePersistedSettings = useChatRuntimeStore(
    (state) => state.hydratePersistedSettings,
  );
  useEffect(() => {
    void hydratePersistedSettings();
    hydrateModelDisclaimerPreference().catch(() => undefined);
  }, [hydratePersistedSettings]);
  return null;
}


function CredentialBootstrapGate({
  active,
  children,
}: {
  active: boolean;
  children: ReactNode;
}) {
  const [ready, setReady] = useState(false);
  const runRevision = useRef(0);

  useEffect(() => {
    if (!active) {
      runRevision.current += 1;
      setReady(false);
      return;
    }
    let mounted = true;
    const reconcile = () => {
      const revision = ++runRevision.current;
      if (!hasAuthToken()) {
        setReady(false);
        return;
      }
      setReady(false);
      void bootstrapPersistedCredentials().finally(() => {
        if (
          mounted &&
          revision === runRevision.current &&
          hasAuthToken()
        ) {
          setReady(true);
        }
      });
    };

    window.addEventListener(AUTH_SESSION_CLEARED_EVENT, reconcile);
    window.addEventListener(AUTH_SESSION_STORED_EVENT, reconcile);
    reconcile();
    return () => {
      mounted = false;
      runRevision.current += 1;
      window.removeEventListener(AUTH_SESSION_CLEARED_EVENT, reconcile);
      window.removeEventListener(AUTH_SESSION_STORED_EVENT, reconcile);
    };
  }, [active]);
  useEffect(() => {
    if (active && ready) return startLlamaCppAutoReload();
  }, [active, ready]);

  return (
    <>
      <SettingsDialogMount active={active && ready} />
      {active && !ready ? <RouteFallback /> : children}
    </>
  );
}

const CHAT_ONLY_ALLOWED = new Set([
  "/",
  "/chat",
  "/projects",
  "/library",
  "/hub",
  "/login",
  "/signup",
  "/change-password",
  // Reachable on chat-only hosts so the page shows its own reason; it self-gates.
  "/export",
  // Must stay reachable on chat-only hosts: the overlay "Expand" and Settings API card link here.
  "/api-monitor",
]);

// Paths that render their own "still checking" state and self-gate once the verdict lands.
// The redirect below is one-way, so acting on the pre-measurement guess strands a healthy host
// on /chat; these two wait it out instead. Everything else keeps the old behaviour.
// /video is allowed outright below, so this is in practice what keeps /studio off the guess. It
// stays listed so that admission is the only thing /video depends on, not both.
const SELF_GATED_WHILE_UNKNOWN = ["/studio", "/video"];

function waitsOutUnknownVerdict(pathname: string): boolean {
  return SELF_GATED_WHILE_UNKNOWN.some(
    (base) => pathname === base || pathname.startsWith(`${base}/`),
  );
}

function isChatOnlyAllowed(pathname: string): boolean {
  if (CHAT_ONLY_ALLOWED.has(pathname)) return true;
  if (pathname === "/data-recipes" || pathname.startsWith("/data-recipes/"))
    return true;
  // Images runs on CPU/MPS via sd.cpp; chat-only is about training/export.
  if (pathname === "/images" || pathname.startsWith("/images/")) return true;
  if (pathname === "/audio" || pathname.startsWith("/audio/")) return true;
  // Video self-gates on the backend's video verdict, and works on chat-only Apple Silicon.
  if (pathname === "/video" || pathname.startsWith("/video/")) return true;
  return false;
}

export const Route = createRootRoute({
  beforeLoad: async ({ location }) => {
    await fetchDeviceType();
    const { isChatOnly, capabilitiesUnknown } = usePlatformStore.getState();
    const unmeasured = capabilitiesUnknown();
    if (
      isChatOnly() &&
      !isChatOnlyAllowed(location.pathname) &&
      !(unmeasured && waitsOutUnknownVerdict(location.pathname))
    ) {
      throw redirect({ to: "/chat" });
    }
  },
  component: RootLayout,
});

const HIDDEN_NAVBAR_ROUTES = ["/login", "/change-password"];

const DEFAULT_DOCUMENT_TITLE = "Unsloth";

function RootLayout() {
  const t = useT();
  const pathname = useRouterState({ select: (s) => s.location.pathname });
  const hideNavbar = HIDDEN_NAVBAR_ROUTES.includes(pathname);
  const routeOwnsReloadReadiness =
    pathname === "/hub" ||
    pathname === "/projects" ||
    pathname === "/export" ||
    pathname === "/studio" ||
    pathname === "/api-monitor" ||
    pathname === "/login" ||
    pathname === "/change-password" ||
    pathname === "/data-recipes" ||
    pathname.startsWith("/data-recipes/");
  const isAuthFlowRoute = useMatches({
    select: (matches) => matches.some((match) => match.staticData.isAuthFlow),
  });
  const chatOnlyMeasured = usePlatformStore(
    (s) => s.isChatOnly() && !s.capabilitiesUnknown(),
  );
  const chatOnlyReason = usePlatformStore((s) => s.chatOnlyReason);
  const videoDisabled =
    videoNavHint(chatOnlyMeasured, chatOnlyReason) !== undefined;
  // Exact match: a prefix would treat /chatty as chat.
  const isChatRoute = pathname === "/chat";
  const { pinned, setPinned, togglePinned } = useSidebarPin();
  const navigate = useNavigate();

  // ChatPage is mounted persistently so generation survives leaving the tab; search frozen off-route.
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
  // Freeze the last /chat search and latch "mounted" via render-phase setState (React's "adjust
  // state during render" pattern), avoiding effects/refs. Empty until /chat is visited:
  // location.search is the raw URL's, not the matched route's, so seeding it would let another
  // route's ?project= stand in for a chat the user has never opened. The adjustment below fills it
  // on the first /chat render, so landing straight on /chat loses nothing.
  const [frozenChatSearch, setFrozenChatSearch] = useState<ChatSearch>({});
  const [chatMounted, setChatMounted] = useState(isChatRoute);
  if (isChatRoute && frozenChatSearch !== liveChatSearch) {
    setFrozenChatSearch(liveChatSearch);
  }
  if (isChatRoute && !chatMounted) {
    setChatMounted(true);
  }
  const chatSearch = isChatRoute ? liveChatSearch : frozenChatSearch;
  const shouldMountChat = isChatRoute || chatMounted;

  // `active` lags the matches by a render, so ImagesPage reads ?model= from its own match.
  const isImagesRoute = pathname === "/images";
  const [imagesMounted, setImagesMounted] = useState(isImagesRoute);
  if (isImagesRoute && !imagesMounted) {
    setImagesMounted(true);
  }
  const shouldMountImages = isImagesRoute || imagesMounted;

  const isVideoRoute = pathname === "/video";
  const [videoMounted, setVideoMounted] = useState(isVideoRoute);
  if (isVideoRoute && !videoMounted) {
    setVideoMounted(true);
  }
  const shouldMountVideo = isVideoRoute || videoMounted;

  const isAudioRoute = pathname === "/audio";
  const isLibraryRoute = pathname === "/library";
  const [audioMounted, setAudioMounted] = useState(isAudioRoute);
  if (isAudioRoute && !audioMounted) {
    setAudioMounted(true);
  }
  const shouldMountAudio = isAudioRoute || audioMounted;
  // All four pages render their own full-height shell: no outer inset or scroll.
  const isChatLike = isChatRoute || isImagesRoute || isVideoRoute || isAudioRoute;
  // Uses Navbar's hook, not the md breakpoint: a narrowed desktop window keeps the desktop navbar.
  const nonChatTopInset = useIsMobileShell()
    ? "pt-14"
    : "pt-[calc(var(--studio-non-chat-content-top-inset,var(--studio-content-top-inset,0px))-var(--studio-non-chat-scroller-top,0px))] [--studio-titlebar-height:var(--studio-non-chat-content-top-inset,var(--studio-content-top-inset,0px))]";

  useTrainingUnloadGuard();
  useTypeToActivate();
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

  // Settings predating the server override map live only locally; backfill once after auth.
  useEffect(() => {
    if (isAuthFlowRoute) {
      return;
    }
    void backfillModelOverrides();
    void hydratePins();
  }, [isAuthFlowRoute]);

  useEffect(() => {
    if (isAuthFlowRoute) {
      useSettingsDialogStore.getState().closeDialog();
    }
  }, [isAuthFlowRoute]);

  useShortcut(
    "openSettings",
    () => useSettingsDialogStore.getState().openDialog(),
    { enabled: !isAuthFlowRoute },
  );
  useShortcut(
    "openKeyboardShortcuts",
    () =>
      useSettingsDialogStore.getState().openDialog("keyboard-shortcuts"),
    { enabled: !isAuthFlowRoute },
  );
  const startNewChat = (options?: {
    incognito?: boolean;
    standalone?: boolean;
  }) => {
    clearNewChatDraft();
    const chatRuntime = useChatRuntimeStore.getState();
    // The project on screen, which on Chat is the runtime's. The page keeps that in step with the
    // route, the inferred ones included: a thread or a compare pair opened without ?project= still
    // belongs to its project, and the page's own New chat button starts the next chat there.
    // Reading the search param instead would leave that project without being asked to. Off Chat
    // the page is hidden rather than unmounted, so its project is one the user cannot see and a new
    // chat belongs to none.
    const openProjectId = isChatRoute ? chatRuntime.activeProjectId : null;
    const projectId = options?.standalone ? null : openProjectId;
    chatRuntime.setActiveThreadId(null);
    chatRuntime.setActiveProjectId(projectId);
    chatRuntime.setIncognito(Boolean(options?.incognito));
    void navigate({
      to: "/chat",
      search: projectId ? { project: projectId } : { new: createNavigationNonce() },
    });
  };

  // Gated like the workspace chords below: /login has no shell, and /chat
  // bounces straight back off requireAuth.
  const routeShortcutEnabled = !isAuthFlowRoute && !settingsDialogOpen;
  useShortcut("newChat", () => startNewChat(), {
    enabled: routeShortcutEnabled,
  });
  useShortcut(
    "newTemporaryChat",
    () => startNewChat({ incognito: true, standalone: true }),
    { enabled: routeShortcutEnabled },
  );
  useShortcut("newStandaloneChat", () => startNewChat({ standalone: true }), {
    enabled: routeShortcutEnabled,
  });

  const pathLeasesSupported = useNativePathLeasesSupported();
  const ragUnavailable = useRagAvailabilityStore((s) => s.isUnavailable());
  const openingFolder = useOpeningFolder();
  const desktopShellReady = useDesktopShellReady();
  const sidebarMounted = useShortcutAvailable("toggleSidebar", isTauri);
  const findMounted = useShortcutAvailable("findInPage", isTauri);
  const previousChatMounted = useShortcutAvailable("previousChat", isTauri);
  const nextChatMounted = useShortcutAvailable("nextChat", isTauri);
  const viaShortcut = (id: Parameters<typeof triggerShortcut>[0], mounted: boolean) =>
    mounted ? () => void triggerShortcut(id) : null;
  const isOwner = useIsAccountOwner();
  const helpAction = (action: HelpAction) =>
    isAuthFlowRoute || !helpActionAvailable(action, isOwner) ? null : () => runHelpAction(action);
  const goTo = (to: string) => () => void navigate({ to });
  const goAction = (enabled: boolean, go: () => void) => (enabled ? go : null);
  // Go > Settings: the pages this account can open.
  const settingsActions = Object.fromEntries(
    SETTINGS_TABS.map((tab) => [
      `settings-${tab}`,
      !isAuthFlowRoute && settingsTabVisible(tab, isOwner)
        ? () => useSettingsDialogStore.getState().openDialog(tab)
        : null,
    ]),
  ) as Record<SettingsMenuAction, (() => void) | null>;
  useAppMenuActions({
    "new-chat": routeShortcutEnabled ? () => startNewChat() : null,
    "new-temporary-chat": routeShortcutEnabled
      ? () => startNewChat({ incognito: true, standalone: true })
      : null,
    "open-folder":
      routeShortcutEnabled && pathLeasesSupported && !ragUnavailable && !openingFolder
        ? () =>
            void openFolderAsProject().then((project) => {
              if (!project) return;
              const chatRuntime = useChatRuntimeStore.getState();
              chatRuntime.setActiveThreadId(null);
              chatRuntime.setActiveProjectId(project.id);
              void navigate({ to: "/chat", search: { project: project.id } });
            })
        : null,
    "toggle-sidebar": viaShortcut("toggleSidebar", sidebarMounted),
    "find": viaShortcut("findInPage", findMounted),
    "previous-chat": viaShortcut("previousChat", previousChatMounted),
    "next-chat": viaShortcut("nextChat", nextChatMounted),
    "back": routeShortcutEnabled ? () => window.history.back() : null,
    "forward": routeShortcutEnabled ? () => window.history.forward() : null,
    "zoom-in": () => zoomInterfaceFromMenu(1),
    "zoom-out": () => zoomInterfaceFromMenu(-1),
    "actual-size": () => zoomInterfaceFromMenu(0),
    "help-documentation": helpAction("help-documentation"),
    "help-keyboard-shortcuts": helpAction("help-keyboard-shortcuts"),
    "help-whats-new": helpAction("help-whats-new"),
    "help-troubleshooting": helpAction("help-troubleshooting"),
    "help-system-status": helpAction("help-system-status"),
    "help-send-feedback": helpAction("help-send-feedback"),
    "go-chat": goAction(
      routeShortcutEnabled,
      () => void navigate({ to: "/chat", search: chatSearch }),
    ),
    "go-projects": goAction(routeShortcutEnabled, goTo("/projects")),
    "go-library": goAction(routeShortcutEnabled, goTo("/library")),
    "go-hub": goAction(routeShortcutEnabled, goTo("/hub")),
    "go-train": goAction(routeShortcutEnabled && !chatOnlyMeasured, goTo("/studio")),
    "go-recipes": goAction(routeShortcutEnabled, goTo("/data-recipes")),
    "go-images": goAction(routeShortcutEnabled, goTo("/images")),
    "go-video": goAction(routeShortcutEnabled && !videoDisabled, goTo("/video")),
    "go-audio": goAction(routeShortcutEnabled, goTo("/audio")),
    "go-export": goAction(routeShortcutEnabled, goTo("/export")),
    ...settingsActions,
  }, desktopShellReady);

  // Carry the frozen search back: a bare /chat is a fresh chat and would drop the thread.
  useShortcut(
    "switchToChat",
    () => void navigate({ to: "/chat", search: chatSearch }),
    { enabled: routeShortcutEnabled },
  );
  useShortcut("switchToProjects", goTo("/projects"), {
    enabled: routeShortcutEnabled,
  });
  useShortcut("switchToHub", goTo("/hub"), {
    enabled: routeShortcutEnabled,
  });
  // Train's chord waits for a measured verdict, or it would bounce off /studio to /chat.
  useShortcut("switchToTrain", goTo("/studio"), {
    enabled: routeShortcutEnabled && !chatOnlyMeasured,
  });
  useShortcut("switchToRecipes", goTo("/data-recipes"), {
    enabled: routeShortcutEnabled,
  });
  useShortcut("switchToImages", goTo("/images"), {
    enabled: routeShortcutEnabled,
  });
  // /video only checks auth, so gate the chord on its own measured predicate.
  useShortcut("switchToVideo", goTo("/video"), {
    enabled: routeShortcutEnabled && !videoDisabled,
  });
  useShortcut("switchToAudio", goTo("/audio"), {
    enabled: routeShortcutEnabled,
  });
  useShortcut("switchToExport", goTo("/export"), {
    enabled: routeShortcutEnabled,
  });

  useEffect(() => {
    if (isChatRoute) return;
    const chatRuntime = useChatRuntimeStore.getState();
    // Clearing the thread id mid-generation remounts the provider and cancels the stream.
    const anyRunning = Object.values(chatRuntime.runningByThreadId).some(
      Boolean,
    );
    if (anyRunning) return;
    chatRuntime.setActiveProjectId(null);
    chatRuntime.setActiveThreadId(null);
    chatRuntime.setIncognito(false);
  }, [isChatRoute]);

  const content = (
    <>
      <PersonalizationSyncMount />
      <InterfaceZoom />
      <ReloadSnapshotPrivacy />
      {!isAuthFlowRoute && <ChatSettingsHydrationMount />}
      {!isAuthFlowRoute && <LowDiskNoticeMount />}
      {!isAuthFlowRoute && <ApiMonitorOverlay />}
      <HfTokenWarningDialog />
      <RemoteCodeConsentDialog />
      <TransformersUpgradeDialog />
      <LlmCompressorConsentDialog />
      {/* At the root, not under /chat: a swap can start from the Hub too. */}
      <StopRunningChatsDialog />
      <ImageViewer />
      {!hideNavbar && <CommandPalette />}
      {hideNavbar ? (
        <main className="flex-1 pt-[var(--studio-hidden-route-top-inset,0px)] [--studio-titlebar-height:var(--studio-hidden-route-top-inset,0px)]">
          <RouteBoundary readyWhenCommitted={!routeOwnsReloadReadiness}>
            <Outlet />
          </RouteBoundary>
        </main>
      ) : (
        <SidebarProvider
          pinned={pinned}
          setPinned={setPinned}
          togglePinned={togglePinned}
          className="!min-h-0 h-[calc(100dvh-var(--studio-titlebar-height,0px))] overflow-hidden"
        >
          <AppSidebar />
          <SidebarEdgeTrigger />
          <SidebarInset
            className={
              isChatLike
                ? "overflow-hidden"
                :
                  isLibraryRoute
                  ? "mt-[var(--studio-non-chat-scroller-top,0px)] overflow-y-auto [scrollbar-gutter:stable]"
                  : "mt-[var(--studio-non-chat-scroller-top,0px)] overflow-y-auto"
            }
          >
            <Navbar />
            <div
              {...{ [FIND_SCOPE_ATTRIBUTE]: "" }}
              className={`relative flex min-h-0 min-w-0 flex-1 basis-0 flex-col ${isChatLike ? "overflow-hidden" : "overflow-visible"} ${isChatLike ? "" : nonChatTopInset}`}
            >
              {/* The find bar floats over this region and searches it: the workspace on screen,
                  without the sidebar, the navbar, or the off-route workspaces parked here under
                  `inert`. Gated off behind a modal, which owns Escape while it is up. */}
              <FindInPage enabled={routeShortcutEnabled} />
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
              {/* `active` force-closes body-portaled overlays so none bleed over another tab while hidden. */}
              {shouldMountImages && (
                <div
                  className={
                    isImagesRoute
                      ? "flex min-h-0 min-w-0 flex-1 basis-0 flex-col overflow-hidden"
                      : "hidden"
                  }
                  inert={!isImagesRoute || undefined}
                >
                  <Suspense fallback={<RouteFallback />}>
                    <InitialReadyPage>
                      {(signalReady) => (
                        <ImagesPage active={isImagesRoute} onInitialReady={signalReady} />
                      )}
                    </InitialReadyPage>
                  </Suspense>
                </div>
              )}
              {shouldMountVideo && (
                <div
                  className={
                    isVideoRoute
                      ? "flex min-h-0 min-w-0 flex-1 basis-0 flex-col overflow-hidden"
                      : "hidden"
                  }
                  inert={!isVideoRoute || undefined}
                >
                  <Suspense fallback={<RouteFallback />}>
                    <InitialReadyPage>
                      {(signalReady) => (
                        <VideoPage active={isVideoRoute} onInitialReady={signalReady} />
                      )}
                    </InitialReadyPage>
                  </Suspense>
                </div>
              )}
              {shouldMountAudio && (
                <div
                  className={
                    isAudioRoute
                      ? "flex min-h-0 min-w-0 flex-1 basis-0 flex-col overflow-hidden"
                      : "hidden"
                  }
                  inert={!isAudioRoute || undefined}
                >
                  <Suspense fallback={<RouteFallback />}>
                    <InitialReadyPage>
                      {(signalReady) => (
                        <AudioPage active={isAudioRoute} onInitialReady={signalReady} />
                      )}
                    </InitialReadyPage>
                  </Suspense>
                </div>
              )}
              {/* Use mode="popLayout" instead of "wait" to prevent UI freezes when
                  switching from heavy pages (like Export with many checkpoints).
                  "popLayout" allows the new route to mount immediately while the
                  old one animates out, avoiding blocking on expensive exit renders.
                  See issue #5850. */}
              {!isChatRoute && !isImagesRoute && !isVideoRoute && !isAudioRoute && (
                <AnimatePresence initial={false} mode="popLayout">
                  <motion.div
                    key={pathname}
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                    transition={{ duration: 0.06 }}
                    className="flex min-h-0 min-w-0 flex-1 basis-0 flex-col overflow-visible"
                  >
                    <RouteBoundary readyWhenCommitted={!routeOwnsReloadReadiness}>
                      <Outlet />
                    </RouteBoundary>
                  </motion.div>
                </AnimatePresence>
              )}
            </div>
          </SidebarInset>
        </SidebarProvider>
      )}
      {/* This side-effect-only mount stays last so it cannot shift existing React useId paths. */}
      {!isAuthFlowRoute && <HubSourceNoticeMount />}
    </>
  );

  return (
    <AppProvider>
      <CredentialBootstrapGate active={!isAuthFlowRoute}>
        <SharedRunConfigLinkHandler
          chatSearch={shouldMountChat ? chatSearch : null}
        />
        {content}
      </CredentialBootstrapGate>
    </AppProvider>
  );
}
