// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { AppSidebar } from "@/components/app-sidebar";
import { Navbar } from "@/components/navbar";
import { SidebarInset, SidebarProvider } from "@/components/ui/sidebar";
import { fetchDeviceType, usePlatformStore } from "@/config/env";
import { videoNavHint } from "@/config/hardware-verdict";
import { ApiMonitorOverlay } from "@/features/api-monitor/api-monitor-overlay";
import {
  AUTH_SESSION_CLEARED_EVENT,
  AUTH_SESSION_STORED_EVENT,
  hasAuthToken,
} from "@/features/auth";
import {
  ChatPage,
  type ChatSearch,
  clearNewChatDraft,
  StopRunningChatsDialog,
  useChatRuntimeStore,
} from "@/features/chat";
import { useExportRuntimeLifecycle } from "@/features/export";
import { HfTokenWarningDialog } from "@/features/hf-auth";
import { bootstrapPersistedCredentials } from "@/features/credentials/bootstrap";
import { backfillModelOverrides } from "@/features/model-picker/api/migrate-model-overrides";
import { usePersonalizationSync } from "@/features/profile";
import { RemoteCodeConsentDialog } from "@/features/security";
import {
  SettingsDialog,
  useSettingsDialogStore,
  useShortcut,
} from "@/features/settings";
import { useTrainingUnloadGuard } from "@/features/training";
import { TransformersUpgradeDialog } from "@/features/transformers-upgrade";
import { useSidebarPin } from "@/hooks/use-sidebar-pin";
import { type TranslationKey, useT } from "@/i18n";
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

// Retires the retained reload shell (public/reload-snapshot.js). It rides
// inside the route's own Suspense boundary, so a lazy page that is still
// resolving keeps the shell up instead of uncovering RouteFallback.
function signalReloadSnapshotReady() {
  window.dispatchEvent(new Event("unsloth:app-shell-ready"));
}

function ReloadSnapshotReady() {
  useLayoutEffect(() => {
    signalReloadSnapshotReady();
  }, []);
  return null;
}

// reload-snapshot.js runs outside React during pageswap. Mirror the in-memory
// privacy state onto the document so Temporary Chat is never serialized even
// briefly into sessionStorage.
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

// ImagesPage is mounted persistently below (not via the /images route) so an in-flight batch survives leaving the tab,
// mirroring ChatPage. Kept lazy so its bundle still loads only on the first /images visit.
const ImagesPage = lazy(() =>
  import("@/features/images").then((m) => ({ default: m.ImagesPage })),
);

// VideoPage gets the same persistent mount so an in-flight generation survives leaving the tab; still lazy on first /video visit.
const VideoPage = lazy(() =>
  import("@/features/video").then((m) => ({ default: m.VideoPage })),
);

// AudioPage gets the same persistent mount so an in-flight generation keeps its UI state; still lazy on first /audio visit.
const AudioPage = lazy(() =>
  import("@/features/audio").then((m) => ({ default: m.AudioPage })),
);

function PersonalizationSyncMount() {
  usePersonalizationSync(hasAuthToken());
  return null;
}

// The chat settings are the installation's, and the Models page and the model
// picker read them too, so hydration cannot wait for ChatPage to mount.
function ChatSettingsHydrationMount() {
  const hydratePersistedSettings = useChatRuntimeStore(
    (state) => state.hydratePersistedSettings,
  );
  useEffect(() => {
    void hydratePersistedSettings();
  }, [hydratePersistedSettings]);
  return null;
}


function CredentialBootstrapGate({ children }: { children: ReactNode }) {
  const [ready, setReady] = useState(false);
  const runRevision = useRef(0);

  useEffect(() => {
    let active = true;
    const reconcile = () => {
      const revision = ++runRevision.current;
      if (!hasAuthToken()) {
        setReady(false);
        return;
      }
      setReady(false);
      void bootstrapPersistedCredentials().finally(() => {
        if (
          active &&
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
      active = false;
      runRevision.current += 1;
      window.removeEventListener(AUTH_SESSION_CLEARED_EVENT, reconcile);
      window.removeEventListener(AUTH_SESSION_STORED_EVENT, reconcile);
    };
  }, []);
  return ready ? children : <RouteFallback />;
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
  // Chat-only hosts serve the API like any other, so the monitor must be reachable there
  // or the overlay's "Expand" and the Settings API card redirect to /chat.
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
  if (pathname === "/data-recipes" || pathname.startsWith("/data-recipes/")) return true;
  if (pathname == "/notebooks" || pathname.startsWith("/notebooks")) return true;
  if (pathname === "/data-recipes" || pathname.startsWith("/data-recipes/"))
    return true;
  // Images runs on CPU/MPS via the native sd.cpp engine, the very no-GPU setup it was added for. The chat-only flag is about training/export, so it must not redirect /images.
  if (pathname === "/images" || pathname.startsWith("/images/")) return true;
  // Audio inference is CPU-capable too: GGUF TTS through llama.cpp and STT through the whisper.cpp / mtmd sidecars.
  if (pathname === "/audio" || pathname.startsWith("/audio/")) return true;
  // Video follows /export: the page explains an unsupported host itself from the backend's video
  // verdict, and on Apple Silicon a chat-only host is where video works anyway. So a direct link
  // or a reload must reach VideoPage's gate, which self-gates on videoSupported.
  if (pathname === "/video" || pathname.startsWith("/video/")) return true;
  return false;
}

export const Route = createRootRoute({
  beforeLoad: async ({ location }) => {
    // Fetch platform info before the chat-only guard. fetchDeviceType caches,
    // so later navigations are instant.
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

// Fallback when no matched route declares a `staticData.title`.
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
  // Measured, not guessed: the same pair the sidebar reads to gray Train out.
  const chatOnlyMeasured = usePlatformStore(
    (s) => s.isChatOnly() && !s.capabilitiesUnknown(),
  );
  const chatOnlyReason = usePlatformStore((s) => s.chatOnlyReason);
  // Video is the other row the sidebar grays out, on the two verdicts its
  // pipelines cannot run on at all. Same hint the row reads, so the two cannot
  // disagree about which hosts they are.
  const videoDisabled =
    videoNavHint(chatOnlyMeasured, chatOnlyReason) !== undefined;
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
  // Empty until /chat is visited: location.search is the raw URL's, not the
  // matched route's, so seeding it would let another route's ?project= stand
  // in for a chat the user has never opened. The adjustment below fills it on
  // the first /chat render, so landing straight on /chat loses nothing.
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

  // Same persistent mount for /images so a long batch keeps generating off-tab. Mounts lazily on first visit, then stays
  // mounted, hidden+inert while off-route. `active` is a visibility flag only: it lags the matches by a render, so ImagesPage
  // reads ?model= from its own match instead of trusting it.
  const isImagesRoute = pathname === "/images";
  const [imagesMounted, setImagesMounted] = useState(isImagesRoute);
  if (isImagesRoute && !imagesMounted) {
    setImagesMounted(true);
  }
  const shouldMountImages = isImagesRoute || imagesMounted;

  // Same persistent mount for /video so a long generation keeps running off-tab. Mounts lazily on first visit, then stays mounted, hidden+inert while off-route.
  const isVideoRoute = pathname === "/video";
  const [videoMounted, setVideoMounted] = useState(isVideoRoute);
  if (isVideoRoute && !videoMounted) {
    setVideoMounted(true);
  }
  const shouldMountVideo = isVideoRoute || videoMounted;

  // Same persistent mount for /audio so generation UI state survives leaving the tab.
  const isAudioRoute = pathname === "/audio";
  const [audioMounted, setAudioMounted] = useState(isAudioRoute);
  if (isAudioRoute && !audioMounted) {
    setAudioMounted(true);
  }
  const shouldMountAudio = isAudioRoute || audioMounted;
  // Chat, Images, Video and Audio each render their own full-height shell, so all four want the chat-style layout: no outer pt-14 inset, no outer
  // scroll. Keying off isChatRoute alone pushed the picker down and clipped the gallery. Container padding/overflow only; keep-alive stays per route.
  const isChatLike = isChatRoute || isImagesRoute || isVideoRoute || isAudioRoute;

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

  // Settings predating the server override map live only here, so an API load would use
  // app defaults. Backfill once, after auth.
  useEffect(() => {
    if (isAuthFlowRoute) {
      return;
    }
    void backfillModelOverrides();
  }, [isAuthFlowRoute]);

  useEffect(() => {
    if (isAuthFlowRoute) {
      useSettingsDialogStore.getState().closeDialog();
    }
  }, [isAuthFlowRoute]);

  // Chords come from the shortcuts store (Settings -> Shortcuts), so a rebind
  // applies without a reload. The auth flow has no shell to act on.
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
  /** Every "new chat" chord lands here. `incognito` skips history,
   *  `standalone` leaves the open project. */
  const startNewChat = (options?: {
    incognito?: boolean;
    standalone?: boolean;
  }) => {
    clearNewChatDraft(); // fresh chat starts empty, no bleed from the last one
    const chatRuntime = useChatRuntimeStore.getState();
    // The project on screen, which on Chat is the runtime's. The page keeps
    // that in step with the route, the inferred ones included: a thread or a
    // compare pair opened without ?project= still belongs to its project, and
    // the page's own New chat button starts the next chat there. Reading the
    // search param instead would leave that project without being asked to.
    // Off Chat the page is hidden rather than unmounted, so its project is one
    // the user cannot see and a new chat belongs to none.
    const openProjectId = isChatRoute ? chatRuntime.activeProjectId : null;
    const projectId = options?.standalone ? null : openProjectId;
    chatRuntime.setActiveThreadId(null);
    chatRuntime.setActiveProjectId(projectId);
    chatRuntime.setIncognito(Boolean(options?.incognito));
    void navigate({
      to: "/chat",
      search: projectId ? { project: projectId } : { new: crypto.randomUUID() },
    });
  };

  // Gated like the workspace chords below: /login has no shell, and /chat
  // bounces straight back off requireAuth.
  useShortcut("newChat", () => startNewChat(), { enabled: !isAuthFlowRoute });
  useShortcut(
    "newTemporaryChat",
    () => startNewChat({ incognito: true, standalone: true }),
    { enabled: !isAuthFlowRoute },
  );
  useShortcut("newStandaloneChat", () => startNewChat({ standalone: true }), {
    enabled: !isAuthFlowRoute,
  });

  // Workspaces. The shell is mounted on every route, so the chords live here.
  const goTo = (to: string) => () => void navigate({ to });
  // Carry the frozen search back: a bare /chat is a fresh chat, so switching
  // away and back would drop the thread, compare pair or project.
  useShortcut(
    "switchToChat",
    () => void navigate({ to: "/chat", search: chatSearch }),
    { enabled: !isAuthFlowRoute },
  );
  useShortcut("switchToProjects", goTo("/projects"), {
    enabled: !isAuthFlowRoute,
  });
  useShortcut("switchToHub", goTo("/hub"), { enabled: !isAuthFlowRoute });
  // Train is the one workspace the chat-only guard turns away, so its chord is
  // the one that has to ask first: firing it on a host without the hardware
  // would bounce off /studio and land the user on /chat, away from whatever
  // they had open. The sidebar disables the row on the same measured check,
  // and only once measured, since the guess is what the row waits out too.
  useShortcut("switchToTrain", goTo("/studio"), {
    enabled: !isAuthFlowRoute && !chatOnlyMeasured,
  });
  useShortcut("switchToRecipes", goTo("/data-recipes"), {
    enabled: !isAuthFlowRoute,
  });
  useShortcut("switchToImages", goTo("/images"), { enabled: !isAuthFlowRoute });
  // /video checks auth and nothing else, so an ungated chord would put the
  // unsupported-hardware gate where the user's workspace was. Train's chord
  // waits on the same measurement; this one has its own predicate to wait on.
  useShortcut("switchToVideo", goTo("/video"), {
    enabled: !isAuthFlowRoute && !videoDisabled,
  });
  useShortcut("switchToAudio", goTo("/audio"), { enabled: !isAuthFlowRoute });
  useShortcut("switchToExport", goTo("/export"), { enabled: !isAuthFlowRoute });

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

  const content = (
    <>
      <PersonalizationSyncMount />
      <ReloadSnapshotPrivacy />
      {!isAuthFlowRoute && <ChatSettingsHydrationMount />}
      {!isAuthFlowRoute && <SettingsDialog />}
      {/* Opens itself when API traffic arrives; hides on the full monitor page. */}
      {!isAuthFlowRoute && <ApiMonitorOverlay />}
      <HfTokenWarningDialog />
      <RemoteCodeConsentDialog />
      <TransformersUpgradeDialog />
      {/* At the root, not under /chat: a swap can start from the Hub too. */}
      <StopRunningChatsDialog />
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
          <SidebarInset
            className={isChatLike ? "overflow-hidden" : "overflow-y-auto"}
          >
            <Navbar />
            <div
              className={`relative flex min-h-0 min-w-0 flex-1 basis-0 flex-col ${isChatLike ? "overflow-hidden" : "overflow-visible"} ${isChatLike ? "" : "pt-14 md:pt-[var(--studio-non-chat-content-top-inset,var(--studio-content-top-inset,0px))] md:[--studio-titlebar-height:var(--studio-non-chat-content-top-inset,var(--studio-content-top-inset,0px))]"}`}
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
              {/* Same keep-alive treatment for Images so a long batch keeps generating off-tab; `active` force-closes its body-portaled overlays (model selector, recipe popover, aspect dropdown) so none bleed over another tab while hidden. */}
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
                    <ImagesPage
                      active={isImagesRoute}
                      onInitialReady={signalReloadSnapshotReady}
                    />
                  </Suspense>
                </div>
              )}
              {/* Same keep-alive treatment for Video so a long generation keeps running off-tab; `active` force-closes its body-portaled overlays so none bleed over another tab while hidden. */}
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
                    <VideoPage
                      active={isVideoRoute}
                      onInitialReady={signalReloadSnapshotReady}
                    />
                  </Suspense>
                </div>
              )}
              {/* Same keep-alive treatment for Audio so generation and training UI state survive off-tab; `active` force-closes its body-portaled overlays so none bleed over another tab while hidden. */}
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
                    <AudioPage
                      active={isAudioRoute}
                      onInitialReady={signalReloadSnapshotReady}
                    />
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
    </>
  );

  return (
    <AppProvider>
      {!isAuthFlowRoute ? (
        <CredentialBootstrapGate>{content}</CredentialBootstrapGate>
      ) : (
        content
      )}
    </AppProvider>
  );
}
