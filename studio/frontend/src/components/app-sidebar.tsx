// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  Sidebar,
  SidebarContent,
  SidebarFooter,
  SidebarGroup,
  SidebarGroupContent,
  SidebarGroupLabel,
  SidebarHeader,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
  useSidebar,
} from "@/components/ui/sidebar";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuGroup,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuShortcut,
  DropdownMenuSub,
  DropdownMenuSubContent,
  DropdownMenuSubTrigger,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Spinner } from "@/components/ui/spinner";
import { Switch } from "@/components/ui/switch";
import { useAnimatedThemeToggle } from "@/components/ui/animated-theme-toggler";
import {
  getClientPlatform,
  shouldUseCustomWindowTitlebar,
  shouldUseNativeMacWindowTitlebar,
} from "@/components/tauri/window-titlebar";
import { cn } from "@/lib/utils";
import { isTauri } from "@/lib/api-base";
import { useWebUpdateCheck } from "@/hooks/use-web-update-check";
import {
  Archive03Icon,
  ArrowRight02Icon,
  BadgeInfoIcon,
  BubbleChatIcon,
  ChefHatIcon,
  CloudIcon,
  CpuIcon,
  CursorInfo02Icon,
  DashboardCircleIcon,
  Delete02Icon,
  Download01Icon,
  DownloadSquare01Icon,
  Edit03Icon,
  FolderAddIcon,
  FolderExportIcon,
  Folder01Icon,
  Globe02Icon,
  HelpCircleIcon,
  Logout05Icon,
  Message01Icon,
  MoreVerticalIcon,
  PaintBrush02Icon,
  Search01Icon,
  PinIcon,
  PinOffIcon,
  PlusSignIcon,
  PowerIcon,
  PencilEdit02Icon,
  LayoutAlignLeftIcon,
  Notebook01Icon,
  Settings02Icon,
  Sun03Icon,
  TestTube01Icon,
  UserIcon,
  ZapIcon,
} from "@hugeicons/core-free-icons";
import {
  Tooltip,
  TooltipContent,
} from "@/components/ui/tooltip";
import { Tooltip as TooltipPrimitive } from "radix-ui";
import { HugeiconsIcon } from "@hugeicons/react";
import { ChevronDown, Moon } from "lucide-react";
import {
  Link,
  useNavigate,
  useRouter,
  useRouterState,
} from "@tanstack/react-router";
import {
  archiveChatItem,
  ChatSearchDialog,
  clearNewChatDraft,
  createChatProject,
  deleteChatProject,
  deleteChatItem,
  listStoredChatThreads,
  moveChatItemToProject,
  notifyChatHistoryUpdated,
  renameChatItem,
  renameChatProject,
  useChatRuntimeStore,
  useChatProjects,
  useChatSearchStore,
  useChatSidebarItems,
  usePinnedChatsStore,
  usePinnedProjectsStore,
  useChatPreferencesStore,
  type ProjectRecord,
  type SidebarItem,
} from "@/features/chat";
import {
  useAppearanceCustomStore,
  useSettingsDialogStore,
} from "@/features/settings";
import { useEffectiveProfile, UserAvatar } from "@/features/profile";
import { fetchDeviceType, usePlatformStore } from "@/config/env";
import { clearAuthTokens, logout } from "@/features/auth";
import { TOUR_OPEN_EVENT } from "@/features/tour";
import {
  deleteTrainingRun,
  emitTrainingRunDeleted,
  emitTrainingRunUpdated,
  getTrainingRunDisplayTitle,
  removeTrainingUnloadGuard,
  renameTrainingRun,
  useTrainingCompletionWatch,
  useTrainingHistorySidebarItems,
  useTrainingRuntimeStore,
} from "@/features/training";
import type { TrainingRunSummary } from "@/features/training";
import { useExportRuntimeStore } from "@/features/export";
import {
  Fragment,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from "react";
import { isDownloadCancelled } from "@/lib/native-files";
import { toast } from "@/lib/toast";
import { ShutdownDialog } from "@/components/shutdown-dialog";
import { translate, useT, type TranslationKey } from "@/i18n";

const EMPHASIS_MARKER = "__UNSLOTH_I18N_EMPHASIS_MARKER__";

type AppT = ReturnType<typeof useT>;

function renderEmphasizedTranslation(
  t: AppT,
  key: TranslationKey,
  emphasizedValue: string,
): ReactNode {
  const translated = t(key, { name: EMPHASIS_MARKER });
  const parts = translated.split(EMPHASIS_MARKER);
  if (parts.length === 1) return translated;

  const nodes: ReactNode[] = [];
  parts.forEach((part, index) => {
    if (part.length > 0) nodes.push(part);
    if (index < parts.length - 1) {
      nodes.push(<em key={`emphasis-${index}`}>{emphasizedValue}</em>);
    }
  });
  return nodes;
}

function getTourId(pathname: string): string | null {
  if (pathname.startsWith("/studio")) return "studio";
  if (pathname.startsWith("/export")) return "export";
  if (pathname.startsWith("/chat")) return "chat";
  return null;
}

// Optional user-menu shortcuts that jump straight to a settings tab; the id
// doubles as the settings dialog tab id.
const SETTINGS_TAB_MENU_ITEMS: Record<
  "profile" | "appearance" | "resources" | "chat" | "connections",
  { icon: typeof ZapIcon; labelKey: TranslationKey }
> = {
  profile: { icon: UserIcon, labelKey: "settings.tabs.profile" },
  appearance: { icon: PaintBrush02Icon, labelKey: "settings.tabs.appearance" },
  resources: { icon: CpuIcon, labelKey: "settings.tabs.resources" },
  chat: { icon: Message01Icon, labelKey: "settings.tabs.chat" },
  connections: { icon: CloudIcon, labelKey: "settings.tabs.connections" },
};

// TestTube01Icon's last 2 paths are interior bubbles; slice to the first
// 3 (outline + cap + liquid line) to drop them. Original export untouched.
const TestTubeOutlineIcon = TestTube01Icon.slice(
  0,
  3,
) as typeof TestTube01Icon;


type ConversationExportFormat = "raw-jsonl" | "csv" | "sharegpt-jsonl";

// A pinned project shows this many recent chats before "Show more".
const PINNED_PROJECT_CHAT_LIMIT = 4;

const CHAT_EXPORT_OPTIONS: Array<{
  label: string;
  format: ConversationExportFormat;
}> = [
  { label: "Raw JSONL", format: "raw-jsonl" },
  { label: "CSV", format: "csv" },
  { label: "ShareGPT JSONL", format: "sharegpt-jsonl" },
];

async function exportConversationByFormat(
  threadId: string,
  format: ConversationExportFormat,
): Promise<void> {
  const exports = await import(
    "@/features/chat/prompt-storage/prompt-storage-dialog"
  );
  switch (format) {
    case "raw-jsonl":
      return exports.exportConversationRawJsonl(threadId);
    case "csv":
      return exports.exportConversationCsv(threadId);
    case "sharegpt-jsonl":
      return exports.exportConversationShareGPT(threadId);
  }
}


function runStatusDotClass(status: TrainingRunSummary["status"]): string {
  switch (status) {
    case "running":
      return "bg-blue-500 animate-pulse";
    case "completed":
      return "bg-emerald-500";
    case "stopped":
      return "bg-amber-500";
    case "error":
      return "bg-red-500";
    default:
      return "bg-muted-foreground";
  }
}

function formatRelativeShort(iso: string): string {
  const then = new Date(iso).getTime();
  if (Number.isNaN(then)) return "";
  const diffMs = Date.now() - then;
  const s = Math.max(0, Math.floor(diffMs / 1000));
  if (s < 60) return `${s}s`;
  const m = Math.floor(s / 60);
  if (m < 60) return `${m}m`;
  const h = Math.floor(m / 60);
  if (h < 24) return `${h}h`;
  const d = Math.floor(h / 24);
  return `${d}d`;
}

function createNavigationNonce(): string {
  if (typeof globalThis.crypto?.randomUUID === "function") {
    return globalThis.crypto.randomUUID();
  }
  return `${Date.now()}-${Math.random().toString(36).slice(2, 10)}`;
}

function preloadSilently(request: Promise<unknown>): void {
  void request.catch(() => undefined);
}

function NavItem({
  icon,
  label,
  active,
  disabled,
  onClick,
  children,
  dataTour,
  className,
  spinner,
  tooltip,
  onIntent,
}: {
  icon: typeof ZapIcon;
  label: string;
  active: boolean;
  disabled?: boolean;
  onClick: () => void;
  children?: ReactNode;
  dataTour?: string;
  className?: string;
  spinner?: boolean;
  onIntent?: () => void;
  // Overrides the hover tooltip (defaults to `label`). Used to explain why a
  // disabled item (e.g. Train/Export on a chat-only host) is greyed out.
  tooltip?: string;
}) {
  return (
    <SidebarMenuItem className={className}>
      <div className="relative">
        <SidebarMenuButton
          tooltip={tooltip ?? label}
          disabled={disabled}
          onClick={onClick}
          onPointerEnter={disabled ? undefined : onIntent}
          onFocus={disabled ? undefined : onIntent}
          isActive={active}
          data-tour={dataTour}
          className="sidebar-nav-btn h-[33px] rounded-full gap-[8.5px] pl-3 pr-2.5 font-medium group-data-[collapsible=icon]:px-2.5 group-data-[collapsible=icon]:!w-[32px] group-data-[collapsible=icon]:mx-auto"
        >
          <HugeiconsIcon icon={icon} strokeWidth={1.75} className="size-icon! shrink-0 group-hover/menu-button:animate-icon-pop" />
          <span className="text-[14.5px] leading-[19px] tracking-nav">{label}</span>
          {spinner && (
            <Spinner className="ml-auto size-3.5 shrink-0 text-muted-foreground group-data-[collapsible=icon]:hidden" />
          )}
        </SidebarMenuButton>
        {spinner && (
          // Collapsed (icon-only) rail: small spinner badge over the icon corner.
          <Spinner className="pointer-events-none absolute right-1 top-1 hidden size-2.5 text-muted-foreground group-data-[collapsible=icon]:block" />
        )}
      </div>
      {children}
    </SidebarMenuItem>
  );
}

export function AppSidebar() {
  const t = useT();
  const { isDark, toggleTheme, anchorRef } = useAnimatedThemeToggle();
  const sidebarMenu = useAppearanceCustomStore(
    (s) => s.customization.sidebarMenu,
  );
  const [usesCustomTitlebar] = useState(shouldUseCustomWindowTitlebar);
  const [usesNativeMacTitlebar] = useState(shouldUseNativeMacWindowTitlebar);
  // Mac uses Cmd, others use Ctrl. Not Tauri-gated, so it's right on web too.
  const [isMacPlatform] = useState(() => getClientPlatform().includes("mac"));
  const { pathname, search } = useRouterState({
    select: (s) => ({
      pathname: s.location.pathname,
      search: s.location.search as Record<string, string | undefined>,
    }),
  });
  const { togglePinned, isMobile, setOpenMobile } = useSidebar();
  const navigate = useNavigate();
  const router = useRouter();

  // Web update detection: `webUpdate` is non-null only when the installed
  // (PyPI) version is behind the latest release, so the card is hidden by
  // default.
  const { status: webUpdate } = useWebUpdateCheck();
  const showUpdateCard = Boolean(webUpdate);
  const updateVersion = webUpdate?.latestVersion ?? null;

  // Auto-close mobile Sheet after navigation
  const closeMobileIfOpen = () => {
    if (isMobile) setOpenMobile(false);
  };

  const chatOnly = usePlatformStore((s) => s.isChatOnly());
  const chatOnlyReason = usePlatformStore((s) => s.chatOnlyReason);
  // Explain a greyed-out Train (chat-only host) on hover instead of disabling silently. Export is
  // no longer disabled here: it stays navigable so its page can show a precise grayed-out reason.
  const trainDisabledHint: string | undefined = !chatOnly
    ? undefined
    : chatOnlyReason === "mlx_unavailable"
      ? "Training needs MLX. Run `unsloth studio update` to enable Train."
      : chatOnlyReason === "intel_mac"
        ? "Training needs Apple Silicon or a GPU. Intel Macs are chat-only."
        : chatOnlyReason === "no_gpu"
          ? "Training needs an NVIDIA or AMD GPU."
          : undefined;

  // The backend MLX self-heal (utils/mlx_repair) can reinstall MLX in the
  // background and flip chat_only false without a restart. The platform store
  // cached the initial /api/health, so re-poll while we are chat-only for the
  // recoverable mlx_unavailable case; the effect stops once Train/Export become
  // available (chatOnly flips false and this effect's guard returns early).
  useEffect(() => {
    if (!chatOnly || chatOnlyReason !== "mlx_unavailable") return;
    const id = window.setInterval(() => {
      void fetchDeviceType({ force: true }).catch(() => undefined);
    }, 15000);
    return () => window.clearInterval(id);
  }, [chatOnly, chatOnlyReason]);

  const [shutdownOpen, setShutdownOpen] = useState(false);

  const isChatRoute = pathname.startsWith("/chat");
  const isStudioRoute = pathname === "/studio" || pathname.startsWith("/studio/");
  const [chatOpen, setChatOpen] = useState(true);

  const [trainOpen, setTrainOpen] = useState(true);
  const [runsOpen, setRunsOpen] = useState(true);

  useEffect(() => {
    if (!isChatRoute) return;
    queueMicrotask(() => setChatOpen(true));
  }, [isChatRoute]);
  useEffect(() => {
    if (!isStudioRoute) return;
    queueMicrotask(() => setRunsOpen(true));
  }, [isStudioRoute]);

  const scrollRef = useRef<HTMLDivElement | null>(null);
  const [scrolled, setScrolled] = useState(false);
  // Bottom fade hides at the very bottom / for short lists so the last row
  // isn't washed out (Gemini-style).
  const [canScrollDown, setCanScrollDown] = useState(false);
  // Driven only from onScroll + a content-change effect below. No
  // ResizeObserver: its callback-driven setState caused a render loop (React
  // #185). Both setters bail out when unchanged, so neither path can loop.
  const syncScrollState = (el: HTMLDivElement) => {
    const nextScrolled = el.scrollTop > 0;
    setScrolled((prev) => (prev === nextScrolled ? prev : nextScrolled));
    const nextCanScrollDown =
      el.scrollHeight - el.scrollTop - el.clientHeight > 1;
    setCanScrollDown((prev) =>
      prev === nextCanScrollDown ? prev : nextCanScrollDown,
    );
  };

  const isRecipesRoute = pathname.startsWith("/data-recipes");
  const isNotebooksRoute =
    pathname === "/notebooks" || pathname.startsWith("/notebooks/");
  const isExportRoute = pathname === "/export" || pathname.startsWith("/export/");
  const { displayTitle, avatarDataUrl } = useEffectiveProfile();

  const { projects } = useChatProjects();
  const activeProjectId = isChatRoute
    ? ((search.project as string | undefined) ?? null)
    : null;
  const {
    items: allChatItems,
    archivedItems: archivedChatItems,
    loaded: chatItemsLoaded,
  } = useChatSidebarItems({
    enabled: !isStudioRoute,
    requireMessages: false,
  });
  const pinnedIds = usePinnedChatsStore((s) => s.pinnedIds);
  const togglePinnedChat = usePinnedChatsStore((s) => s.togglePin);
  const unpinChat = usePinnedChatsStore((s) => s.unpin);
  const confirmDeleteChats = useChatPreferencesStore(
    (s) => s.confirmDeleteChats,
  );
  const pinnedIdSet = useMemo(() => new Set(pinnedIds), [pinnedIds]);
  const recentChatItems = useMemo(
    () =>
      allChatItems.filter(
        (item) => !item.projectId && !pinnedIdSet.has(item.id),
      ),
    [allChatItems, pinnedIdSet],
  );
  const [pinnedOpen, setPinnedOpen] = useState(true);
  // "Projects" section: projects the user pinned, in pin order (most recent
  // first). The section only appears once at least one project is pinned.
  const pinnedProjectIds = usePinnedProjectsStore((s) => s.pinnedIds);
  const unpinProject = usePinnedProjectsStore((s) => s.unpin);
  const pinnedProjectRecords = useMemo(() => {
    const byId = new Map(projects.map((p) => [p.id, p]));
    return pinnedProjectIds
      .map((id) => byId.get(id))
      .filter((p): p is ProjectRecord => Boolean(p));
  }, [projects, pinnedProjectIds]);
  // Pinned chats, in pin order (most recent first). Includes chats that live
  // inside a project: pinning promotes a chat into this list, and it is removed
  // from the project's nested list below so it never shows twice.
  const pinnedChatItems = useMemo(() => {
    const byId = new Map(allChatItems.map((item) => [item.id, item]));
    return pinnedIds
      .map((id) => byId.get(id))
      .filter((item): item is SidebarItem => Boolean(item));
  }, [allChatItems, pinnedIds]);
  // A pinned project reveals its recent chats (most recent first) nested below.
  // Pinned chats are excluded here since they render in the pinned-chats list.
  const chatsByProjectId = useMemo(() => {
    const map = new Map<string, SidebarItem[]>();
    for (const item of allChatItems) {
      if (!item.projectId) continue;
      if (pinnedIdSet.has(item.id)) continue;
      const list = map.get(item.projectId);
      if (list) list.push(item);
      else map.set(item.projectId, [item]);
    }
    for (const list of map.values())
      list.sort((a, b) => b.updatedAt - a.updatedAt);
    return map;
  }, [allChatItems, pinnedIdSet]);
  // Default expanded (not collapsed); the row toggles this. Show-more reveals
  // chats past the first PINNED_PROJECT_CHAT_LIMIT.
  const [collapsedProjectIds, setCollapsedProjectIds] = useState<Set<string>>(
    () => new Set(),
  );
  const [expandedChatProjectIds, setExpandedChatProjectIds] = useState<
    Set<string>
  >(() => new Set());
  const toggleProjectCollapsed = (id: string) =>
    setCollapsedProjectIds((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  const toggleProjectShowAll = (id: string) =>
    setExpandedChatProjectIds((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  const storeThreadId = useChatRuntimeStore((s) => s.activeThreadId);
  const setActiveThreadId = useChatRuntimeStore((s) => s.setActiveThreadId);
  const anyChatRunning = useChatRuntimeStore((s) =>
    Object.values(s.runningByThreadId).some(Boolean),
  );
  // The thread currently generating (if any), so "Return to Chat" lands on the
  // live chat rather than an empty new-chat draft left active after New Chat.
  const runningThreadId = useChatRuntimeStore((s) => {
    const entry = Object.entries(s.runningByThreadId).find(([, on]) => on);
    return entry ? entry[0] : null;
  });
  const activeThreadId = isChatRoute
    ? (search.thread as string | undefined) ??
      (search.compare as string | undefined) ??
      storeThreadId ??
      undefined
    : undefined;

  // Training runs: surfaced as sidebar "Recents" on Train, Recipes, and Export,
  // falling back to chat recents when there are no runs yet.
  const trainingRecentsRoute = isStudioRoute || isRecipesRoute || isExportRoute;
  const { items: runItems } = useTrainingHistorySidebarItems(
    !chatOnly && trainingRecentsRoute,
  );
  const showTrainingRecents =
    !chatOnly && trainingRecentsRoute && runItems.length > 0;
  const activeJobId = useTrainingRuntimeStore((s) => s.jobId);
  const currentRunViewActive = useTrainingRuntimeStore((s) => s.currentRunViewActive);
  const selectedHistoryRunId = useTrainingRuntimeStore((s) => s.selectedHistoryRunId);
  const setSelectedHistoryRunId = useTrainingRuntimeStore((s) => s.setSelectedHistoryRunId);
  // Running or starting up. Drives the Train spinner + New Chat / Return to Chat swap.
  const trainingInProgress = useTrainingRuntimeStore((s) => s.isTrainingRunning || s.isStarting);
  // Export runs in the background (parallel with training/inference); reflect it
  // on the Export nav item so it is visible from any tab.
  const exportInProgress = useExportRuntimeStore((s) => s.isExporting);
  // On any non-chat tab (Train, Export, Recipes, Projects, Hub, ...) offer a way
  // back to the live chat instead of starting a new one, whenever a chat is
  // running or its thread is still active, or a training / export is in progress.
  const showReturnToChat =
    !isChatRoute &&
    (trainingInProgress || exportInProgress || anyChatRunning || storeThreadId != null);
  // The Train-page status poll doesn't run off-route; keep state fresh so the spinner
  // clears even if a run finishes while the user is on another tab.
  useTrainingCompletionWatch();

  // Recompute bottom-fade on mount and whenever list height can change
  // (items load, sections toggle, route switch) - onScroll never fires for
  // short, non-scrolling lists. Guarded setState below can't loop.
  useEffect(() => {
    const el = scrollRef.current;
    if (!el) return;
    const next = el.scrollHeight - el.scrollTop - el.clientHeight > 1;
    setCanScrollDown((prev) => (prev === next ? prev : next));
  }, [
    recentChatItems.length,
    runItems.length,
    projects.length,
    chatOpen,
    trainOpen,
    runsOpen,
    pinnedOpen,
    isStudioRoute,
  ]);

  const chatDisabled = trainingInProgress;
  const showSidebarBrand = !usesCustomTitlebar;
  const showCompactMacBrand = showSidebarBrand && usesNativeMacTitlebar;

  function chatSearchForProject(projectId: string | null) {
    if (projectId) {
      return { project: projectId };
    }
    return {
      new: createNavigationNonce(),
    };
  }

  function openNewChat(projectId = activeProjectId) {
    clearNewChatDraft();
    setActiveThreadId(null);
    useChatRuntimeStore.getState().setActiveProjectId(projectId);
    // The normal new-chat affordance is always a regular, saved chat --
    // only the toolbar toggle starts a temporary one.
    useChatRuntimeStore.getState().setIncognito(false);
    navigate({ to: "/chat", search: chatSearchForProject(projectId) });
    closeMobileIfOpen();
  }

  function openProject(projectId: string) {
    setActiveThreadId(null);
    useChatRuntimeStore.getState().setActiveProjectId(projectId);
    navigate({ to: "/chat", search: { project: projectId } });
    closeMobileIfOpen();
  }

  async function handleDeleteThread(item: Parameters<typeof deleteChatItem>[0]) {
    await deleteChatItem(item, activeThreadId, (view) => {
      navigate({
        to: "/chat",
        search: item.projectId
          ? { project: item.projectId }
          : { new: view.newThreadNonce },
      });
    });
  }

  // Shared chat delete: same error toast and pin cleanup whether or not the
  // confirm dialog is used.
  async function deleteChatWithCleanup(item: SidebarItem) {
    try {
      await handleDeleteThread(item);
      unpinChat(item.id);
    } catch (err) {
      toast.error(translate("shell.toast.failedToDeleteChat"), {
        description: err instanceof Error ? err.message : undefined,
      });
    }
  }

  async function handleArchiveThread(item: SidebarItem) {
    try {
      await archiveChatItem(item, activeThreadId, (view) => {
        navigate({
          to: "/chat",
          search: item.projectId
            ? { project: item.projectId }
            : { new: view.newThreadNonce },
        });
      });
      const toastId = toast(
        <button
          type="button"
          onClick={() => {
            toast.dismiss(toastId);
            useSettingsDialogStore.getState().openArchivedChats();
          }}
          className="w-full cursor-pointer text-left"
        >
          You can view archived chats in Settings
        </button>,
        { closeButton: true },
      );
    } catch (err) {
      toast.error("Failed to archive chat", {
        description: err instanceof Error ? err.message : undefined,
      });
    }
  }

  type RenameTarget =
    | { kind: "chat"; item: SidebarItem; current: string }
    | { kind: "project"; project: ProjectRecord; current: string }
    | { kind: "run"; run: TrainingRunSummary; current: string };
  const [renamingTarget, setRenamingTarget] = useState<RenameTarget | null>(
    null,
  );
  const [renameDraft, setRenameDraft] = useState("");
  // Skips the inline rename input's blur-commit when Enter/Escape already handled it.
  const skipRenameBlurRef = useRef(false);
  // Optimistic title shown while the debounced sidebar refresh catches up after
  // a rename, so the old name does not flash back in.
  const [pendingRename, setPendingRename] = useState<{
    id: string;
    title: string;
  } | null>(null);
  useEffect(() => {
    if (!pendingRename) return;
    const match = allChatItems.find((i) => i.id === pendingRename.id);
    if (!match || match.title !== pendingRename.title) return;
    queueMicrotask(() => {
      setPendingRename((current) =>
        current?.id === pendingRename.id && current.title === pendingRename.title
          ? null
          : current,
      );
    });
  }, [allChatItems, pendingRename]);
  const [creatingProject, setCreatingProject] = useState(false);
  const [projectNameDraft, setProjectNameDraft] = useState("");
  const [projectCreateMoveTarget, setProjectCreateMoveTarget] =
    useState<SidebarItem | null>(null);
  const renameTrimmed = renameDraft.trim();
  const nextRunDisplayName = renameTrimmed.length > 0 ? renameTrimmed : null;
  const renameDirty =
    renamingTarget !== null &&
    (renamingTarget.kind === "chat"
      ? renameTrimmed.length > 0 && renameTrimmed !== renamingTarget.current
      : renamingTarget.kind === "project"
        ? renameTrimmed.length > 0 && renameTrimmed !== renamingTarget.current
      : renameTrimmed.length > 0
        ? renameTrimmed !== renamingTarget.current
        : renamingTarget.run.display_name != null);

  function openRenameChat(item: SidebarItem) {
    setRenameDraft(item.title);
    setRenamingTarget({ kind: "chat", item, current: item.title });
  }
  function openRenameRun(run: TrainingRunSummary) {
    const current = getTrainingRunDisplayTitle(run);
    setRenameDraft(current);
    setRenamingTarget({ kind: "run", run, current });
  }
  async function commitRename() {
    const target = renamingTarget;
    if (!target || !renameDirty) return;
    setRenamingTarget(null);
    if (target.kind === "chat") {
      setPendingRename({ id: target.item.id, title: renameTrimmed });
      try {
        await renameChatItem(target.item, renameTrimmed);
      } catch (err) {
        setPendingRename(null);
        toast.error(translate("shell.toast.failedToRenameChat"), {
          description: err instanceof Error ? err.message : undefined,
        });
      }
      return;
    }
    if (target.kind === "project") {
      try {
        await renameChatProject(target.project.id, renameTrimmed);
      } catch (err) {
        toast.error("Failed to rename project", {
          description: err instanceof Error ? err.message : undefined,
        });
      }
      return;
    }
    try {
      const updated = await renameTrainingRun(target.run.id, nextRunDisplayName);
      emitTrainingRunUpdated(updated);
    } catch (err) {
      toast.error(translate("shell.toast.failedToRenameRun"), {
        description: err instanceof Error ? err.message : undefined,
      });
    }
  }

  // Inline chat rename commits on Enter or blur, cancels on Escape.
  function handleInlineRenameKeyDown(
    event: React.KeyboardEvent<HTMLInputElement>,
  ) {
    if (event.key === "Enter") {
      event.preventDefault();
      skipRenameBlurRef.current = true;
      // Commit when changed; otherwise just close, so a no-op Enter does not
      // leave the row stuck as an input with its blur suppressed.
      if (renameDirty) void commitRename();
      else setRenamingTarget(null);
    } else if (event.key === "Escape") {
      event.preventDefault();
      skipRenameBlurRef.current = true;
      setRenamingTarget(null);
    }
  }

  function handleInlineRenameBlur() {
    if (skipRenameBlurRef.current) {
      skipRenameBlurRef.current = false;
      return;
    }
    if (renameDirty) void commitRename();
    else setRenamingTarget(null);
  }

  type DeleteTarget =
    | { kind: "chat"; item: SidebarItem }
    | { kind: "project"; project: ProjectRecord }
    | { kind: "run"; run: TrainingRunSummary };
  const [confirmingDelete, setConfirmingDelete] =
    useState<DeleteTarget | null>(null);
  const [deleteProjectFiles, setDeleteProjectFiles] = useState(false);

  async function commitDelete() {
    const target = confirmingDelete;
    if (!target) return;
    const shouldDeleteProjectFiles =
      target.kind === "project" && deleteProjectFiles;
    setConfirmingDelete(null);
    // Reset so the next project delete never inherits this checkbox.
    setDeleteProjectFiles(false);
    if (target.kind === "chat") {
      await deleteChatWithCleanup(target.item);
      return;
    }
    if (target.kind === "project") {
      try {
        await deleteChatProject(target.project.id, {
          deleteFiles: shouldDeleteProjectFiles,
        });
        // Refresh chat history so the project's reparented chats don't linger
        // as stale top-level rows.
        notifyChatHistoryUpdated();
        // activeProjectId is only the ?project= param; on a thread-only URL the
        // project is resolved from the thread into the runtime store, so check
        // that too or we strand the user on a now-deleted thread. Only redirect
        // from a chat route: the runtime store value can be stale elsewhere.
        const runtimeProjectId =
          useChatRuntimeStore.getState().activeProjectId;
        if (
          isChatRoute &&
          (activeProjectId === target.project.id ||
            runtimeProjectId === target.project.id)
        ) {
          useChatRuntimeStore.getState().setActiveProjectId(null);
          navigate({ to: "/chat", search: { new: createNavigationNonce() } });
        }
      } catch (err) {
        toast.error("Failed to delete project", {
          description: err instanceof Error ? err.message : undefined,
        });
      }
      return;
    }
    if (target.run.status === "running") {
      toast.error(t("shell.toast.cannotDeleteRunningRun"));
      return;
    }
    try {
      await deleteTrainingRun(target.run.id);
      if (selectedHistoryRunId === target.run.id) {
        setSelectedHistoryRunId(null);
      }
      emitTrainingRunDeleted(target.run.id);
    } catch (err) {
      toast.error(translate("shell.toast.failedToDeleteRun"), {
        description: err instanceof Error ? err.message : undefined,
      });
    }
  }

  async function commitCreateProject() {
    const name = projectNameDraft.trim();
    if (!name) return;
    const moveTarget = projectCreateMoveTarget;
    try {
      const project = await createChatProject(name);
      if (moveTarget) {
        await moveChatItemToProject(moveTarget, project.id);
        if (activeThreadId === moveTarget.id) {
          useChatRuntimeStore.getState().setActiveProjectId(project.id);
        }
      }
      setCreatingProject(false);
      setProjectNameDraft("");
      setProjectCreateMoveTarget(null);
      if (moveTarget) {
        return;
      } else {
        openProject(project.id);
      }
    } catch (err) {
      toast.error(moveTarget ? "Failed to create and move chat" : "Failed to create project", {
        description: err instanceof Error ? err.message : undefined,
      });
    }
  }

  async function moveChatToProject(item: SidebarItem, projectId: string | null) {
    if (item.projectId === projectId) return;
    try {
      await moveChatItemToProject(item, projectId);
      if (activeThreadId === item.id) {
        useChatRuntimeStore.getState().setActiveProjectId(projectId);
      }
    } catch (err) {
      toast.error("Failed to move chat", {
        description: err instanceof Error ? err.message : undefined,
      });
    }
  }

  function renderChatSidebarItem(
    item: SidebarItem,
    variant: "project" | "recent",
  ) {
    const isPinned = pinnedIdSet.has(item.id);
    const itemClass =
      variant === "project"
        ? "group/project-chat-item relative"
        : "group/recent-item relative";
    const actionClass =
      variant === "project"
        ? "sidebar-row-action group-hover/project-chat-item:opacity-100 group-hover/project-chat-item:pointer-events-auto focus-visible:opacity-100 focus-visible:pointer-events-auto"
        : "sidebar-row-action group-hover/recent-item:opacity-100 group-hover/recent-item:pointer-events-auto focus-visible:opacity-100 focus-visible:pointer-events-auto";
    const buttonClass = cn(
      "sidebar-nav-btn h-[33px] cursor-pointer rounded-full pr-4 text-[14.5px] leading-[19px] tracking-nav font-medium",
      // pl-3 (12px) over the content's pl-1.5 (6px) = 18px, aligning the
      // title with the nav items above.
      variant === "project" ? "pl-[39px]" : "pl-3",
      // Pinned chats carry a chat icon, so add the nav-item icon gap.
      isPinned && variant !== "project" && "gap-[8.5px]",
      variant === "project"
        ? // Room for the hover pin quick-action plus the kebab.
          "group-hover/project-chat-item:pr-14 group-has-[.sidebar-row-action[data-state=open]]/project-chat-item:pr-8"
        : isPinned
          ? // Pinned rows show an extra unpin button on hover, so reserve more room
            // (pr-8 when the menu is open keeps the unpin button clear of the title).
            "group-hover/recent-item:pr-16 group-has-[.sidebar-row-action[data-state=open]]/recent-item:pr-8"
          : // Hover room for the kebab only; title keeps one more character.
            "group-hover/recent-item:pr-6 group-has-[.sidebar-row-action[data-state=open]]/recent-item:pr-6",
    );

    const isRenamingThis =
      renamingTarget?.kind === "chat" && renamingTarget.item.id === item.id;

    // Inline rename edits the title in place as a rounded pill, no dialog.
    if (isRenamingThis) {
      return (
        <SidebarMenuItem key={item.id} className={itemClass}>
          <input
            autoFocus
            value={renameDraft}
            onChange={(event) => setRenameDraft(event.target.value)}
            onKeyDown={handleInlineRenameKeyDown}
            onBlur={handleInlineRenameBlur}
            onFocus={(event) => event.currentTarget.select()}
            maxLength={120}
            aria-label={translate("shell.dialog.renameChat.placeholder")}
            className={cn(
              // No pill or box; edit in place as plain highlighted text.
              "text-foreground h-[33px] w-full border-0 bg-transparent pr-4 text-[14.5px] leading-[19px] font-medium tracking-nav outline-none",
              variant === "project" ? "pl-[39px]" : "pl-3",
            )}
          />
        </SidebarMenuItem>
      );
    }

    return (
      <SidebarMenuItem key={item.id} className={itemClass}>
        <SidebarMenuButton
          data-testid="recent-thread"
          data-thread-type={item.type}
          data-thread-id={item.id}
          isActive={activeThreadId === item.id}
          className={buttonClass}
          onClick={() => {
            navigate({
              to: "/chat",
              search:
                item.type === "single"
                  ? {
                      thread: item.id,
                      ...(item.projectId ? { project: item.projectId } : {}),
                    }
                  : {
                      compare: item.id,
                      ...(item.projectId ? { project: item.projectId } : {}),
                    },
            });
            closeMobileIfOpen();
          }}
        >
          {isPinned && variant !== "project" && (
            <HugeiconsIcon icon={BubbleChatIcon} strokeWidth={1.75} className="size-icon! shrink-0" />
          )}
          <span className="truncate">
            {pendingRename?.id === item.id ? pendingRename.title : item.title}
          </span>
        </SidebarMenuButton>
        {variant === "project" && (
          <button
            type="button"
            onClick={(e) => {
              e.stopPropagation();
              togglePinnedChat(item.id);
            }}
            aria-label={isPinned ? "Unpin chat" : "Pin chat"}
            className="sidebar-row-action is-unpin-action group-hover/project-chat-item:opacity-100 group-hover/project-chat-item:pointer-events-auto focus-visible:opacity-100 focus-visible:pointer-events-auto"
          >
            <span className="sidebar-row-action-glyph">
              <HugeiconsIcon icon={isPinned ? PinOffIcon : PinIcon} strokeWidth={1.75} className="size-icon" />
            </span>
          </button>
        )}
        {variant === "recent" && isPinned && (
          <button
            type="button"
            onClick={(e) => {
              e.stopPropagation();
              togglePinnedChat(item.id);
            }}
            aria-label="Unpin chat"
            className="sidebar-row-action is-unpin-action group-hover/recent-item:opacity-100 group-hover/recent-item:pointer-events-auto focus-visible:opacity-100 focus-visible:pointer-events-auto"
          >
            <span className="sidebar-row-action-glyph">
              <HugeiconsIcon icon={PinOffIcon} strokeWidth={1.75} className="size-icon" />
            </span>
          </button>
        )}
        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <button
              type="button"
              onClick={(e) => e.stopPropagation()}
              aria-label="Chat options"
              className={actionClass}
            >
              <span className="sidebar-row-action-glyph">
                <HugeiconsIcon icon={MoreVerticalIcon} strokeWidth={1.75} className="size-icon" />
              </span>
            </button>
          </DropdownMenuTrigger>
          <DropdownMenuContent
            side="bottom"
            align="start"
            sideOffset={0}
            className="unsloth-plus-menu menu-flat-destructive w-56"
          >
            <DropdownMenuItem onSelect={() => openRenameChat(item)}>
              <HugeiconsIcon icon={Edit03Icon} strokeWidth={1.75} className="size-icon" />
              <span>Rename</span>
            </DropdownMenuItem>
            <DropdownMenuItem onSelect={() => togglePinnedChat(item.id)}>
              <HugeiconsIcon icon={isPinned ? PinOffIcon : PinIcon} strokeWidth={1.75} className="size-icon" />
              <span>{isPinned ? "Unpin chat" : "Pin chat"}</span>
            </DropdownMenuItem>
            <DropdownMenuSub>
              <DropdownMenuSubTrigger>
                <HugeiconsIcon icon={FolderExportIcon} strokeWidth={1.75} className="size-icon" />
                <span>Move to project</span>
              </DropdownMenuSubTrigger>
              <DropdownMenuSubContent
                sideOffset={0}
                alignOffset={-4}
                className="unsloth-plus-menu w-52"
              >
                <DropdownMenuItem
                  onSelect={() => {
                    setProjectCreateMoveTarget(item);
                    setProjectNameDraft("");
                    setCreatingProject(true);
                  }}
                >
                  <HugeiconsIcon icon={FolderAddIcon} strokeWidth={1.75} className="size-icon" />
                  <span>New project</span>
                </DropdownMenuItem>
                <DropdownMenuItem
                  disabled={!item.projectId}
                  onSelect={() => void moveChatToProject(item, null)}
                >
                  <span>Recents</span>
                </DropdownMenuItem>
                {projects.map((project) => (
                  <DropdownMenuItem
                    key={project.id}
                    disabled={item.projectId === project.id}
                    onSelect={() => void moveChatToProject(item, project.id)}
                  >
                    <HugeiconsIcon icon={Folder01Icon} strokeWidth={1.75} className="size-icon" />
                    <span className="truncate">{project.name}</span>
                  </DropdownMenuItem>
                ))}
              </DropdownMenuSubContent>
            </DropdownMenuSub>
            <DropdownMenuSub>
              <DropdownMenuSubTrigger>
                <HugeiconsIcon icon={Download01Icon} strokeWidth={1.75} className="size-icon" />
                <span>Export</span>
              </DropdownMenuSubTrigger>
              <DropdownMenuSubContent sideOffset={8} alignOffset={-4} className="unsloth-plus-menu w-52">
                {CHAT_EXPORT_OPTIONS.map(({ label, format }) => (
                  <DropdownMenuItem
                    key={label}
                    onSelect={async () => {
                      try {
                        const ids = item.type === "single"
                          ? [item.id]
                          : (await listStoredChatThreads({ pairId: item.id })).map((t) => t.id);
                        for (const id of ids) {
                          await exportConversationByFormat(id, format);
                        }
                      } catch (error) {
                        if (!isDownloadCancelled(error)) {
                          toast.error("Export failed.");
                        }
                      }
                    }}
                  >
                    {label}
                  </DropdownMenuItem>
                ))}
                <DropdownMenuSeparator />
                {/* Bulk export and import live in Settings -> Data. */}
                <DropdownMenuItem
                  onSelect={() =>
                    useSettingsDialogStore.getState().openDialog("data")
                  }
                >
                  Export all chats…
                </DropdownMenuItem>
              </DropdownMenuSubContent>
            </DropdownMenuSub>
            <DropdownMenuSeparator />
            <DropdownMenuItem onSelect={() => void handleArchiveThread(item)}>
              <HugeiconsIcon icon={Archive03Icon} strokeWidth={1.75} className="size-icon" />
              <span>Archive</span>
            </DropdownMenuItem>
            <DropdownMenuItem
              variant="destructive"
              onSelect={() =>
                confirmDeleteChats
                  ? setConfirmingDelete({ kind: "chat", item })
                  : void deleteChatWithCleanup(item)
              }
            >
              <HugeiconsIcon icon={Delete02Icon} strokeWidth={1.75} className="size-icon" />
              <span>Delete</span>
            </DropdownMenuItem>
          </DropdownMenuContent>
        </DropdownMenu>
      </SidebarMenuItem>
    );
  }

  return (
    <>
    <Sidebar
      collapsible="icon"
      variant="sidebar"
      className="font-heading group-data-[collapsible=icon]:[&_[data-sidebar=sidebar]]:bg-white dark:group-data-[collapsible=icon]:[&_[data-sidebar=sidebar]]:bg-background"
    >
      <SidebarHeader
        className={cn(
          "relative",
          showSidebarBrand
            ? showCompactMacBrand
              ? "h-[var(--studio-chat-header-height,48px)] pl-[calc(var(--studio-mac-traffic-light-inset,78px)+6px)] pr-2 pt-[var(--studio-chat-header-padding-top,9px)] pb-0 group-data-[collapsible=icon]:h-[calc(var(--studio-mac-titlebar-height,34px)+var(--studio-chat-control-height,33px)+8px)] group-data-[collapsible=icon]:px-0 group-data-[collapsible=icon]:pt-[calc(var(--studio-mac-titlebar-height,34px)+8px)]"
              : "pl-[17px] pr-3 pt-[14px] pb-[8px] group-data-[collapsible=icon]:px-0"
            : "h-[var(--studio-custom-titlebar-height,34px)] shrink-0 p-0",
        )}
      >
        {showSidebarBrand && (
          <>
            {usesNativeMacTitlebar && (
              <div
                data-tauri-drag-region
                aria-hidden="true"
                className="absolute inset-x-0 top-0 z-0 h-[var(--studio-mac-titlebar-height,34px)] select-none"
              />
            )}
            <div
              className={cn(
                "relative z-10 flex items-center gap-[8.5px] group-data-[collapsible=icon]:hidden",
                showCompactMacBrand &&
                  "h-[var(--studio-chat-control-height,33px)] justify-end gap-2",
                !showCompactMacBrand && "justify-between",
              )}
            >
              {!showCompactMacBrand && (
                <Link
                  to="/chat"
                  onClick={(event) => {
                    event.preventDefault();
                    if (chatDisabled) return;
                    openNewChat(null);
                  }}
                  className={cn(
                    "flex items-center gap-[6px] select-none transition-opacity",
                    chatDisabled && "pointer-events-none opacity-50",
                  )}
                  aria-label={t("shell.aria.home")}
                  aria-disabled={chatDisabled}
                  tabIndex={chatDisabled ? -1 : undefined}
                >
                  <img
                    src="/circle-logo-small.png"
                    alt="Unsloth"
                    className="h-[34px] w-[34px] rounded-full object-cover"
                  />
                  <span className="font-heading text-[21px] font-semibold tracking-[0em] leading-none text-black dark:text-white dark:tracking-[0.02em]">
                    unsloth
                  </span>
                  <span className="nav-badge ml-0.5 inline-flex items-center justify-center rounded-full border border-nav-beta-border px-[5px] pt-[3px] pb-[2px] text-[8px] font-medium leading-none tracking-[0.04em] text-nav-fg-muted antialiased subpixel-antialiased shadow-[0_1px_2px_rgba(0,0,0,0.06)] dark:shadow-[0_1px_2px_rgba(0,0,0,0.35)]">
                    {t("shell.beta")}
                  </span>
                </Link>
              )}
              <div className="flex items-center gap-0.5">
                <Tooltip>
                  <TooltipPrimitive.Trigger asChild>
                    <button
                      type="button"
                      onClick={() => {
                        useChatSearchStore.getState().open();
                        closeMobileIfOpen();
                      }}
                      className="inline-flex h-[33px] w-[32px] cursor-pointer items-center justify-center rounded-[10px] text-nav-icon-idle dark:text-nav-fg-muted transition-colors hover:bg-nav-surface-hover hover:text-black dark:hover:text-white focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
                      aria-label={t("shell.navigation.search")}
                    >
                      <HugeiconsIcon icon={Search01Icon} strokeWidth={1.75} className="size-icon" />
                    </button>
                  </TooltipPrimitive.Trigger>
                  <TooltipContent
                    side="bottom"
                    sideOffset={6}
                    className="tooltip-compact flex items-center gap-1.5"
                    hidden={isMobile}
                  >
                    {t("shell.navigation.search")}
                    <kbd className="rounded bg-black/10 px-1 py-px text-[10px] font-medium leading-none dark:bg-white/15">
                      {isMacPlatform ? "⌘K" : "Ctrl+K"}
                    </kbd>
                  </TooltipContent>
                </Tooltip>
                {!isMobile && (
                  <Tooltip>
                    <TooltipPrimitive.Trigger asChild>
                      <button
                        type="button"
                        onClick={togglePinned}
                        className="inline-flex h-[33px] w-[32px] cursor-pointer items-center justify-center rounded-[10px] text-nav-icon-idle dark:text-nav-fg-muted transition-colors hover:bg-nav-surface-hover hover:text-black dark:hover:text-white focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
                        aria-label={t("shell.aria.closeSidebar")}
                      >
                        <HugeiconsIcon icon={LayoutAlignLeftIcon} strokeWidth={1.75} className="size-icon" />
                      </button>
                    </TooltipPrimitive.Trigger>
                    <TooltipContent
                      side="bottom"
                      sideOffset={6}
                      className="tooltip-compact"
                    >
                      {t("shell.aria.closeSidebar")}
                    </TooltipContent>
                  </Tooltip>
                )}
              </div>
            </div>
            {!isMobile && (
              <div className="relative z-10 hidden group-data-[collapsible=icon]:flex h-[33px] items-center justify-center w-full">
                <Tooltip>
                  <TooltipPrimitive.Trigger asChild>
                    <button
                      type="button"
                      onClick={togglePinned}
                      className="inline-flex h-[33px] w-[32px] cursor-pointer items-center justify-center rounded-[10px] text-nav-fg transition-colors hover:bg-nav-surface-hover hover:text-black dark:hover:text-white focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
                      aria-label={t("shell.aria.openSidebar")}
                    >
                      <HugeiconsIcon icon={LayoutAlignLeftIcon} strokeWidth={1.75} className="size-icon" />
                    </button>
                  </TooltipPrimitive.Trigger>
                  <TooltipContent
                    side="right"
                    sideOffset={8}
                    className="tooltip-compact"
                  >
                    {t("shell.aria.openSidebar")}
                  </TooltipContent>
                </Tooltip>
              </div>
            )}
          </>
        )}
      </SidebarHeader>

      {/* Uniform pl-1.5 pr-2 keeps every hover pill the same width, inset from the edge. */}
      <SidebarGroup
        className={cn(
          "group-data-[collapsible=icon]:px-0 pl-1.5 pr-2 shrink-0 transition-[padding]",
          showCompactMacBrand ? "pt-0" : "pt-[9px]",
          // Scrolled: New Chat is pinned, give a little gap below it.
          scrolled ? "pb-[5px]" : "pb-px",
        )}
      >
        <SidebarGroupContent>
          <SidebarMenu>
            <NavItem
              icon={PencilEdit02Icon}
              label={
                showReturnToChat
                  ? t("shell.navigation.returnToChat")
                  : t("shell.navigation.newChat")
              }
              active={
                isChatRoute &&
                !search.thread &&
                !search.compare &&
                !search.project
              }
              onClick={() => {
                if (showReturnToChat) {
                  // Prefer the running thread so we return to the live generation,
                  // not the empty new chat that became active after New Chat.
                  if (runningThreadId && runningThreadId !== storeThreadId) {
                    navigate({ to: "/chat", search: { thread: runningThreadId } });
                  } else {
                    navigate({ to: "/chat" });
                  }
                  closeMobileIfOpen();
                  return;
                }
                openNewChat(null);
              }}
            />
            {/* Search sits in the header when the brand row is shown (mac/web).
                Hide this row there, but keep it in the collapsed rail. On custom
                titlebars (win/linux) there's no header button, so keep the row. */}
            <NavItem
              icon={Search01Icon}
              label={t("shell.navigation.search")}
              active={false}
              className={
                showSidebarBrand
                  ? "hidden group-data-[collapsible=icon]:block"
                  : undefined
              }
              onClick={() => {
                useChatSearchStore.getState().open();
                closeMobileIfOpen();
              }}
            />
          </SidebarMenu>
        </SidebarGroupContent>
      </SidebarGroup>

      <SidebarContent
        ref={scrollRef}
        onScroll={(e) => syncScrollState(e.currentTarget)}
        // Collapsible groups animate their height; re-measure the fade once the
        // open/close animation settles, not on the (still-animating) state flip.
        onAnimationEnd={(e) => {
          if (
            e.animationName === "collapsible-down" ||
            e.animationName === "collapsible-up"
          ) {
            syncScrollState(e.currentTarget);
          }
        }}
        className={cn(
          // pb-2 keeps the last row's rounded highlight clear of the
          // overflow clip edge so its bottom corners aren't shaved off.
          "sidebar-scroll-fade gap-0 overflow-y-auto overscroll-contain min-h-0 pb-2",
          scrolled && "is-scrolled",
        )}
      >
        <SidebarGroup className="group-data-[collapsible=icon]:px-0 pl-1.5 pr-2 py-0 shrink-0">
          <SidebarGroupContent>
            <SidebarMenu>
              <NavItem
                icon={Folder01Icon}
                label="Projects"
                active={
                  pathname === "/projects" || pathname.startsWith("/projects/")
                }
                onClick={() => {
                  navigate({ to: "/projects" });
                  closeMobileIfOpen();
                }}
                onIntent={() => {
                  preloadSilently(router.preloadRoute({ to: "/projects" }));
                }}
                className="group/projects-item relative"
              >
                <button
                  type="button"
                  aria-label="New project"
                  onClick={(e) => {
                    e.stopPropagation();
                    setProjectCreateMoveTarget(null);
                    setProjectNameDraft("");
                    setCreatingProject(true);
                  }}
                  className="sidebar-row-action group-hover/projects-item:opacity-100 group-hover/projects-item:pointer-events-auto focus-visible:opacity-100 focus-visible:pointer-events-auto group-data-[collapsible=icon]:hidden"
                >
                  <span className="sidebar-row-action-glyph">
                    <HugeiconsIcon
                      icon={PlusSignIcon}
                      strokeWidth={1.75}
                      className="size-4"
                    />
                  </span>
                </button>
              </NavItem>
              <NavItem
                icon={DashboardCircleIcon}
                label={t("shell.navigation.hub")}
                active={pathname === "/hub" || pathname.startsWith("/hub/")}
                onClick={() => {
                  navigate({ to: "/hub" });
                  closeMobileIfOpen();
                }}
                onIntent={() => {
                  preloadSilently(router.preloadRoute({ to: "/hub" }));
                }}
              />
              {/* Train has a labelled section when expanded; plain icon here only when collapsed. */}
              <NavItem
                icon={TestTubeOutlineIcon}
                label={t("shell.navigation.train")}
                active={
                  pathname === "/studio" || pathname.startsWith("/studio/")
                }
                disabled={chatOnly}
                tooltip={trainDisabledHint}
                spinner={trainingInProgress}
                onClick={() => {
                  if (chatOnly) return;
                  navigate({ to: "/studio" });
                  closeMobileIfOpen();
                }}
                onIntent={() => {
                  preloadSilently(router.preloadRoute({ to: "/studio" }));
                }}
                className="hidden group-data-[collapsible=icon]:block"
              />
            </SidebarMenu>
          </SidebarGroupContent>
        </SidebarGroup>

        <Collapsible open={trainOpen} onOpenChange={setTrainOpen} asChild>
          <SidebarGroup data-tour="navbar" className="group-data-[collapsible=icon]:hidden px-0 py-0">
            <SidebarGroupLabel className={cn("sidebar-sticky-label sidebar-sticky-label-following", scrolled && "is-scrolled")} asChild>
              <CollapsibleTrigger className="cursor-pointer flex w-full items-center gap-1 group/sb-collap">
                {t("shell.navigation.train")}
                <ChevronDown className="size-3.5 opacity-0 transition-[transform,opacity] duration-200 group-hover/sb-collap:opacity-100 group-focus-visible/sb-collap:opacity-100 data-[state=open]:rotate-0 [[data-state=closed]_&]:rotate-[-90deg] [[data-state=closed]_&]:opacity-100" />
              </CollapsibleTrigger>
            </SidebarGroupLabel>
            <CollapsibleContent>
              <SidebarGroupContent className="pl-1.5 pr-2">
                <SidebarMenu>
                  <NavItem
                    icon={TestTubeOutlineIcon}
                    label={t("shell.navigation.train")}
                    active={pathname === "/studio" || pathname.startsWith("/studio/")}
                    disabled={chatOnly}
                    tooltip={trainDisabledHint}
                    spinner={trainingInProgress}
                    onClick={() => {
                      if (chatOnly) return;
                      navigate({ to: "/studio" });
                      closeMobileIfOpen();
                    }}
                    onIntent={() => {
                      preloadSilently(router.preloadRoute({ to: "/studio" }));
                    }}
                  />
                  <NavItem
                    icon={Notebook01Icon}
                    label={t("shell.navigation.notebooks")}
                    active={isNotebooksRoute}
                    onClick={() => {
                      navigate({ to: "/notebooks" });
                      closeMobileIfOpen();
                    }}
                  />
                  <NavItem
                    icon={ChefHatIcon}
                    label={t("shell.navigation.recipes")}
                    active={isRecipesRoute}
                    onClick={() => {
                      navigate({ to: "/data-recipes" });
                      closeMobileIfOpen();
                    }}
                    onIntent={() => {
                      preloadSilently(
                        router.preloadRoute({ to: "/data-recipes" }),
                      );
                      preloadSilently(
                        import("@/features/data-recipes").then((module) =>
                          module.preloadRecipes(),
                        ),
                      );
                    }}
                  />
                  <NavItem
                    icon={DownloadSquare01Icon}
                    label={t("shell.navigation.export")}
                    active={pathname === "/export" || pathname.startsWith("/export/")}
                    spinner={exportInProgress}
                    onClick={() => {
                      navigate({ to: "/export" });
                      closeMobileIfOpen();
                    }}
                    onIntent={() => {
                      preloadSilently(router.preloadRoute({ to: "/export" }));
                      preloadSilently(
                        import(
                          "@/features/export/export-navigation-cache"
                        ).then((module) => module.preloadExportData()),
                      );
                    }}
                  />
                </SidebarMenu>
              </SidebarGroupContent>
            </CollapsibleContent>
          </SidebarGroup>
        </Collapsible>

        {/* Pinned: pinned projects (with their chats) and pinned chats */}
        {!isStudioRoute &&
          !showTrainingRecents &&
          (pinnedProjectRecords.length > 0 ||
            pinnedChatItems.length > 0) && (
            <Collapsible open={pinnedOpen} onOpenChange={setPinnedOpen} asChild>
              <SidebarGroup className="group-data-[collapsible=icon]:hidden px-0 py-0">
                <SidebarGroupLabel className={cn("sidebar-sticky-label sidebar-sticky-label-following", scrolled && "is-scrolled")} asChild>
                  <CollapsibleTrigger className="cursor-pointer flex w-full items-center gap-1 group/sb-collap">
                    Pinned
                    <ChevronDown className="size-3.5 opacity-0 transition-[transform,opacity] duration-200 group-hover/sb-collap:opacity-100 group-focus-visible/sb-collap:opacity-100 data-[state=open]:rotate-0 [[data-state=closed]_&]:rotate-[-90deg] [[data-state=closed]_&]:opacity-100" />
                  </CollapsibleTrigger>
                </SidebarGroupLabel>
                <CollapsibleContent>
                  <SidebarGroupContent className="pl-1.5 pr-2">
                    <SidebarMenu>
                      {pinnedProjectRecords.map((project) => {
                        const projectChats =
                          chatsByProjectId.get(project.id) ?? [];
                        const expanded = !collapsedProjectIds.has(project.id);
                        const showAll = expandedChatProjectIds.has(project.id);
                        const visibleChats =
                          expanded && !showAll
                            ? projectChats.slice(0, PINNED_PROJECT_CHAT_LIMIT)
                            : projectChats;
                        return (
                        <Fragment key={project.id}>
                        <SidebarMenuItem
                          className="group/recent-item relative"
                        >
                          <SidebarMenuButton
                            // Highlight the folder only on the project home; when
                            // a chat inside it is open, only that chat row is active.
                            isActive={activeProjectId === project.id && !activeThreadId}
                            onClick={() => toggleProjectCollapsed(project.id)}
                            className="sidebar-nav-btn h-[33px] rounded-full gap-[8.5px] pl-3 pr-2.5 font-medium group-hover/recent-item:pr-16 group-has-[.sidebar-row-action[data-state=open]]/recent-item:pr-8"
                          >
                            <HugeiconsIcon icon={Folder01Icon} strokeWidth={1.75} className="size-icon! shrink-0" />
                            <span className="truncate text-[14.5px] leading-[19px] tracking-nav">{project.name}</span>
                          </SidebarMenuButton>
                          {/* New chat in this project */}
                          <button
                            type="button"
                            aria-label="New chat"
                            onClick={(e) => {
                              e.stopPropagation();
                              openNewChat(project.id);
                            }}
                            className="sidebar-row-action is-unpin-action group-hover/recent-item:opacity-100 group-hover/recent-item:pointer-events-auto focus-visible:opacity-100 focus-visible:pointer-events-auto"
                          >
                            <span className="sidebar-row-action-glyph">
                              <HugeiconsIcon icon={PencilEdit02Icon} strokeWidth={1.75} className="size-icon" />
                            </span>
                          </button>
                          {/* Project options */}
                          <DropdownMenu>
                            <DropdownMenuTrigger asChild>
                              <button
                                type="button"
                                onClick={(e) => e.stopPropagation()}
                                aria-label="Project options"
                                className="sidebar-row-action group-hover/recent-item:opacity-100 group-hover/recent-item:pointer-events-auto focus-visible:opacity-100 focus-visible:pointer-events-auto"
                              >
                                <span className="sidebar-row-action-glyph">
                                  <HugeiconsIcon icon={MoreVerticalIcon} strokeWidth={1.75} className="size-icon" />
                                </span>
                              </button>
                            </DropdownMenuTrigger>
                            <DropdownMenuContent
                              side="bottom"
                              align="start"
                              sideOffset={0}
                              className="unsloth-plus-menu menu-flat-destructive w-56"
                            >
                              <DropdownMenuItem onSelect={() => openProject(project.id)}>
                                <HugeiconsIcon icon={Folder01Icon} strokeWidth={1.75} className="size-icon" />
                                <span>Project home</span>
                              </DropdownMenuItem>
                              <DropdownMenuItem onSelect={() => openNewChat(project.id)}>
                                <HugeiconsIcon icon={PencilEdit02Icon} strokeWidth={1.75} className="size-icon" />
                                <span>New chat</span>
                              </DropdownMenuItem>
                              <DropdownMenuItem
                                onSelect={() => {
                                  // Seed the shared draft so the dialog opens
                                  // with the current name, not stale text.
                                  setRenameDraft(project.name);
                                  setRenamingTarget({
                                    kind: "project",
                                    project,
                                    current: project.name,
                                  });
                                }}
                              >
                                <HugeiconsIcon icon={Edit03Icon} strokeWidth={1.75} className="size-icon" />
                                <span>Rename project</span>
                              </DropdownMenuItem>
                              <DropdownMenuItem onSelect={() => unpinProject(project.id)}>
                                <HugeiconsIcon icon={PinOffIcon} strokeWidth={1.75} className="size-icon" />
                                <span>Unpin project</span>
                              </DropdownMenuItem>
                              <DropdownMenuSeparator />
                              <DropdownMenuItem
                                variant="destructive"
                                onSelect={() => {
                                  // Start each delete with the file toggle off:
                                  // Cancel closes programmatically and skips the
                                  // dialog onOpenChange reset.
                                  setDeleteProjectFiles(false);
                                  setConfirmingDelete({ kind: "project", project });
                                }}
                              >
                                <HugeiconsIcon icon={Delete02Icon} strokeWidth={1.75} className="size-icon" />
                                <span>Delete project</span>
                              </DropdownMenuItem>
                            </DropdownMenuContent>
                          </DropdownMenu>
                        </SidebarMenuItem>
                        {expanded &&
                          visibleChats.map((chat) =>
                            renderChatSidebarItem(chat, "project"),
                          )}
                        {expanded &&
                          projectChats.length > PINNED_PROJECT_CHAT_LIMIT && (
                            <SidebarMenuItem>
                              <SidebarMenuButton
                                onClick={() => toggleProjectShowAll(project.id)}
                                // Force the muted token: .sidebar-nav-btn's own
                                // color rule outweighs a plain text utility, so
                                // Show more would otherwise match the chat rows.
                                className="sidebar-nav-btn h-[30px] rounded-full pl-9 pr-4 font-medium text-nav-fg-muted!"
                              >
                                <span className="text-[13px] leading-[18px] tracking-nav">
                                  {showAll ? "Show less" : "Show more"}
                                </span>
                              </SidebarMenuButton>
                            </SidebarMenuItem>
                          )}
                        </Fragment>
                        );
                      })}
                      {pinnedChatItems.map((item) =>
                        renderChatSidebarItem(item, "recent"),
                      )}
                    </SidebarMenu>
                  </SidebarGroupContent>
                </CollapsibleContent>
              </SidebarGroup>
            </Collapsible>
          )}

        {!isStudioRoute && !showTrainingRecents && (
          <Collapsible open={chatOpen} onOpenChange={setChatOpen} asChild>
            <SidebarGroup className="group-data-[collapsible=icon]:hidden px-0 py-0">
              <SidebarGroupLabel className={cn("sidebar-sticky-label sidebar-sticky-label-following", scrolled && "is-scrolled")} asChild>
                <CollapsibleTrigger className="cursor-pointer flex w-full items-center gap-1 group/sb-collap">
                  {t("shell.navigation.recents")}
                  <ChevronDown className="size-3.5 opacity-0 transition-[transform,opacity] duration-200 group-hover/sb-collap:opacity-100 group-focus-visible/sb-collap:opacity-100 data-[state=open]:rotate-0 [[data-state=closed]_&]:rotate-[-90deg] [[data-state=closed]_&]:opacity-100" />
                </CollapsibleTrigger>
              </SidebarGroupLabel>
              <CollapsibleContent>
                <SidebarGroupContent className="pl-1.5 pr-2">
                  <SidebarMenu>
                    {recentChatItems.map((item) =>
                      renderChatSidebarItem(item, "recent"),
                    )}
                  </SidebarMenu>
                  {/* "No chats yet" only when there is truly no history:
                      project-scoped and archived threads leave Recents empty
                      but still count as existing chats. */}
                  {chatItemsLoaded &&
                    allChatItems.length === 0 &&
                    archivedChatItems.length === 0 && (
                      <p className="px-3 py-2 text-xs text-muted-foreground">
                        {t("shell.navigation.noChatsYet")}
                      </p>
                    )}
                </SidebarGroupContent>
              </CollapsibleContent>
            </SidebarGroup>
          </Collapsible>
        )}

        {showTrainingRecents && (
          <Collapsible open={runsOpen} onOpenChange={setRunsOpen} asChild>
          <SidebarGroup className="group-data-[collapsible=icon]:hidden px-0 py-0">
            <SidebarGroupLabel className={cn("sidebar-sticky-label sidebar-sticky-label-following", scrolled && "is-scrolled")} asChild>
              <CollapsibleTrigger className="cursor-pointer flex w-full items-center gap-1 group/sb-collap">
                {t("shell.navigation.recents")}
                <ChevronDown className="size-3.5 opacity-0 transition-[transform,opacity] duration-200 group-hover/sb-collap:opacity-100 group-focus-visible/sb-collap:opacity-100 data-[state=open]:rotate-0 [[data-state=closed]_&]:rotate-[-90deg] [[data-state=closed]_&]:opacity-100" />
              </CollapsibleTrigger>
            </SidebarGroupLabel>
            <CollapsibleContent>
              <SidebarGroupContent className="pl-1.5 pr-2">
                <SidebarMenu>
                  {runItems.map((run) => {
                    // Explicit selection wins. Otherwise highlight the active
                    // job only while the "Current Run" tab is the view, keeping
                    // the Configure tab unhighlighted even though activeJobId
                    // stays pinned to the last job.
                    const isActiveRun =
                      selectedHistoryRunId != null
                        ? run.id === selectedHistoryRunId
                        : currentRunViewActive && run.id === activeJobId;
                    return (
                      <SidebarMenuItem
                        key={run.id}
                        className="group/run-item relative"
                      >
                        <SidebarMenuButton
                          isActive={isActiveRun}
                          className="sidebar-nav-btn h-auto flex-col items-start gap-0.5 py-[5px] rounded-[14px] pl-3 pr-7 text-[14.5px] tracking-nav font-medium"
                          onClick={() => {
                            setSelectedHistoryRunId(run.id);
                            // From Recipes/Export, jump to Train so the run's
                            // history opens (studio reacts to selectedHistoryRunId).
                            if (!isStudioRoute) navigate({ to: "/studio" });
                            closeMobileIfOpen();
                          }}
                        >
                          <div className="flex w-full items-center gap-[8.5px]">
                            <span
                              className={cn(
                                "size-1.5 shrink-0 rounded-full",
                                runStatusDotClass(run.status),
                              )}
                              aria-hidden
                            />
                            <span className="truncate">
                              {getTrainingRunDisplayTitle(run)}
                            </span>
                            <span className="ml-auto mr-0.5 shrink-0 text-[10px] text-muted-foreground">
                              {formatRelativeShort(run.started_at)}
                            </span>
                          </div>
                          <span className="w-full truncate pl-3.5 text-xs text-muted-foreground">
                            {run.dataset_name}
                          </span>
                        </SidebarMenuButton>
                        <DropdownMenu>
                          <DropdownMenuTrigger asChild>
                            <button
                              type="button"
                              onClick={(e) => e.stopPropagation()}
                              aria-label={t("shell.aria.runOptions")}
                              className="sidebar-row-action group-hover/run-item:opacity-100 group-hover/run-item:pointer-events-auto focus-visible:opacity-100 focus-visible:pointer-events-auto"
                            >
                              <span className="sidebar-row-action-glyph">
                                <HugeiconsIcon icon={MoreVerticalIcon} strokeWidth={1.75} className="size-icon" />
                              </span>
                            </button>
                          </DropdownMenuTrigger>
                          <DropdownMenuContent
                            side="bottom"
                            align="end"
                            sideOffset={0}
                            className="app-user-menu menu-soft-surface menu-flat-destructive ring-0 w-44 py-2 font-heading rounded-full border-0"
                          >
                            <DropdownMenuItem onSelect={() => openRenameRun(run)}>
                              <HugeiconsIcon icon={Edit03Icon} strokeWidth={1.75} className="size-icon" />
                              <span>{t("common.rename")}</span>
                            </DropdownMenuItem>
                            <DropdownMenuItem
                              variant="destructive"
                              disabled={run.status === "running"}
                              onSelect={() =>
                                setConfirmingDelete({ kind: "run", run })
                              }
                            >
                              <HugeiconsIcon icon={Delete02Icon} strokeWidth={1.75} className="size-icon" />
                              <span>{t("common.delete")}</span>
                            </DropdownMenuItem>
                          </DropdownMenuContent>
                        </DropdownMenu>
                      </SidebarMenuItem>
                    );
                  })}
                </SidebarMenu>
              </SidebarGroupContent>
            </CollapsibleContent>
          </SidebarGroup>
          </Collapsible>
        )}
      </SidebarContent>

      <SidebarFooter
        className={cn(
          "relative pb-3 group-data-[collapsible=icon]:px-0",
          // Tighter top with the update card so the fade hugs it; fuller top
          // for the profile on its own.
          showUpdateCard ? "pt-1.5" : "pt-2.5",
        )}
      >
        {/* Fade above the profile box, shown only when there's more list below
            the fold; at the bottom (or short lists) it fades so the last row
            shows fully (Gemini-style). right-2 keeps it clear of the 8px scrollbar gutter. */}
        <div
          aria-hidden="true"
          className={cn(
            "pointer-events-none absolute left-0 right-2 bottom-full bg-gradient-to-t from-[var(--sidebar)] to-[rgb(from_var(--sidebar)_r_g_b/0)] transition-opacity duration-200",
            // Shorter fade when the update card sits above the profile so the
            // list reads closer to it.
            showUpdateCard ? "h-3" : "h-10",
            canScrollDown ? "opacity-100" : "opacity-0",
          )}
        />
        <SidebarMenu className="gap-3 group-data-[collapsible=icon]:gap-2.5">
          {/* Update affordance — shows only when a newer version is available. */}
          {showUpdateCard && (
            <SidebarMenuItem>
              <button
                type="button"
                aria-label={t("shell.updateAvailable")}
                onClick={() => {
                  useSettingsDialogStore
                    .getState()
                    .openDialog("about", { scrollTarget: "about-updates" });
                  closeMobileIfOpen();
                }}
                className="flex h-[44px] w-full items-center gap-[9px] rounded-[14px] border border-border/60 bg-transparent px-2 py-[3px] text-left transition-colors hover:bg-nav-surface-hover focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring group-data-[collapsible=icon]:mx-auto group-data-[collapsible=icon]:h-[34px] group-data-[collapsible=icon]:w-[34px] group-data-[collapsible=icon]:justify-center group-data-[collapsible=icon]:gap-0 group-data-[collapsible=icon]:rounded-full group-data-[collapsible=icon]:p-0"
              >
                <span
                  aria-hidden="true"
                  className="flex size-[32px] shrink-0 items-center justify-center group-data-[collapsible=icon]:size-full"
                >
                  <HugeiconsIcon
                    icon={BadgeInfoIcon}
                    strokeWidth={1.75}
                    className="size-[21px] text-nav-fg"
                  />
                </span>
                <div className="flex min-w-0 flex-col gap-px leading-tight group-data-[collapsible=icon]:hidden">
                  <span className="truncate font-heading text-[13.5px] font-semibold text-nav-fg">
                    {t("shell.updateAvailable")}
                  </span>
                  {updateVersion && (
                    <span className="truncate text-[11.5px] text-muted-foreground">
                      v{updateVersion}
                    </span>
                  )}
                </div>
                <span
                  aria-hidden="true"
                  className="ml-auto flex size-[32px] shrink-0 items-center justify-center text-muted-foreground group-data-[collapsible=icon]:hidden"
                >
                  <HugeiconsIcon
                    icon={ArrowRight02Icon}
                    className="size-[17px]"
                    strokeWidth={1.75}
                  />
                </span>
              </button>
            </SidebarMenuItem>
          )}
          <SidebarMenuItem>
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <SidebarMenuButton
                  size="lg"
                  aria-label={t("shell.accountMenu", { name: displayTitle })}
                  className="sidebar-nav-btn !h-[44px] -my-[3px] gap-[9px] pl-2 pr-[45px] py-[3px] rounded-[14px] group-data-[collapsible=icon]:!size-[34px] group-data-[collapsible=icon]:!rounded-full group-data-[collapsible=icon]:!p-0 group-data-[collapsible=icon]:mx-auto group-data-[collapsible=icon]:justify-center"
                >
                  <div className="flex shrink-0 items-center">
                    <UserAvatar
                      name={displayTitle}
                      imageUrl={avatarDataUrl}
                      size="sm"
                      className="!size-[32px] group-data-[collapsible=icon]:!rounded-full"
                    />
                  </div>
                  {/* min-w-0 so long names truncate instead of overflowing;
                      pr on the button reserves room for the settings cog */}
                  <div className="flex min-w-0 flex-1 flex-col gap-px leading-tight group-data-[collapsible=icon]:hidden">
                    <span className="truncate font-heading text-[13.5px] tracking-[0.025em] dark:tracking-[0.04em] font-semibold text-nav-fg">{displayTitle}</span>
                    <span className="truncate text-[11.5px] tracking-nav text-muted-foreground">Unsloth</span>
                  </div>
                </SidebarMenuButton>
              </DropdownMenuTrigger>
              <DropdownMenuContent
                side="top"
                align="center"
                sideOffset={8}
                className="app-user-menu menu-soft-surface-up ring-0 w-[16rem] px-2.5 py-2.5 font-heading rounded-[20px] border-0"
              >
                <DropdownMenuGroup>
                  <DropdownMenuItem
                    onSelect={() => useSettingsDialogStore.getState().openDialog()}
                  >
                    <HugeiconsIcon icon={Settings02Icon} strokeWidth={1.75} className="size-icon" />
                    <span>{t("shell.navigation.settings")}</span>
                    <DropdownMenuShortcut>⌘,</DropdownMenuShortcut>
                  </DropdownMenuItem>
                  {/* Optional items follow the order and visibility set in
                      Appearance settings; Settings above and the block after
                      the separator are pinned. */}
                  {sidebarMenu.map((item) => {
                    if (!item.visible) return null;
                    if (item.id === "api") {
                      return (
                        <DropdownMenuItem
                          key={item.id}
                          onSelect={() => useSettingsDialogStore.getState().openDialog("api-keys")}
                        >
                          <HugeiconsIcon icon={Globe02Icon} strokeWidth={1.75} className="size-[18px]" />
                          <span>{t("shell.navigation.api")}</span>
                        </DropdownMenuItem>
                      );
                    }
                    if (item.id === "darkMode") {
                      return (
                        <DropdownMenuItem
                          key={item.id}
                          ref={anchorRef as React.Ref<HTMLDivElement>}
                          onSelect={(e) => { e.preventDefault(); toggleTheme(); }}
                        >
                          {isDark ? <HugeiconsIcon icon={Sun03Icon} strokeWidth={1.75} className="size-icon" /> : <Moon strokeWidth={1.75} className="size-icon" />}
                          <span>
                            {isDark
                              ? t("shell.navigation.lightMode")
                              : t("shell.navigation.darkMode")}
                          </span>
                        </DropdownMenuItem>
                      );
                    }
                    if (item.id === "guidedTour") {
                      if (!getTourId(pathname)) return null;
                      return (
                        <DropdownMenuItem
                          key={item.id}
                          onSelect={() => {
                            const tourId = getTourId(pathname);
                            if (!tourId) return;
                            window.dispatchEvent(
                              new CustomEvent(TOUR_OPEN_EVENT, {
                                detail: { id: tourId },
                              }),
                            );
                          }}
                        >
                          <HugeiconsIcon icon={CursorInfo02Icon} strokeWidth={1.75} className="size-icon" />
                          <span>{t("shell.navigation.guidedTour")}</span>
                        </DropdownMenuItem>
                      );
                    }
                    // Remaining ids are settings tabs shown by their tab name.
                    const settingsTabId = item.id;
                    const tab = SETTINGS_TAB_MENU_ITEMS[settingsTabId];
                    return (
                      <DropdownMenuItem
                        key={item.id}
                        onSelect={() => useSettingsDialogStore.getState().openDialog(settingsTabId)}
                      >
                        <HugeiconsIcon icon={tab.icon} strokeWidth={1.75} className="size-icon" />
                        <span>{t(tab.labelKey)}</span>
                      </DropdownMenuItem>
                    );
                  })}
                </DropdownMenuGroup>
                <DropdownMenuSeparator className="mx-1! my-2.5! h-0! border-t border-border/70 bg-transparent!" />
                <DropdownMenuItem
                  onSelect={() => useSettingsDialogStore.getState().openDialog("about")}
                >
                  <HugeiconsIcon icon={HelpCircleIcon} strokeWidth={1.75} className="size-icon" />
                  <span>{t("common.help")}</span>
                </DropdownMenuItem>
                {!isTauri && (
                  <DropdownMenuItem
                    onSelect={async () => {
                      // Best-effort server revocation; ignore network errors so
                      // the local clear still runs and the user lands on /login.
                      try {
                        await logout();
                      } catch {
                        clearAuthTokens();
                      }
                      void navigate({ to: "/login" });
                    }}
                  >
                    <HugeiconsIcon icon={Logout05Icon} strokeWidth={1.75} className="size-icon" />
                    <span>{t("shell.navigation.logOut")}</span>
                  </DropdownMenuItem>
                )}
                {!isTauri && (
                  <DropdownMenuItem onSelect={() => setShutdownOpen(true)}>
                    <HugeiconsIcon icon={PowerIcon} strokeWidth={1.75} className="size-icon" />
                    <span>{t("common.shutdown")}</span>
                  </DropdownMenuItem>
                )}
              </DropdownMenuContent>
            </DropdownMenu>
            {/* settings cog; sibling of the trigger (buttons cannot nest),
                overlaid on the row's right edge, opens settings directly */}
            <button
              type="button"
              aria-label={t("shell.navigation.settings")}
              onClick={() => useSettingsDialogStore.getState().openDialog()}
              className="absolute right-2 top-1/2 flex size-[32px] -translate-y-1/2 items-center justify-center rounded-full text-muted-foreground transition-colors hover:bg-black/10 hover:text-foreground dark:hover:bg-white/10 focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring group-data-[collapsible=icon]:hidden"
            >
              <HugeiconsIcon
                icon={Settings02Icon}
                strokeWidth={1.5}
                className="!size-[18px]"
              />
            </button>
          </SidebarMenuItem>
        </SidebarMenu>
      </SidebarFooter>
    </Sidebar>
    <ChatSearchDialog />
    {!isTauri && (
      <ShutdownDialog
        open={shutdownOpen}
        onOpenChange={setShutdownOpen}
        onAfterShutdown={removeTrainingUnloadGuard}
      />
    )}
    <Dialog
      open={confirmingDelete !== null}
      onOpenChange={(open) => {
        if (!open) {
          setConfirmingDelete(null);
          setDeleteProjectFiles(false);
        }
      }}
    >
      <DialogContent className="menu-flat-destructive corner-squircle dialog-soft-surface sm:max-w-md">
        <DialogHeader>
          <DialogTitle>
            {confirmingDelete?.kind === "run"
              ? t("shell.dialog.deleteRun.title")
              : confirmingDelete?.kind === "project"
                ? "Delete project"
                : t("shell.dialog.deleteChat.title")}
          </DialogTitle>
          <DialogDescription>
            {confirmingDelete?.kind === "run" ? (
              renderEmphasizedTranslation(
                t,
                "shell.dialog.deleteRun.description",
                getTrainingRunDisplayTitle(confirmingDelete.run),
              )
            ) : confirmingDelete?.kind === "chat" ? (
              renderEmphasizedTranslation(
                t,
                "shell.dialog.deleteChat.description",
                confirmingDelete.item.title,
              )
            ) : confirmingDelete?.kind === "project" ? (
              <>
                Delete{" "}
                <span className="font-medium text-foreground">
                  &quot;{confirmingDelete.project.name}&quot;
                </span>
                ? Its chats will be permanently deleted.
              </>
            ) : null}
          </DialogDescription>
        </DialogHeader>
        {confirmingDelete?.kind === "project" ? (
          <div className="flex items-start justify-between gap-4 rounded-md border border-border/60 bg-muted/35 px-3 py-2.5">
            <label htmlFor="delete-project-files" className="min-w-0 space-y-1">
              <span className="block text-sm font-medium text-foreground">
                Delete files and sandbox folder
              </span>
              <span className="block break-words text-xs leading-5 text-muted-foreground">
                {confirmingDelete.project.rootPath
                  ? confirmingDelete.project.rootPath
                  : "The project workspace folder will be removed from disk."}
              </span>
            </label>
            <Switch
              id="delete-project-files"
              checked={deleteProjectFiles}
              onCheckedChange={setDeleteProjectFiles}
              aria-label="Delete project files and sandbox folder"
            />
          </div>
        ) : null}
        <DialogFooter className="flex-wrap gap-2 sm:justify-end">
          <Button
            type="button"
            variant="ghost"
            onClick={() => setConfirmingDelete(null)}
          >
            {t("common.cancel")}
          </Button>
          <Button
            type="button"
            variant="destructive"
            onClick={() => void commitDelete()}
          >
            {confirmingDelete?.kind === "project" && deleteProjectFiles
              ? "Delete all"
              : t("common.delete")}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
    <Dialog
      open={renamingTarget !== null && renamingTarget.kind !== "chat"}
      onOpenChange={(open) => {
        if (!open) setRenamingTarget(null);
      }}
    >
      <DialogContent className="corner-squircle dialog-soft-surface sm:max-w-md">
        <DialogHeader>
          <DialogTitle>
            {renamingTarget?.kind === "run"
              ? t("shell.dialog.renameRun.title")
              : renamingTarget?.kind === "project"
                ? "Rename project"
                : t("shell.dialog.renameChat.title")}
          </DialogTitle>
        </DialogHeader>
        <Input
          value={renameDraft}
          onChange={(event) => setRenameDraft(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === "Enter") {
              event.preventDefault();
              void commitRename();
            }
          }}
          autoFocus
          maxLength={120}
          placeholder={
            renamingTarget?.kind === "run"
              ? t("shell.dialog.renameRun.placeholder")
              : renamingTarget?.kind === "project"
                ? "Project name"
                : t("shell.dialog.renameChat.placeholder")
          }
          aria-label={
            renamingTarget?.kind === "run"
              ? t("shell.dialog.renameRun.placeholder")
              : renamingTarget?.kind === "project"
                ? "Project name"
                : t("shell.dialog.renameChat.placeholder")
          }
          className="focus-visible:border-input focus-visible:ring-0"
        />
        <DialogFooter className="flex-wrap gap-2 sm:justify-end">
          <Button
            type="button"
            variant="ghost"
            onClick={() => setRenamingTarget(null)}
          >
            {t("common.cancel")}
          </Button>
          <Button
            type="button"
            onClick={() => void commitRename()}
            disabled={!renameDirty}
          >
            {t("common.save")}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
    <Dialog
      open={creatingProject}
      onOpenChange={(open) => {
        setCreatingProject(open);
        if (!open) {
          setProjectNameDraft("");
          setProjectCreateMoveTarget(null);
        }
      }}
    >
      <DialogContent className="corner-squircle dialog-soft-surface sm:max-w-md">
        <DialogHeader>
          <DialogTitle>
            {projectCreateMoveTarget ? "Move to new project" : "New project"}
          </DialogTitle>
        </DialogHeader>
        <Input
          value={projectNameDraft}
          onChange={(event) => setProjectNameDraft(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === "Enter") {
              event.preventDefault();
              void commitCreateProject();
            }
          }}
          autoFocus
          maxLength={120}
          placeholder="Project name"
          aria-label="Project name"
          className="focus-visible:border-input focus-visible:ring-0"
        />
        <DialogFooter className="flex-wrap gap-2 sm:justify-end">
          <Button
            type="button"
            variant="ghost"
            onClick={() => {
              setCreatingProject(false);
              setProjectCreateMoveTarget(null);
            }}
          >
            Cancel
          </Button>
          <Button
            type="button"
            onClick={() => void commitCreateProject()}
            disabled={!projectNameDraft.trim()}
          >
            Create
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
    </>
  );
}
