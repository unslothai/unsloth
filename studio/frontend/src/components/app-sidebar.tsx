import { resolveNavRowState } from "@/components/nav-row-state";
import { ShutdownDialog } from "@/components/shutdown-dialog";
import {
  DesktopTitlebarNavigation,
  getClientPlatform,
  shouldUseCustomWindowTitlebar,
  shouldUseNativeMacWindowTitlebar,
} from "@/components/tauri/window-titlebar";
import { useAnimatedThemeToggle } from "@/components/ui/animated-theme-toggler";
import { Button } from "@/components/ui/button";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
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
import { Input } from "@/components/ui/input";
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
import { Spinner } from "@/components/ui/spinner";
import { Switch } from "@/components/ui/switch";
import { Tooltip, TooltipContent } from "@/components/ui/tooltip";
import { PRODUCT_NAME } from "@/config/branding";
import {
  FEATURE_AGENTS_NAV,
  FEATURE_API_MONITOR,
  FEATURE_AUDIO,
  FEATURE_EXPORT,
  FEATURE_FILES_NAV,
  FEATURE_IMAGES,
  FEATURE_MANAGEMENT_NAV,
  FEATURE_MEMORY_NAV,
  FEATURE_PROJECTS,
  FEATURE_RECIPES,
  FEATURE_SEARCH_NAV,
  FEATURE_TRAIN,
  FEATURE_VIDEO,
} from "@/config/disabled-features";
import { fetchDeviceType, usePlatformStore } from "@/config/env";
import { getProductCapability } from "@/config/platform-capabilities";
import { clearAuthTokens, logout } from "@/features/auth";
import {
  CONVERSATION_MARKDOWN_FORMAT,
  CONVERSATION_MARKDOWN_LABEL,
  ChatSearchDialog,
  type ProjectRecord,
  type SidebarItem,
  archiveChatItem,
  clearNewChatDraft,
  deleteChatItem,
  deleteChatProject,
  listStoredChatThreads,
  moveChatItemToProject,
  notifyChatHistoryUpdated,
  renameChatItem,
  renameChatProject,
  useChatPreferencesStore,
  useChatProjects,
  useChatRuntimeStore,
  useChatSearchStore,
  useChatSidebarItems,
  usePinnedChatsStore,
  usePinnedProjectsStore,
  usePromptQueueUI,
} from "@/features/chat";
import { NewProjectDialog } from "@/features/chat/components/new-project-dialog";
import { useExportRuntimeStore } from "@/features/export";
// Deep imports on purpose: the Images index re-exports ImagesPage, which would undo its code split.
/* eslint-disable no-restricted-imports */
import {
  isWorkflowEnabled,
  useImageWorkflowStore,
} from "@/features/images/stores/image-workflow-store";
import { WORKFLOW_TABS, type WorkflowId } from "@/features/images/workflows";
import { UserAvatar, useEffectiveProfile } from "@/features/profile";
import {
  useAppearanceCustomStore,
  useSettingsDialogStore,
} from "@/features/settings";
import type { SidebarNavItemId } from "@/features/settings";
import { TOUR_OPEN_EVENT } from "@/features/tour";
import {
  deleteTrainingRun,
  emitTrainingRunDeleted,
  emitTrainingRunUpdated,
  getTrainingRunDisplayTitle,
  isTrainingStartPending,
  removeTrainingUnloadGuard,
  renameTrainingRun,
  useTrainingCompletionWatch,
  useTrainingHistorySidebarItems,
  useTrainingRuntimeStore,
} from "@/features/training";
import type { TrainingRunSummary } from "@/features/training";
import { useWebUpdateCheck } from "@/hooks/use-web-update-check";
import { type TranslationKey, translate, useT } from "@/i18n";
import { isTauri } from "@/lib/api-base";
import { TestTubeOutlineIcon } from "@/lib/hugeicons-derived";
import { isDownloadCancelled } from "@/lib/native-files";
import { toast } from "@/lib/toast";
/* eslint-enable no-restricted-imports */
import { cn } from "@/lib/utils";
import {
  Archive03Icon,
  ArrowRight02Icon,
  AudioWave01Icon,
  BadgeInfoIcon,
  BotIcon,
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
  FlimSlateIcon,
  Folder01Icon,
  FolderAddIcon,
  FolderExportIcon,
  Globe02Icon,
  HelpCircleIcon,
  Image03Icon,
  LayoutAlignLeftIcon,
  Logout05Icon,
  Message01Icon,
  MoreHorizontalIcon,
  MoreVerticalIcon,
  PaintBrush02Icon,
  PencilEdit02Icon,
  PinIcon,
  PinOffIcon,
  PlusSignIcon,
  PowerIcon,
  Search01Icon,
  Settings02Icon,
  Sun03Icon,
  UserIcon,
  type ZapIcon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import {
  Link,
  useNavigate,
  useRouter,
  useRouterState,
} from "@tanstack/react-router";
import { ChevronDown, Moon } from "lucide-react";
import { Tooltip as TooltipPrimitive } from "radix-ui";
import {
  Fragment,
  type ReactNode,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";

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

// Optional user-menu shortcuts that jump to a settings tab; the id is the tab id.
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

// One navigable row, rendered as a NavItem or a MoreMenuItem depending on its pin state.
type NavRowDef = {
  icon: typeof ZapIcon;
  label: string;
  active: boolean;
  disabled?: boolean;
  tooltip?: string;
  spinner?: boolean;
  // The capability that decides `disabled` has not been measured yet. A row in this state is
  // neither enabled-looking nor blacked out: resolveNavRowState renders it with the spinner.
  pending?: boolean;
  // What that spinning row says on hover; detection can take minutes on a cold host.
  pendingTooltip?: string;
  badge?: string;
  onClick: () => void;
  onIntent?: () => void;
  className?: string;
  children?: ReactNode;
};

type ConversationExportFormat =
  | "raw-jsonl"
  | "csv"
  | "sharegpt-jsonl"
  | typeof CONVERSATION_MARKDOWN_FORMAT;

// A pinned project shows this many recent chats before "Show more".
const PINNED_PROJECT_CHAT_LIMIT = 4;

const CHAT_EXPORT_OPTIONS: Array<{
  label: string;
  format: ConversationExportFormat;
}> = [
  { label: "Raw JSONL", format: "raw-jsonl" },
  { label: "CSV", format: "csv" },
  { label: "ShareGPT JSONL", format: "sharegpt-jsonl" },
  { label: CONVERSATION_MARKDOWN_LABEL, format: CONVERSATION_MARKDOWN_FORMAT },
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
    case CONVERSATION_MARKDOWN_FORMAT:
      return exports.exportConversationMarkdown(threadId);
    default: {
      // Exhaustive: a new format is a build error, not a menu item that does nothing.
      const unhandled: never = format;
      throw new Error(`Unhandled export format: ${String(unhandled)}`);
    }
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

// "New" pill for recent tabs. Same recipe as the brand "beta" badge.
function NavBadge({ label, className }: { label: string; className?: string }) {
  return (
    <span
      className={cn(
        "nav-badge inline-flex shrink-0 items-center justify-center rounded-full border border-nav-beta-border px-[5px] pt-[3px] pb-[2px] text-[calc(0.5rem*var(--ui-font-scale,1))] font-medium uppercase leading-none tracking-[0.04em] text-nav-fg-muted antialiased subpixel-antialiased shadow-[0_1px_2px_rgba(0,0,0,0.06)] dark:shadow-[0_1px_2px_rgba(0,0,0,0.35)]",
        className,
      )}
    >
      {label}
    </span>
  );
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
  alwaysTooltip,
  onIntent,
  badge,
  overlay,
  testId,
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
  // Stable hook for the UI smokes, which assert a row spins rather than greys out
  // while its capability verdict is still unmeasured.
  testId?: string;
  // Overrides the hover tooltip; explains why a disabled item is greyed out.
  tooltip?: string;
  // Show that tooltip on the expanded row too, not just the collapsed rail where it
  // stands in for the hidden label.
  alwaysTooltip?: boolean;
  // Trailing "New" pill text.
  badge?: string;
  // Absolutely-positioned extras over the row, e.g. a disclosure chevron.
  overlay?: ReactNode;
}) {
  return (
    <SidebarMenuItem className={className}>
      <div className="relative">
        <SidebarMenuButton
          tooltip={tooltip ?? label}
          alwaysTooltip={alwaysTooltip && Boolean(tooltip)}
          disabled={disabled}
          onClick={onClick}
          onPointerEnter={disabled ? undefined : onIntent}
          onFocus={disabled ? undefined : onIntent}
          isActive={active}
          data-tour={dataTour}
          data-testid={testId}
          data-spinner={spinner ? "true" : undefined}
          className="sidebar-nav-btn h-[33px] rounded-full gap-[8.5px] pl-3 pr-2.5 font-medium group-data-[collapsible=icon]:px-2.5 group-data-[collapsible=icon]:!w-[32px] group-data-[collapsible=icon]:mx-auto"
        >
          <HugeiconsIcon
            icon={icon}
            strokeWidth={1.75}
            className="size-icon! shrink-0 translate-x-0.5 group-hover/menu-button:animate-icon-pop"
          />
          <span className="text-ui-14p5 leading-ui-19 tracking-nav">
            {label}
          </span>
          {badge && (
            <NavBadge
              label={badge}
              className="ml-auto group-data-[collapsible=icon]:hidden"
            />
          )}
          {spinner && (
            // mr-1.5 over the row's pr-2.5 = 16px, matching the chat rows' pr-4: one spinner column.
            <Spinner className="ml-auto mr-1.5 size-3.5 shrink-0 text-muted-foreground group-data-[collapsible=icon]:hidden" />
          )}
        </SidebarMenuButton>
        {spinner && (
          // Collapsed (icon-only) rail: small spinner badge over the icon corner.
          <Spinner className="pointer-events-none absolute right-1 top-1 hidden size-2.5 text-muted-foreground group-data-[collapsible=icon]:block" />
        )}
        {overlay}
      </div>
      {children}
    </SidebarMenuItem>
  );
}

function getSidebarItemThreadIds(item: SidebarItem) {
  return item.threadIds?.length ? item.threadIds : [item.id];
}

const WORKFLOW_UNAVAILABLE = "The loaded model cannot do this";

// Re-read cadences for the hardware verdict below. An unmeasured verdict holds Train and Video
// on a spinner, so it is re-read sooner than the background MLX self-heal check, which only
// pays off after the user has repaired an install. A re-read outstanding longer than the stall
// window is given up on rather than latching the poll off: the backend it is waiting on is the
// one case this has to recover from.
const VERDICT_UNKNOWN_POLL_MS = 3000;
const SELF_HEAL_POLL_MS = 15000;
const VERDICT_POLL_STALL_MS = 30000;

/** One workflow in the list under the Images row. */
function WorkflowChoice({
  tab,
  active,
  enabled,
  onSelect,
}: {
  tab: (typeof WORKFLOW_TABS)[number];
  active: boolean;
  enabled: boolean;
  onSelect: () => void;
}) {
  return (
    <button
      type="button"
      disabled={!enabled}
      title={enabled ? undefined : WORKFLOW_UNAVAILABLE}
      onClick={onSelect}
      // Weight and colour come from the nav rows above; only the size sets a submenu apart.
      className={cn(
        "flex h-[29px] w-full items-center gap-2 rounded-full pl-3 pr-2.5 text-left font-medium text-nav-fg transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring",
        active ? "bg-sidebar-accent" : "hover:bg-sidebar-accent/60",
        !enabled && "opacity-40 hover:bg-transparent",
      )}
    >
      <HugeiconsIcon
        icon={tab.icon}
        strokeWidth={1.75}
        className="size-4 shrink-0"
      />
      <span className="min-w-0 flex-1 truncate text-ui-13 tracking-nav">
        {tab.label}
      </span>
    </button>
  );
}

/** Expands the workflow list on rows that do not list it outright, i.e. off the Images page. */
function ImagesNavDisclosure() {
  const expanded = useImageWorkflowStore((s) => s.navExpanded);
  const setExpanded = useImageWorkflowStore((s) => s.setNavExpanded);
  return (
    // Row action, so it gets the shared hover circle. Shown on row hover, kept while open.
    <button
      type="button"
      aria-label={expanded ? "Hide workflows" : "Show workflows"}
      aria-expanded={expanded}
      onClick={(e) => {
        e.stopPropagation();
        setExpanded(!expanded);
      }}
      className={cn(
        "sidebar-row-action group-hover/images-item:opacity-100 group-hover/images-item:pointer-events-auto focus-visible:opacity-100 focus-visible:pointer-events-auto",
        expanded && "is-disclosure-open",
      )}
    >
      <span className="sidebar-row-action-glyph">
        <ChevronDown
          className={cn(
            "size-3.5 transition-transform duration-200",
            !expanded && "-rotate-90",
          )}
        />
      </span>
    </button>
  );
}

/**
 * The Images workflows, listed under their nav row. Open by default on the Images page, since
 * they are that page's switcher; elsewhere they stay folded until the row's chevron asks.
 */
function ImagesWorkflowList({
  active,
  collapsed,
  onPick,
}: {
  active: boolean;
  collapsed: boolean;
  onPick: (id: WorkflowId) => void;
}) {
  const workflow = useImageWorkflowStore((s) => s.workflow);
  const supported = useImageWorkflowStore((s) => s.supported);
  const pageMode = useImageWorkflowStore((s) => s.pageMode);
  const expanded = useImageWorkflowStore((s) => s.navExpanded);
  if (collapsed) return null;
  if (active ? pageMode === "train" : !expanded) return null;
  // Nothing here is current unless the page is actually showing a workflow.
  const current = active && pageMode === "create" ? workflow : null;
  return (
    <div className="mt-0.5 flex flex-col gap-px pl-5">
      {WORKFLOW_TABS.map((tab) => (
        <WorkflowChoice
          key={tab.id}
          tab={tab}
          active={current === tab.id}
          enabled={isWorkflowEnabled(tab.id, supported)}
          onSelect={() => onPick(tab.id)}
        />
      ))}
    </div>
  );
}

// A NavItem's affordances in dropdown-item form, for the "More" flyout.
function MoreMenuItem({
  icon,
  label,
  active,
  disabled,
  tooltip,
  badge,
  spinner,
  onSelect,
  onIntent,
}: {
  icon: typeof ZapIcon;
  label: string;
  active: boolean;
  disabled?: boolean;
  tooltip?: string;
  badge?: string;
  spinner?: boolean;
  onSelect: () => void;
  onIntent?: () => void;
}) {
  return (
    <DropdownMenuItem
      disabled={disabled}
      // Whenever there is one: gated on `disabled` it dropped the tooltip of a row that is
      // still being measured, which is enabled and has something to say.
      title={tooltip}
      onSelect={onSelect}
      onPointerEnter={disabled ? undefined : onIntent}
      onFocus={disabled ? undefined : onIntent}
      className={cn(active && "bg-accent/60")}
    >
      <HugeiconsIcon icon={icon} strokeWidth={1.75} />
      <span className="min-w-0 flex-1 truncate">{label}</span>
      {badge && <NavBadge label={badge} />}
      {spinner && (
        <Spinner className="size-3.5 shrink-0 text-muted-foreground" />
      )}
    </DropdownMenuItem>
  );
}

export function AppSidebar() {
  const t = useT();
  const { isDark, toggleTheme, anchorRef } = useAnimatedThemeToggle();
  const sidebarMenu = useAppearanceCustomStore(
    (s) => s.customization.sidebarMenu,
  );
  const sidebarNav = useAppearanceCustomStore(
    (s) => s.customization.sidebarNav,
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
  const {
    pinned,
    togglePinned,
    isMobile,
    setOpenMobile,
    state: sidebarState,
  } = useSidebar();
  const navigate = useNavigate();
  const router = useRouter();
  const imagesPageMode = useImageWorkflowStore((s) => s.pageMode);

  // `webUpdate` is non-null only when the installed (PyPI) version is behind the latest release.
  const { status: webUpdate } = useWebUpdateCheck();
  const showUpdateCard = Boolean(webUpdate);
  const updateVersion = webUpdate?.latestVersion ?? null;

  const closeMobileIfOpen = () => {
    if (isMobile) setOpenMobile(false);
  };

  const chatOnly = usePlatformStore((s) => s.isChatOnly());
  const chatOnlyReason = usePlatformStore((s) => s.chatOnlyReason);
  const detectionDeferred = usePlatformStore((s) => s.detectionDeferred);
  const platformDeviceType = usePlatformStore((s) => s.deviceType);
  // Until /api/health answers, `chatOnly` is the browser-platform guess, so every Mac painted
  // Train and Video blacked out on load and only recovered once the backend reported. Gate the
  // rows on a measured verdict and let them spin until it lands.
  const capabilitiesUnknown = usePlatformStore((s) => s.capabilitiesUnknown());
  const chatOnlyMeasured = chatOnly && !capabilitiesUnknown;
  // Explain a greyed-out Train (chat-only host) on hover. Export stays navigable so its page
  // can show a precise reason.
  const trainDisabledHint: string | undefined = chatOnlyMeasured
    ? chatOnlyReason === "mlx_unavailable"
      ? "Training needs MLX. Run `unsloth studio update` to enable Train."
      : chatOnlyReason === "intel_mac"
        ? "Training needs Apple Silicon or a GPU. Intel Macs are chat-only."
        : chatOnlyReason === "no_gpu"
          ? "Training needs an NVIDIA or AMD GPU."
          : undefined
    : undefined;
  // Video's hint is derived the same way. It used to be hardcoded to "needs an NVIDIA or AMD
  // GPU", which is wrong on an Apple Silicon host: video has no Apple path at all, so the row
  // says so rather than pointing at hardware the user cannot add.
  const videoDisabledHint: string | undefined = chatOnlyMeasured
    ? platformDeviceType === "mac"
      ? "Video generation on macOS is coming soon."
      : chatOnlyReason === "no_gpu"
        ? "Video generation needs an NVIDIA or AMD GPU."
        : undefined
    : undefined;

  // Two things can change the verdict after the first /api/health. The backend MLX self-heal
  // (utils/mlx_repair) can reinstall MLX and flip chat_only false without a restart, and
  // detection can land after fetchDeviceType gave up waiting for it. The platform store cached
  // that first reply, so re-poll for both; the guard below stops it once neither applies.
  useEffect(() => {
    // Also while deferred: under the kill switch health settles nothing, so a GPU host would stay chat-only.
    const selfHealSettled =
      !chatOnly || (chatOnlyReason !== "mlx_unavailable" && !detectionDeferred);
    // And on any platform while the verdict itself is out. fetchDeviceType spends its bounded
    // wait at most once per page load, so a host that detects slower than that keeps the
    // provisional reply, and nothing else is scheduled to re-read it: the rows above would spin
    // and /studio would hold its loading panel for the rest of the session. This is the only
    // recovery poll in the app, and the sidebar is mounted on every route that gates on the
    // verdict (studio-page and video-page read the same store, so they recover with it).
    if (selfHealSettled && !capabilitiesUnknown) return;
    let pollingSince = 0;
    // Which read currently owns the guard. A read that outlived the stall window is replaced,
    // and the replacement takes the guard with it; without an owner the abandoned read's
    // `finally` would clear a guard it no longer holds and let the next tick stack another
    // forced read onto the slow backend, every interval, which is the pile-up this prevents.
    let pollOwner = 0;
    const id = window.setInterval(
      () => {
        // A backend still importing torch answers slowly, so skip while a re-read is outstanding
        // rather than stacking them against it. Bounded, or a request that never settles would
        // hold the poll off for good.
        if (pollingSince && Date.now() - pollingSince < VERDICT_POLL_STALL_MS)
          return;
        const owned = ++pollOwner;
        pollingSince = Date.now();
        void fetchDeviceType({ force: true })
          .catch(() => undefined)
          .finally(() => {
            if (owned === pollOwner) pollingSince = 0;
          });
      },
      capabilitiesUnknown ? VERDICT_UNKNOWN_POLL_MS : SELF_HEAL_POLL_MS,
    );
    return () => window.clearInterval(id);
  }, [capabilitiesUnknown, chatOnly, chatOnlyReason, detectionDeferred]);

  const [shutdownOpen, setShutdownOpen] = useState(false);

  const isChatRoute = pathname.startsWith("/chat");
  const isStudioRoute =
    pathname === "/studio" || pathname.startsWith("/studio/");
  const [chatOpen, setChatOpen] = useState(true);

  // "More" flyout. Opens on click or hover; close is delayed so the pointer can cross the gap to the panel.
  const [moreOpen, setMoreOpen] = useState(false);
  const moreCloseTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const openMore = useCallback(() => {
    if (moreCloseTimer.current) {
      clearTimeout(moreCloseTimer.current);
      moreCloseTimer.current = null;
    }
    setMoreOpen(true);
  }, []);
  const closeMoreSoon = useCallback(() => {
    if (moreCloseTimer.current) clearTimeout(moreCloseTimer.current);
    moreCloseTimer.current = setTimeout(() => setMoreOpen(false), 180);
  }, []);
  useEffect(
    () => () => {
      if (moreCloseTimer.current) clearTimeout(moreCloseTimer.current);
    },
    [],
  );
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
  // Bottom fade hides at the very bottom / for short lists so the last row isn't washed out.
  const [canScrollDown, setCanScrollDown] = useState(false);
  // Rail width: 0 where scrollbars overlay (macOS default) or the list fits,
  // the platform's thin rail where they are classic. Only rows inside the
  // scroller lose it, so the rows outside pad by it to keep one edge. Written
  // to the DOM, and only on a change: state here would loop (React #185).
  const railWidthRef = useRef<number | null>(null);
  const measureScrollRail = useCallback((el: HTMLDivElement) => {
    const rail = el.offsetWidth - el.clientWidth;
    if (rail === railWidthRef.current) return;
    railWidthRef.current = rail;
    el.parentElement?.style.setProperty("--sidebar-rail", `${rail}px`);
  }, []);

  // A callback ref, not an effect: the mobile Sheet unmounts its subtree on
  // close and the breakpoint swaps it for the desktop one, so the scroller is a
  // new node each time and an effect keyed on a stable callback never re-runs.
  // Still runs before paint.
  const railObserverRef = useRef<ResizeObserver | null>(null);
  const attachScroller = useCallback(
    (el: HTMLDivElement | null) => {
      railObserverRef.current?.disconnect();
      railObserverRef.current = null;
      scrollRef.current = el;
      // Per node: a new parent has no variable yet even at the same rail, and
      // the cache would otherwise skip the write.
      railWidthRef.current = null;
      if (!el) return;
      measureScrollRail(el);
      // Watch the box, not renders: the Images disclosure and the project
      // toggles change the row count without rendering this component, and a
      // scrollbar appearing shrinks the content box by its own width. Safe
      // where the earlier observer was not: it writes a variable, never state,
      // so there is no render to feed back (React #185).
      const observer = new ResizeObserver(() => measureScrollRail(el));
      observer.observe(el);
      railObserverRef.current = observer;
    },
    [measureScrollRail],
  );

  // Driven only from onScroll + a content-change effect below. No
  // ResizeObserver: its callback-driven setState caused a render loop (React
  // #185). Both setters bail out when unchanged, so neither path can loop.
  const syncScrollState = useCallback((el: HTMLDivElement) => {
    const nextScrolled = el.scrollTop > 0;
    setScrolled((prev) => (prev === nextScrolled ? prev : nextScrolled));
    const nextCanScrollDown =
      el.scrollHeight - el.scrollTop - el.clientHeight > 1;
    setCanScrollDown((prev) =>
      prev === nextCanScrollDown ? prev : nextCanScrollDown,
    );
  }, []);

  const isRecipesRoute = pathname.startsWith("/data-recipes");
  const isExportRoute =
    pathname === "/export" || pathname.startsWith("/export/");
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
  // Projects the user pinned, in pin order; the section appears once at least one is pinned.
  const pinnedProjectIds = usePinnedProjectsStore((s) => s.pinnedIds);
  const unpinProject = usePinnedProjectsStore((s) => s.unpin);
  const pinnedProjectRecords = useMemo(() => {
    const byId = new Map(projects.map((p) => [p.id, p]));
    return pinnedProjectIds
      .map((id) => byId.get(id))
      .filter((p): p is ProjectRecord => Boolean(p));
  }, [projects, pinnedProjectIds]);
  // Pinned chats, in pin order. Includes chats inside a project: pinning promotes a chat here
  // and removes it from the project's nested list, so it never shows twice.
  const pinnedChatItems = useMemo(() => {
    const byId = new Map(allChatItems.map((item) => [item.id, item]));
    return pinnedIds
      .map((id) => byId.get(id))
      .filter((item): item is SidebarItem => Boolean(item));
  }, [allChatItems, pinnedIds]);
  // A pinned project nests its recent chats; pinned chats are excluded (they render above).
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
  // Default expanded; the row toggles this. Show-more reveals chats past the limit.
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
  // The whole map, so each row can show its own spinner.
  const runningByThreadId = useChatRuntimeStore((s) => s.runningByThreadId);
  // Rows, not raw thread ids: a compare conversation runs two pane threads but is one row.
  const runningChatCount = useMemo(() => {
    const running = new Set(
      Object.entries(runningByThreadId)
        .filter(([, on]) => on)
        .map(([id]) => id),
    );
    if (running.size === 0) return 0;
    let rows = 0;
    for (const item of allChatItems) {
      const ids = item.type === "compare" ? (item.threadIds ?? []) : [item.id];
      let claimed = false;
      for (const id of ids) {
        if (running.delete(id)) claimed = true;
      }
      if (claimed) rows += 1;
    }
    // Anything left belongs to no known row (a first turn mid-persist); count it as one.
    return rows + running.size;
  }, [runningByThreadId, allChatItems]);
  const anyChatRunning = runningChatCount > 0;
  // Where "Return to Chat" lands: the newest running chat, not an empty New Chat draft (map
  // insertion order is start order). Compare rows resolve back to the pair id /chat expects.
  const runningTarget = useMemo(() => {
    const ids = Object.entries(runningByThreadId)
      .filter(([, on]) => on)
      .map(([id]) => id);
    const id = ids.length > 0 ? ids[ids.length - 1] : null;
    if (!id) return null;
    const pair = allChatItems.find(
      (item) => item.type === "compare" && (item.threadIds ?? []).includes(id),
    );
    return pair
      ? { id: pair.id, compare: true as const }
      : { id, compare: false as const };
  }, [runningByThreadId, allChatItems]);
  const activeThreadId = isChatRoute
    ? ((search.thread as string | undefined) ??
      (search.compare as string | undefined) ??
      storeThreadId ??
      undefined)
    : undefined;
  const queueByThreadId = usePromptQueueUI((s) => s.byThreadId);
  const [unreadThreadIds, setUnreadThreadIds] = useState<Set<string>>(
    () => new Set(),
  );
  const previousRunningByThreadIdRef = useRef<Record<string, boolean>>({});
  const activeVisibleThreadIds = useMemo(() => {
    if (!activeThreadId) {
      return [];
    }
    const activeItem = allChatItems.find((item) => item.id === activeThreadId);
    return activeItem ? getSidebarItemThreadIds(activeItem) : [activeThreadId];
  }, [activeThreadId, allChatItems]);
  const activeVisibleThreadIdKey = activeVisibleThreadIds.join("\n");

  useEffect(() => {
    const activeVisibleThreadIdSet = new Set(
      activeVisibleThreadIdKey ? activeVisibleThreadIdKey.split("\n") : [],
    );
    const previousRunningByThreadId = previousRunningByThreadIdRef.current;
    const completedThreadIds: string[] = [];

    for (const [threadId, wasRunning] of Object.entries(
      previousRunningByThreadId,
    )) {
      if (
        wasRunning &&
        !runningByThreadId[threadId] &&
        !activeVisibleThreadIdSet.has(threadId)
      ) {
        completedThreadIds.push(threadId);
      }
    }

    if (completedThreadIds.length > 0 || activeVisibleThreadIdSet.size > 0) {
      queueMicrotask(() => {
        setUnreadThreadIds((current) => {
          let next: Set<string> | null = null;
          const mutable = () => {
            next ??= new Set(current);
            return next;
          };

          for (const threadId of completedThreadIds) {
            if (!current.has(threadId)) {
              mutable().add(threadId);
            }
          }

          for (const threadId of activeVisibleThreadIdSet) {
            if (current.has(threadId)) {
              mutable().delete(threadId);
            }
          }

          return next ?? current;
        });
      });
    }
    previousRunningByThreadIdRef.current = runningByThreadId;
  }, [activeVisibleThreadIdKey, runningByThreadId]);

  // Training runs surface as sidebar "Recents" on Train/Recipes/Export, else chat recents.
  // With Train switched off, those rows would only navigate to a redirecting route, so drop them.
  const trainingRecentsRoute =
    FEATURE_TRAIN && (isStudioRoute || isRecipesRoute || isExportRoute);
  const { items: runItems } = useTrainingHistorySidebarItems(
    !chatOnly && trainingRecentsRoute,
  );
  const showTrainingRecents =
    !chatOnly && trainingRecentsRoute && runItems.length > 0;
  const activeJobId = useTrainingRuntimeStore((s) => s.jobId);
  const currentRunViewActive = useTrainingRuntimeStore(
    (s) => s.currentRunViewActive,
  );
  const selectedHistoryRunId = useTrainingRuntimeStore(
    (s) => s.selectedHistoryRunId,
  );
  const setSelectedHistoryRunId = useTrainingRuntimeStore(
    (s) => s.setSelectedHistoryRunId,
  );
  // Running or starting up. Drives the Train spinner + New Chat / Return to Chat swap.
  const trainingInProgress = useTrainingRuntimeStore(isTrainingStartPending);
  // Export runs in the background; reflect it on the Export nav item from any tab.
  const exportInProgress = useExportRuntimeStore((s) => s.isExporting);
  // On any non-chat tab, offer a way back to the live chat instead of starting a new one
  // whenever a chat is running or its thread is active, or a training / export is in progress.
  const showReturnToChat =
    !isChatRoute &&
    (trainingInProgress ||
      exportInProgress ||
      anyChatRunning ||
      storeThreadId != null);
  // The Train-page status poll doesn't run off-route; keep state fresh so the spinner clears.
  useTrainingCompletionWatch();

  // Recompute bottom-fade on mount and whenever list height can change: onScroll never fires
  // for short, non-scrolling lists. Guarded setState below can't loop.
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
    runsOpen,
    pinnedOpen,
    isStudioRoute,
    // The update card grows the footer, so the scroll area shrinks under it.
    showUpdateCard,
  ]);

  // Resizing changes clientHeight without firing onScroll, so the fade would
  // stay hidden while rows are still clipped. Window events only: no element
  // observer, so this can't feed back into the loop that caused React #185.
  useEffect(() => {
    const onResize = () => {
      const el = scrollRef.current;
      if (el) syncScrollState(el);
    };
    window.addEventListener("resize", onResize);
    return () => window.removeEventListener("resize", onResize);
  }, [syncScrollState]);

  const chatDisabled = trainingInProgress;
  const usesDesktopTitlebar = usesCustomTitlebar || usesNativeMacTitlebar;

  // One box for every row pill, so a hover pill has the same edges wherever it
  // lands. Rows outside the list scroller add the rail width it does not lose,
  // so both end on the same edge whether or not the scrollbar takes space.
  // Logical sides, since the rail is on the inline end and moves under rtl.
  const rowPadding = usesDesktopTitlebar
    ? "ps-[5px] pe-[calc(var(--sidebar-rail,0px)+5px)]"
    : "ps-1.5 pe-[calc(var(--sidebar-rail,0px)+6px)]";

  // Inside it the rail already sits in that space, so this is just the gap
  // between a pill and the scrollbar, matched to the gap on the other side.
  const scrollRowPadding = usesDesktopTitlebar ? "px-[5px]" : "px-1.5";

  // One definition per row, so pinned rows and the flyout can't drift apart.
  const navRows: Record<SidebarNavItemId, NavRowDef> = {
    projects: {
      icon: Folder01Icon,
      label: t("shell.navigation.projects"),
      active: pathname === "/projects" || pathname.startsWith("/projects/"),
      onClick: () => {
        navigate({ to: "/projects" });
        closeMobileIfOpen();
      },
      onIntent: () => {
        preloadSilently(router.preloadRoute({ to: "/projects" }));
      },
      className: "group/projects-item relative",
      // The inline "new project" affordance only fits a real row.
      children: (
        <button
          type="button"
          aria-label="New project"
          onClick={(e) => {
            e.stopPropagation();
            // NewProjectDialog owns its own name field, so opening it is just the move target plus the open flag.
            setProjectCreateMoveTarget(null);
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
      ),
    },
    hub: {
      icon: DashboardCircleIcon,
      label: t("shell.navigation.hub"),
      active: pathname === "/hub" || pathname.startsWith("/hub/"),
      onClick: () => {
        navigate({ to: "/hub" });
        closeMobileIfOpen();
      },
      onIntent: () => {
        preloadSilently(router.preloadRoute({ to: "/hub" }));
      },
    },
    agents: {
      icon: BotIcon,
      label: t("shell.navigation.agents"),
      active: pathname === "/agents" || pathname.startsWith("/agents/"),
      disabled: !getProductCapability("agents").available,
      tooltip:
        getProductCapability("agents").reason ??
        t("shell.navigation.agentsUnavailable"),
      onClick: () => {
        navigate({ to: "/agents" });
        closeMobileIfOpen();
      },
      onIntent: () => {
        preloadSilently(router.preloadRoute({ to: "/agents" }));
      },
    },
    files: {
      icon: Folder01Icon,
      label: t("shell.navigation.files"),
      active: pathname === "/files" || pathname.startsWith("/files/"),
      disabled: !getProductCapability("files").available,
      tooltip: getProductCapability("files").reason ?? undefined,
      onClick: () => {
        navigate({ to: "/files" });
        closeMobileIfOpen();
      },
      onIntent: () => {
        preloadSilently(router.preloadRoute({ to: "/files" }));
      },
    },
    memory: {
      icon: CloudIcon,
      label: t("shell.navigation.memory"),
      active: pathname === "/memory" || pathname.startsWith("/memory/"),
      disabled: !getProductCapability("memory").available,
      tooltip: getProductCapability("memory").reason ?? undefined,
      onClick: () => {
        navigate({ to: "/memory" });
        closeMobileIfOpen();
      },
      onIntent: () => {
        preloadSilently(router.preloadRoute({ to: "/memory" }));
      },
    },
    search: {
      icon: Search01Icon,
      label: t("shell.navigation.searchApps"),
      active: pathname === "/search" || pathname.startsWith("/search/"),
      disabled: !getProductCapability("search").available,
      tooltip: getProductCapability("search").reason ?? undefined,
      onClick: () => {
        navigate({ to: "/search" });
        closeMobileIfOpen();
      },
      onIntent: () => {
        preloadSilently(router.preloadRoute({ to: "/search" }));
      },
    },
    management: {
      icon: Settings02Icon,
      label: t("shell.navigation.management"),
      active: pathname === "/management" || pathname.startsWith("/management/"),
      disabled: !getProductCapability("management").available,
      tooltip: getProductCapability("management").reason ?? undefined,
      onClick: () => {
        navigate({ to: "/management" });
        closeMobileIfOpen();
      },
      onIntent: () => {
        preloadSilently(router.preloadRoute({ to: "/management" }));
      },
    },
    images: {
      icon: Image03Icon,
      label: t("shell.navigation.images"),
      // No "New" pill: the row's trailing slot holds the workflow disclosure instead.
      active: pathname === "/images" || pathname.startsWith("/images/"),
      onClick: () => {
        navigate({ to: "/images" });
        closeMobileIfOpen();
      },
      onIntent: () => {
        preloadSilently(router.preloadRoute({ to: "/images" }));
      },
    },
    train: {
      icon: TestTubeOutlineIcon,
      label: t("shell.navigation.train"),
      active: pathname === "/studio" || pathname.startsWith("/studio/"),
      disabled: chatOnlyMeasured,
      tooltip: trainDisabledHint,
      spinner: trainingInProgress,
      pending: capabilitiesUnknown,
      pendingTooltip: t("shell.navigation.trainChecking"),
      onClick: () => {
        if (chatOnlyMeasured) return;
        navigate({ to: "/studio" });
        closeMobileIfOpen();
      },
      onIntent: () => {
        preloadSilently(router.preloadRoute({ to: "/studio" }));
      },
    },
    // Video is diffusers-only, so a chat-only host cannot load it: disable with a hint instead of bouncing off the root guard.
    video: {
      icon: FlimSlateIcon,
      label: t("shell.navigation.video"),
      active: pathname === "/video" || pathname.startsWith("/video/"),
      disabled: chatOnlyMeasured,
      tooltip: videoDisabledHint,
      pending: capabilitiesUnknown,
      pendingTooltip: t("shell.navigation.videoChecking"),
      onClick: () => {
        navigate({ to: "/video" });
        closeMobileIfOpen();
      },
      onIntent: () => {
        preloadSilently(router.preloadRoute({ to: "/video" }));
      },
    },
    audio: {
      icon: AudioWave01Icon,
      label: t("shell.navigation.audio"),
      active: pathname === "/audio" || pathname.startsWith("/audio/"),
      onClick: () => {
        navigate({ to: "/audio" });
        closeMobileIfOpen();
      },
      onIntent: () => {
        preloadSilently(router.preloadRoute({ to: "/audio" }));
      },
    },
    recipes: {
      icon: ChefHatIcon,
      label: t("shell.navigation.recipes"),
      active: isRecipesRoute,
      onClick: () => {
        navigate({ to: "/data-recipes" });
        closeMobileIfOpen();
      },
      onIntent: () => {
        preloadSilently(router.preloadRoute({ to: "/data-recipes" }));
        preloadSilently(
          import("@/features/data-recipes").then((module) =>
            module.preloadRecipes(),
          ),
        );
      },
    },
    export: {
      icon: DownloadSquare01Icon,
      label: t("shell.navigation.export"),
      active: pathname === "/export" || pathname.startsWith("/export/"),
      spinner: exportInProgress,
      onClick: () => {
        navigate({ to: "/export" });
        closeMobileIfOpen();
      },
      onIntent: () => {
        preloadSilently(router.preloadRoute({ to: "/export" }));
        preloadSilently(
          import("@/features/export/export-navigation-cache").then((module) =>
            module.preloadExportData(),
          ),
        );
      },
    },
    // The monitor page, not the API keys dialog the profile menu opens.
    api: {
      icon: Globe02Icon,
      label: t("shell.navigation.api"),
      active:
        pathname === "/api-monitor" || pathname.startsWith("/api-monitor/"),
      onClick: () => {
        navigate({ to: "/api-monitor" });
        closeMobileIfOpen();
      },
      onIntent: () => {
        preloadSilently(router.preloadRoute({ to: "/api-monitor" }));
      },
    },
  };
  // Switched-off destinations drop out of both the inline rows and "More" (see config/disabled-features).
  const enabledNav = sidebarNav.filter(
    (item) =>
      (item.id !== "projects" || FEATURE_PROJECTS) &&
      (item.id !== "agents" || FEATURE_AGENTS_NAV) &&
      (item.id !== "files" || FEATURE_FILES_NAV) &&
      (item.id !== "memory" || FEATURE_MEMORY_NAV) &&
      (item.id !== "search" || FEATURE_SEARCH_NAV) &&
      (item.id !== "management" || FEATURE_MANAGEMENT_NAV) &&
      (item.id !== "images" || FEATURE_IMAGES) &&
      (item.id !== "train" || FEATURE_TRAIN) &&
      (item.id !== "video" || FEATURE_VIDEO) &&
      (item.id !== "audio" || FEATURE_AUDIO) &&
      (item.id !== "recipes" || FEATURE_RECIPES) &&
      (item.id !== "export" || FEATURE_EXPORT) &&
      (item.id !== "api" || FEATURE_API_MONITOR),
  );
  const unpinnedNavIds = enabledNav
    .filter((item) => !item.pinned)
    .map((item) => item.id);
  const overflowNavIds = unpinnedNavIds;
  const inlineNavIds = enabledNav
    .filter((item) => item.pinned)
    .map((item) => item.id);
  // Mirrors ImagesWorkflowList's own test: it decides which row owns the highlight.
  const imagesWorkflowsListed =
    sidebarState !== "collapsed" &&
    !(navRows.images.active && imagesPageMode === "train");

  const showSidebarBrand = true;

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
    // The normal new-chat affordance is always a saved chat; only the toolbar toggle is temporary.
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

  async function handleDeleteThread(
    item: Parameters<typeof deleteChatItem>[0],
    args: { deleteFiles?: boolean } = {},
  ) {
    await deleteChatItem(
      item,
      activeThreadId,
      (view) => {
        navigate({
          to: "/chat",
          search: item.projectId
            ? { project: item.projectId }
            : { new: view.newThreadNonce },
        });
      },
      args,
    );
  }

  // Shared chat delete: same error toast and pin cleanup with or without the confirm dialog.
  async function deleteChatWithCleanup(
    item: SidebarItem,
    args: { deleteFiles?: boolean } = {},
  ) {
    try {
      await handleDeleteThread(item, args);
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
  // Optimistic title while the debounced sidebar refresh catches up, so the old name doesn't flash.
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
        current?.id === pendingRename.id &&
        current.title === pendingRename.title
          ? null
          : current,
      );
    });
  }, [allChatItems, pendingRename]);
  const [creatingProject, setCreatingProject] = useState(false);
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
      const updated = await renameTrainingRun(
        target.run.id,
        nextRunDisplayName,
      );
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
      // Commit when changed, else just close, so a no-op Enter doesn't leave the row an input.
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
  const [confirmingDelete, setConfirmingDelete] = useState<DeleteTarget | null>(
    null,
  );
  const [deleteFilesOnDelete, setDeleteFilesOnDelete] = useState(false);

  /** Always through here: a stale switch would delete an unrelated sandbox. */
  function openDeleteDialog(target: DeleteTarget) {
    setDeleteFilesOnDelete(false);
    setConfirmingDelete(target);
  }

  /** Only where a sandbox can actually be removed. A training run has none.
   *  A chat in a project still has one: anything it wrote before the move is in
   *  its own folder, and deletion never touches the project workspace. */
  function deleteTargetHasFiles(target: DeleteTarget | null): boolean {
    if (!target) return false;
    return target.kind === "project" || target.kind === "chat";
  }

  async function commitDelete() {
    const target = confirmingDelete;
    if (!target) return;
    const shouldDeleteProjectFiles =
      target.kind === "project" && deleteFilesOnDelete;
    const shouldDeleteChatFiles =
      deleteTargetHasFiles(target) &&
      target.kind === "chat" &&
      deleteFilesOnDelete;
    setConfirmingDelete(null);
    // Reset so the next delete never inherits this switch.
    setDeleteFilesOnDelete(false);
    if (target.kind === "chat") {
      await deleteChatWithCleanup(target.item, {
        deleteFiles: shouldDeleteChatFiles,
      });
      return;
    }
    if (target.kind === "project") {
      try {
        await deleteChatProject(target.project.id, {
          deleteFiles: shouldDeleteProjectFiles,
        });
        // Refresh chat history so the project's reparented chats don't linger as stale rows.
        notifyChatHistoryUpdated();
        // activeProjectId is only the ?project= param; on a thread-only URL the project comes from
        // the runtime store, so check that too or the user is stranded on a deleted thread. Only
        // redirect from a chat route: the runtime store value can be stale elsewhere.
        const runtimeProjectId = useChatRuntimeStore.getState().activeProjectId;
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

  // "New project" from a chat's menu moves that chat in and stays put; otherwise open the
  // project, unless a slow upload outlasted the route the user was on.
  async function afterCreateProject(
    project: ProjectRecord,
    { stayedOnRoute }: { stayedOnRoute: boolean },
  ) {
    const moveTarget = projectCreateMoveTarget;
    setProjectCreateMoveTarget(null);
    if (!moveTarget) {
      if (stayedOnRoute) openProject(project.id);
      return;
    }
    try {
      await moveChatItemToProject(moveTarget, project.id);
      if (activeThreadId === moveTarget.id) {
        useChatRuntimeStore.getState().setActiveProjectId(project.id);
      }
    } catch (err) {
      toast.error("Failed to move chat to the new project", {
        description: err instanceof Error ? err.message : undefined,
      });
    }
  }

  async function moveChatToProject(
    item: SidebarItem,
    projectId: string | null,
  ) {
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

  function clearChatNotifications(item: SidebarItem) {
    const threadIds = getSidebarItemThreadIds(item);
    setUnreadThreadIds((current) => {
      if (!threadIds.some((threadId) => current.has(threadId))) {
        return current;
      }
      const next = new Set(current);
      for (const threadId of threadIds) {
        next.delete(threadId);
      }
      return next;
    });
  }

  function renderChatSidebarItem(
    item: SidebarItem,
    variant: "project" | "recent",
  ) {
    const threadIds = getSidebarItemThreadIds(item);
    const isPinned = pinnedIdSet.has(item.id);
    // A compare row's id is the pair id while runningByThreadId is per pane thread; aggregate.
    const isGenerating =
      item.type === "compare"
        ? (item.threadIds ?? []).some((id) => Boolean(runningByThreadId[id]))
        : Boolean(runningByThreadId[item.id]);
    const hasQueuedActivity = threadIds.some((threadId) =>
      Boolean(queueByThreadId[threadId]),
    );
    // Active generation and queued work share the in-row spinner slot so they cannot drift.
    const showQueuedActivity = hasQueuedActivity && !isGenerating;
    const showWorkSpinner = isGenerating || showQueuedActivity;
    const hasUnreadActivity =
      !isGenerating &&
      !hasQueuedActivity &&
      threadIds.some((threadId) => unreadThreadIds.has(threadId));
    const hasSecondaryRowAction =
      variant === "project" || (variant === "recent" && isPinned);
    const itemClass =
      variant === "project"
        ? "group/project-chat-item relative"
        : "group/recent-item relative";
    const actionClass =
      variant === "project"
        ? "sidebar-row-action sidebar-touch-reveal group-hover/project-chat-item:opacity-100 group-hover/project-chat-item:pointer-events-auto focus-visible:opacity-100 focus-visible:pointer-events-auto"
        : "sidebar-row-action sidebar-touch-reveal group-hover/recent-item:opacity-100 group-hover/recent-item:pointer-events-auto focus-visible:opacity-100 focus-visible:pointer-events-auto";
    const buttonClass = cn(
      "sidebar-nav-btn h-[30px] cursor-pointer rounded-full py-0 pr-4 text-ui-14p5 leading-ui-19 tracking-nav font-medium",
      // pl-3 (12px) over the content's pl-1.5 (6px) = 18px, aligning with the nav items above.
      variant === "project" ? "pl-[39px]" : "pl-3",
      // Pinned chats carry a chat icon, so add the nav-item icon gap.
      isPinned && variant !== "project" && "gap-[8.5px]",
      showWorkSpinner
        ? hasSecondaryRowAction
          ? "pr-16"
          : undefined
        : hasUnreadActivity
          ? "pr-7"
          : undefined,
      variant === "project"
        ? showWorkSpinner
          ? undefined
          : // Room for the hover pin quick-action plus the kebab.
            "group-hover/project-chat-item:pr-14 group-has-[.sidebar-row-action[data-state=open]]/project-chat-item:pr-8 [@media(pointer:coarse)]:pr-14"
        : isPinned
          ? showWorkSpinner
            ? undefined
            : // Pinned rows show an extra unpin button on hover, so reserve more room
              // (pr-8 when the menu is open keeps the unpin button clear of the title).
              "group-hover/recent-item:pr-16 group-has-[.sidebar-row-action[data-state=open]]/recent-item:pr-8 [@media(pointer:coarse)]:pr-16"
          : showWorkSpinner
            ? // A spinner glyph cannot truncate, so clear the kebab's 30px inset (pr-1.5 + size-6).
              "group-hover/recent-item:pr-8 group-has-[.sidebar-row-action[data-state=open]]/recent-item:pr-8 [@media(pointer:coarse)]:pr-10"
            : // Hover room for the kebab only; title keeps one more character.
              // Touch rows clear the full always-visible kebab hit area (pr-10).
              "group-hover/recent-item:pr-6 group-has-[.sidebar-row-action[data-state=open]]/recent-item:pr-6 [@media(pointer:coarse)]:pr-10",
      // A focused kebab is revealed without hover, so a spinner row reserves the same room.
      showWorkSpinner &&
        (variant === "project"
          ? "group-has-[.sidebar-row-action:focus-visible]/project-chat-item:pr-14"
          : isPinned
            ? "group-has-[.sidebar-row-action:focus-visible]/recent-item:pr-16"
            : "group-has-[.sidebar-row-action:focus-visible]/recent-item:pr-8"),
    );

    const isRenamingThis =
      renamingTarget?.kind === "chat" && renamingTarget.item.id === item.id;

    // Inline rename edits the title in place as a rounded pill, no dialog.
    if (isRenamingThis) {
      return (
        <SidebarMenuItem key={item.id} className={itemClass}>
          <input
            autoFocus={true}
            value={renameDraft}
            onChange={(event) => setRenameDraft(event.target.value)}
            onKeyDown={handleInlineRenameKeyDown}
            onBlur={handleInlineRenameBlur}
            onFocus={(event) => event.currentTarget.select()}
            maxLength={120}
            aria-label={translate("shell.dialog.renameChat.placeholder")}
            className={cn(
              // No pill or box; edit in place as plain highlighted text.
              "text-foreground h-[30px] w-full border-0 bg-transparent py-0 pr-4 text-ui-14p5 leading-ui-19 font-medium tracking-nav outline-none",
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
          data-generating={isGenerating ? "true" : undefined}
          aria-busy={isGenerating || undefined}
          isActive={activeThreadId === item.id}
          className={buttonClass}
          onClick={() => {
            clearChatNotifications(item);
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
            <HugeiconsIcon
              icon={BubbleChatIcon}
              strokeWidth={1.75}
              className="size-icon! shrink-0"
            />
          )}
          <span className="truncate">
            {pendingRename?.id === item.id ? pendingRename.title : item.title}
          </span>
          {showWorkSpinner && (
            <Spinner
              data-testid="chat-row-spinner"
              // role="status" + label: announced, not motion-only.
              label={
                isGenerating
                  ? translate("shell.navigation.chatGenerating")
                  : "Queued"
              }
              className="ml-auto size-3.5 shrink-0 text-muted-foreground"
            />
          )}
        </SidebarMenuButton>
        {hasUnreadActivity ? (
          <span
            className={cn(
              "pointer-events-none absolute right-2 top-1/2 z-10 flex size-4 -translate-y-1/2 items-center justify-center text-muted-foreground transition-opacity",
              variant === "project"
                ? "group-hover/project-chat-item:opacity-0 group-has-[.sidebar-row-action[data-state=open]]/project-chat-item:opacity-0"
                : "group-hover/recent-item:opacity-0 group-has-[.sidebar-row-action[data-state=open]]/recent-item:opacity-0",
            )}
            aria-hidden={true}
          >
            {/* Neutral: a finished reply is news, not a fault. */}
            <span className="size-2 rounded-full bg-muted-foreground/60" />
          </span>
        ) : null}
        {variant === "project" && (
          <button
            type="button"
            onClick={(e) => {
              e.stopPropagation();
              togglePinnedChat(item.id);
            }}
            aria-label={isPinned ? "Unpin chat" : "Pin chat"}
            className="sidebar-row-action sidebar-touch-reveal is-unpin-action group-hover/project-chat-item:opacity-100 group-hover/project-chat-item:pointer-events-auto focus-visible:opacity-100 focus-visible:pointer-events-auto"
          >
            <span className="sidebar-row-action-glyph">
              <HugeiconsIcon
                icon={isPinned ? PinOffIcon : PinIcon}
                strokeWidth={1.75}
                className="size-icon"
              />
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
            className="sidebar-row-action sidebar-touch-reveal is-unpin-action group-hover/recent-item:opacity-100 group-hover/recent-item:pointer-events-auto focus-visible:opacity-100 focus-visible:pointer-events-auto"
          >
            <span className="sidebar-row-action-glyph">
              <HugeiconsIcon
                icon={PinOffIcon}
                strokeWidth={1.75}
                className="size-icon"
              />
            </span>
          </button>
        )}
        <DropdownMenu>
          <DropdownMenuTrigger asChild={true}>
            <button
              type="button"
              onClick={(e) => e.stopPropagation()}
              aria-label="Chat options"
              className={actionClass}
            >
              <span className="sidebar-row-action-glyph">
                <HugeiconsIcon
                  icon={MoreVerticalIcon}
                  strokeWidth={1.75}
                  className="size-icon"
                />
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
              <HugeiconsIcon
                icon={Edit03Icon}
                strokeWidth={1.75}
                className="size-icon"
              />
              <span>Rename</span>
            </DropdownMenuItem>
            <DropdownMenuItem onSelect={() => togglePinnedChat(item.id)}>
              <HugeiconsIcon
                icon={isPinned ? PinOffIcon : PinIcon}
                strokeWidth={1.75}
                className="size-icon"
              />
              <span>{isPinned ? "Unpin chat" : "Pin chat"}</span>
            </DropdownMenuItem>
            <DropdownMenuSub>
              <DropdownMenuSubTrigger>
                <HugeiconsIcon
                  icon={FolderExportIcon}
                  strokeWidth={1.75}
                  className="size-icon"
                />
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
                    setCreatingProject(true);
                  }}
                >
                  <HugeiconsIcon
                    icon={FolderAddIcon}
                    strokeWidth={1.75}
                    className="size-icon"
                  />
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
                    <HugeiconsIcon
                      icon={Folder01Icon}
                      strokeWidth={1.75}
                      className="size-icon"
                    />
                    <span className="truncate">{project.name}</span>
                  </DropdownMenuItem>
                ))}
              </DropdownMenuSubContent>
            </DropdownMenuSub>
            <DropdownMenuSub>
              <DropdownMenuSubTrigger>
                <HugeiconsIcon
                  icon={Download01Icon}
                  strokeWidth={1.75}
                  className="size-icon"
                />
                <span>Export</span>
              </DropdownMenuSubTrigger>
              <DropdownMenuSubContent
                sideOffset={8}
                alignOffset={-4}
                className="unsloth-plus-menu w-52"
              >
                {CHAT_EXPORT_OPTIONS.map(({ label, format }) => (
                  <DropdownMenuItem
                    key={label}
                    onSelect={async () => {
                      try {
                        const ids =
                          item.type === "single"
                            ? [item.id]
                            : (
                                await listStoredChatThreads({ pairId: item.id })
                              ).map((t) => t.id);
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
              <HugeiconsIcon
                icon={Archive03Icon}
                strokeWidth={1.75}
                className="size-icon"
              />
              <span>Archive</span>
            </DropdownMenuItem>
            <DropdownMenuItem
              variant="destructive"
              onSelect={() =>
                confirmDeleteChats
                  ? openDeleteDialog({ kind: "chat", item })
                  : void deleteChatWithCleanup(item)
              }
            >
              <HugeiconsIcon
                icon={Delete02Icon}
                strokeWidth={1.75}
                className="size-icon"
              />
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
        collapseToZero={isTauri}
        variant="sidebar"
        className={cn(
          // Rail background comes from --sidebar-surface (index.css) so the footer fade can match it.
          "font-heading group-data-[collapsible=icon]:[&_[data-sidebar=sidebar]]:bg-[var(--sidebar-surface)]",
          usesNativeMacTitlebar &&
            "group-data-[collapsible=icon]:[&_[data-sidebar=sidebar]]:border-r-0",
        )}
      >
        <SidebarHeader
          className={cn(
            "relative",
            usesDesktopTitlebar
              ? "shrink-0 p-0 pt-[calc(var(--studio-desktop-titlebar-height,34px)+17px)]"
              : "pl-3 pr-3 pt-[14px] pb-[8px] group-data-[collapsible=icon]:px-0",
          )}
        >
          {showSidebarBrand && (
            <>
              {usesNativeMacTitlebar && !isMobile && (
                <div
                  data-tauri-drag-region={true}
                  className="absolute inset-x-0 top-0 z-10 flex h-[var(--studio-desktop-titlebar-height,34px)] items-start pt-px pl-[calc(var(--studio-mac-traffic-light-inset,78px)+6px)] select-none group-data-[collapsible=icon]:hidden"
                >
                  <DesktopTitlebarNavigation
                    expanded={pinned}
                    onToggleSidebar={togglePinned}
                  />
                </div>
              )}
              <div
                data-tauri-drag-region={usesNativeMacTitlebar || undefined}
                className={cn(
                  "relative z-10 flex items-center gap-[8.5px] group-data-[collapsible=icon]:hidden",
                  usesDesktopTitlebar
                    ? "justify-between pl-4 pr-3"
                    : "justify-between",
                )}
              >
                <Link
                  to="/chat"
                  onClick={(event) => {
                    event.preventDefault();
                    if (chatDisabled) return;
                    openNewChat(null);
                  }}
                  className={cn(
                    // min-w-0 so a narrow sidebar truncates the wordmark instead of pushing the search icon over.
                    "flex min-w-0 items-center gap-[6px] select-none transition-opacity",
                    chatDisabled && "pointer-events-none opacity-50",
                  )}
                  aria-label={t("shell.aria.home")}
                  aria-disabled={chatDisabled}
                  tabIndex={chatDisabled ? -1 : undefined}
                >
                  <span className="relative -top-px min-w-0 truncate font-heading text-[calc(13px+0.5rem*var(--ui-font-scale,1))] font-semibold leading-tight tracking-[0em] text-black dark:text-white dark:tracking-[0.02em]">
                    {PRODUCT_NAME}
                  </span>
                  <span className="nav-badge ml-0.5 inline-flex shrink-0 items-center justify-center rounded-full border border-nav-beta-border px-[5px] pt-[3px] pb-[2px] text-[calc(0.5rem*var(--ui-font-scale,1))] font-medium leading-none tracking-[0.04em] text-nav-fg-muted antialiased subpixel-antialiased shadow-[0_1px_2px_rgba(0,0,0,0.06)] dark:shadow-[0_1px_2px_rgba(0,0,0,0.35)]">
                    {t("shell.beta")}
                  </span>
                </Link>
                <div className="flex shrink-0 items-center gap-0.25">
                  <Tooltip>
                    <TooltipPrimitive.Trigger asChild={true}>
                      <button
                        type="button"
                        onClick={() => {
                          useChatSearchStore.getState().open();
                          closeMobileIfOpen();
                        }}
                        className="inline-flex size-[30px] cursor-pointer items-center justify-center rounded-[10px] text-nav-icon-idle dark:text-nav-fg-muted transition-colors hover:bg-nav-surface-hover hover:text-black dark:hover:text-white focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
                        aria-label={t("shell.navigation.search")}
                      >
                        <HugeiconsIcon
                          icon={Search01Icon}
                          strokeWidth={1.75}
                          className="size-icon"
                        />
                      </button>
                    </TooltipPrimitive.Trigger>
                    <TooltipContent
                      side="bottom"
                      sideOffset={6}
                      className="tooltip-compact flex items-center gap-1.5"
                      hidden={isMobile}
                    >
                      {t("shell.navigation.search")}
                      <kbd className="rounded bg-black/10 px-1 py-px text-ui-10 font-medium leading-none dark:bg-white/15">
                        {isMacPlatform ? "⌘K" : "Ctrl+K"}
                      </kbd>
                    </TooltipContent>
                  </Tooltip>
                  {!isMobile && !usesDesktopTitlebar && (
                    <Tooltip>
                      <TooltipPrimitive.Trigger asChild={true}>
                        <button
                          type="button"
                          onClick={togglePinned}
                          className="inline-flex size-[30px] cursor-pointer items-center justify-center rounded-[10px] text-nav-icon-idle dark:text-nav-fg-muted transition-colors hover:bg-nav-surface-hover hover:text-black dark:hover:text-white focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
                          aria-label={t("shell.aria.closeSidebar")}
                        >
                          <HugeiconsIcon
                            icon={LayoutAlignLeftIcon}
                            strokeWidth={1.75}
                            className="size-icon"
                          />
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
              {!isMobile && (!usesDesktopTitlebar || usesNativeMacTitlebar) && (
                <div className="relative z-10 hidden group-data-[collapsible=icon]:flex h-[33px] items-center justify-center w-full">
                  <Tooltip>
                    <TooltipPrimitive.Trigger asChild={true}>
                      <button
                        type="button"
                        onClick={togglePinned}
                        className="inline-flex size-[30px] cursor-pointer items-center justify-center rounded-[10px] text-nav-fg transition-colors hover:bg-nav-surface-hover hover:text-black dark:hover:text-white focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
                        aria-label={t("shell.aria.openSidebar")}
                      >
                        <HugeiconsIcon
                          icon={LayoutAlignLeftIcon}
                          strokeWidth={1.75}
                          className="size-icon"
                        />
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

        <SidebarGroup
          className={cn(
            "group-data-[collapsible=icon]:px-0 shrink-0 transition-[padding]",
            rowPadding,
            usesDesktopTitlebar ? "pt-[11px]" : "pt-[9px]",
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
                    ? runningChatCount > 1
                      ? // Name the count rather than imply a single live chat.
                        t("shell.navigation.returnToChats", {
                          count: runningChatCount,
                        })
                      : t("shell.navigation.returnToChat")
                    : t("shell.navigation.newChat")
                }
                // Off-route this row is the only sign chats are still running.
                spinner={anyChatRunning && !isChatRoute}
                // An action, not a destination, so it never marks itself active: the active pill is the
                // hover pill, and on a blank new chat it left the row looking permanently hovered.
                active={false}
                onClick={() => {
                  if (showReturnToChat) {
                    // Prefer the running thread so we return to the live generation, not the empty new chat.
                    if (runningTarget && runningTarget.id !== storeThreadId) {
                      navigate({
                        to: "/chat",
                        search: runningTarget.compare
                          ? { compare: runningTarget.id }
                          : { thread: runningTarget.id },
                      });
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
          ref={attachScroller}
          onScroll={(e) => syncScrollState(e.currentTarget)}
          // Collapsible groups animate their height; re-measure the fade once the animation settles.
          onAnimationEnd={(e) => {
            if (
              e.animationName === "collapsible-down" ||
              e.animationName === "collapsible-up"
            ) {
              syncScrollState(e.currentTarget);
            }
          }}
          className={cn(
            // pb-2 keeps the last row's rounded highlight clear of the overflow clip edge.
            "sidebar-scroll-fade gap-0 overflow-y-auto overscroll-contain min-h-0 pb-2",
            scrolled && "is-scrolled",
          )}
        >
          <SidebarGroup
            data-tour="navbar"
            className={cn(
              "group-data-[collapsible=icon]:px-0 py-0 shrink-0",
              scrollRowPadding,
            )}
          >
            <SidebarGroupContent>
              <SidebarMenu>
                {/* Order and pin state come from Settings -> Appearance ->
                  Sidebar navigation. */}
                {inlineNavIds.map((id) => {
                  const row = navRows[id];
                  // A row whose capability is still unmeasured spins instead of blacking out.
                  const rowState = resolveNavRowState(row);
                  return (
                    <NavItem
                      key={id}
                      icon={row.icon}
                      label={row.label}
                      badge={row.badge}
                      // While the workflows are listed, the current one carries the highlight, not the Images row.
                      active={
                        id === "images" && imagesWorkflowsListed
                          ? false
                          : row.active
                      }
                      disabled={rowState.disabled}
                      tooltip={rowState.tooltip}
                      alwaysTooltip={rowState.pending}
                      spinner={rowState.spinner}
                      testId={`nav-row-${id}`}
                      onClick={row.onClick}
                      onIntent={row.onIntent}
                      className={cn(
                        row.className,
                        id === "images" && "group/images-item",
                      )}
                      // Off the Images page the list is folded, so the row offers a way to open it.
                      overlay={
                        id === "images" &&
                        !row.active &&
                        sidebarState !== "collapsed" ? (
                          <ImagesNavDisclosure />
                        ) : undefined
                      }
                    >
                      {/* Images carries its workflows as rows beneath it. */}
                      {id === "images" ? (
                        <ImagesWorkflowList
                          active={row.active}
                          collapsed={sidebarState === "collapsed"}
                          onPick={(workflowId) => {
                            useImageWorkflowStore
                              .getState()
                              .setWorkflow(workflowId);
                            navigate({ to: "/images" });
                            closeMobileIfOpen();
                          }}
                        />
                      ) : (
                        row.children
                      )}
                    </NavItem>
                  );
                })}
                {/* Unpinned destinations, behind one row. */}
                {overflowNavIds.length > 0 && (
                  <SidebarMenuItem
                    onPointerEnter={openMore}
                    onPointerLeave={closeMoreSoon}
                  >
                    <DropdownMenu
                      open={moreOpen}
                      onOpenChange={setMoreOpen}
                      modal={false}
                    >
                      {/* Tooltip wraps the trigger rather than using the button's `tooltip` prop: that returns a Tooltip root, so DropdownMenuTrigger asChild would miss the DOM node. */}
                      <Tooltip>
                        <TooltipPrimitive.Trigger asChild={true}>
                          <DropdownMenuTrigger asChild={true}>
                            <SidebarMenuButton
                              // More is a container, not a destination: no active style just because the current page
                              // lives inside it. Keeps the row highlighted while the panel is open, after the pointer
                              // has left. Not data-state: the tooltip and menu triggers both write that one.
                              data-menu-open={moreOpen ? "true" : undefined}
                              className="sidebar-nav-btn h-[33px] rounded-full gap-[8.5px] pl-3 pr-2.5 font-medium group-data-[collapsible=icon]:px-2.5 group-data-[collapsible=icon]:!w-[32px] group-data-[collapsible=icon]:mx-auto"
                            >
                              <HugeiconsIcon
                                icon={MoreHorizontalIcon}
                                strokeWidth={1.75}
                                className="size-icon! shrink-0 group-hover/menu-button:animate-icon-pop"
                              />
                              <span className="text-ui-14p5 leading-ui-19 tracking-nav">
                                {t("shell.navigation.more")}
                              </span>
                            </SidebarMenuButton>
                          </DropdownMenuTrigger>
                        </TooltipPrimitive.Trigger>
                        {/* Collapsed rail only; expanded rows show their label. */}
                        <TooltipContent
                          side="right"
                          align="center"
                          className="tooltip-compact"
                          hidden={isMobile || sidebarState !== "collapsed"}
                        >
                          {t("shell.navigation.more")}
                        </TooltipContent>
                      </Tooltip>
                      <DropdownMenuContent
                        side="right"
                        align="start"
                        sideOffset={6}
                        onPointerEnter={openMore}
                        onPointerLeave={closeMoreSoon}
                        className="w-48 p-1"
                      >
                        {overflowNavIds.map((id) => {
                          const row = navRows[id];
                          // Same pending handling as the inline rows above.
                          const rowState = resolveNavRowState(row);
                          return (
                            <MoreMenuItem
                              key={id}
                              icon={row.icon}
                              label={row.label}
                              badge={row.badge}
                              active={row.active}
                              disabled={rowState.disabled}
                              tooltip={rowState.tooltip}
                              spinner={rowState.spinner}
                              onSelect={row.onClick}
                              onIntent={row.onIntent}
                            />
                          );
                        })}
                        {/* Way out of the flyout: jump straight to the control that
                          decides what lives here vs. on the sidebar itself.
                          my-1 matches the menu's own p-1, so the gap either side
                          of the rule equals the one under the last row. */}
                        <DropdownMenuSeparator className="mx-1! my-1! h-0! border-t border-border/70 bg-transparent! dark:border-white/15" />
                        <DropdownMenuItem
                          onSelect={() =>
                            useSettingsDialogStore
                              .getState()
                              .openDialog("appearance", {
                                scrollTarget: "appearance-sidebar-nav",
                              })
                          }
                        >
                          <HugeiconsIcon
                            icon={Settings02Icon}
                            strokeWidth={1.75}
                          />
                          <span className="min-w-0 flex-1 truncate">
                            {t("shell.navigation.customizeSidebar")}
                          </span>
                        </DropdownMenuItem>
                      </DropdownMenuContent>
                    </DropdownMenu>
                  </SidebarMenuItem>
                )}
              </SidebarMenu>
            </SidebarGroupContent>
          </SidebarGroup>

          {/* Pinned: pinned projects (with their chats) and pinned chats */}
          {!isStudioRoute &&
            !showTrainingRecents &&
            (pinnedProjectRecords.length > 0 || pinnedChatItems.length > 0) && (
              <Collapsible
                open={pinnedOpen}
                onOpenChange={setPinnedOpen}
                asChild={true}
              >
                <SidebarGroup className="group-data-[collapsible=icon]:hidden px-0 py-0">
                  <SidebarGroupLabel
                    className={cn(
                      "sidebar-sticky-label sidebar-sticky-label-following",
                      scrolled && "is-scrolled",
                    )}
                    asChild={true}
                  >
                    <CollapsibleTrigger className="cursor-pointer flex w-full items-center gap-1 group/sb-collap">
                      Pinned
                      <ChevronDown className="size-3.5 opacity-0 transition-[transform,opacity] duration-200 group-hover/sb-collap:opacity-100 group-focus-visible/sb-collap:opacity-100 data-[state=open]:rotate-0 [[data-state=closed]_&]:rotate-[-90deg] [[data-state=closed]_&]:opacity-100" />
                    </CollapsibleTrigger>
                  </SidebarGroupLabel>
                  <CollapsibleContent>
                    <SidebarGroupContent className={scrollRowPadding}>
                      <SidebarMenu>
                        {pinnedProjectRecords.map((project) => {
                          const projectChats =
                            chatsByProjectId.get(project.id) ?? [];
                          const expanded = !collapsedProjectIds.has(project.id);
                          const showAll = expandedChatProjectIds.has(
                            project.id,
                          );
                          const visibleChats =
                            expanded && !showAll
                              ? projectChats.slice(0, PINNED_PROJECT_CHAT_LIMIT)
                              : projectChats;
                          return (
                            <Fragment key={project.id}>
                              <SidebarMenuItem className="group/recent-item relative">
                                <SidebarMenuButton
                                  // Highlight the folder only on the project home; with a chat open, only that row is active.
                                  isActive={
                                    activeProjectId === project.id &&
                                    !activeThreadId
                                  }
                                  onClick={() =>
                                    toggleProjectCollapsed(project.id)
                                  }
                                  className="sidebar-nav-btn h-[33px] rounded-full gap-[8.5px] pl-3 pr-2.5 font-medium group-hover/recent-item:pr-16 group-has-[.sidebar-row-action[data-state=open]]/recent-item:pr-8"
                                >
                                  <HugeiconsIcon
                                    icon={Folder01Icon}
                                    strokeWidth={1.75}
                                    className="size-icon! shrink-0"
                                  />
                                  <span className="truncate text-ui-14p5 leading-ui-19 tracking-nav">
                                    {project.name}
                                  </span>
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
                                    <HugeiconsIcon
                                      icon={PencilEdit02Icon}
                                      strokeWidth={1.75}
                                      className="size-icon"
                                    />
                                  </span>
                                </button>
                                {/* Project options */}
                                <DropdownMenu>
                                  <DropdownMenuTrigger asChild={true}>
                                    <button
                                      type="button"
                                      onClick={(e) => e.stopPropagation()}
                                      aria-label="Project options"
                                      className="sidebar-row-action group-hover/recent-item:opacity-100 group-hover/recent-item:pointer-events-auto focus-visible:opacity-100 focus-visible:pointer-events-auto"
                                    >
                                      <span className="sidebar-row-action-glyph">
                                        <HugeiconsIcon
                                          icon={MoreVerticalIcon}
                                          strokeWidth={1.75}
                                          className="size-icon"
                                        />
                                      </span>
                                    </button>
                                  </DropdownMenuTrigger>
                                  <DropdownMenuContent
                                    side="bottom"
                                    align="start"
                                    sideOffset={0}
                                    className="unsloth-plus-menu menu-flat-destructive w-56"
                                  >
                                    <DropdownMenuItem
                                      onSelect={() => openProject(project.id)}
                                    >
                                      <HugeiconsIcon
                                        icon={Folder01Icon}
                                        strokeWidth={1.75}
                                        className="size-icon"
                                      />
                                      <span>Project home</span>
                                    </DropdownMenuItem>
                                    <DropdownMenuItem
                                      onSelect={() => openNewChat(project.id)}
                                    >
                                      <HugeiconsIcon
                                        icon={PencilEdit02Icon}
                                        strokeWidth={1.75}
                                        className="size-icon"
                                      />
                                      <span>New chat</span>
                                    </DropdownMenuItem>
                                    <DropdownMenuItem
                                      onSelect={() => {
                                        // Seed the shared draft so the dialog opens with the current name, not stale text.
                                        setRenameDraft(project.name);
                                        setRenamingTarget({
                                          kind: "project",
                                          project,
                                          current: project.name,
                                        });
                                      }}
                                    >
                                      <HugeiconsIcon
                                        icon={Edit03Icon}
                                        strokeWidth={1.75}
                                        className="size-icon"
                                      />
                                      <span>Rename project</span>
                                    </DropdownMenuItem>
                                    <DropdownMenuItem
                                      onSelect={() => unpinProject(project.id)}
                                    >
                                      <HugeiconsIcon
                                        icon={PinOffIcon}
                                        strokeWidth={1.75}
                                        className="size-icon"
                                      />
                                      <span>Unpin project</span>
                                    </DropdownMenuItem>
                                    <DropdownMenuSeparator />
                                    <DropdownMenuItem
                                      variant="destructive"
                                      onSelect={() => {
                                        // Start each delete with the file toggle off: Cancel closes programmatically and skips the
                                        openDeleteDialog({
                                          kind: "project",
                                          project,
                                        });
                                      }}
                                    >
                                      <HugeiconsIcon
                                        icon={Delete02Icon}
                                        strokeWidth={1.75}
                                        className="size-icon"
                                      />
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
                                projectChats.length >
                                  PINNED_PROJECT_CHAT_LIMIT && (
                                  <SidebarMenuItem>
                                    <SidebarMenuButton
                                      onClick={() =>
                                        toggleProjectShowAll(project.id)
                                      }
                                      // Force the muted token: .sidebar-nav-btn's own color rule outweighs a plain text utility,
                                      // so Show more would otherwise match the chat rows.
                                      className="sidebar-nav-btn h-[30px] rounded-full pl-9 pr-4 font-medium text-nav-fg-muted!"
                                    >
                                      <span className="text-ui-13 leading-ui-18 tracking-nav">
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
            <Collapsible
              open={chatOpen}
              onOpenChange={setChatOpen}
              asChild={true}
            >
              <SidebarGroup className="group-data-[collapsible=icon]:hidden px-0 py-0">
                <SidebarGroupLabel
                  className={cn(
                    "sidebar-sticky-label sidebar-sticky-label-following",
                    scrolled && "is-scrolled",
                    usesDesktopTitlebar && "translate-x-[2px]",
                  )}
                  asChild={true}
                >
                  <CollapsibleTrigger className="cursor-pointer flex w-full items-center gap-1 group/sb-collap">
                    {t("shell.navigation.recents")}
                    <ChevronDown className="size-3.5 opacity-0 transition-[transform,opacity] duration-200 group-hover/sb-collap:opacity-100 group-focus-visible/sb-collap:opacity-100 data-[state=open]:rotate-0 [[data-state=closed]_&]:rotate-[-90deg] [[data-state=closed]_&]:opacity-100" />
                  </CollapsibleTrigger>
                </SidebarGroupLabel>
                <CollapsibleContent>
                  <SidebarGroupContent className={scrollRowPadding}>
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
            <Collapsible
              open={runsOpen}
              onOpenChange={setRunsOpen}
              asChild={true}
            >
              <SidebarGroup className="group-data-[collapsible=icon]:hidden px-0 py-0">
                <SidebarGroupLabel
                  className={cn(
                    "sidebar-sticky-label sidebar-sticky-label-following",
                    scrolled && "is-scrolled",
                  )}
                  asChild={true}
                >
                  <CollapsibleTrigger className="cursor-pointer flex w-full items-center gap-1 group/sb-collap">
                    {t("shell.navigation.recents")}
                    <ChevronDown className="size-3.5 opacity-0 transition-[transform,opacity] duration-200 group-hover/sb-collap:opacity-100 group-focus-visible/sb-collap:opacity-100 data-[state=open]:rotate-0 [[data-state=closed]_&]:rotate-[-90deg] [[data-state=closed]_&]:opacity-100" />
                  </CollapsibleTrigger>
                </SidebarGroupLabel>
                <CollapsibleContent>
                  <SidebarGroupContent className={scrollRowPadding}>
                    <SidebarMenu>
                      {runItems.map((run) => {
                        // Explicit selection wins. Otherwise highlight the active job only while the "Current Run"
                        // tab is the view, keeping Configure unhighlighted even though activeJobId stays pinned.
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
                              className="sidebar-nav-btn h-auto flex-col items-start gap-0.5 py-[5px] rounded-[14px] pl-3 pr-7 text-ui-14p5 tracking-nav font-medium"
                              onClick={() => {
                                setSelectedHistoryRunId(run.id);
                                // From Recipes/Export, jump to Train so the run's history opens.
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
                                  aria-hidden={true}
                                />
                                <span className="truncate">
                                  {getTrainingRunDisplayTitle(run)}
                                </span>
                                <span className="ml-auto mr-0.5 shrink-0 text-ui-10 text-muted-foreground">
                                  {formatRelativeShort(run.started_at)}
                                </span>
                              </div>
                              <span className="w-full truncate pl-3.5 text-xs text-muted-foreground">
                                {run.dataset_name}
                              </span>
                            </SidebarMenuButton>
                            <DropdownMenu>
                              <DropdownMenuTrigger asChild={true}>
                                <button
                                  type="button"
                                  onClick={(e) => e.stopPropagation()}
                                  aria-label={t("shell.aria.runOptions")}
                                  className="sidebar-row-action group-hover/run-item:opacity-100 group-hover/run-item:pointer-events-auto focus-visible:opacity-100 focus-visible:pointer-events-auto"
                                >
                                  <span className="sidebar-row-action-glyph">
                                    <HugeiconsIcon
                                      icon={MoreVerticalIcon}
                                      strokeWidth={1.75}
                                      className="size-icon"
                                    />
                                  </span>
                                </button>
                              </DropdownMenuTrigger>
                              <DropdownMenuContent
                                side="bottom"
                                align="end"
                                sideOffset={0}
                                className="app-user-menu menu-soft-surface menu-flat-destructive ring-0 w-44 py-2 font-heading rounded-full border-0"
                              >
                                <DropdownMenuItem
                                  onSelect={() => openRenameRun(run)}
                                >
                                  <HugeiconsIcon
                                    icon={Edit03Icon}
                                    strokeWidth={1.75}
                                    className="size-icon"
                                  />
                                  <span>{t("common.rename")}</span>
                                </DropdownMenuItem>
                                <DropdownMenuItem
                                  variant="destructive"
                                  disabled={run.status === "running"}
                                  onSelect={() =>
                                    openDeleteDialog({ kind: "run", run })
                                  }
                                >
                                  <HugeiconsIcon
                                    icon={Delete02Icon}
                                    strokeWidth={1.75}
                                    className="size-icon"
                                  />
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
            "relative pb-[11px] group-data-[collapsible=icon]:px-0",
            // Same box as the rows above.
            rowPadding,
            // pt-[3px] cancels the profile button's -3px margin, so the 8px
            // above it is whatever sits over the footer edge (the fade plateau,
            // or the list's pb-2 once the fade is hidden) and 8px sits below.
            showUpdateCard ? "pt-1" : "pt-[3px]",
          )}
        >
          {/* Fade above the profile box, shown only when there's more list below
            the fold; at the bottom (or short lists) it fades so the last row
            shows fully (Gemini-style). Stops at the rail: the thumb ends its
            travel in this band and a full-width gradient washed it out. */}
          <div
            aria-hidden="true"
            className={cn(
              // The scroll area hard-clips at the fade's bottom edge, so a plain
              // ramp is still part-transparent there and slices the last row
              // mid-glyph. from-[8px] holds it opaque just across the clip, and
              // matches the list's pb-2 so the gap is the same once it hides.
              "pointer-events-none absolute start-0 end-[var(--sidebar-rail,0px)] bottom-full bg-gradient-to-t from-[var(--sidebar-surface)] from-[8px] to-[rgb(from_var(--sidebar-surface)_r_g_b/0)] transition-opacity duration-200",
              // Shorter fade with the update card so the list reads closer to
              // it, but still tall enough to clear a row.
              showUpdateCard ? "h-7" : "h-10",
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
                    <span className="truncate font-heading text-ui-13p5 font-semibold text-nav-fg">
                      {t("shell.updateAvailable")}
                    </span>
                    {updateVersion && (
                      <span className="truncate text-ui-11p5 text-muted-foreground">
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
            {/* Collapsed rail has no room for the cog on the profile row, so it
              sits above the avatar instead. */}
            <NavItem
              className="hidden group-data-[collapsible=icon]:block"
              icon={Settings02Icon}
              label={t("shell.navigation.settings")}
              active={false}
              onClick={() => {
                useSettingsDialogStore.getState().openDialog();
                closeMobileIfOpen();
              }}
            />
            <SidebarMenuItem>
              <DropdownMenu>
                <DropdownMenuTrigger asChild={true}>
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
                      <span className="truncate font-heading text-ui-13p5 tracking-[0.025em] dark:tracking-[0.04em] font-semibold text-nav-fg">
                        {displayTitle}
                      </span>
                      <span className="truncate text-ui-11p5 tracking-nav text-muted-foreground">
                        Rag Platform
                      </span>
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
                      onSelect={() =>
                        useSettingsDialogStore.getState().openDialog()
                      }
                    >
                      <HugeiconsIcon
                        icon={Settings02Icon}
                        strokeWidth={1.75}
                        className="size-icon"
                      />
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
                            onSelect={() =>
                              useSettingsDialogStore
                                .getState()
                                .openDialog("api-keys")
                            }
                          >
                            <HugeiconsIcon
                              icon={Globe02Icon}
                              strokeWidth={1.75}
                              className="size-[18px]"
                            />
                            <span>{t("shell.navigation.api")}</span>
                          </DropdownMenuItem>
                        );
                      }
                      if (item.id === "darkMode") {
                        return (
                          <DropdownMenuItem
                            key={item.id}
                            ref={anchorRef as React.Ref<HTMLDivElement>}
                            onSelect={(e) => {
                              e.preventDefault();
                              toggleTheme();
                            }}
                          >
                            {isDark ? (
                              <HugeiconsIcon
                                icon={Sun03Icon}
                                strokeWidth={1.75}
                                className="size-icon"
                              />
                            ) : (
                              <Moon strokeWidth={1.75} className="size-icon" />
                            )}
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
                            <HugeiconsIcon
                              icon={CursorInfo02Icon}
                              strokeWidth={1.75}
                              className="size-icon"
                            />
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
                          onSelect={() =>
                            useSettingsDialogStore
                              .getState()
                              .openDialog(settingsTabId)
                          }
                        >
                          <HugeiconsIcon
                            icon={tab.icon}
                            strokeWidth={1.75}
                            className="size-icon"
                          />
                          <span>{t(tab.labelKey)}</span>
                        </DropdownMenuItem>
                      );
                    })}
                  </DropdownMenuGroup>
                  <DropdownMenuSeparator className="mx-1! my-2.5! h-0! border-t border-border/70 bg-transparent! dark:border-white/15" />
                  <DropdownMenuItem
                    onSelect={() =>
                      useSettingsDialogStore.getState().openDialog("about")
                    }
                  >
                    <HugeiconsIcon
                      icon={HelpCircleIcon}
                      strokeWidth={1.75}
                      className="size-icon"
                    />
                    <span>{t("common.help")}</span>
                  </DropdownMenuItem>
                  {!isTauri && (
                    <DropdownMenuItem
                      onSelect={async () => {
                        // Best-effort server revocation; ignore network errors so the local clear still runs.
                        try {
                          await logout();
                        } catch {
                          clearAuthTokens();
                        }
                        void navigate({ to: "/login" });
                      }}
                    >
                      <HugeiconsIcon
                        icon={Logout05Icon}
                        strokeWidth={1.75}
                        className="size-icon"
                      />
                      <span>{t("shell.navigation.logOut")}</span>
                    </DropdownMenuItem>
                  )}
                  {!isTauri && (
                    <DropdownMenuItem onSelect={() => setShutdownOpen(true)}>
                      <HugeiconsIcon
                        icon={PowerIcon}
                        strokeWidth={1.75}
                        className="size-icon"
                      />
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
            setDeleteFilesOnDelete(false);
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
          {deleteTargetHasFiles(confirmingDelete) ? (
            <div className="flex items-start justify-between gap-4 rounded-md border border-border/60 bg-muted/35 px-3 py-2.5">
              <label
                htmlFor="delete-files-on-delete"
                className="min-w-0 space-y-1"
              >
                <span className="block text-sm font-medium text-foreground">
                  Delete files and sandbox folder
                </span>
                <span className="block break-words text-xs leading-5 text-muted-foreground">
                  {confirmingDelete?.kind === "project"
                    ? (confirmingDelete.project.rootPath ??
                      "The project workspace folder will be removed from disk.")
                    : "This chat's own sandbox folder is removed from disk. Files it wrote inside a project stay in that project's workspace."}
                </span>
              </label>
              <Switch
                id="delete-files-on-delete"
                checked={deleteFilesOnDelete}
                onCheckedChange={setDeleteFilesOnDelete}
                aria-label="Delete files and sandbox folder"
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
              {deleteTargetHasFiles(confirmingDelete) && deleteFilesOnDelete
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
            autoFocus={true}
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
      <NewProjectDialog
        open={creatingProject}
        onOpenChange={(open) => {
          setCreatingProject(open);
          if (!open) setProjectCreateMoveTarget(null);
        }}
        title={
          projectCreateMoveTarget ? "Move to new project" : "Create project"
        }
        submitLabel={
          projectCreateMoveTarget ? "Create and move" : "Create project"
        }
        onCreated={afterCreateProject}
      />
    </>
  );
}
