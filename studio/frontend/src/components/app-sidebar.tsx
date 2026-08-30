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
  ContextMenu,
  ContextMenuContent,
  ContextMenuItem,
  ContextMenuLabel,
  ContextMenuSeparator,
  ContextMenuTrigger,
} from "@/components/ui/context-menu";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuGroup,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuRadioGroup,
  DropdownMenuRadioItem,
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
  DesktopTitlebarNavigation,
  shouldUseCustomWindowTitlebar,
  shouldUseNativeMacWindowTitlebar,
} from "@/components/tauri/window-titlebar";
// Deep imports on purpose: the Images index re-exports ImagesPage, which would undo its code split.
/* eslint-disable no-restricted-imports */
import {
  isWorkflowEnabled,
  useImageWorkflowStore,
} from "@/features/images/stores/image-workflow-store";
import { WORKFLOW_TABS, type WorkflowId } from "@/features/images/workflows";
/* eslint-enable no-restricted-imports */
import { cn } from "@/lib/utils";
import { copyToClipboardFrom } from "@/lib/copy-to-clipboard";
import { isTauri } from "@/lib/api-base";
import { useWebUpdateCheck } from "@/hooks/use-web-update-check";
import {
  Archive03Icon,
  ArrowDown01Icon,
  ArrowRight02Icon,
  ArrowUp01Icon,
  BadgeInfoIcon,
  BookOpen01Icon,
  BubbleChatIcon,
  ChefHatIcon,
  CloudIcon,
  CpuIcon,
  CursorInfo02Icon,
  DashboardCircleIcon,
  AudioWave01Icon,
  Delete02Icon,
  Download01Icon,
  DownloadSquare01Icon,
  Edit03Icon,
  FolderAddIcon,
  FolderExportIcon,
  FolderOpenIcon,
  Folder01Icon,
  FlimSlateIcon,
  Globe02Icon,
  HelpCircleIcon,
  Image03Icon,
  Logout05Icon,
  Message01Icon,
  MoreHorizontalIcon,
  MoreVerticalIcon,
  PaintBrush02Icon,
  Search01Icon,
  PinIcon,
  PinOffIcon,
  PlusSignIcon,
  PowerIcon,
  PencilEdit02Icon,
  LayoutAlignLeftIcon,
  Settings02Icon,
  Sun03Icon,
  UserIcon,
  ZapIcon,
} from "@hugeicons/core-free-icons";
import { TestTubeOutlineIcon } from "@/lib/hugeicons-derived";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
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
  deleteChatProject,
  deleteChatItem,
  listStoredChatMessages,
  listStoredChatThreads,
  moveChatItemToProject,
  allRecordedSandboxSessionIds,
  notifyChatHistoryUpdated,
  renameChatItem,
  renameChatProject,
  useChatRuntimeStore,
  useChatProjects,
  useChatSearchStore,
  useChatSidebarItems,
  usePinnedChatsStore,
  usePinnedProjectsStore,
  rangeBetween,
  toggleSelected,
  useChatPreferencesStore,
  usePromptQueueUI,
  useSidebarOrganizationStore,
  applyManualOrder,
  dropEdgeFor,
  showsInRecents,
  moveIdBy,
  projectOrderScope,
  reorderIds,
  PINNED_ORDER_SCOPE,
  PROJECT_ORDER_SCOPE,
  RECENTS_ORDER_SCOPE,
  type SidebarChatSort,
  type SidebarOrganizeBy,
  CONVERSATION_MARKDOWN_FORMAT,
  CONVERSATION_MARKDOWN_LABEL,
  type ProjectRecord,
  type SidebarItem,
  type ChatNavigationState,
  adjacentChatItem,
  countUnreadRows,
  nextAttentionChatItem,
  openChatItemById,
  recentChatItemAtSlot,
  useChatNavigationStore,
} from "@/features/chat";
import { sandboxSessionIdFor } from "@/components/assistant-ui/sandbox-files";
import {
  revealSandbox,
  sandboxHasFiles,
} from "@/components/assistant-ui/sandbox-reveal";
import { NewProjectDialog } from "@/features/chat/components/new-project-dialog";
import {
  useAppearanceCustomStore,
  useSettingsDialogStore,
  isSurfaceBackgrounded,
  useShortcutLabel,
  Shortcut,
  type ShortcutId,
  useShortcut,
} from "@/features/settings";
import type { SidebarNavItemId } from "@/features/settings";
import { useEffectiveProfile, UserAvatar } from "@/features/profile";
import { resolveNavRowState } from "@/components/nav-row-state";
import { fetchDeviceType, usePlatformStore } from "@/config/env";
import { videoNavHint } from "@/config/hardware-verdict";
import { clearAuthTokens, logout } from "@/features/auth";
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
import { useExportRuntimeStore } from "@/features/export";
import {
  Fragment,
  useCallback,
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

/** The ⌥⌘1-6 Recents slots, as a constant so the list below is fixed-length. */
const RECENT_SLOT_NUMBERS = [1, 2, 3, 4, 5, 6] as const;

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
  | "messages-jsonl"
  | "csv"
  | "sharegpt-jsonl"
  | typeof CONVERSATION_MARKDOWN_FORMAT;

// An expanded project shows this many recent chats before "Show more".
const PROJECT_CHAT_LIMIT = 4;
// And the Projects section shows this many folders before its own "Show more".
const SIDEBAR_PROJECT_LIMIT = 5;

// The shared radio item ticks on the right; these read as settings, so tick first.
const menuRadioItemClass =
  "pl-9 pr-3 [&>[data-slot=dropdown-menu-radio-item-indicator]]:right-auto [&>[data-slot=dropdown-menu-radio-item-indicator]]:left-3";

// Whether cmd or ctrl adds a row to the selection. This is the user's own
// keyboard, not the host Unsloth runs on, so it reads the browser rather than
// the platform store: a Mac browser on a Linux host still uses cmd. Ctrl is
// left alone on macOS, where ctrl click is the right click chord.
const SELECT_WITH_META =
  typeof navigator !== "undefined" &&
  /mac/i.test(navigator.platform || navigator.userAgent);

// Insertion cue on the edge the row will land on, inset to the pill.
const DROP_CUE_BASE =
  "before:absolute before:inset-x-2 before:h-0.5 before:rounded-full before:bg-primary/70 before:content-['']";
const DROP_CUE_TOP = `${DROP_CUE_BASE} before:-top-px`;
const DROP_CUE_BOTTOM = `${DROP_CUE_BASE} before:-bottom-px`;

// Every list offers the same three orders.
const CHAT_SORT_OPTIONS: Array<{
  value: SidebarChatSort;
  key: TranslationKey;
}> = [
  { value: "priority", key: "shell.organize.priority" },
  { value: "updated", key: "shell.organize.lastUpdated" },
  { value: "manual", key: "shell.organize.manualOrder" },
];
const ORGANIZE_OPTIONS: Array<{
  value: SidebarOrganizeBy;
  key: TranslationKey;
}> = [
  { value: "project", key: "shell.organize.byProject" },
  { value: "list", key: "shell.organize.inOneList" },
];

const CHAT_EXPORT_OPTIONS: Array<{
  label: string;
  format: ConversationExportFormat;
}> = [
  { label: "Training JSONL", format: "raw-jsonl" },
  { label: "Message JSONL", format: "messages-jsonl" },
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
    case "messages-jsonl":
      return exports.exportConversationMessagesJsonl(threadId);
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

async function saveChatToProjectSources(
  item: SidebarItem,
  projectId: string,
): Promise<void> {
  const { saveChatItemAsProjectSource } = await import(
    "@/features/chat/prompt-storage/prompt-storage-dialog"
  );
  await saveChatItemAsProjectSource(item, projectId);
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
/**
 * "Open chat folder" for a browser session, where the backend's file manager is
 * not the user's. Radix's `disabled` takes the row's pointer events away and a
 * tooltip is blocked while the menu owns the screen, so the row stays enabled,
 * refuses the select itself, and drives a controlled tooltip off a
 * pointer-events-none anchor (as the MCP rows do).
 *
 * The reason is carried twice: the tooltip hangs off an aria-hidden anchor and
 * opens on hover, which a screen reader never reaches and a touch device does
 * not have, so `title` describes the row itself and selecting it (tap or Enter)
 * opens the hint rather than doing nothing.
 */
function OpenChatFolderUnavailableItem() {
  const [hintOpen, setHintOpen] = useState(false);

  return (
    <DropdownMenuItem
      aria-disabled={true}
      title="Opening the folder needs the desktop app. In a browser, save a file from the card that created it."
      className="relative opacity-50"
      onSelect={(event) => {
        event.preventDefault();
        setHintOpen(true);
      }}
      onPointerEnter={() => setHintOpen(true)}
      onPointerLeave={() => setHintOpen(false)}
      onFocus={() => setHintOpen(true)}
      onBlur={() => setHintOpen(false)}
    >
      <HugeiconsIcon icon={FolderOpenIcon} strokeWidth={1.75} className="size-icon" />
      <span>Open chat folder</span>
      <Tooltip open={hintOpen}>
        {/* Our wrapper, not the raw primitive: it registers the trigger element,
            without which the tooltip counts itself blocked by the open menu. */}
        <TooltipTrigger asChild={true}>
          <span
            aria-hidden={true}
            className="pointer-events-none absolute inset-y-0 right-0 w-0"
          />
        </TooltipTrigger>
        <TooltipContent side="right" className="max-w-[220px]">
          Opening the folder needs the desktop app. In a browser, save a file
          from the card that created it.
        </TooltipContent>
      </Tooltip>
    </DropdownMenuItem>
  );
}

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
          <HugeiconsIcon icon={icon} strokeWidth={1.75} className="size-icon! shrink-0 translate-x-0.5 group-hover/menu-button:animate-icon-pop" />
          <span className="text-ui-14p5 leading-ui-19 tracking-nav">{label}</span>
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
/** How long a selection chord keeps a repeat press off the open chat. */
const SELECTION_ACTION_GRACE_MS = 750;
/** The sidebar's own element, present on desktop and inside the mobile drawer. */
const SIDEBAR_SELECTOR = '[data-slot="sidebar"]';
const VERDICT_UNKNOWN_POLL_MS = 3000;
const SELF_HEAL_POLL_MS = 15000;
const VERDICT_POLL_STALL_MS = 30000;
// The backend refreshes its physical GPU inventory on a 60s TTL and can reclassify a host
// without a restart: attach an eGPU to a CPU-torch machine and no_gpu becomes
// torch_cpu_build, or a driver finishes restarting and the host stops being chat-only at
// all. Nothing else re-reads the verdict, so without this the new hint is unreachable for
// the rest of the session. Matched to that TTL rather than to SELF_HEAL_POLL_MS: polling
// faster than the backend can change its answer is pure request traffic.
const INVENTORY_POLL_MS = 60000;
// The verdicts the inventory can still move. Everything else describes something a probe
// cannot change (an Intel Mac stays an Intel Mac), and polling those forever would be a
// request a minute for the life of the session on hosts working exactly as intended.
const INVENTORY_SENSITIVE_REASONS = new Set([
  "no_gpu",
  "torch_cpu_build",
  "torch_cuda_unavailable",
]);

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
      {spinner && <Spinner className="size-3.5 shrink-0 text-muted-foreground" />}
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
  // Read from the shortcuts store, not the shipped default: a rebound or
  // cleared action must not leave the hint advertising a dead chord. Both
  // already render in the platform's own notation.
  const searchShortcutLabel = useShortcutLabel("searchChats");
  const settingsShortcutLabel = useShortcutLabel("openSettings");
  const { pathname, search, href } = useRouterState({
    select: (s) => ({
      pathname: s.location.pathname,
      search: s.location.search as Record<string, string | undefined>,
      // Pathname and search as one string, so a dep can follow it without a
      // fresh object every render.
      href: s.location.href,
    }),
  });
  const {
    pinned,
    togglePinned,
    isMobile,
    openMobile,
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
  // SidebarProvider is mounted at the route root and outlives the navigation,
  // and the workspace chords register up there too, above it, so they cannot
  // call what every row below calls by hand. Closing on the route covers both,
  // and anything else that navigates from outside this file.
  useEffect(() => {
    if (isMobile) setOpenMobile(false);
  }, [href, isMobile, setOpenMobile]);

  const chatOnly = usePlatformStore((s) => s.isChatOnly());
  const chatOnlyReason = usePlatformStore((s) => s.chatOnlyReason);
  const chatOnlyDetail = usePlatformStore((s) => s.chatOnlyDetail);
  const detectionDeferred = usePlatformStore((s) => s.detectionDeferred);
  // Until /api/health answers, `chatOnly` is the browser-platform guess, so every Mac painted
  // Train and Video blacked out on load and only recovered once the backend reported. Gate the
  // rows on a measured verdict and let them spin until it lands.
  const capabilitiesUnknown = usePlatformStore((s) => s.capabilitiesUnknown());
  const chatOnlyMeasured = chatOnly && !capabilitiesUnknown;
  // Explain a greyed-out Train (chat-only host) on hover. Export stays navigable so its page
  // can show a precise reason.
  const trainDisabledHint: string | undefined = !chatOnlyMeasured
    ? undefined
    : chatOnlyReason === "mlx_unavailable"
      ? // The gate is all-or-nothing across mlx, mlx-lm and mlx-vlm, and a resolver
        // backtrack leaves a stack that is present but unusable. Naming the package
        // that is missing, too old, or refusing to import is what makes this
        // actionable to someone whose `unsloth studio update` has already run.
        chatOnlyDetail
        ? `Training needs MLX: ${chatOnlyDetail}. Run \`unsloth studio update\` to enable Train.`
        : "Training needs MLX. Run `unsloth studio update` to enable Train."
      : chatOnlyReason === "intel_mac"
        ? "Training needs Apple Silicon or a GPU. Intel Macs are chat-only."
        : chatOnlyReason === "torch_cpu_build" ||
            chatOnlyReason === "torch_cuda_unavailable"
          ? // The host HAS GPUs; this PyTorch cannot open them. "Get a GPU" is both wrong
            // and unactionable here, so name the installed build and point at the repair.
            chatOnlyDetail
            ? `Training needs a working PyTorch GPU build. This machine's GPUs were detected but PyTorch ${chatOnlyDetail} cannot use them; repair the installation.`
            : "Training needs a working PyTorch GPU build. This machine's GPUs were detected but PyTorch cannot use them; repair the installation."
          : chatOnlyReason === "no_gpu"
            ? "Training needs an NVIDIA or AMD GPU."
            : undefined;
  // Everything without a hint reaches VideoPage, which answers from the backend's video verdict.
  const videoDisabledHint = videoNavHint(chatOnlyMeasured, chatOnlyReason);
  const videoDisabled = videoDisabledHint !== undefined;

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
    // verdict (studio-page reads the same store, so it recovers with it; video-page reads the
    // backend's video verdict instead and needs nothing from here).
    const inventorySensitive =
      chatOnly && INVENTORY_SENSITIVE_REASONS.has(chatOnlyReason ?? "");
    if (selfHealSettled && !capabilitiesUnknown && !inventorySensitive) return;
    let pollingSince = 0;
    // Which read currently owns the guard. A read that outlived the stall window is replaced,
    // and the replacement takes the guard with it; without an owner the abandoned read's
    // `finally` would clear a guard it no longer holds and let the next tick stack another
    // forced read onto the slow backend, every interval, which is the pile-up this prevents.
    let pollOwner = 0;
    const id = window.setInterval(() => {
      // A backend still importing torch answers slowly, so skip while a re-read is outstanding
      // rather than stacking them against it. Bounded, or a request that never settles would
      // hold the poll off for good.
      if (pollingSince && Date.now() - pollingSince < VERDICT_POLL_STALL_MS) return;
      const owned = ++pollOwner;
      pollingSince = Date.now();
      void fetchDeviceType({ force: true })
        .catch(() => undefined)
        .finally(() => {
          if (owned === pollOwner) pollingSince = 0;
        });
    }, capabilitiesUnknown
      ? VERDICT_UNKNOWN_POLL_MS
      : selfHealSettled
        ? INVENTORY_POLL_MS
        : SELF_HEAL_POLL_MS);
    return () => window.clearInterval(id);
  }, [capabilitiesUnknown, chatOnly, chatOnlyReason, detectionDeferred]);

  const [shutdownOpen, setShutdownOpen] = useState(false);

  const isChatRoute = pathname.startsWith("/chat");
  const isStudioRoute = pathname === "/studio" || pathname.startsWith("/studio/");
  const [chatOpen, setChatOpen] = useState(true);

  // Hover previews the flyout; a primary click pins that preview open. The trigger owns pointer
  // clicks so Radix cannot interpret the already-hover-open menu as a request to close it.
  const [moreHoverOpen, setMoreHoverOpen] = useState(false);
  const [morePinnedOpen, setMorePinnedOpen] = useState(false);
  const moreOpen = moreHoverOpen || morePinnedOpen;
  const moreCloseTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const clearMoreCloseTimer = useCallback(() => {
    if (!moreCloseTimer.current) return;
    clearTimeout(moreCloseTimer.current);
    moreCloseTimer.current = null;
  }, []);
  const openMorePreview = useCallback(() => {
    clearMoreCloseTimer();
    setMoreHoverOpen(true);
  }, [clearMoreCloseTimer]);
  const closeMorePreviewSoon = useCallback(() => {
    clearMoreCloseTimer();
    moreCloseTimer.current = setTimeout(() => setMoreHoverOpen(false), 180);
  }, [clearMoreCloseTimer]);
  const handleMoreOpenChange = useCallback((next: boolean) => {
    if (next) {
      setMorePinnedOpen(true);
      return;
    }
    setMorePinnedOpen(false);
    setMoreHoverOpen(false);
  }, []);
  useEffect(
    () => () => {
      clearMoreCloseTimer();
    },
    [clearMoreCloseTimer],
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
  const isExportRoute = pathname === "/export" || pathname.startsWith("/export/");
  // Training runs surface as sidebar "Recents" on Train/Recipes/Export, else chat recents.
  // Read up here because the chat lists below only exist when this is off.
  const trainingRecentsRoute = isStudioRoute || isRecipesRoute || isExportRoute;
  const { items: runItems } = useTrainingHistorySidebarItems(
    !chatOnly && trainingRecentsRoute,
  );
  const showTrainingRecents =
    !chatOnly && trainingRecentsRoute && runItems.length > 0;
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
  const setPinnedChats = usePinnedChatsStore((s) => s.setPinned);
  const unpinChat = usePinnedChatsStore((s) => s.unpin);
  const confirmDeleteChats = useChatPreferencesStore(
    (s) => s.confirmDeleteChats,
  );
  const alwaysDeleteChatFiles = useChatPreferencesStore(
    (s) => s.alwaysDeleteChatFiles,
  );
  const pinnedIdSet = useMemo(() => new Set(pinnedIds), [pinnedIds]);
  // Which row every thread belongs to, listed or not. The published lists carry
  // only what is on screen, so this is what keeps an unread Compare row behind
  // a collapsed section counting as one chat.
  const rowIdByThreadId = useMemo(() => {
    const map: Record<string, string> = {};
    for (const item of allChatItems) {
      for (const threadId of getSidebarItemThreadIds(item)) map[threadId] = item.id;
    }
    return map;
  }, [allChatItems]);
  const organizeBy = useSidebarOrganizationStore((s) => s.organizeBy);
  const chatSort = useSidebarOrganizationStore((s) => s.chatSort);
  const pinnedSort = useSidebarOrganizationStore((s) => s.pinnedSort);
  const manualOrder = useSidebarOrganizationStore((s) => s.manualOrder);
  const setOrganizeBy = useSidebarOrganizationStore((s) => s.setOrganizeBy);
  const setChatSort = useSidebarOrganizationStore((s) => s.setChatSort);
  const setPinnedSort = useSidebarOrganizationStore((s) => s.setPinnedSort);
  const setManualOrder = useSidebarOrganizationStore((s) => s.setManualOrder);
  // With the Projects section on, a project chat lives in its folder and
  // repeating it here would be noise. With it off there are no folders, so
  // Recents is where those chats go, and a new project chat still lands
  // somewhere visible. Pinned chats are held back either way: the Pinned
  // section renders those.
  const recentChatItems = useMemo(
    () =>
      allChatItems.filter(
        (item) =>
          !pinnedIdSet.has(item.id) && showsInRecents(item.projectId, organizeBy),
      ),
    [allChatItems, pinnedIdSet, organizeBy],
  );
  const [pinnedOpen, setPinnedOpen] = useState(true);
  const [projectsOpen, setProjectsOpen] = useState(true);
  const [showAllProjects, setShowAllProjects] = useState(false);
  // Pinning a project now sorts it to the top of Projects, not into its own section.
  const pinnedProjectIds = usePinnedProjectsStore((s) => s.pinnedIds);
  const toggleProjectPin = usePinnedProjectsStore((s) => s.togglePin);
  const pinnedProjectIdSet = useMemo(
    () => new Set(pinnedProjectIds),
    [pinnedProjectIds],
  );
  // Pinned chats, in pin order. A pinned project chat also stays in its folder.
  const pinnedChatItems = useMemo(() => {
    const byId = new Map(allChatItems.map((item) => [item.id, item]));
    return pinnedIds
      .map((id) => byId.get(id))
      .filter((item): item is SidebarItem => Boolean(item));
  }, [allChatItems, pinnedIds]);
  // Chats per project, newest first. Pinned ones stay: a chat belongs to its
  // project either way, and these rows are mirrored in Recents regardless.
  const chatsByProjectId = useMemo(() => {
    const map = new Map<string, SidebarItem[]>();
    for (const item of allChatItems) {
      if (!item.projectId) continue;
      const list = map.get(item.projectId);
      if (list) list.push(item);
      else map.set(item.projectId, [item]);
    }
    for (const list of map.values())
      list.sort((a, b) => b.updatedAt - a.updatedAt);
    return map;
  }, [allChatItems]);
  // Every project gets a folder: pinned first in pin order, then by activity,
  // then whatever the user dragged, which outranks both. Activity comes from
  // the member chats, since a project's own updatedAt only moves when its name,
  // instructions or archived flag are edited.
  const sidebarProjectRecords = useMemo(() => {
    const lastActivityAt = (project: ProjectRecord) => {
      let latest = project.updatedAt ?? project.createdAt;
      for (const chat of chatsByProjectId.get(project.id) ?? []) {
        if (chat.updatedAt > latest) latest = chat.updatedAt;
      }
      return latest;
    };
    const byId = new Map(projects.map((p) => [p.id, p]));
    const pinned = pinnedProjectIds
      .map((id) => byId.get(id))
      .filter((p): p is ProjectRecord => Boolean(p));
    const rest = projects
      .filter((p) => !pinnedProjectIdSet.has(p.id))
      .sort((a, b) => lastActivityAt(b) - lastActivityAt(a));
    return applyManualOrder(
      [...pinned, ...rest],
      manualOrder[PROJECT_ORDER_SCOPE],
      (project) => project.id,
    );
  }, [
    projects,
    pinnedProjectIds,
    pinnedProjectIdSet,
    manualOrder,
    chatsByProjectId,
  ]);
  // Memoised for its identity, not for the slice. It feeds the rendered-row set
  // the selection guard depends on, and that effect sets state: React re-renders
  // once to find the bail-out, which would rebuild this array and schedule the
  // effect again, without end.
  const visibleProjectRecords = useMemo(
    () =>
      showAllProjects
        ? sidebarProjectRecords
        : sidebarProjectRecords.slice(0, SIDEBAR_PROJECT_LIMIT),
    [showAllProjects, sidebarProjectRecords],
  );
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
    ? (search.thread as string | undefined) ??
      (search.compare as string | undefined) ??
      storeThreadId ??
      undefined
    : undefined;
  const queueByThreadId = usePromptQueueUI((s) => s.byThreadId);
  // In the navigation store, not local state: the unread chords register
  // outside this tree.
  const unreadThreadIds = useChatNavigationStore((s) => s.unreadThreadIds);
  const markThreadsUnread = useChatNavigationStore((s) => s.markThreadsUnread);
  const clearThreadsUnread = useChatNavigationStore(
    (s) => s.clearThreadsUnread,
  );
  const noteViewed = useChatNavigationStore((s) => s.noteViewed);
  const previousRunningByThreadIdRef = useRef<Record<string, boolean>>({});
  const activeVisibleThreadIds = useMemo(() => {
    if (!activeThreadId) {
      return [];
    }
    const activeItem = allChatItems.find((item) => item.id === activeThreadId);
    return activeItem ? getSidebarItemThreadIds(activeItem) : [activeThreadId];
  }, [activeThreadId, allChatItems]);
  const activeVisibleThreadIdKey = activeVisibleThreadIds.join("\n");

  // "Priority" lifts rows wanting attention (generating, queued, unread), then
  // falls back to recency, which "Last updated" sorts by outright.
  const chatPriorityRank = useCallback(
    (item: SidebarItem) => {
      const ids = getSidebarItemThreadIds(item);
      if (ids.some((id) => runningByThreadId[id])) return 0;
      if (ids.some((id) => queueByThreadId[id])) return 1;
      if (ids.some((id) => unreadThreadIds.has(id))) return 2;
      return 3;
    },
    [runningByThreadId, queueByThreadId, unreadThreadIds],
  );
  const sortChatItems = useCallback(
    (
      items: SidebarItem[],
      scope: string,
      mode: SidebarChatSort,
    ): SidebarItem[] => {
      if (mode === "manual") {
        // Incoming order is the list's own rule, so undragged rows keep it:
        // newest first in Recents, pin order in Pinned.
        return applyManualOrder(items, manualOrder[scope], (item) => item.id);
      }
      if (mode === "priority") {
        return [...items].sort(
          (a, b) =>
            chatPriorityRank(a) - chatPriorityRank(b) ||
            b.updatedAt - a.updatedAt,
        );
      }
      return [...items].sort((a, b) => b.updatedAt - a.updatedAt);
    },
    [manualOrder, chatPriorityRank],
  );
  const sortedRecentChatItems = useMemo(
    () => sortChatItems(recentChatItems, RECENTS_ORDER_SCOPE, chatSort),
    [recentChatItems, sortChatItems, chatSort],
  );
  const sortedPinnedChatItems = useMemo(
    () => sortChatItems(pinnedChatItems, PINNED_ORDER_SCOPE, pinnedSort),
    [pinnedChatItems, sortChatItems, pinnedSort],
  );
  // The open chat heads the ⌃Tab stack, however it was opened.
  useEffect(() => {
    if (activeThreadId) noteViewed(activeThreadId);
  }, [activeThreadId, noteViewed]);
  // A walk ends when its modifiers come up, as an app switcher's does. A no-op
  // when no walk is running.
  useEffect(() => {
    const end = () => useChatNavigationStore.getState().endTraversal();
    const onKeyUp = (event: KeyboardEvent) => {
      if (event.ctrlKey || event.metaKey || event.altKey || event.shiftKey) {
        return;
      }
      end();
    };
    window.addEventListener("keyup", onKeyUp);
    // Losing the window ends it too: ⌘Tab away mid-walk and the release lands
    // elsewhere, leaving the walk frozen, so the next press carries on instead
    // of toggling back and the stack never takes the chat it landed on.
    window.addEventListener("blur", end);
    return () => {
      window.removeEventListener("keyup", onKeyUp);
      window.removeEventListener("blur", end);
    };
  }, []);

  const sortedChatsByProjectId = useMemo(() => {
    const map = new Map<string, SidebarItem[]>();
    for (const [projectId, items] of chatsByProjectId) {
      map.set(
        projectId,
        sortChatItems(items, projectOrderScope(projectId), chatSort),
      );
    }
    return map;
  }, [chatsByProjectId, sortChatItems, chatSort]);
  // One id array per list, shared by every row in it. Built per row, these
  // would be N arrays of length N on each render.
  const recentRowIds = useMemo(
    () => sortedRecentChatItems.map((item) => item.id),
    [sortedRecentChatItems],
  );
  const pinnedRowIds = useMemo(
    () => sortedPinnedChatItems.map((item) => item.id),
    [sortedPinnedChatItems],
  );
  // Whole lists, not the visible slices, so a drop cannot lose what a
  // collapsed "Show more" is hiding.
  // The project chats actually on screen. Grouping by project keeps them out of
  // Recents, so without these the chords cannot see them at all. Same rule the
  // Projects section renders by, collapsed folders and per-folder limit included.
  // All three chat groups leave the tree on Train/Recipes/Export and are hidden
  // on the icon rail, so a chord must not reach what they hold. Same rule a
  // collapsed section follows, read off the sidebar as a whole. The mobile
  // sheet is not part of it: it carries no collapsible state, and gating on it
  // would strand the chords on any window narrow enough to count as mobile.
  const chatListsOnScreen =
    !isStudioRoute &&
    !showTrainingRecents &&
    (isMobile || sidebarState !== "collapsed");
  // Selecting needs the rows, not just the lists. A closed mobile sheet unmounts
  // them as the icon rail does, so Select All would build a selection with
  // nothing on screen and Archive, Pin and Mark unread would take it over the
  // open chat. Navigation is deliberately exempt: it moves the chat the user IS
  // looking at, and the sheet is closed for most of its life on a narrow window.
  const chatRowsOnScreen = chatListsOnScreen && (!isMobile || openMobile);
  const renderedProjectChatItems = useMemo(() => {
    if (!chatListsOnScreen || organizeBy !== "project" || !projectsOpen)
      return [];
    const out: SidebarItem[] = [];
    for (const project of visibleProjectRecords) {
      if (collapsedProjectIds.has(project.id)) continue;
      const chats = sortedChatsByProjectId.get(project.id) ?? [];
      out.push(
        ...(expandedChatProjectIds.has(project.id)
          ? chats
          : chats.slice(0, PROJECT_CHAT_LIMIT)),
      );
    }
    return out;
  }, [
    chatListsOnScreen,
    organizeBy,
    projectsOpen,
    visibleProjectRecords,
    collapsedProjectIds,
    expandedChatProjectIds,
    sortedChatsByProjectId,
  ]);
  // A collapsed section is not on screen either, so its rows are not walked or
  // selected any more than a collapsed folder's are.
  const visiblePinnedItems = useMemo(
    () => (chatListsOnScreen && pinnedOpen ? sortedPinnedChatItems : []),
    [chatListsOnScreen, pinnedOpen, sortedPinnedChatItems],
  );
  const visibleRecentItems = useMemo(
    () => (chatListsOnScreen && chatOpen ? sortedRecentChatItems : []),
    [chatListsOnScreen, chatOpen, sortedRecentChatItems],
  );
  // Every row on screen, in one set. The three arrays above already fold in
  // each section's disclosure, each folder's, and the per-folder limit, so a
  // selection can be held to what is rendered without restating any of it.
  const renderedChatIds = useMemo(() => {
    const ids = new Set<string>();
    for (const item of visiblePinnedItems) ids.add(item.id);
    for (const item of renderedProjectChatItems) ids.add(item.id);
    for (const item of visibleRecentItems) ids.add(item.id);
    return ids;
  }, [visiblePinnedItems, renderedProjectChatItems, visibleRecentItems]);
  // The folder rows, selectable in their own right and leaving the screen on
  // their own terms: the section closes, the sidebar organizes by date, or a
  // "show less" takes back the overflow. The chat sets above say nothing about
  // that, since a folder with no chats in view is still a row.
  const renderedProjectIds = useMemo(() => {
    if (!chatListsOnScreen || organizeBy !== "project" || !projectsOpen) {
      return new Set<string>();
    }
    return new Set(visibleProjectRecords.map((project) => project.id));
  }, [chatListsOnScreen, organizeBy, projectsOpen, visibleProjectRecords]);
  // Rows wanting attention, most urgent first. Same rule the Priority sort uses.
  const attentionItemIds = useMemo(
    () =>
      [...visiblePinnedItems, ...renderedProjectChatItems, ...visibleRecentItems]
        .filter((item) => chatPriorityRank(item) < 3)
        .sort(
          (a, b) =>
            chatPriorityRank(a) - chatPriorityRank(b) ||
            b.updatedAt - a.updatedAt,
        )
        .map((item) => item.id),
    [
      visiblePinnedItems,
      renderedProjectChatItems,
      visibleRecentItems,
      chatPriorityRank,
    ],
  );
  // Publish the finished order, so the chords cannot disagree with the screen.
  const publishLists = useChatNavigationStore((s) => s.publishLists);
  useEffect(() => {
    publishLists({
      pinnedItems: visiblePinnedItems,
      projectItems: renderedProjectChatItems,
      recentItems: visibleRecentItems,
      attentionItemIds,
      activeItemId: activeThreadId ?? null,
    });
  }, [
    publishLists,
    visiblePinnedItems,
    renderedProjectChatItems,
    visibleRecentItems,
    attentionItemIds,
    activeThreadId,
  ]);

  const projectRowIds = useMemo(
    () => sidebarProjectRecords.map((project) => project.id),
    [sidebarProjectRecords],
  );
  const projectChatRowIds = useMemo(() => {
    const map = new Map<string, string[]>();
    for (const [projectId, items] of sortedChatsByProjectId) {
      map.set(
        projectId,
        items.map((item) => item.id),
      );
    }
    return map;
  }, [sortedChatsByProjectId]);
  // How many nested rows the Projects section renders, so the bottom fade can
  // re-measure when regrouping or a disclosure changes the list height.
  const projectChatRowCount = useMemo(() => {
    if (organizeBy !== "project") return 0;
    let rows = 0;
    for (const project of visibleProjectRecords) {
      if (collapsedProjectIds.has(project.id)) continue;
      const chats = sortedChatsByProjectId.get(project.id) ?? [];
      rows += expandedChatProjectIds.has(project.id)
        ? chats.length
        : Math.min(chats.length, PROJECT_CHAT_LIMIT);
      // The "Show more" row counts too.
      if (chats.length > PROJECT_CHAT_LIMIT) rows += 1;
    }
    return rows;
  }, [
    organizeBy,
    visibleProjectRecords,
    collapsedProjectIds,
    expandedChatProjectIds,
    sortedChatsByProjectId,
  ]);

  // Multi-select. Ids only: a row that gets deleted elsewhere drops out of
  // selectedChatItems on its own rather than leaving a ghost to act on.
  const [selectedChatIds, setSelectedChatIds] = useState<ReadonlySet<string>>(
    () => new Set(),
  );
  // Which list the anchor belongs to, since one chat can sit in two lists.
  const selectionAnchorRef = useRef<{ scope: string; id: string } | null>(null);
  const selectedChatItems = useMemo(
    () => allChatItems.filter((item) => selectedChatIds.has(item.id)),
    [allChatItems, selectedChatIds],
  );
  const selectionCount = selectedChatItems.length;

  // Projects select separately: a mixed selection has no shared bulk action,
  // so picking one kind drops the other.
  const [selectedProjectIds, setSelectedProjectIds] = useState<
    ReadonlySet<string>
  >(() => new Set());
  const projectAnchorRef = useRef<string | null>(null);
  const selectedProjectRecords = useMemo(
    () => projects.filter((project) => selectedProjectIds.has(project.id)),
    [projects, selectedProjectIds],
  );
  const projectSelectionCount = selectedProjectRecords.length;

  // Each kind drops the other, anchor included: a stale anchor would shift-select
  // a range from a row the user can no longer see selected.
  const dropChatSelection = useCallback(() => {
    selectionAnchorRef.current = null;
    setSelectedChatIds((prev) => (prev.size === 0 ? prev : new Set()));
  }, []);
  const dropProjectSelection = useCallback(() => {
    projectAnchorRef.current = null;
    setSelectedProjectIds((prev) => (prev.size === 0 ? prev : new Set()));
  }, []);

  const clearSelection = useCallback(() => {
    dropChatSelection();
    dropProjectSelection();
  }, [dropChatSelection, dropProjectSelection]);

  // Emptying the published lists is not enough: Archive, Pin, Mark unread and
  // Delete all prefer the selection over the open chat, and a selection shows
  // nowhere but the rows, its count living in their context menus. Carried onto
  // Train or behind the icon rail it would be invisible and still be what the
  // chords hit. Same reason opening a row drops it.
  // Sections close one at a time, though, and a "show less" takes back only
  // its own overflow, so the rest of the selection is still on screen and
  // still worth acting on. Drop what went and keep what stayed.
  useEffect(() => {
    if (!chatRowsOnScreen) {
      clearSelection();
      return;
    }
    const anchor = selectionAnchorRef.current;
    if (anchor && !renderedChatIds.has(anchor.id)) {
      selectionAnchorRef.current = null;
    }
    setSelectedChatIds((prev) => {
      if (prev.size === 0) return prev;
      const kept = new Set<string>();
      for (const id of prev) {
        if (renderedChatIds.has(id)) kept.add(id);
      }
      return kept.size === prev.size ? prev : kept;
    });
    // Folder rows go the same way. Left behind, a project whose section the
    // user closed keeps the selection alive with nothing on screen, and the
    // tool card's Escape steps aside for it: the press that should have
    // declined a call clears a selection instead.
    const projectAnchor = projectAnchorRef.current;
    if (projectAnchor && !renderedProjectIds.has(projectAnchor)) {
      projectAnchorRef.current = null;
    }
    setSelectedProjectIds((prev) => {
      if (prev.size === 0) return prev;
      const kept = new Set<string>();
      for (const id of prev) {
        if (renderedProjectIds.has(id)) kept.add(id);
      }
      return kept.size === prev.size ? prev : kept;
    });
  }, [chatRowsOnScreen, clearSelection, renderedChatIds, renderedProjectIds]);

  /** Select every chat row on screen, pinned block included. */
  const selectAllChats = useCallback(() => {
    if (!chatRowsOnScreen) return;
    const ids = [
      ...visiblePinnedItems,
      ...renderedProjectChatItems,
      ...visibleRecentItems,
    ].map((item) => item.id);
    if (ids.length === 0) return;
    dropProjectSelection();
    // No anchor: a shift click after this one has no row to reach back to.
    selectionAnchorRef.current = null;
    setSelectedChatIds(new Set(ids));
  }, [
    chatRowsOnScreen,
    visiblePinnedItems,
    renderedProjectChatItems,
    visibleRecentItems,
    dropProjectSelection,
  ]);

  // Escape leaves a selection, as it does the menus. A passive listener rather
  // than one that consumes the key: dictation's Escape reads defaultPrevented
  // first, and a stale selection must not outrank a live recording. Declining a
  // tool call is the Escape that must not double up, and it steps aside on
  // `selectionActive` below.
  const selectionActive = selectionCount > 0 || projectSelectionCount > 0;
  useEffect(() => {
    if (!selectionActive) return;
    const onKeyDown = (event: KeyboardEvent) => {
      // Bare, and only bare. Escape with a modifier is somebody else's chord,
      // ⇧Esc for Clear all unreads among them, and dropping the selection under
      // one would leave Archive or Pin pointing elsewhere. defaultPrevented for
      // the same reason: a menu closing on Escape is not a request to lose it.
      if (event.key !== "Escape" || event.defaultPrevented) return;
      if (event.metaKey || event.ctrlKey || event.altKey || event.shiftKey) {
        return;
      }
      clearSelection();
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [selectionActive, clearSelection]);
  const setSelectionActive = useChatNavigationStore((s) => s.setSelectionActive);
  useEffect(() => {
    setSelectionActive(selectionActive);
    return () => setSelectionActive(false);
  }, [selectionActive, setSelectionActive]);

  /** True when the click was a selection gesture, so the row must not navigate. */
  function handleSelectionClick(
    event: React.MouseEvent,
    item: SidebarItem,
    list: { scope: string; ids: string[] },
  ): boolean {
    dropProjectSelection();
    if (SELECT_WITH_META ? event.metaKey : event.ctrlKey) {
      setSelectedChatIds((prev) => toggleSelected(prev, item.id));
      selectionAnchorRef.current = { scope: list.scope, id: item.id };
      return true;
    }
    if (!event.shiftKey) return false;
    const anchor = selectionAnchorRef.current;
    const sameList = anchor?.scope === list.scope;
    // Shift without an anchor in this list starts one here.
    if (!sameList) selectionAnchorRef.current = { scope: list.scope, id: item.id };
    setSelectedChatIds(
      new Set(
        sameList && anchor
          ? rangeBetween(list.ids, anchor.id, item.id)
          : [item.id],
      ),
    );
    return true;
  }

  /** Right-clicking outside the selection acts on that row alone. */
  function selectForContextMenu(
    item: SidebarItem,
    list: { scope: string; ids: string[] },
  ) {
    // Before the early return: the menu that opens acts on chats, so folders
    // left highlighted would misreport what a bulk action is about to touch.
    dropProjectSelection();
    if (selectedChatIds.has(item.id)) return;
    selectionAnchorRef.current = { scope: list.scope, id: item.id };
    setSelectedChatIds(new Set([item.id]));
  }

  /** True when the click selected folders instead of opening one. */
  function handleProjectSelectionClick(
    event: React.MouseEvent,
    projectId: string,
  ): boolean {
    const additive = SELECT_WITH_META ? event.metaKey : event.ctrlKey;
    if (!additive && !event.shiftKey) return false;
    dropChatSelection();
    if (additive) {
      setSelectedProjectIds((prev) => toggleSelected(prev, projectId));
      projectAnchorRef.current = projectId;
      return true;
    }
    const anchorId = projectAnchorRef.current;
    if (!anchorId) projectAnchorRef.current = projectId;
    setSelectedProjectIds(
      new Set(
        anchorId ? rangeBetween(projectRowIds, anchorId, projectId) : [projectId],
      ),
    );
    return true;
  }

  function selectProjectForContextMenu(projectId: string) {
    dropChatSelection();
    if (selectedProjectIds.has(projectId)) return;
    projectAnchorRef.current = projectId;
    setSelectedProjectIds(new Set([projectId]));
  }

  const allSelectedProjectsPinned =
    projectSelectionCount > 0 &&
    selectedProjectRecords.every((project) =>
      pinnedProjectIdSet.has(project.id),
    );

  function pinSelectedProjects(pinned: boolean) {
    for (const project of selectedProjectRecords) {
      if (pinnedProjectIdSet.has(project.id) !== pinned) {
        toggleProjectPin(project.id);
      }
    }
    clearSelection();
  }

  function deleteSelectedProjects() {
    if (projectSelectionCount === 0) return;
    openDeleteDialog({ kind: "projects", projects: selectedProjectRecords });
  }

  const allSelectedPinned =
    selectionCount > 0 &&
    selectedChatItems.every((item) => pinnedIdSet.has(item.id));

  function pinSelected(pinned: boolean) {
    setPinnedChats(
      selectedChatItems.map((item) => item.id),
      pinned,
    );
    clearSelection();
  }

  function markSelectedUnread() {
    const threadIds = selectedChatItems.flatMap(getSidebarItemThreadIds);
    clearSelection();
    markThreadsUnread(threadIds, rowIdByThreadId);
  }

  async function archiveSelected() {
    const items = selectedChatItems;
    clearSelection();
    // Sequential: each archive can reset the active thread, and two of those
    // racing would fight over where the chat pane lands.
    let archived = 0;
    let failure: unknown;
    for (const item of items) {
      // Per item, so one bad chat does not strand the rest of the batch
      // unarchived with the selection already gone.
      try {
        await archiveChatItem(item, activeThreadId, (view) => {
          navigate({
            to: "/chat",
            search: item.projectId
              ? { project: item.projectId }
              : { new: view.newThreadNonce },
          });
        });
        archived += 1;
      } catch (err) {
        failure = err;
      }
    }
    // One notice for the batch, not one per chat. A partial batch gets both:
    // where the archived ones went, and that the rest did not make it.
    if (archived > 0) showArchivedChatsToast();
    if (archived < items.length) {
      toast.error(translate("settings.data.failedToArchiveChats"), {
        description: failure instanceof Error ? failure.message : undefined,
      });
    }
  }

  function deleteSelected() {
    const items = selectedChatItems;
    if (items.length === 0) return;
    if (confirmDeleteChats) {
      openDeleteDialog({ kind: "chats", items });
      return;
    }
    clearSelection();
    void (async () => {
      for (const item of items) {
        await deleteChatWithCleanup(item, {
          deleteFiles: alwaysDeleteChatFiles,
        });
      }
    })();
  }

  // A row must know which list it is dragged within: the same chat can sit in
  // its project and in Recents, and each list keeps its own order.
  const [draggingRow, setDraggingRow] = useState<{
    id: string;
    scope: string;
  } | null>(null);
  const [dropTargetRowId, setDropTargetRowId] = useState<string | null>(null);
  const manualDragEnabled = chatSort === "manual";
  const pinnedDragEnabled = pinnedSort === "manual";

  /** The cue class for this row, or undefined when it is not the drop target. */
  function dropCueClass(
    scope: string | undefined,
    orderedIds: string[] | undefined,
    rowId: string,
  ): string | undefined {
    if (
      scope === undefined ||
      dropTargetRowId !== rowId ||
      draggingRow?.scope !== scope ||
      draggingRow.id === rowId
    ) {
      return undefined;
    }
    return dropEdgeFor(orderedIds ?? [], draggingRow.id, rowId) === "bottom"
      ? DROP_CUE_BOTTOM
      : DROP_CUE_TOP;
  }

  /**
   * Menu path to the same reorder dragging does. Touch browsers never fire
   * dragstart and a keyboard cannot drag, so a manually ordered list is
   * unreorderable without this.
   */
  function renderMoveRowItems(
    scope: string,
    orderedIds: string[],
    rowId: string,
    // Passed in, not searched for: this runs for every row on every render.
    at: number,
  ) {
    const move = (delta: number) => {
      const next = moveIdBy(orderedIds, rowId, delta);
      if (next !== orderedIds) setManualOrder(scope, next);
    };
    return (
      <>
        <DropdownMenuItem disabled={at <= 0} onSelect={() => move(-1)}>
          <HugeiconsIcon icon={ArrowUp01Icon} strokeWidth={1.75} className="size-icon" />
          <span>{t("shell.organize.moveUp")}</span>
        </DropdownMenuItem>
        <DropdownMenuItem
          disabled={at === -1 || at >= orderedIds.length - 1}
          onSelect={() => move(1)}
        >
          <HugeiconsIcon icon={ArrowDown01Icon} strokeWidth={1.75} className="size-icon" />
          <span>{t("shell.organize.moveDown")}</span>
        </DropdownMenuItem>
      </>
    );
  }

  /** Drag handlers for one row of a reorderable list. */
  function rowDragProps(scope: string, orderedIds: string[], rowId: string) {
    return {
      draggable: true,
      onDragStart: (event: React.DragEvent) => {
        // Firefox needs a payload to drag at all.
        event.dataTransfer.effectAllowed = "move";
        event.dataTransfer.setData("text/plain", rowId);
        setDraggingRow({ id: rowId, scope });
      },
      onDragEnd: () => {
        setDraggingRow(null);
        setDropTargetRowId(null);
      },
      onDragOver: (event: React.DragEvent) => {
        if (draggingRow?.scope !== scope) return;
        event.preventDefault();
        event.dataTransfer.dropEffect = "move";
        if (dropTargetRowId !== rowId) setDropTargetRowId(rowId);
      },
      onDragLeave: () => {
        setDropTargetRowId((prev) => (prev === rowId ? null : prev));
      },
      onDrop: (event: React.DragEvent) => {
        event.preventDefault();
        const dragged = draggingRow;
        setDraggingRow(null);
        setDropTargetRowId(null);
        // Cross-list drops are ignored: the row is not in this list's order.
        if (!dragged || dragged.scope !== scope) return;
        const next = reorderIds(orderedIds, dragged.id, rowId);
        if (next !== orderedIds) setManualOrder(scope, next);
      },
    };
  }

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
        // A run that finished in the open chat is read, so clear after mark.
        markThreadsUnread(completedThreadIds, rowIdByThreadId);
        clearThreadsUnread([...activeVisibleThreadIdSet]);
      });
    }
    previousRunningByThreadIdRef.current = runningByThreadId;
  }, [
    activeVisibleThreadIdKey,
    runningByThreadId,
    markThreadsUnread,
    clearThreadsUnread,
    rowIdByThreadId,
  ]);

  const activeJobId = useTrainingRuntimeStore((s) => s.jobId);
  const currentRunViewActive = useTrainingRuntimeStore((s) => s.currentRunViewActive);
  const selectedHistoryRunId = useTrainingRuntimeStore((s) => s.selectedHistoryRunId);
  const setSelectedHistoryRunId = useTrainingRuntimeStore((s) => s.setSelectedHistoryRunId);
  // Running or starting up. Drives the Train spinner + New Chat / Return to Chat swap.
  const trainingInProgress = useTrainingRuntimeStore(isTrainingStartPending);
  // Export runs in the background; reflect it on the Export nav item from any tab.
  const exportInProgress = useExportRuntimeStore((s) => s.isExporting);
  // On any non-chat tab, offer a way back to the live chat instead of starting a new one
  // whenever a chat is running or its thread is active, or a training / export is in progress.
  const showReturnToChat =
    !isChatRoute &&
    (trainingInProgress || exportInProgress || anyChatRunning || storeThreadId != null);
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
    // Regrouping, collapsing a folder or revealing more adds and removes rows
    // with no scroll and no collapsible animation to re-measure off.
    projectsOpen,
    projectChatRowCount,
    visibleProjectRecords.length,
    // Pinning a project chat adds a Pinned row while Recents and the folder
    // counts both stay put, so nothing else here moves.
    pinnedChatItems.length,
    // And with no chats in them, folders appear and disappear on their own.
    organizeBy,
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

  // Navigation rows share one box, so a hover pill has the same edges wherever
  // it lands. Rows outside the list scroller add the rail width it does not
  // lose, so both end on the same edge whether or not the scrollbar takes
  // space. Logical sides, since the rail moves under rtl.
  const rowPadding = usesDesktopTitlebar
    ? "ps-[5px] pe-[calc(var(--sidebar-rail,0px)+5px)]"
    : "ps-1.5 pe-[calc(var(--sidebar-rail,0px)+6px)]";

  // Inside the scroller the rail already occupies that space. The profile
  // footer also uses this padding deliberately: its width is independent of
  // whether the unrelated recent-chat list currently has a scrollbar.
  const unrailedRowPadding = usesDesktopTitlebar ? "px-[5px]" : "px-1.5";

  // Header actions end where a hovered row's "…" does: unrailedRowPadding + the
  // action's own pr-1.5. 12px normally (the pr-3 class default), 11px here.
  const headerRightPadding = usesDesktopTitlebar
    ? "sidebar-sticky-label-desktop"
    : null;
  // Recents alone is nudged 2px right there, and carries its padding with it.
  const recentsHeaderRightPadding = usesDesktopTitlebar
    ? "sidebar-sticky-label-desktop-recents"
    : null;

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
    // A host with no video device at all is disabled with a hint instead of bouncing off the root guard.
    video: {
      icon: FlimSlateIcon,
      label: t("shell.navigation.video"),
      active: pathname === "/video" || pathname.startsWith("/video/"),
      disabled: videoDisabled,
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
      active: pathname === "/api-monitor" || pathname.startsWith("/api-monitor/"),
      onClick: () => {
        navigate({ to: "/api-monitor" });
        closeMobileIfOpen();
      },
      onIntent: () => {
        preloadSilently(router.preloadRoute({ to: "/api-monitor" }));
      },
    },
  };
  const unpinnedNavIds = sidebarNav
    .filter((item) => !item.pinned)
    .map((item) => item.id);
  // More needs two or more rows to be worth a click; with exactly one unpinned, the menu and that row are both dropped.
  const overflowNavIds = unpinnedNavIds.length > 1 ? unpinnedNavIds : [];
  const inlineNavIds = sidebarNav
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

  function showArchivedChatsToast() {
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
      showArchivedChatsToast();
    } catch (err) {
      toast.error("Failed to archive chat", {
        description: err instanceof Error ? err.message : undefined,
      });
    }
  }

  type RenameTarget =
    // `inline` is the row's own pill, and a chord has no row under the cursor
    // and may have none on screen at all, so it opens the dialog instead.
    | { kind: "chat"; item: SidebarItem; current: string; inline: boolean }
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
        current?.id === pendingRename.id && current.title === pendingRename.title
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

  function openRenameChat(item: SidebarItem, inline = true) {
    setRenameDraft(item.title);
    setRenamingTarget({ kind: "chat", item, current: item.title, inline });
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
    | { kind: "chats"; items: SidebarItem[] }
    | { kind: "project"; project: ProjectRecord }
    | { kind: "projects"; projects: ProjectRecord[] }
    | { kind: "run"; run: TrainingRunSummary };
  const [confirmingDelete, setConfirmingDelete] =
    useState<DeleteTarget | null>(null);
  const [deleteFilesOnDelete, setDeleteFilesOnDelete] = useState(false);

  /** Always through here: a stale switch would delete an unrelated sandbox. */
  function openDeleteDialog(target: DeleteTarget) {
    // Chats follow the preference, so the switch shows what will happen and can
    // still be turned off for this one delete. A project workspace is a bigger
    // thing to remove, so it keeps asking from scratch.
    const chats = target.kind === "chat" || target.kind === "chats";
    setDeleteFilesOnDelete(chats && alwaysDeleteChatFiles);
    setConfirmingDelete(target);
  }

  /** Only where a sandbox can actually be removed. A training run has none.
   *  A chat in a project still has one: anything it wrote before the move is in
   *  its own folder, and deletion never touches the project workspace. */
  function deleteTargetHasFiles(target: DeleteTarget | null): boolean {
    if (!target) return false;
    return target.kind !== "run";
  }

  async function commitDelete() {
    const target = confirmingDelete;
    if (!target) return;
    const shouldDeleteProjectFiles =
      (target.kind === "project" || target.kind === "projects") &&
      deleteFilesOnDelete;
    const shouldDeleteChatFiles =
      (target.kind === "chat" || target.kind === "chats") && deleteFilesOnDelete;
    setConfirmingDelete(null);
    // Reset so the next delete never inherits this switch.
    setDeleteFilesOnDelete(false);
    if (target.kind === "chat") {
      await deleteChatWithCleanup(target.item, {
        deleteFiles: shouldDeleteChatFiles,
      });
      return;
    }
    if (target.kind === "chats") {
      clearSelection();
      // Sequential, so one failure's toast is not buried by the next delete.
      for (const item of target.items) {
        await deleteChatWithCleanup(item, {
          deleteFiles: shouldDeleteChatFiles,
        });
      }
      return;
    }
    if (target.kind === "projects") {
      clearSelection();
      // Only what actually went: a failed delete leaves that project in place,
      // and redirecting off it would strand the user for nothing.
      const deletedIds = new Set<string>();
      for (const project of target.projects) {
        try {
          await deleteChatProject(project.id, {
            deleteFiles: shouldDeleteProjectFiles,
          });
          deletedIds.add(project.id);
        } catch (err) {
          toast.error("Failed to delete project", {
            description: err instanceof Error ? err.message : undefined,
          });
        }
      }
      // The same cleanup one delete does, once for the batch: refresh history so
      // member chats do not linger as rows, and leave a deleted project's page.
      // Unconditional, since a delete that threw may still have removed chats.
      notifyChatHistoryUpdated();
      const runtimeProjectId = useChatRuntimeStore.getState().activeProjectId;
      if (
        isChatRoute &&
        ((activeProjectId !== null && deletedIds.has(activeProjectId)) ||
          (runtimeProjectId !== null && deletedIds.has(runtimeProjectId)))
      ) {
        useChatRuntimeStore.getState().setActiveProjectId(null);
        navigate({ to: "/chat", search: { new: createNavigationNonce() } });
      }
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

  function clearChatNotifications(item: SidebarItem) {
    clearThreadsUnread(getSidebarItemThreadIds(item));
  }

  /** The chat as markdown, on the clipboard rather than in a download. */
  async function copyChatItemAsMarkdown(item: SidebarItem) {
    const empty = { value: false };
    // The read runs inside the write: Safari drops the gesture across an
    // await, and a chord has no second one to fall back on.
    const copied = await copyToClipboardFrom(async () => {
      const { buildChatItemMarkdown } = await import(
        "@/features/chat/prompt-storage/prompt-storage-dialog"
      );
      // A compare row is two threads: keep both, each under its model's name.
      const markdown = await buildChatItemMarkdown(item);
      if (!markdown) {
        empty.value = true;
        throw new Error("nothing to copy");
      }
      return markdown;
    });
    if (copied) {
      toast.success("Chat copied as Markdown");
    } else if (empty.value) {
      toast.info("No exportable content.");
    } else {
      // Denied permission, an unfocused document, a failed history read: the
      // payload was fine and the write was not, and a chord has no cursor to
      // show that with.
      toast.error("Could not copy this chat.");
    }
  }

  /** The sandbox sessions this chat's stored tool results name, if any. */
  async function recordedSandboxSessionIds(ids: string[]): Promise<string[]> {
    const recorded: string[] = [];
    // Every id a thread names, not just its latest: one chat that ran a tool,
    // moved between projects and ran another wrote to two folders on its own,
    // and the newest would answer for both.
    // One at a time, not Promise.all: this file's export contract forbids a
    // concurrent await here, and two panes win nothing.
    for (const threadId of ids) {
      recorded.push(
        ...allRecordedSandboxSessionIds(await listStoredChatMessages(threadId)),
      );
    }
    return [...new Set(recorded)];
  }

  /**
   * The folders this chat's files are actually in: what its tool results name,
   * or, for a chat old enough that they name nothing, what is on disk.
   *
   * Chats stored before results carried a session recorded nothing, so one that
   * ran loose and has since joined a project would be answered with the project
   * workspace. Its thread sandbox is the only other candidate, and files there
   * are this chat's, which is what makes them worth probing: a chat can join a
   * project, record that session, and move back out, and current membership
   * says nothing about where its older files went. The project workspace is
   * not probed, because it belongs to every chat in the project alike.
   *
   * A union rather than a fallback. One recorded id is not evidence that the
   * others are recorded too: a chat can have run a tool before recording
   * existed, moved into a project, and run another one since, and taking the
   * recorded id alone would answer for both folders while hiding the older.
   *
   * Shared by "Open chat folder" and "Copy session id", which drifted apart
   * once: the copy path skipped the probe and reported success on a folder the
   * chat had never written to.
   */
  async function sandboxSessionIdsHolding(ids: string[]): Promise<string[]> {
    const recorded = await recordedSandboxSessionIds(ids);
    // Thread folders only. A project sandbox is shared by every chat in the
    // project, so files there are no evidence that THIS chat wrote them, and
    // counting one would report a second folder for any chat that joined a
    // project someone else had already used. Both callers already fall back to
    // the folder membership gives them when nothing here names one, which is
    // the honest answer where there is no evidence either way.
    const held: string[] = [];
    for (const candidate of ids) {
      // Already named, so there is nothing a probe could add.
      if (recorded.includes(candidate)) continue;
      if (await sandboxHasFiles(candidate)) held.push(candidate);
    }
    return [...new Set([...recorded, ...held])];
  }

  /** The sandbox session this chat's tool calls write into. */
  async function copyChatSessionId(item: SidebarItem) {
    const threadIds = getSidebarItemThreadIds(item);
    const ids = threadIds.length > 0 ? threadIds : [item.id];
    // The chat's own history names the folder it wrote to, which is not where
    // current membership points once it has moved between projects. Same read
    // as "Open chat folder", and the same answer when it names two, whether
    // that is a compare row's two panes or one thread that outlived a move.
    const refusal: { value: { title: string; description?: string } | null } = {
      value: null,
    };
    // Read inside the write, for the same reason as the markdown copy above.
    const copied = await copyToClipboardFrom(async () => {
      let sessionId: string | undefined;
      try {
        const distinct = await sandboxSessionIdsHolding(ids);
        if (distinct.length > 1) {
          refusal.value = {
            title: "This chat wrote to more than one folder.",
            description: "Copy the session id from a tool card instead.",
          };
          throw new Error("more than one folder");
        }
        sessionId = distinct[0];
      } catch (error) {
        refusal.value ??= {
          title: "Could not read this chat's session id.",
          description: error instanceof Error ? error.message : String(error),
        };
        throw error;
      }
      // Nothing recorded means no tool has run yet, so the id it would get is
      // the one current membership gives it.
      sessionId ??=
        item.type === "single" || item.projectId
          ? sandboxSessionIdFor(ids[0], item.projectId)
          : undefined;
      if (!sessionId) {
        refusal.value = { title: "This chat has no single session id" };
        throw new Error("no session id");
      }
      return sessionId;
    });
    if (copied) {
      toast.success("Session id copied");
      return;
    }
    if (refusal.value) {
      toast.error(refusal.value.title, {
        description: refusal.value.description,
      });
      return;
    }
    // No refusal means the id was found and the clipboard write itself failed.
    toast.error("Could not copy the session id.");
  }

  /** Open a chat row. Shared by the row and the navigation chords. */
  function openChatItem(item: SidebarItem) {
    // Archive/pin/unread act on the selection when there is one, so leaving it
    // behind would point them at rows the user just navigated away from.
    clearSelection();
    clearChatNotifications(item);
    noteViewed(item.id);
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
  }
  // Through a ref: openChatItem is render-scoped, so registering it directly
  // would rewrite the store every render.
  const openChatItemRef = useRef(openChatItem);
  useEffect(() => {
    openChatItemRef.current = openChatItem;
  });
  // The sidebar is unmounted on the auth routes, so this is where a sign-out
  // reaches: the store outlives the component that filled it, and the unread
  // set and the walk belong to the account that just left.
  useEffect(
    () => () => useChatNavigationStore.getState().resetAccountState(),
    [],
  );
  useEffect(() => {
    const setOpenChatItem = useChatNavigationStore.getState().setOpenChatItem;
    setOpenChatItem((item) => openChatItemRef.current(item));
    return () => setOpenChatItem(null);
  }, []);

  // --- Chat shortcuts ----------------------------------------------------
  // The sidebar is on every shell route and holds the list, the handlers and
  // the router, so the chat chords register here.
  const activeChatItem = useMemo(
    () => allChatItems.find((item) => item.id === activeThreadId) ?? null,
    [allChatItems, activeThreadId],
  );
  /** Run `fn` on the open chat, or say why nothing happened. */
  const withActiveChat = (fn: (item: SidebarItem) => void) => {
    if (!activeChatItem) {
      toast.info("Open a chat first");
      return;
    }
    fn(activeChatItem);
  };
  const goToChat = (pick: (state: ChatNavigationState) => SidebarItem | null) =>
    openChatItemById(pick(useChatNavigationStore.getState()));

  // With rows selected these act on the selection, matching the context menu;
  // otherwise on the open chat. Acting on a selection clears it, so without
  // this latch a second press would land on the open chat, which the user
  // never selected.
  // Keyed by action: the press to hold back is a repeat of the one that just
  // took the selection, not a different command the user chose deliberately.
  const selectionActedRef = useRef<{ id: ShortcutId; at: number } | null>(null);
  const actOnSelection = (id: ShortcutId, fn: () => void) => {
    selectionActedRef.current = { id, at: Date.now() };
    fn();
  };
  const followsSelectionAction = (id: ShortcutId) => {
    const last = selectionActedRef.current;
    return (
      last?.id === id && Date.now() - last.at < SELECTION_ACTION_GRACE_MS
    );
  };
  // A project selection is not a chat selection: none of the chords below is on
  // its context menu, so with only projects selected they stand aside rather
  // than falling through to the open chat. Delete already behaves this way.
  const projectsOnlySelected = () =>
    selectionCount === 0 && projectSelectionCount > 0;
  // A dialog leaves the sidebar mounted and inert behind it, and these chords
  // are window-level, so Settings over Chat would archive or rename the chat
  // behind it. Asked at press time, since `enabled` is read at render.
  //
  // Backgrounded, not "not in the foreground": reading a missing element as
  // covered would kill these chords on the mobile drawer, which unmounts.
  const sidebarCovered = () => {
    if (isSurfaceBackgrounded(SIDEBAR_SELECTOR)) return true;
    // With the mobile drawer closed the sidebar is unmounted, so the check
    // above has nothing to read. The app root is always mounted and Radix
    // aria-hides it for a modal's life. A fallback only: an open drawer is
    // itself a dialog hiding the root, and the sidebar in it is the foreground.
    return (
      typeof document !== "undefined" &&
      document.querySelector(SIDEBAR_SELECTOR) === null &&
      isSurfaceBackgrounded("#root")
    );
  };
  useShortcut("archiveChat", () => {
    if (sidebarCovered()) return;
    if (projectsOnlySelected()) return;
    if (selectionCount > 0) {
      actOnSelection("archiveChat", () => void archiveSelected());
      return;
    }
    if (followsSelectionAction("archiveChat")) return;
    withActiveChat((item) => void handleArchiveThread(item));
  });
  useShortcut("markChatUnread", () => {
    if (sidebarCovered()) return;
    if (projectsOnlySelected()) return;
    if (selectionCount > 0) {
      actOnSelection("markChatUnread", markSelectedUnread);
      return;
    }
    if (followsSelectionAction("markChatUnread")) return;
    withActiveChat((item) =>
      markThreadsUnread(getSidebarItemThreadIds(item), rowIdByThreadId),
    );
  });
  useShortcut("togglePinChat", () => {
    if (sidebarCovered()) return;
    if (projectsOnlySelected()) return;
    if (selectionCount > 0) {
      actOnSelection("togglePinChat", () => pinSelected(!allSelectedPinned));
      return;
    }
    if (followsSelectionAction("togglePinChat")) return;
    withActiveChat((item) => togglePinnedChat(item.id));
  });
  // A selection made behind a dialog is invisible and still what the mutating
  // chords hit once the dialog closes, which is the thing those chords being
  // guarded was meant to prevent.
  useShortcut("selectAllChats", () => {
    if (sidebarCovered()) return;
    selectAllChats();
  });
  useShortcut("clearChatSelection", clearSelection);
  useShortcut("deleteSelectedChats", () => {
    if (sidebarCovered()) return;
    if (selectionCount > 0) deleteSelected();
  });
  // Through the dialog: the row's inline pill is rendered by the row, and the
  // open chat may be behind a collapsed section, past a folder's "show more",
  // or on a route with no chat list at all, where the chord would look dead and
  // leave a rename waiting to appear the moment the row came back.
  useShortcut("renameChat", () => {
    if (sidebarCovered()) return;
    withActiveChat((item) => openRenameChat(item, false));
  });
  // The clipboard is outside the app, so writing a hidden chat's contents into
  // it from behind a dialog is not something the user can take back by closing
  // the dialog.
  useShortcut("copyChatAsMarkdown", () => {
    if (sidebarCovered()) return;
    withActiveChat((item) => void copyChatItemAsMarkdown(item));
  });
  useShortcut("copySessionId", () => {
    if (sidebarCovered()) return;
    withActiveChat((item) => void copyChatSessionId(item));
  });

  // These four walk the list, so holding them steps through it, the way an
  // arrow key does. The rest are one-shot and ignore auto-repeat.
  useShortcut("nextChat", () => goToChat((s) => adjacentChatItem(s, 1)), {
    repeats: true,
  });
  useShortcut("previousChat", () => goToChat((s) => adjacentChatItem(s, -1)), {
    repeats: true,
  });
  // The walk holds the stack still while it runs, so a modifier held down
  // reaches the third chat back and beyond; releasing it ends the walk and
  // puts the chat it landed on at the top.
  const walkRecentlyViewed = (delta: number) =>
    openChatItemById(useChatNavigationStore.getState().stepRecentlyViewed(delta));
  useShortcut("nextRecentlyViewedChat", () => walkRecentlyViewed(1), {
    repeats: true,
  });
  useShortcut("previousRecentlyViewedChat", () => walkRecentlyViewed(-1), {
    repeats: true,
  });
  useShortcut("nextChatNeedingAttention", () =>
    goToChat(nextAttentionChatItem),
  );
  // No undo and no menu item anywhere, so it says what it did.
  useShortcut("clearAllUnreads", () => {
    if (sidebarCovered()) return;
    const state = useChatNavigationStore.getState();
    const cleared = countUnreadRows(state);
    if (state.unreadThreadIds.size === 0) {
      toast.info("No unread chats");
      return;
    }
    state.clearAllUnreads();
    toast.success(`Cleared ${cleared} unread ${cleared === 1 ? "chat" : "chats"}`);
  });

  // The six slots register as <Shortcut> elements: a loop of hooks would
  // break the rules of hooks.
  const slotShortcuts = (
    <>
      {RECENT_SLOT_NUMBERS.map((slot) => (
        <Shortcut
          key={`recent-${slot}`}
          id={`goToRecentChat${slot}` as ShortcutId}
          onTrigger={() => goToChat((s) => recentChatItemAtSlot(s, slot))}
        />
      ))}
    </>
  );

  useShortcut(
    "logOut",
    () => {
      // Desktop signs out through the OS account menu, not here.
      if (isTauri) return;
      void (async () => {
        try {
          await logout();
        } catch {
          clearAuthTokens();
        }
        void navigate({ to: "/login" });
      })();
    },
    { enabled: !isTauri },
  );

  // The "..." every list header carries. Only chat lists regroup, so that half
  // is opt-in; Pinned takes the sort half alone.
  function renderSidebarHeaderMenu(options: {
    ariaLabel: string;
    sortLabel: string;
    sortValue: SidebarChatSort;
    onSortChange: (next: SidebarChatSort) => void;
    includeOrganize?: boolean;
  }) {
    return (
      <DropdownMenu>
        <DropdownMenuTrigger asChild>
          <button
            type="button"
            aria-label={options.ariaLabel}
            className="sidebar-header-action"
          >
            <HugeiconsIcon icon={MoreHorizontalIcon} strokeWidth={1.75} className="size-icon" />
          </button>
        </DropdownMenuTrigger>
        <DropdownMenuContent
          side="bottom"
          align="end"
          sideOffset={2}
          className="unsloth-plus-menu w-56"
        >
          {options.includeOrganize && (
            <>
              <DropdownMenuLabel>
                {t("shell.organize.sidebarHeading")}
              </DropdownMenuLabel>
              <DropdownMenuRadioGroup
                value={organizeBy}
                onValueChange={(value) =>
                  setOrganizeBy(value as SidebarOrganizeBy)
                }
              >
                {ORGANIZE_OPTIONS.map((option) => (
                  <DropdownMenuRadioItem
                    key={option.value}
                    value={option.value}
                    className={menuRadioItemClass}
                  >
                    {t(option.key)}
                  </DropdownMenuRadioItem>
                ))}
              </DropdownMenuRadioGroup>
            </>
          )}
          <DropdownMenuLabel>{options.sortLabel}</DropdownMenuLabel>
          <DropdownMenuRadioGroup
            value={options.sortValue}
            onValueChange={(value) =>
              options.onSortChange(value as SidebarChatSort)
            }
          >
            {CHAT_SORT_OPTIONS.map((option) => (
              <DropdownMenuRadioItem
                key={option.value}
                value={option.value}
                className={menuRadioItemClass}
              >
                {t(option.key)}
              </DropdownMenuRadioItem>
            ))}
          </DropdownMenuRadioGroup>
        </DropdownMenuContent>
      </DropdownMenu>
    );
  }

  /** Bulk actions for selected folders, on right-click of any project row. */
  function renderProjectContextMenu() {
    if (projectSelectionCount === 0) return null;
    return (
      <ContextMenuContent className="unsloth-plus-menu menu-flat-destructive w-52">
        {projectSelectionCount > 1 && (
          <ContextMenuLabel>
            {t("shell.selection.countSelected", {
              count: projectSelectionCount,
            })}
          </ContextMenuLabel>
        )}
        <ContextMenuItem
          onSelect={() => pinSelectedProjects(!allSelectedProjectsPinned)}
        >
          <HugeiconsIcon icon={allSelectedProjectsPinned ? PinOffIcon : PinIcon} strokeWidth={1.75} className="size-icon" />
          <span>
            {allSelectedProjectsPinned
              ? t("shell.selection.unpinProjects")
              : t("shell.selection.pinProjects")}
          </span>
        </ContextMenuItem>
        <ContextMenuSeparator />
        <ContextMenuItem
          variant="destructive"
          onSelect={() => deleteSelectedProjects()}
        >
          <HugeiconsIcon icon={Delete02Icon} strokeWidth={1.75} className="size-icon" />
          <span>{t("shell.selection.deleteProjects")}</span>
        </ContextMenuItem>
      </ContextMenuContent>
    );
  }

  /** Bulk actions for the current selection, on right-click of any chat row. */
  function renderChatContextMenu() {
    if (selectionCount === 0) return null;
    return (
      <ContextMenuContent className="unsloth-plus-menu menu-flat-destructive w-52">
        {selectionCount > 1 && (
          <ContextMenuLabel>
            {t("shell.selection.countSelected", { count: selectionCount })}
          </ContextMenuLabel>
        )}
        <ContextMenuItem onSelect={() => pinSelected(!allSelectedPinned)}>
          <HugeiconsIcon icon={allSelectedPinned ? PinOffIcon : PinIcon} strokeWidth={1.75} className="size-icon" />
          <span>
            {allSelectedPinned
              ? t("shell.selection.unpinChats")
              : t("shell.selection.pinChats")}
          </span>
        </ContextMenuItem>
        <ContextMenuItem onSelect={() => void archiveSelected()}>
          <HugeiconsIcon icon={Archive03Icon} strokeWidth={1.75} className="size-icon" />
          <span>{t("shell.selection.archiveChats")}</span>
        </ContextMenuItem>
        <ContextMenuItem onSelect={() => markSelectedUnread()}>
          <HugeiconsIcon icon={BubbleChatIcon} strokeWidth={1.75} className="size-icon" />
          <span>{t("shell.selection.markUnread")}</span>
        </ContextMenuItem>
        <ContextMenuSeparator />
        <ContextMenuItem variant="destructive" onSelect={() => deleteSelected()}>
          <HugeiconsIcon icon={Delete02Icon} strokeWidth={1.75} className="size-icon" />
          <span>{t("shell.selection.deleteChats")}</span>
        </ContextMenuItem>
      </ContextMenuContent>
    );
  }

  function renderChatSidebarItem(
    item: SidebarItem,
    variant: "project" | "recent",
    // Manual order only: the list this row drags within, its ids to reorder,
    // and this row's slot in them.
    drag?: { scope: string; orderedIds: string[]; index: number },
    // The list this row is rendered in, for shift-click ranges. Always passed;
    // `drag` only turns up when the list is manually ordered.
    list?: { scope: string; ids: string[] },
  ) {
    const threadIds = getSidebarItemThreadIds(item);
    const isPinned = pinnedIdSet.has(item.id);
    // A compare row outside a project spans two sandboxes, and there is no
    // honest single folder to offer for it.
    const sandboxSessionId =
      item.type === "single" || item.projectId
        ? sandboxSessionIdFor(threadIds[0] ?? item.id, item.projectId)
        : undefined;
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
      renamingTarget?.kind === "chat" &&
      renamingTarget.inline &&
      renamingTarget.item.id === item.id;

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
              "text-foreground h-[30px] w-full border-0 bg-transparent py-0 pr-4 text-ui-14p5 leading-ui-19 font-medium tracking-nav outline-none",
              variant === "project" ? "pl-[39px]" : "pl-3",
            )}
          />
        </SidebarMenuItem>
      );
    }

    return (
      <ContextMenu key={item.id}>
        <ContextMenuTrigger asChild>
          <SidebarMenuItem
            className={cn(
              itemClass,
              draggingRow?.id === item.id && "opacity-50",
              dropCueClass(drag?.scope, drag?.orderedIds, item.id),
            )}
            onContextMenu={() => list && selectForContextMenu(item, list)}
            {...(drag
              ? rowDragProps(drag.scope, drag.orderedIds, item.id)
              : undefined)}
          >
            <SidebarMenuButton
              data-testid="recent-thread"
              data-thread-type={item.type}
              data-thread-id={item.id}
              data-generating={isGenerating ? "true" : undefined}
              aria-busy={isGenerating || undefined}
              isActive={activeThreadId === item.id}
              data-selected={selectedChatIds.has(item.id) ? "true" : undefined}
              className={buttonClass}
              onClick={(event) => {
                if (list && handleSelectionClick(event, item, list)) return;
                openChatItem(item);
              }}
            >
              {isPinned && variant !== "project" && (
                <HugeiconsIcon icon={BubbleChatIcon} strokeWidth={1.75} className="size-icon! shrink-0" />
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
                aria-hidden
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
                className="sidebar-row-action sidebar-touch-reveal is-unpin-action group-hover/recent-item:opacity-100 group-hover/recent-item:pointer-events-auto focus-visible:opacity-100 focus-visible:pointer-events-auto"
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
                {drag &&
                  renderMoveRowItems(
                    drag.scope,
                    drag.orderedIds,
                    item.id,
                    drag.index,
                  )}
                {sandboxSessionId ? (
                  isTauri ? (
                    <DropdownMenuItem
                      title="Open the folder this chat's tool calls read and write"
                      onSelect={() => {
                        void (async () => {
                          try {
                            // A chat moved between projects keeps the sandbox it
                            // wrote to, so its own history names the folder, not
                            // current membership. A failed read is reported
                            // below rather than caught per pane, which would
                            // read as "never ran a tool" and fall back to
                            // membership, the answer the recorded id overrides.
                            const ids =
                              threadIds.length > 0 ? threadIds : [item.id];
                            const distinct = await sandboxSessionIdsHolding(ids);
                            if (distinct.length > 1) {
                              toast.error("This chat wrote to more than one folder.", {
                                description:
                                  "It ran tools on both sides of a move, so open the folder from a tool card instead.",
                              });
                              return;
                            }
                            await revealSandbox(distinct[0] ?? sandboxSessionId);
                          } catch (error) {
                            toast.error("Could not open the chat folder.", {
                              description:
                                error instanceof Error
                                  ? error.message
                                  : String(error),
                            });
                          }
                        })();
                      }}
                    >
                      <HugeiconsIcon icon={FolderOpenIcon} strokeWidth={1.75} className="size-icon" />
                      <span>Open chat folder</span>
                    </DropdownMenuItem>
                  ) : (
                    <OpenChatFolderUnavailableItem />
                  )
                ) : null}
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
                <DropdownMenuSub>
                  <DropdownMenuSubTrigger>
                    <HugeiconsIcon icon={BookOpen01Icon} strokeWidth={1.75} className="size-icon" />
                    <span>Save to project sources</span>
                  </DropdownMenuSubTrigger>
                  <DropdownMenuSubContent sideOffset={8} alignOffset={-4} className="unsloth-plus-menu w-52">
                    {projects.length === 0 && (
                      <DropdownMenuItem disabled>No projects yet</DropdownMenuItem>
                    )}
                    {projects.map((project) => (
                      <DropdownMenuItem
                        key={project.id}
                        onSelect={async () => {
                          try {
                            await saveChatToProjectSources(item, project.id);
                          } catch {
                            toast.error("Failed to save to project sources.");
                          }
                        }}
                      >
                        <HugeiconsIcon icon={Folder01Icon} strokeWidth={1.75} className="size-icon" />
                        <span className="truncate">{project.name}</span>
                      </DropdownMenuItem>
                    ))}
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
                      ? openDeleteDialog({ kind: "chat", item })
                      : void deleteChatWithCleanup(item, {
                          deleteFiles: alwaysDeleteChatFiles,
                        })
                  }
                >
                  <HugeiconsIcon icon={Delete02Icon} strokeWidth={1.75} className="size-icon" />
                  <span>Delete</span>
                </DropdownMenuItem>
              </DropdownMenuContent>
            </DropdownMenu>
          </SidebarMenuItem>
        </ContextMenuTrigger>
        {renderChatContextMenu()}
      </ContextMenu>
    );
  }

  return (
    <>
      {slotShortcuts}
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
                data-tauri-drag-region
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
                    chatDisabled && "pointer-events-none",
                  )}
                  aria-label={t("shell.aria.home")}
                  aria-disabled={chatDisabled}
                  tabIndex={chatDisabled ? -1 : undefined}
                >
                  {/* Logo lockup follows the UI font size at half rate:
                      base + (root scale - 1) * 8px. Exact base sizes at 16px. */}
                  <img
                    src="/circle-logo-small.png"
                    alt="Unsloth"
                    className="relative top-px h-[calc(22px+0.5rem*var(--ui-font-scale,1))] w-[calc(22px+0.5rem*var(--ui-font-scale,1))] shrink-0 rounded-full object-cover"
                  />
                  <span className="relative -top-px truncate font-heading text-[calc(13px+0.5rem*var(--ui-font-scale,1))] font-semibold tracking-[0em] leading-tight text-black dark:text-white dark:tracking-[0.02em]">
                    unsloth
                  </span>
                  <span className="nav-badge ml-0.5 inline-flex shrink-0 items-center justify-center rounded-full border border-nav-beta-border px-[5px] pt-[3px] pb-[2px] text-[calc(0.5rem*var(--ui-font-scale,1))] font-medium leading-none tracking-[0.04em] text-nav-fg-muted antialiased subpixel-antialiased shadow-[0_1px_2px_rgba(0,0,0,0.06)] dark:shadow-[0_1px_2px_rgba(0,0,0,0.35)]">
                    {t("shell.beta")}
                  </span>
                </Link>
              <div className="flex shrink-0 items-center gap-0.25">
                <Tooltip>
                  <TooltipPrimitive.Trigger asChild>
                    <button
                      type="button"
                      onClick={() => {
                        useChatSearchStore.getState().open();
                        closeMobileIfOpen();
                      }}
                      className="inline-flex size-[30px] cursor-pointer items-center justify-center rounded-[10px] text-nav-icon-idle dark:text-nav-fg-muted transition-colors hover:bg-nav-surface-hover hover:text-black dark:hover:text-white focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
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
                    {searchShortcutLabel && (
                      <kbd className="rounded bg-black/10 px-1 py-px text-ui-10 font-medium leading-none dark:bg-white/15">
                        {searchShortcutLabel}
                      </kbd>
                    )}
                  </TooltipContent>
                </Tooltip>
                {!isMobile && !usesDesktopTitlebar && (
                  <Tooltip>
                    <TooltipPrimitive.Trigger asChild>
                      <button
                        type="button"
                        onClick={togglePinned}
                        className="inline-flex size-[30px] cursor-pointer items-center justify-center rounded-[10px] text-nav-icon-idle dark:text-nav-fg-muted transition-colors hover:bg-nav-surface-hover hover:text-black dark:hover:text-white focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
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
            {!isMobile && (!usesDesktopTitlebar || usesNativeMacTitlebar) && (
              <div className="relative z-10 hidden group-data-[collapsible=icon]:flex h-[33px] items-center justify-center w-full">
                <Tooltip>
                  <TooltipPrimitive.Trigger asChild>
                    <button
                      type="button"
                      onClick={togglePinned}
                      className="inline-flex size-[30px] cursor-pointer items-center justify-center rounded-[10px] text-nav-fg transition-colors hover:bg-nav-surface-hover hover:text-black dark:hover:text-white focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
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
                    // Name the count rather than imply a single live chat.
                    ? t("shell.navigation.returnToChats", {
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
            unrailedRowPadding,
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
                      id === "images" && imagesWorkflowsListed ? false : row.active
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
                  onPointerEnter={openMorePreview}
                  onPointerLeave={closeMorePreviewSoon}
                >
                  <DropdownMenu
                    open={moreOpen}
                    onOpenChange={handleMoreOpenChange}
                    modal={false}
                  >
                    {/* Tooltip wraps the trigger rather than using the button's `tooltip` prop: that returns a Tooltip root, so DropdownMenuTrigger asChild would miss the DOM node. */}
                    <Tooltip>
                      <TooltipPrimitive.Trigger asChild>
                        <DropdownMenuTrigger asChild>
                          <SidebarMenuButton
                            // More is a container, not a destination: no active style just because the current page
                            // lives inside it. Keeps the row highlighted while the panel is open, after the pointer
                            // has left. Not data-state: the tooltip and menu triggers both write that one.
                            data-menu-open={moreOpen ? "true" : undefined}
                            onPointerDownCapture={(event) => {
                              if (event.button !== 0 || event.ctrlKey) return;
                              event.preventDefault();
                              event.stopPropagation();
                              event.currentTarget.focus({ preventScroll: true });
                              clearMoreCloseTimer();
                              if (morePinnedOpen) {
                                setMorePinnedOpen(false);
                                setMoreHoverOpen(false);
                              } else {
                                setMorePinnedOpen(true);
                              }
                            }}
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
                      className="w-48 p-1"
                      onPointerEnter={openMorePreview}
                      onPointerLeave={closeMorePreviewSoon}
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
                        <HugeiconsIcon icon={Settings02Icon} strokeWidth={1.75} />
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

        {/* Pinned chats */}
        {!isStudioRoute && !showTrainingRecents && pinnedChatItems.length > 0 && (
          <Collapsible open={pinnedOpen} onOpenChange={setPinnedOpen} asChild>
            <SidebarGroup className="group-data-[collapsible=icon]:hidden px-0 py-0">
              <SidebarGroupLabel className={cn("sidebar-sticky-label sidebar-sticky-label-following group/sidebar-header gap-1", headerRightPadding, scrolled && "is-scrolled")}>
                <CollapsibleTrigger className="cursor-pointer flex min-w-0 flex-1 items-center gap-1 group/sb-collap">
                  Pinned
                  <ChevronDown className="size-3.5 opacity-0 transition-[transform,opacity] duration-200 group-hover/sb-collap:opacity-100 group-focus-visible/sb-collap:opacity-100 data-[state=open]:rotate-0 [[data-state=closed]_&]:rotate-[-90deg] [[data-state=closed]_&]:opacity-100" />
                </CollapsibleTrigger>
                {/* Pinning is the grouping, so no organize half and no "+". */}
                {renderSidebarHeaderMenu({
                  ariaLabel: t("shell.organize.sortPinnedChats"),
                  sortLabel: t("shell.organize.sortPinnedBy"),
                  sortValue: pinnedSort,
                  onSortChange: setPinnedSort,
                })}
              </SidebarGroupLabel>
              <CollapsibleContent>
                <SidebarGroupContent className={unrailedRowPadding}>
                  <SidebarMenu>
                    {sortedPinnedChatItems.map((item, index) =>
                      renderChatSidebarItem(
                        item,
                        "recent",
                        pinnedDragEnabled
                          ? {
                              scope: PINNED_ORDER_SCOPE,
                              orderedIds: pinnedRowIds,
                              index,
                            }
                          : undefined,
                        { scope: PINNED_ORDER_SCOPE, ids: pinnedRowIds },
                      ),
                    )}
                  </SidebarMenu>
                </SidebarGroupContent>
              </CollapsibleContent>
            </SidebarGroup>
          </Collapsible>
        )}

        {/* Projects: one folder per project, its chats nested underneath */}
        {!isStudioRoute &&
          !showTrainingRecents &&
          organizeBy === "project" &&
          sidebarProjectRecords.length > 0 && (
            <Collapsible
              open={projectsOpen}
              onOpenChange={setProjectsOpen}
              asChild
            >
              <SidebarGroup className="group-data-[collapsible=icon]:hidden px-0 py-0">
                {/* Trigger takes the free space; the actions reveal beside it. */}
                <SidebarGroupLabel className={cn("sidebar-sticky-label sidebar-sticky-label-following group/sidebar-header gap-1", headerRightPadding, scrolled && "is-scrolled")}>
                  <CollapsibleTrigger className="cursor-pointer flex min-w-0 flex-1 items-center gap-1 group/sb-collap">
                    {t("shell.navigation.projects")}
                    <ChevronDown className="size-3.5 opacity-0 transition-[transform,opacity] duration-200 group-hover/sb-collap:opacity-100 group-focus-visible/sb-collap:opacity-100 data-[state=open]:rotate-0 [[data-state=closed]_&]:rotate-[-90deg] [[data-state=closed]_&]:opacity-100" />
                  </CollapsibleTrigger>
                  {renderSidebarHeaderMenu({
                    ariaLabel: t("shell.organize.organizeProjects"),
                    includeOrganize: true,
                    sortLabel: t("shell.organize.sortChatsBy"),
                    sortValue: chatSort,
                    onSortChange: setChatSort,
                  })}
                  <button
                    type="button"
                    aria-label="New project"
                    onClick={() => {
                      setProjectCreateMoveTarget(null);
                      setCreatingProject(true);
                    }}
                    className="sidebar-header-action"
                  >
                    <HugeiconsIcon icon={PlusSignIcon} strokeWidth={1.75} className="size-icon" />
                  </button>
                </SidebarGroupLabel>
                <CollapsibleContent>
                  <SidebarGroupContent className={unrailedRowPadding}>
                    <SidebarMenu>
                      {visibleProjectRecords.map((project, projectIndex) => {
                        const projectChats =
                          sortedChatsByProjectId.get(project.id) ?? [];
                        const projectChatIds =
                          projectChatRowIds.get(project.id) ?? [];
                        const expanded = !collapsedProjectIds.has(project.id);
                        const showAll = expandedChatProjectIds.has(project.id);
                        const visibleChats =
                          expanded && !showAll
                            ? projectChats.slice(0, PROJECT_CHAT_LIMIT)
                            : projectChats;
                        const isProjectPinned = pinnedProjectIdSet.has(
                          project.id,
                        );
                        return (
                        <Fragment key={project.id}>
                        {/* Folders drag to reorder whatever the chat sort is. */}
                        <ContextMenu>
                          <ContextMenuTrigger asChild>
                            <SidebarMenuItem
                              className={cn(
                                "group/recent-item relative",
                                draggingRow?.id === project.id && "opacity-50",
                                dropCueClass(
                                  PROJECT_ORDER_SCOPE,
                                  projectRowIds,
                                  project.id,
                                ),
                              )}
                              onContextMenu={() =>
                                selectProjectForContextMenu(project.id)
                              }
                              {...rowDragProps(
                                PROJECT_ORDER_SCOPE,
                                projectRowIds,
                                project.id,
                              )}
                            >
                              <SidebarMenuButton
                                // Highlight the folder only on the project home; with a chat open, only that row is active.
                                isActive={activeProjectId === project.id && !activeThreadId}
                                data-selected={
                                selectedProjectIds.has(project.id)
                                  ? "true"
                                  : undefined
                              }
                              onClick={(event) => {
                                if (
                                  handleProjectSelectionClick(event, project.id)
                                )
                                  return;
                                clearSelection();
                                toggleProjectCollapsed(project.id);
                              }}
                                className="sidebar-nav-btn h-[33px] rounded-full gap-[8.5px] pl-3 pr-2.5 font-medium group-hover/recent-item:pr-16 group-has-[.sidebar-row-action[data-state=open]]/recent-item:pr-8 [@media(pointer:coarse)]:pr-16"
                              >
                                <HugeiconsIcon icon={Folder01Icon} strokeWidth={1.75} className="size-icon! shrink-0" />
                                <span className="truncate text-ui-14p5 leading-ui-19 tracking-nav">{project.name}</span>
                              </SidebarMenuButton>
                              {/* New chat in this project */}
                              <button
                                type="button"
                                aria-label="New chat"
                                onClick={(e) => {
                                  e.stopPropagation();
                                  openNewChat(project.id);
                                }}
                                className="sidebar-row-action sidebar-touch-reveal is-unpin-action group-hover/recent-item:opacity-100 group-hover/recent-item:pointer-events-auto focus-visible:opacity-100 focus-visible:pointer-events-auto"
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
                                    className="sidebar-row-action sidebar-touch-reveal group-hover/recent-item:opacity-100 group-hover/recent-item:pointer-events-auto focus-visible:opacity-100 focus-visible:pointer-events-auto"
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
                                      // Seed the shared draft so the dialog opens with the current name, not stale text.
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
                                  {renderMoveRowItems(
                                    PROJECT_ORDER_SCOPE,
                                    projectRowIds,
                                    project.id,
                                    projectIndex,
                                  )}
                                  <DropdownMenuItem onSelect={() => toggleProjectPin(project.id)}>
                                    <HugeiconsIcon icon={isProjectPinned ? PinOffIcon : PinIcon} strokeWidth={1.75} className="size-icon" />
                                    <span>{isProjectPinned ? "Unpin project" : "Pin project"}</span>
                                  </DropdownMenuItem>
                                  <DropdownMenuSeparator />
                                  <DropdownMenuItem
                                    variant="destructive"
                                    onSelect={() => {
                                      // Start each delete with the file toggle off: Cancel closes programmatically and skips the
                                      openDeleteDialog({ kind: "project", project });
                                    }}
                                  >
                                    <HugeiconsIcon icon={Delete02Icon} strokeWidth={1.75} className="size-icon" />
                                    <span>Delete project</span>
                                  </DropdownMenuItem>
                                </DropdownMenuContent>
                              </DropdownMenu>
                            </SidebarMenuItem>
                          </ContextMenuTrigger>
                          {renderProjectContextMenu()}
                        </ContextMenu>
                        {expanded &&
                          visibleChats.map((chat, index) =>
                            renderChatSidebarItem(
                              chat,
                              "project",
                              manualDragEnabled
                                ? {
                                    scope: projectOrderScope(project.id),
                                    orderedIds: projectChatIds,
                                    index,
                                  }
                                : undefined,
                              {
                                scope: projectOrderScope(project.id),
                                ids: projectChatIds,
                              },
                            ),
                          )}
                        {expanded &&
                          projectChats.length > PROJECT_CHAT_LIMIT && (
                            <SidebarMenuItem>
                              <SidebarMenuButton
                                onClick={() => toggleProjectShowAll(project.id)}
                                // Force the muted token: .sidebar-nav-btn's own color rule outweighs a plain text utility,
                                // so Show more would otherwise match the chat rows.
                                className="sidebar-nav-btn h-[30px] rounded-full pl-9 pr-4 font-medium text-nav-fg-muted!"
                              >
                                <span className="text-ui-13 leading-ui-18 tracking-nav">
                                  {t(
                                    showAll
                                      ? "shell.navigation.showLess"
                                      : "shell.navigation.showMore",
                                  )}
                                </span>
                              </SidebarMenuButton>
                            </SidebarMenuItem>
                          )}
                        </Fragment>
                        );
                      })}
                      {/* Long project lists stay one row deep until asked. */}
                      {sidebarProjectRecords.length > SIDEBAR_PROJECT_LIMIT && (
                        <SidebarMenuItem>
                          <SidebarMenuButton
                            onClick={() => setShowAllProjects((prev) => !prev)}
                            className="sidebar-nav-btn h-[30px] rounded-full pl-3 pr-4 font-medium text-nav-fg-muted!"
                          >
                            <span className="text-ui-13 leading-ui-18 tracking-nav">
                              {t(
                                showAllProjects
                                  ? "shell.navigation.showLess"
                                  : "shell.navigation.showMore",
                              )}
                            </span>
                          </SidebarMenuButton>
                        </SidebarMenuItem>
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
              <SidebarGroupLabel
                className={cn(
                  "sidebar-sticky-label sidebar-sticky-label-following group/sidebar-header gap-1",
                  recentsHeaderRightPadding,
                  scrolled && "is-scrolled",
                  usesDesktopTitlebar && "translate-x-[2px]",
                )}
              >
                <CollapsibleTrigger className="cursor-pointer flex min-w-0 flex-1 items-center gap-1 group/sb-collap">
                  {t("shell.navigation.recents")}
                  <ChevronDown className="size-3.5 opacity-0 transition-[transform,opacity] duration-200 group-hover/sb-collap:opacity-100 group-focus-visible/sb-collap:opacity-100 data-[state=open]:rotate-0 [[data-state=closed]_&]:rotate-[-90deg] [[data-state=closed]_&]:opacity-100" />
                </CollapsibleTrigger>
                {renderSidebarHeaderMenu({
                  ariaLabel: t("shell.organize.organizeChats"),
                  includeOrganize: true,
                  sortLabel: t("shell.organize.sortChatsBy"),
                  sortValue: chatSort,
                  onSortChange: setChatSort,
                })}
                {/* Starts a chat outside any project, whatever page is open. */}
                <button
                  type="button"
                  aria-label={t("shell.navigation.newChat")}
                  onClick={() => openNewChat(null)}
                  className="sidebar-header-action"
                >
                  <HugeiconsIcon icon={PencilEdit02Icon} strokeWidth={1.75} className="size-icon" />
                </button>
              </SidebarGroupLabel>
              <CollapsibleContent>
                <SidebarGroupContent className={unrailedRowPadding}>
                  <SidebarMenu>
                    {sortedRecentChatItems.map((item, index) =>
                      renderChatSidebarItem(
                        item,
                        "recent",
                        manualDragEnabled
                          ? {
                              scope: RECENTS_ORDER_SCOPE,
                              orderedIds: recentRowIds,
                              index,
                            }
                          : undefined,
                        { scope: RECENTS_ORDER_SCOPE, ids: recentRowIds },
                      ),
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
              <SidebarGroupContent className={unrailedRowPadding}>
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
                              aria-hidden
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
                                openDeleteDialog({ kind: "run", run })
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
          "relative pb-[11px] group-data-[collapsible=icon]:px-0",
          // The profile is outside the recent-chat scroller and keeps its full
          // width when that scroller gains or loses a scrollbar.
          unrailedRowPadding,
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
                    <span className="truncate font-heading text-ui-13p5 tracking-[0.025em] dark:tracking-[0.04em] font-semibold text-nav-fg">{displayTitle}</span>
                    <span className="truncate text-ui-11p5 tracking-nav text-muted-foreground">Unsloth</span>
                  </div>
                </SidebarMenuButton>
              </DropdownMenuTrigger>
              <DropdownMenuContent
                side="top"
                align="center"
                sideOffset={8}
                className="app-user-menu menu-soft-surface-up ring-0 w-[16rem] rounded-[20px] border border-transparent px-2.5 py-2.5 font-heading dark:border-white/[0.05]"
              >
                <DropdownMenuGroup>
                  <DropdownMenuItem
                    onSelect={() => useSettingsDialogStore.getState().openDialog()}
                  >
                    <HugeiconsIcon icon={Settings02Icon} strokeWidth={1.75} className="size-icon" />
                    <span>{t("shell.navigation.settings")}</span>
                    {settingsShortcutLabel && (
                      <DropdownMenuShortcut>
                        {settingsShortcutLabel}
                      </DropdownMenuShortcut>
                    )}
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
                <DropdownMenuSeparator className="mx-1! my-2.5! h-0! border-t border-border/70 bg-transparent! dark:border-white/15" />
                <DropdownMenuItem
                  onSelect={() => useSettingsDialogStore.getState().openDialog("about")}
                >
                  <HugeiconsIcon icon={HelpCircleIcon} strokeWidth={1.75} className="size-icon" />
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
                : confirmingDelete?.kind === "chats"
                  ? t("shell.selection.deleteTitle")
                  : confirmingDelete?.kind === "projects"
                    ? t("shell.selection.deleteProjectsTitle")
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
            ) : confirmingDelete?.kind === "chats" ? (
              t("shell.selection.deleteDescription", {
                count: confirmingDelete.items.length,
              })
            ) : confirmingDelete?.kind === "projects" ? (
              t("shell.selection.deleteProjectsDescription", {
                count: confirmingDelete.projects.length,
              })
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
            <label htmlFor="delete-files-on-delete" className="min-w-0 space-y-1">
              <span className="block text-sm font-medium text-foreground">
                {t("shell.selection.deleteFilesLabel")}
              </span>
              <span className="block break-words text-xs leading-5 text-muted-foreground">
                {confirmingDelete?.kind === "project"
                  ? (confirmingDelete.project.rootPath ??
                    "The project workspace folder will be removed from disk.")
                  : confirmingDelete?.kind === "projects"
                    ? t("shell.selection.deleteProjectsFilesDescription")
                    : confirmingDelete?.kind === "chats"
                      ? t("shell.selection.deleteFilesDescription")
                    : t("shell.selection.deleteChatFilesDescription")}
              </span>
            </label>
            <Switch
              id="delete-files-on-delete"
              checked={deleteFilesOnDelete}
              onCheckedChange={setDeleteFilesOnDelete}
              aria-label={t("shell.selection.deleteFilesLabel")}
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
      open={
        renamingTarget !== null &&
        (renamingTarget.kind !== "chat" || !renamingTarget.inline)
      }
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
    <NewProjectDialog
      open={creatingProject}
      onOpenChange={(open) => {
        setCreatingProject(open);
        if (!open) setProjectCreateMoveTarget(null);
      }}
      title={
        projectCreateMoveTarget ? "Move to new project" : "Create project"
      }
      submitLabel={projectCreateMoveTarget ? "Create and move" : "Create project"}
      onCreated={afterCreateProject}
    />
    </>
  );
}
