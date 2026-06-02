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
import { Switch } from "@/components/ui/switch";
import { useAnimatedThemeToggle } from "@/components/ui/animated-theme-toggler";
import { cn } from "@/lib/utils";
import {
  ChefHatIcon,
  CursorInfo02Icon,
  Delete02Icon,
  DownloadSquare01Icon,
  Edit03Icon,
  FolderAddIcon,
  Folder01Icon,
  Globe02Icon,
  HelpCircleIcon,
  Logout05Icon,
  MoreVerticalIcon,
  Search01Icon,
  PowerIcon,
  PencilEdit02Icon,
  LayoutAlignLeftIcon,
  Settings02Icon,
  TestTube01Icon,
  ZapIcon,
} from "@hugeicons/core-free-icons";
import {
  Tooltip,
  TooltipContent,
} from "@/components/ui/tooltip";
import { Tooltip as TooltipPrimitive } from "radix-ui";
import { HugeiconsIcon } from "@hugeicons/react";
import { ChevronDown, ChevronsUpDown, MoreHorizontalIcon, Moon, Sun } from "lucide-react";
import { Link, useNavigate, useRouterState } from "@tanstack/react-router";
import {
  ChatSearchDialog,
  createChatProject,
  deleteChatProject,
  deleteChatItem,
  moveChatItemToProject,
  renameChatItem,
  renameChatProject,
  useChatRuntimeStore,
  useChatProjects,
  useChatSearchStore,
  useChatSidebarItems,
  type ProjectRecord,
  type SidebarItem,
} from "@/features/chat";
import { useSettingsDialogStore } from "@/features/settings";
import { useEffectiveProfile, UserAvatar } from "@/features/profile";
import { usePlatformStore } from "@/config/env";
import { clearAuthTokens, logout } from "@/features/auth";
import { TOUR_OPEN_EVENT } from "@/features/tour";
import {
  deleteTrainingRun,
  emitTrainingRunDeleted,
  emitTrainingRunUpdated,
  removeTrainingUnloadGuard,
  renameTrainingRun,
  useTrainingHistorySidebarItems,
  useTrainingRuntimeStore,
} from "@/features/training";
import type { TrainingRunSummary } from "@/features/training";
import { useEffect, useMemo, useRef, useState, type ReactNode } from "react";
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

// Hugeicons' TestTube01Icon ships with two interior bubbles (paths #4
// and #5 of the 5-path definition). Slicing to the first three paths
// keeps the test-tube outline + horizontal cap + liquid line, dropping
// the bubbles. The original export stays untouched, and HugeiconsIcon
// renders this trimmed array exactly the same way.
const TestTubeOutlineIcon = TestTube01Icon.slice(
  0,
  3,
) as typeof TestTube01Icon;

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

function NavItem({
  icon,
  label,
  active,
  disabled,
  onClick,
  children,
  dataTour,
  className,
}: {
  icon: typeof ZapIcon;
  label: string;
  active: boolean;
  disabled?: boolean;
  onClick: () => void;
  children?: ReactNode;
  dataTour?: string;
  className?: string;
}) {
  return (
    <SidebarMenuItem className={className}>
      <div className="relative">
        <SidebarMenuButton
          tooltip={label}
          disabled={disabled}
          onClick={onClick}
          isActive={active}
          data-tour={dataTour}
          className="sidebar-nav-btn h-[33px] rounded-[11px] gap-[8.5px] px-2.5 font-medium group-data-[collapsible=icon]:!w-[32px] group-data-[collapsible=icon]:!rounded-[10px] group-data-[collapsible=icon]:mx-auto"
        >
          <HugeiconsIcon icon={icon} strokeWidth={1.75} className="size-icon! shrink-0 group-hover/menu-button:animate-icon-pop" />
          <span className="text-[14.5px] leading-[19px] tracking-nav">{label}</span>
        </SidebarMenuButton>
      </div>
      {children}
    </SidebarMenuItem>
  );
}

export function AppSidebar() {
  const t = useT();
  const { isDark, toggleTheme, anchorRef } = useAnimatedThemeToggle();
  const { pathname, search } = useRouterState({
    select: (s) => ({
      pathname: s.location.pathname,
      search: s.location.search as Record<string, string | undefined>,
    }),
  });
  const { togglePinned, isMobile, setOpenMobile } = useSidebar();
  const navigate = useNavigate();

  // Auto-close mobile Sheet after navigation
  const closeMobileIfOpen = () => {
    if (isMobile) setOpenMobile(false);
  };

  const isTrainingRunning = useTrainingRuntimeStore((s) => s.isTrainingRunning);
  const chatOnly = usePlatformStore((s) => s.isChatOnly());
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
  // Bottom fade hides at the very bottom (and for short, non-scrolling lists)
  // so the last row isn't washed out - Gemini-style.
  const [canScrollDown, setCanScrollDown] = useState(false);
  // Driven only from onScroll + a content-change effect below. Deliberately NO
  // ResizeObserver: its callback-driven setState created a render loop (React
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
  const { displayTitle, avatarDataUrl } = useEffectiveProfile();

  const { projects } = useChatProjects();
  const activeProjectId = isChatRoute
    ? ((search.project as string | undefined) ?? null)
    : null;
  const { items: allChatItems } = useChatSidebarItems({
    enabled: !isStudioRoute,
    requireMessages: false,
  });
  const recentChatItems = useMemo(
    () => allChatItems.filter((item) => !item.projectId),
    [allChatItems],
  );
  const chatItems = allChatItems;
  const storeThreadId = useChatRuntimeStore((s) => s.activeThreadId);
  const setActiveThreadId = useChatRuntimeStore((s) => s.setActiveThreadId);
  const activeThreadId = isChatRoute
    ? (search.thread as string | undefined) ??
      (search.compare as string | undefined) ??
      storeThreadId ??
      undefined
    : undefined;

  // Training runs
  const { items: runItems } = useTrainingHistorySidebarItems(
    !chatOnly && isStudioRoute,
  );
  const activeJobId = useTrainingRuntimeStore((s) => s.jobId);
  const currentRunViewActive = useTrainingRuntimeStore((s) => s.currentRunViewActive);
  const selectedHistoryRunId = useTrainingRuntimeStore((s) => s.selectedHistoryRunId);
  const setSelectedHistoryRunId = useTrainingRuntimeStore((s) => s.setSelectedHistoryRunId);

  // Recompute the bottom-fade state on mount and whenever the list height can
  // change (items load, sections collapse/expand, route switches the visible
  // list) - onScroll never fires for short, non-scrolling lists. Guarded
  // setState below means this can't loop even if a dep is a fresh reference.
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
    isStudioRoute,
  ]);

  const chatDisabled = isTrainingRunning;

  function chatSearchForProject(projectId: string | null) {
    if (projectId) {
      return { project: projectId };
    }
    return {
      new: createNavigationNonce(),
    };
  }

  function openNewChat(projectId = activeProjectId) {
    if (chatDisabled) return;
    setActiveThreadId(null);
    useChatRuntimeStore.getState().setActiveProjectId(projectId);
    navigate({ to: "/chat", search: chatSearchForProject(projectId) });
    closeMobileIfOpen();
  }

  function openProject(projectId: string) {
    if (chatDisabled) return;
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

  type RenameTarget =
    | { kind: "chat"; item: SidebarItem; current: string }
    | { kind: "project"; project: ProjectRecord; current: string }
    | { kind: "run"; run: TrainingRunSummary; current: string };
  const [renamingTarget, setRenamingTarget] = useState<RenameTarget | null>(
    null,
  );
  const [renameDraft, setRenameDraft] = useState("");
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
    const current = run.display_name ?? run.model_name;
    setRenameDraft(current);
    setRenamingTarget({ kind: "run", run, current });
  }
  async function commitRename() {
    const target = renamingTarget;
    if (!target || !renameDirty) return;
    setRenamingTarget(null);
    if (target.kind === "chat") {
      try {
        await renameChatItem(target.item, renameTrimmed);
      } catch (err) {
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

  type DeleteTarget =
    | { kind: "chat"; item: SidebarItem }
    | { kind: "project"; project: ProjectRecord }
    | { kind: "run"; run: TrainingRunSummary };
  const [confirmingDelete, setConfirmingDelete] =
    useState<DeleteTarget | null>(null);
  const [deleteProjectFiles, setDeleteProjectFiles] = useState(false);

  useEffect(() => {
    if (confirmingDelete?.kind !== "project") {
      setDeleteProjectFiles(false);
    }
  }, [confirmingDelete]);

  async function commitDelete() {
    const target = confirmingDelete;
    if (!target) return;
    const shouldDeleteProjectFiles =
      target.kind === "project" && deleteProjectFiles;
    setConfirmingDelete(null);
    if (target.kind === "chat") {
      try {
        await handleDeleteThread(target.item);
      } catch (err) {
        toast.error(translate("shell.toast.failedToDeleteChat"), {
          description: err instanceof Error ? err.message : undefined,
        });
      }
      return;
    }
    if (target.kind === "project") {
      try {
        await deleteChatProject(target.project.id, {
          deleteFiles: shouldDeleteProjectFiles,
        });
        if (activeProjectId === target.project.id) {
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
    const itemClass =
      variant === "project"
        ? "group/project-chat-item relative"
        : "group/recent-item relative";
    const actionClass =
      variant === "project"
        ? "sidebar-row-action group-hover/project-chat-item:opacity-100 group-hover/project-chat-item:pointer-events-auto focus-visible:opacity-100 focus-visible:pointer-events-auto"
        : "sidebar-row-action group-hover/recent-item:opacity-100 group-hover/recent-item:pointer-events-auto focus-visible:opacity-100 focus-visible:pointer-events-auto";
    const buttonClass = cn(
      "sidebar-nav-btn h-[33px] cursor-pointer rounded-[11px] pr-4 text-[14.5px] leading-[19px] tracking-nav font-medium",
      variant === "project" ? "pl-[37px]" : "pl-2.5",
      variant === "project"
        ? "group-hover/project-chat-item:pr-8 group-has-[.sidebar-row-action[data-state=open]]/project-chat-item:pr-8"
        : "group-hover/recent-item:pr-8 group-has-[.sidebar-row-action[data-state=open]]/recent-item:pr-8",
    );

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
          <span className="truncate">{item.title}</span>
        </SidebarMenuButton>
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
            align="end"
            sideOffset={4}
            className="app-user-menu menu-soft-surface menu-flat-destructive ring-0 w-44 py-2 font-heading rounded-[14px] border-0"
          >
            <DropdownMenuItem onSelect={() => openRenameChat(item)}>
              <HugeiconsIcon icon={Edit03Icon} strokeWidth={1.75} className="size-icon" />
              <span>Rename</span>
            </DropdownMenuItem>
            <DropdownMenuSub>
              <DropdownMenuSubTrigger>
                <HugeiconsIcon icon={Folder01Icon} strokeWidth={1.75} className="size-icon" />
                <span>Move to project</span>
              </DropdownMenuSubTrigger>
              <DropdownMenuSubContent
                sideOffset={8}
                alignOffset={-4}
                className="app-user-menu menu-soft-surface menu-flat-destructive ring-0 w-56 py-2 font-heading rounded-[14px] border-0"
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
            <DropdownMenuItem
              variant="destructive"
              onSelect={() => setConfirmingDelete({ kind: "chat", item })}
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
      <SidebarHeader className="pl-[17px] pr-3 pt-[14px] pb-[8px] group-data-[collapsible=icon]:px-0">
        {/* Expanded: compact logo + close toggle */}
        <div className="flex items-center justify-between gap-[8.5px] group-data-[collapsible=icon]:hidden">
          <Link
            to="/chat"
            onClick={(event) => {
              event.preventDefault();
              if (chatDisabled) return;
              openNewChat(null);
            }}
            className="flex items-center gap-[6px] select-none"
            aria-label={t("shell.aria.home")}
          >
            <img
              src="/circle-logo-small.png"
              alt="Unsloth"
              className="h-[34px] w-[34px] rounded-full object-cover"
            />
            <span className="font-heading text-[21px] font-semibold tracking-[0em] dark:tracking-[0.02em] leading-none text-black dark:text-white">
              unsloth
            </span>
            <span className="nav-badge ml-0.5 inline-flex items-center justify-center rounded-full border border-nav-beta-border px-[5px] pt-[3px] pb-[2px] text-[8px] font-medium leading-none tracking-[0.04em] text-nav-fg-muted antialiased subpixel-antialiased shadow-[0_1px_2px_rgba(0,0,0,0.06)] dark:shadow-[0_1px_2px_rgba(0,0,0,0.35)]">
              {t("shell.beta")}
            </span>
          </Link>
          {!isMobile && (
            <Tooltip>
              <TooltipPrimitive.Trigger asChild>
                <button
                  type="button"
                  onClick={togglePinned}
                  className="inline-flex h-[33px] w-[32px] cursor-pointer items-center justify-center rounded-[10px] text-nav-icon-idle dark:text-nav-fg-muted transition-colors hover:bg-nav-surface-hover hover:text-black dark:hover:text-white focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
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

        {/* Collapsed: panel icon doubles as expand trigger */}
        {!isMobile && (
          <div className="hidden group-data-[collapsible=icon]:flex h-[33px] items-center justify-center w-full">
            <Tooltip>
              <TooltipPrimitive.Trigger asChild>
                <button
                  type="button"
                  onClick={togglePinned}
                  className="inline-flex h-[33px] w-[32px] cursor-pointer items-center justify-center rounded-[10px] text-nav-fg transition-colors hover:bg-nav-surface-hover hover:text-black dark:hover:text-white focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
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
      </SidebarHeader>

      <SidebarGroup className="group-data-[collapsible=icon]:px-0 px-2 pt-[9px] pb-px shrink-0">
        <SidebarGroupContent>
          <SidebarMenu>
            <NavItem
              icon={PencilEdit02Icon}
              label={t("shell.navigation.newChat")}
              active={
                isChatRoute &&
                !search.thread &&
                !search.compare &&
                !search.project
              }
              disabled={chatDisabled}
              onClick={() => openNewChat(null)}
            />
            <NavItem
              icon={Search01Icon}
              label={t("shell.navigation.search")}
              active={false}
              onClick={() => {
                // Search is read-only over chat history and never runs
                // inference, so it stays available while training (unlike
                // New chat, which is gated on `chatDisabled`).
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
        className={cn(
          // pb-2 keeps the last row's rounded highlight clear of the
          // overflow clip edge so its bottom corners aren't shaved off.
          "sidebar-scroll-fade gap-0 overflow-y-auto overscroll-contain min-h-0 pb-2",
          scrolled && "is-scrolled",
        )}
      >
        <SidebarGroup className="group-data-[collapsible=icon]:px-0 px-2 py-0 shrink-0">
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
              />
              {/* Train has its own labelled section when expanded; surface it as
                  a plain icon here only while the sidebar is collapsed. */}
              <NavItem
                icon={TestTubeOutlineIcon}
                label={t("shell.navigation.train")}
                active={
                  pathname === "/studio" || pathname.startsWith("/studio/")
                }
                disabled={chatOnly}
                onClick={() => {
                  if (chatOnly) return;
                  navigate({ to: "/studio" });
                  closeMobileIfOpen();
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
              <SidebarGroupContent className="px-2">
                <SidebarMenu>
                  <NavItem
                    icon={TestTubeOutlineIcon}
                    label={t("shell.navigation.train")}
                    active={pathname === "/studio" || pathname.startsWith("/studio/")}
                    disabled={chatOnly}
                    onClick={() => {
                      if (chatOnly) return;
                      navigate({ to: "/studio" });
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
                  />
                  <NavItem
                    icon={DownloadSquare01Icon}
                    label={t("shell.navigation.export")}
                    active={pathname === "/export" || pathname.startsWith("/export/")}
                    disabled={chatOnly}
                    onClick={() => {
                      if (chatOnly) return;
                      navigate({ to: "/export" });
                      closeMobileIfOpen();
                    }}
                  />
                </SidebarMenu>
              </SidebarGroupContent>
            </CollapsibleContent>
          </SidebarGroup>
        </Collapsible>

        {!isStudioRoute && (
          <Collapsible open={chatOpen} onOpenChange={setChatOpen} asChild>
            <SidebarGroup className="group-data-[collapsible=icon]:hidden px-0 py-0">
              <SidebarGroupLabel className={cn("sidebar-sticky-label sidebar-sticky-label-following", scrolled && "is-scrolled")} asChild>
                <CollapsibleTrigger className="cursor-pointer flex w-full items-center gap-1 group/sb-collap">
                  {t("shell.navigation.recents")}
                  <ChevronDown className="size-3.5 opacity-0 transition-[transform,opacity] duration-200 group-hover/sb-collap:opacity-100 group-focus-visible/sb-collap:opacity-100 data-[state=open]:rotate-0 [[data-state=closed]_&]:rotate-[-90deg] [[data-state=closed]_&]:opacity-100" />
                </CollapsibleTrigger>
              </SidebarGroupLabel>
              <CollapsibleContent>
                <SidebarGroupContent className="px-2">
                  <SidebarMenu>
                    {recentChatItems.map((item) =>
                      renderChatSidebarItem(item, "recent"),
                    )}
                  </SidebarMenu>
                </SidebarGroupContent>
              </CollapsibleContent>
            </SidebarGroup>
          </Collapsible>
        )}

        {isStudioRoute && runItems.length > 0 && !chatOnly && (
          <Collapsible open={runsOpen} onOpenChange={setRunsOpen} asChild>
          <SidebarGroup className="group-data-[collapsible=icon]:hidden px-0 py-0">
            <SidebarGroupLabel className={cn("sidebar-sticky-label sidebar-sticky-label-following", scrolled && "is-scrolled")} asChild>
              <CollapsibleTrigger className="cursor-pointer flex w-full items-center gap-1 group/sb-collap">
                {t("shell.navigation.recents")}
                <ChevronDown className="size-3.5 opacity-0 transition-[transform,opacity] duration-200 group-hover/sb-collap:opacity-100 group-focus-visible/sb-collap:opacity-100 data-[state=open]:rotate-0 [[data-state=closed]_&]:rotate-[-90deg] [[data-state=closed]_&]:opacity-100" />
              </CollapsibleTrigger>
            </SidebarGroupLabel>
            <CollapsibleContent>
              <SidebarGroupContent className="px-2">
                <SidebarMenu>
                  {runItems.map((run) => {
                    // An explicit sidebar selection wins. Otherwise highlight
                    // the active job only while the "Current Run" tab is the
                    // view - that covers a live run (it auto-switches there) and
                    // a just-finished/errored run you're still viewing, while
                    // keeping the Configure tab unhighlighted even though
                    // `activeJobId` stays pinned to the last job.
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
                          className="sidebar-nav-btn h-auto flex-col items-start gap-0.5 py-[5px] rounded-[11px] pl-2.5 pr-7 text-[14.5px] tracking-nav font-medium"
                          onClick={() => {
                            setSelectedHistoryRunId(run.id);
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
                              {run.display_name ?? run.model_name}
                            </span>
                            <span className="ml-auto shrink-0 text-[10px] text-muted-foreground">
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
                                <MoreHorizontalIcon strokeWidth={1.75} className="size-icon" />
                              </span>
                            </button>
                          </DropdownMenuTrigger>
                          <DropdownMenuContent
                            side="bottom"
                            align="end"
                            sideOffset={4}
                            className="app-user-menu menu-soft-surface menu-flat-destructive ring-0 w-44 py-2 font-heading rounded-[14px] border-0"
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

      <SidebarFooter className="relative group-data-[collapsible=icon]:px-0">
        {/* Fade above the profile box, shown only while there's more list below
            the fold; at the very bottom (or for short lists) it fades out so the
            last row shows fully (Gemini-style). `right-2` keeps it clear of the
            8px scrollbar gutter so the scrollbar isn't faded out. */}
        <div
          aria-hidden="true"
          className={cn(
            "pointer-events-none absolute left-0 right-2 bottom-full h-10 bg-gradient-to-t from-[var(--sidebar)] to-transparent transition-opacity duration-200",
            canScrollDown ? "opacity-100" : "opacity-0",
          )}
        />
        <SidebarMenu>
          <SidebarMenuItem>
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <SidebarMenuButton
                  size="lg"
                  aria-label={t("shell.accountMenu", { name: displayTitle })}
                  className="sidebar-nav-btn !h-[40px] gap-[8px] px-2 py-[5px] rounded-[11px]"
                >
                  <div className="shrink-0">
                    <UserAvatar
                      name={displayTitle}
                      imageUrl={avatarDataUrl}
                      size="sm"
                      className="!size-[30px]"
                    />
                  </div>
                  <div className="flex flex-col gap-0.5 leading-tight group-data-[collapsible=icon]:hidden">
                    <span className="truncate font-heading text-[13.5px] tracking-[0.025em] dark:tracking-[0.04em] font-semibold text-nav-fg">{displayTitle}</span>
                    <span className="truncate text-[11.5px] tracking-nav text-muted-foreground">Unsloth</span>
                  </div>
                  <ChevronsUpDown strokeWidth={1.25} className="ml-auto size-4 text-muted-foreground group-data-[collapsible=icon]:hidden" />
                </SidebarMenuButton>
              </DropdownMenuTrigger>
              <DropdownMenuContent
                side="top"
                align="center"
                sideOffset={6}
                className="app-user-menu menu-soft-surface-up ring-0 w-[16rem] px-1.5 py-2.5 font-heading rounded-[20px] border-0"
              >
                <DropdownMenuGroup>
                  <DropdownMenuItem
                    onSelect={() => useSettingsDialogStore.getState().openDialog()}
                  >
                    <HugeiconsIcon icon={Settings02Icon} strokeWidth={1.75} className="size-icon" />
                    <span>{t("shell.navigation.settings")}</span>
                    <DropdownMenuShortcut>⌘,</DropdownMenuShortcut>
                  </DropdownMenuItem>
                  <DropdownMenuItem
                    onSelect={() => useSettingsDialogStore.getState().openDialog("api-keys")}
                  >
                    <HugeiconsIcon icon={Globe02Icon} strokeWidth={1.75} className="size-[18px]" />
                    <span>{t("shell.navigation.api")}</span>
                    <span className="ml-auto rounded-[6px] border border-emerald-500/25 bg-emerald-500/10 px-1.5 py-0.5 text-[10px] leading-none font-semibold text-emerald-700 dark:text-emerald-300">
                      {t("common.new")}
                    </span>
                  </DropdownMenuItem>
                  <DropdownMenuItem
                    ref={anchorRef as React.Ref<HTMLDivElement>}
                    onSelect={(e) => { e.preventDefault(); toggleTheme(); }}
                  >
                    {isDark ? <Sun strokeWidth={1.75} className="size-icon" /> : <Moon strokeWidth={1.75} className="size-icon" />}
                    <span>
                      {isDark
                        ? t("shell.navigation.lightMode")
                        : t("shell.navigation.darkMode")}
                    </span>
                  </DropdownMenuItem>
                  <DropdownMenuItem
                    disabled={!getTourId(pathname)}
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
                </DropdownMenuGroup>
                <DropdownMenuSeparator className="mx-2.5! my-2.5! h-0! border-t border-border/70 bg-transparent!" />
                <DropdownMenuItem
                  onSelect={() => useSettingsDialogStore.getState().openDialog("about")}
                >
                  <HugeiconsIcon icon={HelpCircleIcon} strokeWidth={1.75} className="size-icon" />
                  <span>{t("common.help")}</span>
                </DropdownMenuItem>
                <DropdownMenuItem
                  onSelect={async () => {
                    // Best-effort server-side revocation; ignore network errors
                    // so the local clear path still runs and the user lands on /login.
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
                <DropdownMenuItem onSelect={() => setShutdownOpen(true)}>
                  <HugeiconsIcon icon={PowerIcon} strokeWidth={1.75} className="size-icon" />
                  <span>{t("common.shutdown")}</span>
                </DropdownMenuItem>
              </DropdownMenuContent>
            </DropdownMenu>
          </SidebarMenuItem>
        </SidebarMenu>
      </SidebarFooter>
    </Sidebar>
    <ChatSearchDialog />
    <ShutdownDialog
      open={shutdownOpen}
      onOpenChange={setShutdownOpen}
      onAfterShutdown={removeTrainingUnloadGuard}
    />
    <Dialog
      open={confirmingDelete !== null}
      onOpenChange={(open) => {
        if (!open) {
          setConfirmingDelete(null);
          setDeleteProjectFiles(false);
        }
      }}
    >
      <DialogContent className="menu-flat-destructive corner-squircle border border-border/60 bg-background/98 shadow-none sm:max-w-md">
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
                confirmingDelete.run.display_name ??
                  confirmingDelete.run.model_name,
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
      open={renamingTarget !== null}
      onOpenChange={(open) => {
        if (!open) setRenamingTarget(null);
      }}
    >
      <DialogContent className="corner-squircle border border-border/60 bg-background/98 shadow-none sm:max-w-md">
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
      <DialogContent className="corner-squircle border border-border/60 bg-background/98 shadow-none sm:max-w-md">
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
