// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { ShutdownDialog } from "@/components/shutdown-dialog";
import { useAnimatedThemeToggle } from "@/components/ui/animated-theme-toggler";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
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
import { Tooltip, TooltipContent } from "@/components/ui/tooltip";
import { usePlatformStore } from "@/config/env";
import {
  ChatSearchDialog,
  type SidebarItem,
  deleteChatItem,
  renameChatItem,
  useChatRuntimeStore,
  useChatSearchStore,
  useChatSidebarItems,
} from "@/features/chat";
import { UserAvatar, useEffectiveProfile } from "@/features/profile";
import { useSettingsDialogStore } from "@/features/settings";
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
import { cn } from "@/lib/utils";
import {
  ChefHatIcon,
  CubeIcon,
  CursorInfo02Icon,
  Delete02Icon,
  DownloadSquare01Icon,
  Edit03Icon,
  Globe02Icon,
  HelpCircleIcon,
  LayoutAlignLeftIcon,
  PencilEdit02Icon,
  PowerIcon,
  Search01Icon,
  Settings02Icon,
  type ZapIcon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { Link, useNavigate, useRouterState } from "@tanstack/react-router";
import {
  ChevronDown,
  ChevronsUpDown,
  Moon,
  MoreHorizontalIcon,
  Sun,
} from "lucide-react";
import { Tooltip as TooltipPrimitive } from "radix-ui";
import { useEffect, useState } from "react";
import { toast } from "sonner";

function getTourId(pathname: string): string | null {
  if (pathname.startsWith("/studio")) return "studio";
  if (pathname.startsWith("/export")) return "export";
  if (pathname.startsWith("/chat")) return "chat";
  return null;
}

import { TrainIcon as TestTubeOutlineIcon } from "@/components/icons/train-icon";

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
}: {
  icon: typeof ZapIcon;
  label: string;
  active: boolean;
  disabled?: boolean;
  onClick: () => void;
  children?: React.ReactNode;
  dataTour?: string;
}) {
  return (
    <SidebarMenuItem>
      <div className="relative">
        <SidebarMenuButton
          tooltip={label}
          disabled={disabled}
          onClick={onClick}
          isActive={active}
          data-tour={dataTour}
          className="sidebar-nav-btn h-[35px] rounded-[10px] gap-[8.5px] px-2.5 font-medium group-data-[collapsible=icon]:!w-[32px] group-data-[collapsible=icon]:!rounded-[10px] group-data-[collapsible=icon]:mx-auto"
        >
          <HugeiconsIcon
            icon={icon}
            strokeWidth={1.75}
            className="size-icon! shrink-0 group-hover/menu-button:animate-icon-pop"
          />
          <span className="text-[14.5px] leading-[19px] tracking-nav">
            {label}
          </span>
        </SidebarMenuButton>
      </div>
      {children}
    </SidebarMenuItem>
  );
}

export function AppSidebar() {
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

  // Chat collapsible state — open by default, auto-expand on route entry
  const isChatRoute = pathname.startsWith("/chat");
  const isStudioRoute =
    pathname === "/studio" || pathname.startsWith("/studio/");
  const [chatOpen, setChatOpen] = useState(true);
  const [runsOpen, setRunsOpen] = useState(true);

  useEffect(() => {
    if (isChatRoute) setChatOpen(true);
  }, [isChatRoute]);
  useEffect(() => {
    if (isStudioRoute) setRunsOpen(true);
  }, [isStudioRoute]);

  const { displayTitle, avatarDataUrl } = useEffectiveProfile();

  const { items: chatItems } = useChatSidebarItems();
  const storeThreadId = useChatRuntimeStore((s) => s.activeThreadId);
  const setActiveThreadId = useChatRuntimeStore((s) => s.setActiveThreadId);
  const activeThreadId = isChatRoute
    ? ((search.thread as string | undefined) ??
      (search.compare as string | undefined) ??
      storeThreadId ??
      undefined)
    : undefined;

  // Training runs
  const { items: runItems } = useTrainingHistorySidebarItems(
    !chatOnly && isStudioRoute,
  );
  const activeJobId = useTrainingRuntimeStore((s) => s.jobId);
  const selectedHistoryRunId = useTrainingRuntimeStore(
    (s) => s.selectedHistoryRunId,
  );
  const setSelectedHistoryRunId = useTrainingRuntimeStore(
    (s) => s.setSelectedHistoryRunId,
  );

  const chatDisabled = isTrainingRunning;

  async function handleDeleteThread(
    item: Parameters<typeof deleteChatItem>[0],
  ) {
    await deleteChatItem(item, activeThreadId, (view) => {
      navigate({
        to: "/chat",
        search: { new: view.newThreadNonce },
      });
    });
  }

  type RenameTarget =
    | { kind: "chat"; item: SidebarItem; current: string }
    | { kind: "run"; run: TrainingRunSummary; current: string };
  const [renamingTarget, setRenamingTarget] = useState<RenameTarget | null>(
    null,
  );
  const [renameDraft, setRenameDraft] = useState("");
  const renameTrimmed = renameDraft.trim();
  const nextRunDisplayName = renameTrimmed.length > 0 ? renameTrimmed : null;
  const renameDirty =
    renamingTarget !== null &&
    (renamingTarget.kind === "chat"
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
        toast.error("Failed to rename chat", {
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
      toast.error("Failed to rename run", {
        description: err instanceof Error ? err.message : undefined,
      });
    }
  }

  type DeleteTarget =
    | { kind: "chat"; item: SidebarItem }
    | { kind: "run"; run: TrainingRunSummary };
  const [confirmingDelete, setConfirmingDelete] = useState<DeleteTarget | null>(
    null,
  );
  const [deleteRunArtifacts, setDeleteRunArtifacts] = useState(false);

  useEffect(() => {
    if (confirmingDelete?.kind === "run") {
      setDeleteRunArtifacts(false);
    }
  }, [confirmingDelete]);

  async function commitDelete() {
    const target = confirmingDelete;
    if (!target) return;
    const alsoDeleteArtifacts =
      target.kind === "run" &&
      target.run.artifacts_available &&
      deleteRunArtifacts;
    setConfirmingDelete(null);
    if (target.kind === "chat") {
      try {
        await handleDeleteThread(target.item);
      } catch (err) {
        toast.error("Failed to delete chat", {
          description: err instanceof Error ? err.message : undefined,
        });
      }
      return;
    }
    if (target.run.status === "running") {
      toast.error("Cannot delete a running training run");
      return;
    }
    try {
      await deleteTrainingRun(target.run.id, {
        deleteArtifacts: alsoDeleteArtifacts,
      });
      if (selectedHistoryRunId === target.run.id) {
        setSelectedHistoryRunId(null);
      }
      emitTrainingRunDeleted(target.run.id);
    } catch (err) {
      toast.error("Failed to delete run", {
        description: err instanceof Error ? err.message : undefined,
      });
    }
  }

  return (
    <>
      <Sidebar collapsible="icon" variant="sidebar" className="font-heading">
        <SidebarHeader className="pl-[17px] pr-3 pt-[12px] pb-[8px] group-data-[collapsible=icon]:px-0">
          {/* Expanded: compact logo + close toggle */}
          <div className="flex items-center justify-between gap-[8.5px] group-data-[collapsible=icon]:hidden">
            <Link
              to="/chat"
              onClick={(event) => {
                event.preventDefault();
                if (chatDisabled) return;
                setActiveThreadId(null);
                closeMobileIfOpen();
                void navigate({
                  to: "/chat",
                  search: { new: createNavigationNonce() },
                });
              }}
              className="flex items-center gap-[6px] select-none"
              aria-label="Unsloth home"
            >
              <img
                src="/circle-logo-small.png"
                alt="Unsloth"
                className="h-[34px] w-[34px] rounded-full object-cover"
              />
              <span className="font-heading text-[21px] font-semibold tracking-[-0.01em] dark:tracking-[0.02em] leading-none text-[#232528] dark:text-white">
                unsloth
              </span>
              <span className="nav-badge ml-0.5 inline-flex items-center justify-center rounded-full border border-nav-beta-border px-[5px] pt-[3px] pb-[2px] text-[8px] font-medium leading-none tracking-[0.04em] text-nav-fg-muted antialiased subpixel-antialiased shadow-[0_1px_2px_rgba(0,0,0,0.06)] dark:shadow-[0_1px_2px_rgba(0,0,0,0.35)]">
                BETA
              </span>
            </Link>
            {!isMobile && (
              <Tooltip>
                <TooltipPrimitive.Trigger asChild={true}>
                  <button
                    type="button"
                    onClick={togglePinned}
                    className="inline-flex h-[35px] w-[32px] items-center justify-center rounded-[10px] text-nav-icon-idle dark:text-nav-fg-muted transition-colors hover:bg-nav-surface-hover hover:text-[#232528] dark:hover:text-white focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
                    aria-label="Close sidebar"
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
                  Close sidebar
                </TooltipContent>
              </Tooltip>
            )}
          </div>

          {/* Collapsed: panel icon doubles as expand trigger */}
          {!isMobile && (
            <div className="hidden group-data-[collapsible=icon]:flex h-[35px] items-center justify-center w-full">
              <Tooltip>
                <TooltipPrimitive.Trigger asChild={true}>
                  <button
                    type="button"
                    onClick={togglePinned}
                    className="inline-flex h-[35px] w-[32px] items-center justify-center rounded-[10px] text-nav-fg transition-colors hover:bg-nav-surface-hover hover:text-[#232528] dark:hover:text-white focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
                    aria-label="Open sidebar"
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
                  Open sidebar
                </TooltipContent>
              </Tooltip>
            </div>
          )}
        </SidebarHeader>

        <SidebarGroup className="group-data-[collapsible=icon]:px-0 px-2 pt-[9px] pb-[8px] shrink-0">
          <SidebarGroupContent>
            <SidebarMenu>
              <NavItem
                icon={PencilEdit02Icon}
                label="New Chat"
                active={false}
                disabled={chatDisabled}
                onClick={() => {
                  if (chatDisabled) return;
                  setActiveThreadId(null);
                  navigate({
                    to: "/chat",
                    search: { new: createNavigationNonce() },
                  });
                  closeMobileIfOpen();
                }}
              />
              <NavItem
                icon={Search01Icon}
                label="Search"
                active={false}
                disabled={chatDisabled}
                onClick={() => {
                  if (chatDisabled) return;
                  useChatSearchStore.getState().open();
                  closeMobileIfOpen();
                }}
              />
              <NavItem
                icon={CubeIcon}
                label="Hub"
                active={
                  pathname === "/models" || pathname.startsWith("/models/")
                }
                onClick={() => {
                  navigate({ to: "/models" });
                  closeMobileIfOpen();
                }}
              />
            </SidebarMenu>
          </SidebarGroupContent>
        </SidebarGroup>

        <SidebarGroup
          data-tour="navbar"
          className="group-data-[collapsible=icon]:px-0 px-2 pt-[9px] pb-[20px] shrink-0"
        >
          <SidebarGroupContent>
            <SidebarMenu>
              <NavItem
                icon={TestTubeOutlineIcon}
                label="Train"
                active={
                  pathname === "/studio" || pathname.startsWith("/studio/")
                }
                disabled={chatOnly}
                onClick={() => {
                  if (chatOnly) return;
                  navigate({ to: "/studio" });
                  closeMobileIfOpen();
                }}
              />

              <NavItem
                icon={ChefHatIcon}
                label="Recipes"
                active={pathname.startsWith("/data-recipes")}
                onClick={() => {
                  navigate({ to: "/data-recipes" });
                  closeMobileIfOpen();
                }}
              />

              <NavItem
                icon={DownloadSquare01Icon}
                label="Export"
                active={
                  pathname === "/export" || pathname.startsWith("/export/")
                }
                disabled={chatOnly}
                onClick={() => {
                  if (chatOnly) return;
                  navigate({ to: "/export" });
                  closeMobileIfOpen();
                }}
              />
            </SidebarMenu>
          </SidebarGroupContent>
        </SidebarGroup>

        <SidebarContent className="gap-0 overflow-y-auto overscroll-contain min-h-0 [scrollbar-gutter:stable]">
          {/* Recent Chats — hide on Studio only (Eyera fac13); chatOpen = ec695 clickability */}
          {!isStudioRoute && chatItems.length > 0 && (
            <Collapsible
              open={chatOpen}
              onOpenChange={setChatOpen}
              asChild={true}
            >
              <SidebarGroup className="group-data-[collapsible=icon]:hidden px-0 py-0">
                <SidebarGroupLabel
                  className="sidebar-sticky-label"
                  asChild={true}
                >
                  <CollapsibleTrigger className="cursor-pointer flex items-center justify-between">
                    Recents
                    <ChevronDown className="size-3.5 transition-transform duration-200 data-[state=open]:rotate-0 [[data-state=closed]_&]:rotate-[-90deg]" />
                  </CollapsibleTrigger>
                </SidebarGroupLabel>
                <CollapsibleContent>
                  <SidebarGroupContent className="px-2">
                    <SidebarMenu>
                      {chatItems.map((item) => (
                        <SidebarMenuItem
                          key={item.id}
                          className="group/recent-item relative"
                        >
                          <SidebarMenuButton
                            isActive={activeThreadId === item.id}
                            className="sidebar-nav-btn h-[32px] rounded-[10px] pl-2.5 pr-2.5 group-hover/recent-item:pr-10 group-has-[.sidebar-row-action[data-state=open]]/recent-item:pr-10 text-[14.5px] leading-[19px] tracking-nav font-medium"
                            onClick={() => {
                              navigate({
                                to: "/chat",
                                search:
                                  item.type === "single"
                                    ? { thread: item.id }
                                    : { compare: item.id },
                              });
                              closeMobileIfOpen();
                            }}
                          >
                            <span className="truncate">{item.title}</span>
                          </SidebarMenuButton>
                          <DropdownMenu>
                            <DropdownMenuTrigger asChild={true}>
                              <button
                                type="button"
                                onClick={(e) => e.stopPropagation()}
                                aria-label="Chat options"
                                className="sidebar-row-action group-hover/recent-item:opacity-100 group-hover/recent-item:pointer-events-auto focus-visible:opacity-100 focus-visible:pointer-events-auto"
                              >
                                <span className="sidebar-row-action-glyph">
                                  <MoreHorizontalIcon
                                    strokeWidth={1.75}
                                    className="size-icon"
                                  />
                                </span>
                              </button>
                            </DropdownMenuTrigger>
                            <DropdownMenuContent
                              side="bottom"
                              align="end"
                              sideOffset={4}
                              className="app-user-menu menu-soft-surface menu-flat-destructive ring-0 w-44 py-2 font-heading rounded-[14px] border-0"
                            >
                              <DropdownMenuItem
                                onSelect={() => openRenameChat(item)}
                              >
                                <HugeiconsIcon
                                  icon={Edit03Icon}
                                  strokeWidth={1.75}
                                  className="size-icon"
                                />
                                <span>Rename</span>
                              </DropdownMenuItem>
                              <DropdownMenuItem
                                variant="destructive"
                                onSelect={() =>
                                  setConfirmingDelete({ kind: "chat", item })
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
                      ))}
                    </SidebarMenu>
                  </SidebarGroupContent>
                </CollapsibleContent>
              </SidebarGroup>
            </Collapsible>
          )}

          {/* Recent Runs */}
          {isStudioRoute && runItems.length > 0 && !chatOnly && (
            <Collapsible
              open={runsOpen}
              onOpenChange={setRunsOpen}
              asChild={true}
            >
              <SidebarGroup className="group-data-[collapsible=icon]:hidden px-0 py-0">
                <SidebarGroupLabel
                  className="sidebar-sticky-label"
                  asChild={true}
                >
                  <CollapsibleTrigger className="cursor-pointer flex items-center justify-between">
                    Recents
                    <ChevronDown className="size-3.5 transition-transform duration-200 data-[state=open]:rotate-0 [[data-state=closed]_&]:rotate-[-90deg]" />
                  </CollapsibleTrigger>
                </SidebarGroupLabel>
                <CollapsibleContent>
                  <SidebarGroupContent className="px-2">
                    <SidebarMenu>
                      {runItems.map((run) => {
                        const isActiveRun =
                          selectedHistoryRunId === run.id ||
                          activeJobId === run.id;
                        return (
                          <SidebarMenuItem
                            key={run.id}
                            className="group/run-item relative"
                          >
                            <SidebarMenuButton
                              isActive={isActiveRun}
                              className="sidebar-nav-btn h-auto flex-col items-start gap-0.5 py-[5px] rounded-[10px] pl-2.5 pr-7 text-[14.5px] tracking-nav font-medium"
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
                                  aria-hidden={true}
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
                              <DropdownMenuTrigger asChild={true}>
                                <button
                                  type="button"
                                  onClick={(e) => e.stopPropagation()}
                                  aria-label="Run options"
                                  className="sidebar-row-action group-hover/run-item:opacity-100 group-hover/run-item:pointer-events-auto focus-visible:opacity-100 focus-visible:pointer-events-auto"
                                >
                                  <span className="sidebar-row-action-glyph">
                                    <MoreHorizontalIcon
                                      strokeWidth={1.75}
                                      className="size-icon"
                                    />
                                  </span>
                                </button>
                              </DropdownMenuTrigger>
                              <DropdownMenuContent
                                side="bottom"
                                align="end"
                                sideOffset={4}
                                className="app-user-menu menu-soft-surface menu-flat-destructive ring-0 w-44 py-2 font-heading rounded-[14px] border-0"
                              >
                                <DropdownMenuItem
                                  onSelect={() => openRenameRun(run)}
                                >
                                  <HugeiconsIcon
                                    icon={Edit03Icon}
                                    strokeWidth={1.75}
                                    className="size-icon"
                                  />
                                  <span>Rename</span>
                                </DropdownMenuItem>
                                <DropdownMenuItem
                                  variant="destructive"
                                  disabled={run.status === "running"}
                                  onSelect={() =>
                                    setConfirmingDelete({ kind: "run", run })
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
                      })}
                    </SidebarMenu>
                  </SidebarGroupContent>
                </CollapsibleContent>
              </SidebarGroup>
            </Collapsible>
          )}
        </SidebarContent>

        <SidebarFooter className="border-t border-sidebar-border group-data-[collapsible=icon]:border-transparent group-data-[collapsible=icon]:px-0">
          <SidebarMenu>
            <SidebarMenuItem>
              <DropdownMenu>
                <DropdownMenuTrigger asChild={true}>
                  <SidebarMenuButton
                    size="lg"
                    aria-label={`${displayTitle} account menu`}
                    className="sidebar-nav-btn !h-[50px] gap-[8px] px-2 py-[9px] rounded-[10px]"
                  >
                    <div className="shrink-0">
                      <UserAvatar
                        name={displayTitle}
                        imageUrl={avatarDataUrl}
                        size="sm"
                        className="!size-8"
                      />
                    </div>
                    <div className="flex flex-col gap-0.5 leading-tight group-data-[collapsible=icon]:hidden">
                      <span className="truncate font-heading text-[13.5px] tracking-[0.025em] dark:tracking-[0.04em] font-semibold text-nav-fg">
                        {displayTitle}
                      </span>
                      <span className="truncate text-[11.5px] tracking-nav text-muted-foreground">
                        Unsloth
                      </span>
                    </div>
                    <ChevronsUpDown
                      strokeWidth={1.25}
                      className="ml-auto size-4 text-muted-foreground group-data-[collapsible=icon]:hidden"
                    />
                  </SidebarMenuButton>
                </DropdownMenuTrigger>
                <DropdownMenuContent
                  side="top"
                  align="start"
                  className="app-user-menu menu-soft-surface-up ring-0 w-[15rem] py-2.5 font-heading rounded-[14px] border-0"
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
                      <span>Settings</span>
                      <DropdownMenuShortcut>⌘,</DropdownMenuShortcut>
                    </DropdownMenuItem>
                    <DropdownMenuItem
                      onSelect={() =>
                        useSettingsDialogStore.getState().openDialog("api-keys")
                      }
                    >
                      <HugeiconsIcon
                        icon={Globe02Icon}
                        strokeWidth={1.75}
                        className="size-[18px]"
                      />
                      <span>API</span>
                      <span className="ml-auto rounded-[6px] border border-emerald-500/25 bg-emerald-500/10 px-1.5 py-0.5 text-[10px] leading-none font-semibold text-emerald-700 dark:text-emerald-300">
                        New
                      </span>
                    </DropdownMenuItem>
                    <DropdownMenuItem
                      ref={anchorRef as React.Ref<HTMLDivElement>}
                      onSelect={(e) => {
                        e.preventDefault();
                        toggleTheme();
                      }}
                    >
                      {isDark ? (
                        <Sun strokeWidth={1.75} className="size-icon" />
                      ) : (
                        <Moon strokeWidth={1.75} className="size-icon" />
                      )}
                      <span>{isDark ? "Light Mode" : "Dark Mode"}</span>
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
                      <HugeiconsIcon
                        icon={CursorInfo02Icon}
                        strokeWidth={1.75}
                        className="size-icon"
                      />
                      <span>Guided Tour</span>
                    </DropdownMenuItem>
                  </DropdownMenuGroup>
                  <DropdownMenuSeparator className="mx-2.5! my-2.5! h-0! border-t border-border/70 bg-transparent!" />
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
                    <span>Help</span>
                  </DropdownMenuItem>
                  <DropdownMenuItem onSelect={() => setShutdownOpen(true)}>
                    <HugeiconsIcon
                      icon={PowerIcon}
                      strokeWidth={1.75}
                      className="size-icon"
                    />
                    <span>Shutdown</span>
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
          if (!open) setConfirmingDelete(null);
        }}
      >
        <DialogContent className="menu-flat-destructive corner-squircle border border-border/60 bg-background/98 shadow-none sm:max-w-md">
          <DialogHeader>
            <DialogTitle>
              {confirmingDelete?.kind === "run"
                ? "Delete training run"
                : "Delete chat"}
            </DialogTitle>
            <DialogDescription>
              {confirmingDelete?.kind === "run" ? (
                <>
                  Are you sure you want to delete this run{" "}
                  <em>
                    {confirmingDelete.run.display_name ??
                      confirmingDelete.run.model_name}
                  </em>
                  ? Statistics will be removed from history.
                </>
              ) : confirmingDelete?.kind === "chat" ? (
                <>
                  Are you sure you want to delete this chat{" "}
                  <em>{confirmingDelete.item.title}</em>?
                </>
              ) : null}
            </DialogDescription>
          </DialogHeader>
          {confirmingDelete?.kind === "run" &&
            confirmingDelete.run.artifacts_available && (
              <label className="mt-1 flex cursor-pointer items-start gap-2.5 rounded-md border border-border/60 bg-muted/40 p-3 text-xs leading-relaxed">
                <Checkbox
                  checked={deleteRunArtifacts}
                  onCheckedChange={(checked) =>
                    setDeleteRunArtifacts(checked === true)
                  }
                  className="mt-0.5"
                />
                <span className="flex flex-col gap-0.5">
                  <span className="font-medium text-foreground">
                    Also delete adapter files on disk
                  </span>
                  <span className="text-muted-foreground">
                    Removes the folder from outputs. The model will disappear
                    from the chat picker. Leave unchecked to keep the files.
                  </span>
                </span>
              </label>
            )}
          <DialogFooter className="flex-wrap gap-2 sm:justify-end">
            <Button
              type="button"
              variant="ghost"
              onClick={() => setConfirmingDelete(null)}
            >
              Cancel
            </Button>
            <Button
              type="button"
              variant="destructive"
              onClick={() => void commitDelete()}
            >
              Delete
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
              {renamingTarget?.kind === "run" ? "Rename run" : "Rename chat"}
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
              renamingTarget?.kind === "run" ? "Run name" : "Chat title"
            }
            aria-label={
              renamingTarget?.kind === "run" ? "Run name" : "Chat title"
            }
            className="focus-visible:border-input focus-visible:ring-0"
          />
          <DialogFooter className="flex-wrap gap-2 sm:justify-end">
            <Button
              type="button"
              variant="ghost"
              onClick={() => setRenamingTarget(null)}
            >
              Cancel
            </Button>
            <Button
              type="button"
              onClick={() => void commitRename()}
              disabled={!renameDirty}
            >
              Save
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </>
  );
}
