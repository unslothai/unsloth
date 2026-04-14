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
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { useAnimatedThemeToggle } from "@/components/ui/animated-theme-toggler";
import { cn } from "@/lib/utils";
import {
  Book03Icon,
  ChefHatIcon,
  CursorInfo02Icon,
  Delete02Icon,
  MessageSearch01Icon,
  NewReleasesIcon,
  PackageIcon,
  PencilEdit02Icon,
  Settings02Icon,
  SidebarLeft01Icon,
  SidebarRight01Icon,
  ZapIcon,
} from "@hugeicons/core-free-icons";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { HugeiconsIcon } from "@hugeicons/react";
import { ChevronDown, ChevronsUpDown, Moon, Sun } from "lucide-react";
import { Link, useNavigate, useRouterState } from "@tanstack/react-router";
import { motion } from "motion/react";
import { useTrainingRuntimeStore } from "@/features/training";
import { useSettingsDialogStore } from "@/features/settings";
import { usePlatformStore } from "@/config/env";
import { TOUR_OPEN_EVENT } from "@/features/tour";
import {
  useChatSidebarItems,
  deleteChatItem,
} from "@/features/chat/hooks/use-chat-sidebar-items";
import { useChatRuntimeStore } from "@/features/chat/stores/chat-runtime-store";
import { useTrainingHistorySidebarItems, deleteTrainingRun } from "@/features/training";
import type { TrainingRunSummary } from "@/features/training";
import { useEffect, useState } from "react";

function getTourId(pathname: string): string | null {
  if (pathname.startsWith("/studio")) return "studio";
  if (pathname.startsWith("/export")) return "export";
  if (pathname.startsWith("/chat")) return "chat";
  return null;
}

const NAV_SPRING = { type: "spring", stiffness: 500, damping: 35, mass: 0.5 } as const;

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

function NavItem({
  icon,
  label,
  active,
  disabled,
  onClick,
  children,
}: {
  icon: typeof ZapIcon;
  label: string;
  active: boolean;
  disabled?: boolean;
  onClick: () => void;
  children?: React.ReactNode;
}) {
  return (
    <SidebarMenuItem>
      <div className="relative">
        {active && (
          <motion.div
            layoutId="sidebar-active-indicator"
            className="absolute left-0 top-0 bottom-0 w-[3px] rounded-full bg-primary"
            transition={NAV_SPRING}
          />
        )}
        <SidebarMenuButton
          tooltip={label}
          disabled={disabled}
          onClick={onClick}
          isActive={active}
          className="rounded-none pr-0 pl-4"
        >
          <HugeiconsIcon icon={icon} className="size-5" />
          <span>{label}</span>
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

  // Chat collapsible state — open by default, syncs with route
  const isChatRoute = pathname.startsWith("/chat");
  const isStudioRoute = pathname === "/studio" || pathname.startsWith("/studio/");
  const [chatOpen, setChatOpen] = useState(true);
  const [runsOpen, setRunsOpen] = useState(true);

  const isRecipesRoute = pathname.startsWith("/data-recipes");

  useEffect(() => {
    if (isChatRoute) setChatOpen(true);
  }, [isChatRoute]);

  useEffect(() => {
    if (isStudioRoute) setRunsOpen(true);
  }, [isStudioRoute]);

  const { items: chatItems } = useChatSidebarItems();
  const storeThreadId = useChatRuntimeStore((s) => s.activeThreadId);
  const activeThreadId = (search.thread as string | undefined) ?? (search.compare as string | undefined) ?? storeThreadId ?? undefined;

  // Training runs
  const { items: runItems, refresh: refreshRuns } = useTrainingHistorySidebarItems(!chatOnly);
  const activeJobId = useTrainingRuntimeStore((s) => s.jobId);
  const selectedHistoryRunId = useTrainingRuntimeStore((s) => s.selectedHistoryRunId);
  const setSelectedHistoryRunId = useTrainingRuntimeStore((s) => s.setSelectedHistoryRunId);

  const chatDisabled = isTrainingRunning;

  async function handleDeleteThread(item: Parameters<typeof deleteChatItem>[0]) {
    await deleteChatItem(item, activeThreadId, (view) => {
      navigate({
        to: "/chat",
        search: { new: view.newThreadNonce },
      });
    });
  }

  return (
    <Sidebar collapsible="icon" variant="sidebar">
      <SidebarHeader className="group-data-[collapsible=icon]:px-0">
        {/* Expanded: compact logo + close toggle */}
        <div className="flex items-center justify-between gap-2 px-1 py-1 group-data-[collapsible=icon]:hidden">
          <Link
            to={chatOnly ? "/chat" : "/studio"}
            onClick={closeMobileIfOpen}
            className="flex items-center select-none"
            aria-label="Unsloth home"
          >
            <img
              src="/blacklogo-c.png"
              alt="Unsloth"
              className="h-7 w-auto dark:hidden"
            />
            <img
              src="/whitelogo-c.png"
              alt="Unsloth"
              className="hidden h-7 w-auto dark:block"
            />
          </Link>
          {!isMobile && (
            <button
              type="button"
              onClick={togglePinned}
              className="inline-flex h-7 w-7 items-center justify-center rounded-md text-sidebar-foreground/70 transition-colors hover:bg-sidebar-accent hover:text-sidebar-accent-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
              title="Close sidebar"
              aria-label="Close sidebar"
            >
              <HugeiconsIcon icon={SidebarLeft01Icon} className="size-4" />
            </button>
          )}
        </div>

        {/* Collapsed: sticker with hover-swap to open toggle */}
        {!isMobile && (
          <div className="hidden group-data-[collapsible=icon]:flex relative group/logo items-center justify-center h-9 w-full">
            <Link
              to={chatOnly ? "/chat" : "/studio"}
              onClick={closeMobileIfOpen}
              className="flex items-center justify-center transition-opacity group-hover/logo:opacity-0 pointer-events-auto group-hover/logo:pointer-events-none"
              aria-label="Unsloth home"
            >
              <img src="/sticker.png" alt="Unsloth" className="size-7" />
            </Link>
            <Tooltip>
              <TooltipTrigger asChild>
                <button
                  type="button"
                  onClick={togglePinned}
                  className="absolute inset-0 flex items-center justify-center rounded-md opacity-0 transition-opacity hover:bg-sidebar-accent group-hover/logo:opacity-100 focus-visible:opacity-100 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
                  aria-label="Open sidebar"
                >
                  <HugeiconsIcon
                    icon={SidebarRight01Icon}
                    className="size-5 text-sidebar-foreground/70"
                  />
                </button>
              </TooltipTrigger>
              <TooltipContent side="right" sideOffset={8}>
                Open sidebar
              </TooltipContent>
            </Tooltip>
          </div>
        )}
      </SidebarHeader>

      <SidebarContent>
        {/* Navigation */}
        <SidebarGroup data-tour="navbar" className="group-data-[collapsible=icon]:px-0 px-0">
          <SidebarGroupLabel className="pl-4">Navigation</SidebarGroupLabel>
          <SidebarGroupContent>
            <SidebarMenu>
              <NavItem
                icon={PencilEdit02Icon}
                label="New Chat"
                active={isChatRoute}
                disabled={chatDisabled}
                onClick={() => {
                  if (chatDisabled) return;
                  navigate({ to: "/chat", search: { new: crypto.randomUUID() } });
                  closeMobileIfOpen();
                }}
              />

              <NavItem
                icon={ZapIcon}
                label="Studio"
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
                label="Recipes"
                active={isRecipesRoute}
                onClick={() => {
                  navigate({ to: "/data-recipes" });
                  closeMobileIfOpen();
                }}
              />

              <NavItem
                icon={PackageIcon}
                label="Export"
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
        </SidebarGroup>

        {/* Recent Chats */}
        {chatItems.length > 0 && (
          <Collapsible open={isChatRoute ? chatOpen : true} onOpenChange={setChatOpen} asChild>
          <SidebarGroup className="group-data-[collapsible=icon]:hidden overflow-hidden">
            <SidebarGroupLabel asChild>
              {isChatRoute ? (
                <CollapsibleTrigger className="cursor-pointer flex w-full items-center justify-between">
                  Recent Chats
                  <ChevronDown className="size-3.5 transition-transform duration-200 data-[state=open]:rotate-0 [[data-state=closed]_&]:rotate-[-90deg]" />
                </CollapsibleTrigger>
              ) : (
                <span>Recent Chats</span>
              )}
            </SidebarGroupLabel>
            <CollapsibleContent>
            <SidebarGroupContent className="overflow-y-auto">
              <SidebarMenu>
                {(isChatRoute || isMobile ? chatItems : chatItems.slice(0, 1)).map((item) => (
                  <SidebarMenuItem key={item.id} className="group/recent-item relative">
                    <SidebarMenuButton
                      isActive={activeThreadId === item.id}
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
                    <button
                      type="button"
                      onClick={(e) => {
                        e.stopPropagation();
                        handleDeleteThread(item);
                      }}
                      title="Delete"
                      className="absolute right-1 top-1/2 -translate-y-1/2 flex size-5 items-center justify-center rounded-md text-muted-foreground opacity-0 transition-all hover:bg-destructive/10 hover:text-destructive group-hover/recent-item:opacity-100"
                    >
                      <HugeiconsIcon icon={Delete02Icon} className="size-3" />
                    </button>
                  </SidebarMenuItem>
                ))}
              </SidebarMenu>
            </SidebarGroupContent>
            </CollapsibleContent>
          </SidebarGroup>
          </Collapsible>
        )}

        {/* Recent Runs */}
        {runItems.length > 0 && !chatOnly && (
          <Collapsible open={isStudioRoute ? runsOpen : true} onOpenChange={setRunsOpen} asChild>
          <SidebarGroup className="group-data-[collapsible=icon]:hidden overflow-hidden">
            <SidebarGroupLabel asChild>
              {isStudioRoute ? (
                <CollapsibleTrigger className="cursor-pointer flex w-full items-center justify-between">
                  Recent Runs
                  <ChevronDown className="size-3.5 transition-transform duration-200 data-[state=open]:rotate-0 [[data-state=closed]_&]:rotate-[-90deg]" />
                </CollapsibleTrigger>
              ) : (
                <span>Recent Runs</span>
              )}
            </SidebarGroupLabel>
            <CollapsibleContent>
              <SidebarGroupContent className="overflow-y-auto">
                <SidebarMenu>
                  {(isStudioRoute || isMobile ? runItems : runItems.slice(0, 1)).map((run) => {
                    const isActiveRun =
                      selectedHistoryRunId === run.id || activeJobId === run.id;
                    return (
                      <SidebarMenuItem
                        key={run.id}
                        className="group/run-item relative"
                      >
                        <SidebarMenuButton
                          isActive={isActiveRun}
                          className="h-auto flex-col items-start gap-0.5 py-2"
                          onClick={() => {
                            setSelectedHistoryRunId(run.id);
                            if (!isStudioRoute) {
                              navigate({ to: "/studio" });
                            }
                            closeMobileIfOpen();
                          }}
                        >
                          <div className="flex w-full items-center gap-2">
                            <span
                              className={cn(
                                "size-1.5 shrink-0 rounded-full",
                                runStatusDotClass(run.status),
                              )}
                              aria-hidden
                            />
                            <span className="truncate text-sm font-medium">
                              {run.model_name}
                            </span>
                            <span className="ml-auto shrink-0 text-[10px] text-muted-foreground">
                              {formatRelativeShort(run.started_at)}
                            </span>
                          </div>
                          <span className="w-full truncate pl-3.5 text-xs text-muted-foreground">
                            {run.dataset_name}
                          </span>
                        </SidebarMenuButton>
                        <button
                          type="button"
                          onClick={async (e) => {
                            e.stopPropagation();
                            try {
                              await deleteTrainingRun(run.id);
                              if (selectedHistoryRunId === run.id) {
                                setSelectedHistoryRunId(null);
                              }
                              await refreshRuns();
                            } catch {
                              // ignore — next refresh will reconcile
                            }
                          }}
                          title="Delete"
                          className="absolute right-1 top-1/2 -translate-y-1/2 flex size-5 items-center justify-center rounded-md text-muted-foreground opacity-0 transition-all hover:bg-destructive/10 hover:text-destructive group-hover/run-item:opacity-100"
                        >
                          <HugeiconsIcon icon={Delete02Icon} className="size-3" />
                        </button>
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

      <SidebarFooter>
        {/* Desktop app download — hidden when collapsed */}
        <div className="group-data-[collapsible=icon]:hidden px-4 pb-1">
          <a
            href="https://unsloth.ai/download"
            target="_blank"
            rel="noopener noreferrer"
            className="text-[11px] text-muted-foreground transition-colors hover:text-foreground"
          >
            Desktop app available &rarr;
          </a>
        </div>

        <SidebarMenu>
          <SidebarMenuItem>
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <SidebarMenuButton
                  size="lg"
                  className="data-[state=open]:bg-sidebar-accent data-[state=open]:text-sidebar-accent-foreground"
                >
                  <img
                    src="/Sloth emojis/sloth rounded.png"
                    alt="Unsloth"
                    className="size-8 rounded-lg shrink-0"
                  />
                  <div className="flex flex-col gap-0.5 leading-none group-data-[collapsible=icon]:hidden">
                    <span className="truncate text-sm font-semibold">Unsloth</span>
                    <span className="truncate text-[11px] text-muted-foreground">Studio</span>
                  </div>
                  <ChevronsUpDown className="ml-auto size-4 text-muted-foreground group-data-[collapsible=icon]:hidden" />
                </SidebarMenuButton>
              </DropdownMenuTrigger>
              <DropdownMenuContent
                side="top"
                align="start"
                className="w-56"
              >
                <DropdownMenuGroup>
                  <DropdownMenuItem
                    onSelect={() => useSettingsDialogStore.getState().openDialog()}
                  >
                    <HugeiconsIcon icon={Settings02Icon} className="size-4" />
                    <span>Settings</span>
                    <DropdownMenuShortcut>⌘,</DropdownMenuShortcut>
                  </DropdownMenuItem>
                </DropdownMenuGroup>
                <DropdownMenuSeparator />
                <DropdownMenuGroup>
                  <DropdownMenuItem
                    ref={anchorRef as React.Ref<HTMLDivElement>}
                    onSelect={(e) => { e.preventDefault(); toggleTheme(); }}
                  >
                    {isDark ? <Sun className="size-4" /> : <Moon className="size-4" />}
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
                    <HugeiconsIcon icon={CursorInfo02Icon} className="size-4" />
                    <span>Guided Tour</span>
                  </DropdownMenuItem>
                </DropdownMenuGroup>
                <DropdownMenuSeparator />
                <DropdownMenuGroup>
                  <DropdownMenuItem asChild>
                    <a
                      href="https://unsloth.ai/docs"
                      target="_blank"
                      rel="noopener noreferrer"
                    >
                      <HugeiconsIcon icon={Book03Icon} className="size-4" />
                      <span>Learn More</span>
                    </a>
                  </DropdownMenuItem>
                  <DropdownMenuItem asChild>
                    <a
                      href="https://unsloth.ai/docs/new/changelog"
                      target="_blank"
                      rel="noopener noreferrer"
                    >
                      <HugeiconsIcon
                        icon={NewReleasesIcon}
                        className="size-4"
                      />
                      <span>What's New</span>
                    </a>
                  </DropdownMenuItem>
                </DropdownMenuGroup>
                <DropdownMenuSeparator />
                <DropdownMenuItem asChild>
                  <a
                    href="https://github.com/unslothai/unsloth/issues"
                    target="_blank"
                    rel="noopener noreferrer"
                  >
                    <HugeiconsIcon
                      icon={MessageSearch01Icon}
                      className="size-4"
                    />
                    <span>Feedback</span>
                  </a>
                </DropdownMenuItem>
              </DropdownMenuContent>
            </DropdownMenu>
          </SidebarMenuItem>
        </SidebarMenu>
      </SidebarFooter>
    </Sidebar>
  );
}
