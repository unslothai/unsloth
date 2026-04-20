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
  ColumnInsertIcon,
  CursorInfo02Icon,
  Delete02Icon,
  Download03Icon,
  GemIcon,
  MessageSearch01Icon,
  Search01Icon,
  NewReleasesIcon,
  PowerIcon,
  PencilEdit02Icon,
  LayoutAlignLeftIcon,
  Settings02Icon,
  ZapIcon,
} from "@hugeicons/core-free-icons";
import {
  Tooltip,
  TooltipContent,
} from "@/components/ui/tooltip";
import { Tooltip as TooltipPrimitive } from "radix-ui";
import { HugeiconsIcon } from "@hugeicons/react";
import { ChevronDown, ChevronsUpDown, Moon, Sun } from "lucide-react";
import { Link, useNavigate, useRouterState } from "@tanstack/react-router";
import { useTrainingRuntimeStore } from "@/features/training";
import { useSettingsDialogStore } from "@/features/settings";
import { useEffectiveProfile, UserAvatar } from "@/features/profile";
import { usePlatformStore } from "@/config/env";
import { TOUR_OPEN_EVENT } from "@/features/tour";
import {
  useChatSidebarItems,
  deleteChatItem,
} from "@/features/chat/hooks/use-chat-sidebar-items";
import { useChatRuntimeStore } from "@/features/chat/stores/chat-runtime-store";
import { useChatSearchStore } from "@/features/chat/stores/chat-search-store";
import { ChatSearchDialog } from "@/features/chat/components/chat-search-dialog";
import { useTrainingHistorySidebarItems, deleteTrainingRun } from "@/features/training";
import type { TrainingRunSummary } from "@/features/training";
import { useEffect, useState } from "react";
import { ShutdownDialog } from "@/components/shutdown-dialog";
import { removeTrainingUnloadGuard } from "@/features/training/hooks/use-training-unload-guard";

function getTourId(pathname: string): string | null {
  if (pathname.startsWith("/studio")) return "studio";
  if (pathname.startsWith("/export")) return "export";
  if (pathname.startsWith("/chat")) return "chat";
  return null;
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

function NavItem({
  icon,
  label,
  active,
  disabled,
  onClick,
  children,
  variant = "nav",
  dataTour,
}: {
  icon: typeof ZapIcon;
  label: string;
  active: boolean;
  disabled?: boolean;
  onClick: () => void;
  children?: React.ReactNode;
  variant?: "nav" | "menu";
  dataTour?: string;
}) {
  const isNav = variant === "nav";
  return (
    <SidebarMenuItem>
      <div className="relative">
        <SidebarMenuButton
          tooltip={label}
          disabled={disabled}
          onClick={onClick}
          isActive={active}
          data-tour={dataTour}
          className={
            isNav
              ? "h-[30px] rounded-[8px] gap-2.5 px-2.5 font-medium text-[#383835] dark:text-[#c7c7c4] hover:bg-[#ececec]! dark:hover:bg-[#2e3035]! hover:text-black! dark:hover:text-white! data-active:bg-[#ececec]! dark:data-active:bg-[#2e3035]! data-active:text-black! dark:data-active:text-white! group-data-[collapsible=icon]:!w-[30px] group-data-[collapsible=icon]:!rounded-[9px] group-data-[collapsible=icon]:mx-auto"
              : "h-[30px] rounded-[8px] gap-2.5 px-2.5 font-medium text-[#383835] dark:text-[#c7c7c4] hover:bg-[#ececec]! dark:hover:bg-[#2e3035]! hover:text-black! dark:hover:text-white! data-active:bg-[#ececec]! dark:data-active:bg-[#2e3035]! data-active:text-black! dark:data-active:text-white! group-data-[collapsible=icon]:!w-[30px] group-data-[collapsible=icon]:!rounded-[9px] group-data-[collapsible=icon]:mx-auto"
          }
        >
          <HugeiconsIcon icon={icon} strokeWidth={1.5} className="size-[18px]!" />
          <span className="text-sm">{label}</span>
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
  const isStudioRoute = pathname === "/studio" || pathname.startsWith("/studio/");
  const [chatOpen, setChatOpen] = useState(true);
  const [runsOpen, setRunsOpen] = useState(true);

  useEffect(() => { if (isChatRoute) setChatOpen(true); }, [isChatRoute]);
  useEffect(() => { if (isStudioRoute) setRunsOpen(true); }, [isStudioRoute]);

  const isRecipesRoute = pathname.startsWith("/data-recipes");
  const { displayTitle, avatarDataUrl } = useEffectiveProfile();

  const { items: chatItems } = useChatSidebarItems();
  const storeThreadId = useChatRuntimeStore((s) => s.activeThreadId);
  const setActiveThreadId = useChatRuntimeStore((s) => s.setActiveThreadId);
  const activeThreadId = isChatRoute
    ? (search.thread as string | undefined) ??
      (search.compare as string | undefined) ??
      storeThreadId ??
      undefined
    : undefined;

  // Training runs
  const { items: runItems, refresh: refreshRuns } = useTrainingHistorySidebarItems(
    !chatOnly && isStudioRoute,
  );
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
    <>
    <Sidebar
      collapsible="icon"
      variant="sidebar"
      className="font-heading group-data-[collapsible=icon]:[&_[data-sidebar=sidebar]]:bg-white dark:group-data-[collapsible=icon]:[&_[data-sidebar=sidebar]]:bg-background"
    >
      <SidebarHeader className="pl-[17px] pr-3 pt-[12px] pb-[12px] group-data-[collapsible=icon]:px-0">
        {/* Expanded: compact logo + close toggle */}
        <div className="flex items-center justify-between gap-2 group-data-[collapsible=icon]:hidden">
          <Link
            to="/chat"
            search={{ new: crypto.randomUUID() }}
            onClick={() => {
              setActiveThreadId(null);
              closeMobileIfOpen();
            }}
            className="flex items-center gap-[6px] select-none"
            aria-label="Unsloth home"
          >
            <img
              src="/circle-logo-small.png"
              alt="Unsloth"
              className="h-[34px] w-[34px] rounded-full object-cover"
            />
            <span className="font-heading text-[21px] font-semibold tracking-[-0.01em] dark:tracking-[0.02em] leading-none text-black dark:text-white">
              unsloth
            </span>
            <span
              style={{ fontFamily: '"Inter Variable", ui-sans-serif, system-ui, sans-serif' }}
              className="ml-0.5 inline-flex items-center justify-center rounded-full border border-[#e0ded6] px-[5px] py-[2px] text-[8px] font-medium leading-none tracking-[0.04em] text-[#62605a] antialiased subpixel-antialiased shadow-[0_1px_2px_rgba(0,0,0,0.06)] dark:border-[#3a3c3f] dark:text-[#9d9fa5] dark:shadow-[0_1px_2px_rgba(0,0,0,0.35)]"
            >
              BETA
            </span>
          </Link>
          {!isMobile && (
            <Tooltip>
              <TooltipPrimitive.Trigger asChild>
                <button
                  type="button"
                  onClick={togglePinned}
                  className="inline-flex h-7 w-7 items-center justify-center rounded-[8px] text-[#8f8f8f] dark:text-[#5c5c5c] transition-colors hover:bg-[#ececec] dark:hover:bg-[#2e3035] hover:text-black dark:hover:text-white focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
                  aria-label="Close sidebar"
                >
                  <HugeiconsIcon icon={LayoutAlignLeftIcon} strokeWidth={1.75} className="size-[18px]" />
                </button>
              </TooltipPrimitive.Trigger>
              <TooltipContent side="bottom" sideOffset={6}>
                Close sidebar
              </TooltipContent>
            </Tooltip>
          )}
        </div>

        {/* Collapsed: panel icon doubles as expand trigger */}
        {!isMobile && (
          <div className="hidden group-data-[collapsible=icon]:flex h-[34px] items-center justify-center w-full">
            <Tooltip>
              <TooltipPrimitive.Trigger asChild>
                <button
                  type="button"
                  onClick={togglePinned}
                  className="inline-flex h-7 w-7 items-center justify-center rounded-[8px] text-[#383835] dark:text-[#c7c7c4] transition-colors hover:bg-[#ececec] dark:hover:bg-[#2e3035] hover:text-black dark:hover:text-white focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
                  aria-label="Open sidebar"
                >
                  <HugeiconsIcon icon={LayoutAlignLeftIcon} strokeWidth={1.75} className="size-[18px]" />
                </button>
              </TooltipPrimitive.Trigger>
              <TooltipContent side="right" sideOffset={8}>
                Open sidebar
              </TooltipContent>
            </Tooltip>
          </div>
        )}
      </SidebarHeader>

      <SidebarGroup className="group-data-[collapsible=icon]:px-0 px-2 pt-[8px] pb-[12px] shrink-0">
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
                navigate({ to: "/chat", search: { new: crypto.randomUUID() } });
                closeMobileIfOpen();
              }}
            />
            <NavItem
              icon={ColumnInsertIcon}
              label="Compare"
              active={!!search.compare && !chatItems.some((i) => i.id === search.compare)}
              disabled={chatDisabled}
              dataTour="chat-compare"
              onClick={() => {
                if (chatDisabled) return;
                setActiveThreadId(null);
                navigate({ to: "/chat", search: { compare: crypto.randomUUID() } });
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
          </SidebarMenu>
        </SidebarGroupContent>
      </SidebarGroup>

      <SidebarContent className="gap-0 overflow-y-auto overscroll-contain min-h-0">
        {/* Navigate (no header) */}
        <SidebarGroup data-tour="navbar" className="group-data-[collapsible=icon]:px-0 px-2 pt-[8px] pb-[12px]">
          <SidebarGroupContent>
            <SidebarMenu>
              <NavItem
                icon={GemIcon}
                label="Train"
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
                icon={Download03Icon}
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

        {/* Recent Chats — hide on Studio only (Eyera fac13); chatOpen = ec695 clickability */}
        {!isStudioRoute && chatItems.length > 0 && (
          <Collapsible open={chatOpen} onOpenChange={setChatOpen} asChild>
          <SidebarGroup className="group-data-[collapsible=icon]:hidden overflow-hidden px-2 py-0">
            <SidebarGroupLabel className="pt-2 pb-1.5 pl-2.5 pr-2 text-[12.5px]! font-normal normal-case tracking-normal text-[#62605a] dark:text-[#9d9fa5] focus-visible:ring-0! focus-visible:outline-none" asChild>
              <CollapsibleTrigger className="cursor-pointer flex w-full items-center justify-between">
                Recents
                <ChevronDown className="size-3.5 transition-transform duration-200 data-[state=open]:rotate-0 [[data-state=closed]_&]:rotate-[-90deg]" />
              </CollapsibleTrigger>
            </SidebarGroupLabel>
            <CollapsibleContent>
            <SidebarGroupContent>
              <SidebarMenu>
                {chatItems.map((item) => (
                  <SidebarMenuItem key={item.id} className="group/recent-item relative">
                    <SidebarMenuButton
                      isActive={activeThreadId === item.id}
                      className="h-[30px] rounded-[8px] pl-2.5 pr-7 text-sm font-medium text-[#383835] dark:text-[#c7c7c4] hover:bg-[#ececec]! dark:hover:bg-[#2e3035]! hover:text-black! dark:hover:text-white! data-active:bg-[#ececec]! dark:data-active:bg-[#2e3035]! data-active:text-black! dark:data-active:text-white!"
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
                      className="absolute right-1 top-1/2 -translate-y-1/2 flex size-5 scale-90 items-center justify-center rounded-[8px] text-sidebar-foreground/55 opacity-0 transition-all duration-150 hover:bg-destructive/12 hover:text-destructive group-hover/recent-item:scale-100 group-hover/recent-item:opacity-100"
                    >
                      <HugeiconsIcon icon={Delete02Icon} strokeWidth={2} className="size-3.5" />
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
        {isStudioRoute && runItems.length > 0 && !chatOnly && (
          <Collapsible open={runsOpen} onOpenChange={setRunsOpen} asChild>
          <SidebarGroup className="group-data-[collapsible=icon]:hidden overflow-hidden px-2 py-0">
            <SidebarGroupLabel className="pt-2 pb-1.5 pl-2.5 pr-2 text-[12.5px]! font-normal normal-case tracking-normal text-[#62605a] dark:text-[#9d9fa5] focus-visible:ring-0! focus-visible:outline-none" asChild>
              <CollapsibleTrigger className="cursor-pointer flex w-full items-center justify-between">
                Recents
                <ChevronDown className="size-3.5 transition-transform duration-200 data-[state=open]:rotate-0 [[data-state=closed]_&]:rotate-[-90deg]" />
              </CollapsibleTrigger>
            </SidebarGroupLabel>
            <CollapsibleContent>
              <SidebarGroupContent>
                <SidebarMenu>
                  {runItems.map((run) => {
                    const isActiveRun =
                      selectedHistoryRunId === run.id || activeJobId === run.id;
                    return (
                      <SidebarMenuItem
                        key={run.id}
                        className="group/run-item relative"
                      >
                        <SidebarMenuButton
                          isActive={isActiveRun}
                          className="h-auto flex-col items-start gap-0.5 py-1.5 rounded-[8px] pl-2.5 pr-7 text-sm font-medium text-[#383835] dark:text-[#c7c7c4] hover:bg-[#ececec]! dark:hover:bg-[#2e3035]! hover:text-black! dark:hover:text-white! data-active:bg-[#ececec]! dark:data-active:bg-[#2e3035]! data-active:text-black! dark:data-active:text-white!"
                          onClick={() => {
                            setSelectedHistoryRunId(run.id);
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
                            <span className="truncate text-sm">
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
                          className="absolute right-1 top-1/2 -translate-y-1/2 flex size-5 scale-90 items-center justify-center rounded-[8px] text-sidebar-foreground/55 opacity-0 transition-all duration-150 hover:bg-destructive/12 hover:text-destructive group-hover/run-item:scale-100 group-hover/run-item:opacity-100"
                        >
                          <HugeiconsIcon icon={Delete02Icon} strokeWidth={2} className="size-3.5" />
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

      <SidebarFooter className="border-t border-sidebar-border group-data-[collapsible=icon]:border-t-0 group-data-[collapsible=icon]:px-0">
        <SidebarMenu>
          <SidebarMenuItem>
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <SidebarMenuButton
                  size="lg"
                  aria-label={`${displayTitle} account menu`}
                  className="!h-[50px] gap-[8px] rounded-[8px] text-[#383835] dark:text-[#c7c7c4] hover:bg-[#ececec]! dark:hover:bg-[#2e3035]! hover:text-black! dark:hover:text-white! data-[state=open]:bg-[#ececec]! dark:data-[state=open]:bg-[#2e3035]! data-[state=open]:text-black! dark:data-[state=open]:text-white!"
                >
                  <div className="shrink-0">
                    <UserAvatar
                      name={displayTitle}
                      imageUrl={avatarDataUrl}
                      size="sm"
                      className="!size-8"
                    />
                  </div>
                  <div className="flex flex-col gap-0.5 leading-none group-data-[collapsible=icon]:hidden">
                    <span className="truncate font-heading text-[13px] font-semibold text-[#383835] dark:text-[#c7c7c4]">{displayTitle}</span>
                    <span className="truncate text-[11px] text-muted-foreground">Studio</span>
                  </div>
                  <ChevronsUpDown strokeWidth={1.25} className="ml-auto size-4 text-muted-foreground group-data-[collapsible=icon]:hidden" />
                </SidebarMenuButton>
              </DropdownMenuTrigger>
              <DropdownMenuContent
                side="top"
                align="start"
                className="w-[15rem] font-heading [&_[data-slot=dropdown-menu-item]]:rounded-[8px] [&_[data-slot=dropdown-menu-item]]:font-medium [&_[data-slot=dropdown-menu-item]]:text-[#383835] dark:[&_[data-slot=dropdown-menu-item]]:text-[#c7c7c4] [&_[data-slot=dropdown-menu-item]:focus]:bg-[#ececec] dark:[&_[data-slot=dropdown-menu-item]:focus]:bg-[#2e3035] [&_[data-slot=dropdown-menu-item]:focus]:text-black dark:[&_[data-slot=dropdown-menu-item]:focus]:text-white [&_[data-slot=dropdown-menu-item]:focus_*]:text-black! dark:[&_[data-slot=dropdown-menu-item]:focus_*]:text-white!"
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
                </DropdownMenuGroup>
                <DropdownMenuSeparator />
                <DropdownMenuItem onSelect={() => setShutdownOpen(true)}>
                  <HugeiconsIcon icon={PowerIcon} className="size-4" />
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
    </>
  );
}
