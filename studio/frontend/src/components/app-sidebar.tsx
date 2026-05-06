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
  ChefHatIcon,
  ColumnInsertIcon,
  CursorInfo02Icon,
  Delete02Icon,
  Download03Icon,
  GemIcon,
  Globe02Icon,
  HelpCircleIcon,
  Search01Icon,
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
          className="h-[32px] rounded-[10px] gap-[8.5px] px-2.5 font-medium text-[#383835] dark:text-[#c7c7c4] hover:bg-[#f0f0f0]! dark:hover:bg-[#2a2c2f]! hover:text-black! dark:hover:text-white! data-active:bg-[#f0f0f0]! dark:data-active:bg-[#2a2c2f]! data-active:text-black! dark:data-active:text-white! group-data-[collapsible=icon]:!w-[32px] group-data-[collapsible=icon]:!rounded-[11px] group-data-[collapsible=icon]:mx-auto"
        >
          <HugeiconsIcon icon={icon} strokeWidth={1.75} className="size-[18px]! shrink-0 group-hover/menu-button:animate-icon-pop" />
          <span className="text-[14px] leading-[18px] tracking-[0.01em]">{label}</span>
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
                  className="inline-flex h-7 w-7 items-center justify-center rounded-[10px] text-[#8f8f8f] dark:text-[#5c5c5c] transition-colors hover:bg-[#f0f0f0] dark:hover:bg-[#2a2c2f] hover:text-black dark:hover:text-white focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
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
                  className="inline-flex h-7 w-7 items-center justify-center rounded-[10px] text-[#383835] dark:text-[#c7c7c4] transition-colors hover:bg-[#f0f0f0] dark:hover:bg-[#2a2c2f] hover:text-black dark:hover:text-white focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
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

      <SidebarGroup className="group-data-[collapsible=icon]:px-0 px-2 pt-[10px] pb-[14px] shrink-0">
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
                navigate({ to: "/chat", search: { new: createNavigationNonce() } });
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
                navigate({ to: "/chat", search: { compare: createNavigationNonce() } });
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
        <SidebarGroup data-tour="navbar" className="group-data-[collapsible=icon]:px-0 px-2 pt-[10px] pb-[14px]">
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
                      className="h-[32px] rounded-[10px] pl-2.5 pr-7 text-[14px] leading-[18px] tracking-[0.01em] font-medium text-[#383835] dark:text-[#c7c7c4] hover:bg-[#f0f0f0]! dark:hover:bg-[#2a2c2f]! hover:text-black! dark:hover:text-white! data-active:bg-[#f0f0f0]! dark:data-active:bg-[#2a2c2f]! data-active:text-black! dark:data-active:text-white!"
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
                      className="absolute right-1 top-1/2 -translate-y-1/2 flex size-5 scale-90 items-center justify-center rounded-[10px] text-sidebar-foreground/55 opacity-0 transition-all duration-150 hover:bg-destructive/12 hover:text-destructive group-hover/recent-item:scale-100 group-hover/recent-item:opacity-100"
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
                          className="h-auto flex-col items-start gap-0.5 py-1.5 rounded-[10px] pl-2.5 pr-7 text-[14px] tracking-[0.01em] font-medium text-[#383835] dark:text-[#c7c7c4] hover:bg-[#f0f0f0]! dark:hover:bg-[#2a2c2f]! hover:text-black! dark:hover:text-white! data-active:bg-[#f0f0f0]! dark:data-active:bg-[#2a2c2f]! data-active:text-black! dark:data-active:text-white!"
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
                          className="absolute right-1 top-1/2 -translate-y-1/2 flex size-5 scale-90 items-center justify-center rounded-[10px] text-sidebar-foreground/55 opacity-0 transition-all duration-150 hover:bg-destructive/12 hover:text-destructive group-hover/run-item:scale-100 group-hover/run-item:opacity-100"
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

      <SidebarFooter className="border-t border-sidebar-border group-data-[collapsible=icon]:border-transparent group-data-[collapsible=icon]:px-0">
        <SidebarMenu>
          <SidebarMenuItem>
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <SidebarMenuButton
                  size="lg"
                  aria-label={`${displayTitle} account menu`}
                  className="!h-[50px] gap-[8px] rounded-[10px] text-[#383835] dark:text-[#c7c7c4] hover:bg-[#f0f0f0]! dark:hover:bg-[#2a2c2f]! hover:text-black! dark:hover:text-white! data-[state=open]:bg-[#f0f0f0]! dark:data-[state=open]:bg-[#2a2c2f]! data-[state=open]:text-black! dark:data-[state=open]:text-white!"
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
                    <span className="truncate font-heading text-[13px] tracking-[0.02em] font-semibold text-[#383835] dark:text-[#c7c7c4]">{displayTitle}</span>
                    <span className="truncate text-[11px] tracking-[0.01em] text-muted-foreground">Unsloth</span>
                  </div>
                  <ChevronsUpDown strokeWidth={1.25} className="ml-auto size-4 text-muted-foreground group-data-[collapsible=icon]:hidden" />
                </SidebarMenuButton>
              </DropdownMenuTrigger>
              <DropdownMenuContent
                side="top"
                align="start"
                className="w-[15rem] py-2.5 font-heading [&_[data-slot=dropdown-menu-group]]:flex [&_[data-slot=dropdown-menu-group]]:flex-col [&_[data-slot=dropdown-menu-group]]:gap-px [&_[data-slot=dropdown-menu-item]]:h-[32px] [&_[data-slot=dropdown-menu-item]]:px-2.5! [&_[data-slot=dropdown-menu-item]]:py-0! [&_[data-slot=dropdown-menu-item]]:gap-[8.5px]! [&_[data-slot=dropdown-menu-item]]:rounded-[10px] [&_[data-slot=dropdown-menu-item]]:font-medium [&_[data-slot=dropdown-menu-item]]:text-[14px] [&_[data-slot=dropdown-menu-item]]:leading-[18px] [&_[data-slot=dropdown-menu-item]]:tracking-[0.01em] [&_[data-slot=dropdown-menu-item]]:text-[#383835] dark:[&_[data-slot=dropdown-menu-item]]:text-[#c7c7c4] [&_[data-slot=dropdown-menu-item]_svg]:!size-[18px] [&_[data-slot=dropdown-menu-item]_svg]:shrink-0 [&_[data-slot=dropdown-menu-item]:focus]:bg-[#f0f0f0] dark:[&_[data-slot=dropdown-menu-item]:focus]:bg-[#2a2c2f] [&_[data-slot=dropdown-menu-item]:focus]:text-black dark:[&_[data-slot=dropdown-menu-item]:focus]:text-white [&_[data-slot=dropdown-menu-item]:focus_*]:text-black! dark:[&_[data-slot=dropdown-menu-item]:focus_*]:text-white!"
              >
                <DropdownMenuGroup>
                  <DropdownMenuItem
                    onSelect={() => useSettingsDialogStore.getState().openDialog()}
                  >
                    <HugeiconsIcon icon={Settings02Icon} strokeWidth={1.75} className="size-[18px]" />
                    <span>Settings</span>
                    <DropdownMenuShortcut>⌘,</DropdownMenuShortcut>
                  </DropdownMenuItem>
                  <DropdownMenuItem
                    onSelect={() => useSettingsDialogStore.getState().openDialog("api-keys")}
                  >
                    <HugeiconsIcon icon={Globe02Icon} strokeWidth={1.75} className="size-[18px]" />
                    <span>API</span>
                    <span className="ml-auto rounded-[6px] border border-emerald-500/25 bg-emerald-500/10 px-1.5 py-0.5 text-[10px] leading-none font-semibold text-emerald-700 dark:text-emerald-300">
                      New
                    </span>
                  </DropdownMenuItem>
                  <DropdownMenuItem
                    ref={anchorRef as React.Ref<HTMLDivElement>}
                    onSelect={(e) => { e.preventDefault(); toggleTheme(); }}
                  >
                    {isDark ? <Sun strokeWidth={1.75} className="size-[18px]" /> : <Moon strokeWidth={1.75} className="size-[18px]" />}
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
                    <HugeiconsIcon icon={CursorInfo02Icon} strokeWidth={1.75} className="size-[18px]" />
                    <span>Guided Tour</span>
                  </DropdownMenuItem>
                </DropdownMenuGroup>
                <DropdownMenuSeparator className="mx-2.5! my-2.5! h-0! border-t border-border/70 bg-transparent!" />
                <DropdownMenuItem
                  onSelect={() => useSettingsDialogStore.getState().openDialog("about")}
                >
                  <HugeiconsIcon icon={HelpCircleIcon} strokeWidth={1.75} className="size-[18px]" />
                  <span>Help</span>
                </DropdownMenuItem>
                <DropdownMenuItem onSelect={() => setShutdownOpen(true)}>
                  <HugeiconsIcon icon={PowerIcon} strokeWidth={1.75} className="size-[18px]" />
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
