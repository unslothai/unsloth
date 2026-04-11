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
  SidebarMenuSub,
  SidebarMenuSubButton,
  SidebarMenuSubItem,
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
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { useAnimatedThemeToggle } from "@/components/ui/animated-theme-toggler";
import { cn } from "@/lib/utils";
import {
  Book03Icon,
  BubbleChatIcon,
  ChefHatIcon,
  ColumnInsertIcon,
  CursorInfo02Icon,
  Delete02Icon,
  ArrowLeft02Icon,
  ArrowRight02Icon,
  MessageSearch01Icon,
  NewReleasesIcon,
  PackageIcon,
  PencilEdit02Icon,
  PinIcon,
  PinOffIcon,
  ZapIcon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { ChevronDown, ChevronsUpDown, Moon, Sun } from "lucide-react";
import { Link, useNavigate, useRouter, useRouterState } from "@tanstack/react-router";
import { motion } from "motion/react";
import { useTrainingRuntimeStore } from "@/features/training";
import { usePlatformStore } from "@/config/env";
import { TOUR_OPEN_EVENT } from "@/features/tour";
import {
  useChatSidebarItems,
  deleteChatItem,
} from "@/features/chat/hooks/use-chat-sidebar-items";
import { useChatRuntimeStore } from "@/features/chat/stores/chat-runtime-store";
import { useRecipeSidebarItems } from "@/features/data-recipes/hooks/use-recipe-sidebar-items";
import { useEffect, useRef, useState } from "react";

function getTourId(pathname: string): string | null {
  if (pathname.startsWith("/studio")) return "studio";
  if (pathname.startsWith("/export")) return "export";
  if (pathname.startsWith("/chat")) return "chat";
  return null;
}

const NAV_SPRING = { type: "spring", stiffness: 500, damping: 35, mass: 0.5 } as const;

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
  const { pinned, togglePinned, setHovered } = useSidebar();
  const router = useRouter();
  const navigate = useNavigate();

  const isTrainingRunning = useTrainingRuntimeStore((s) => s.isTrainingRunning);
  const chatOnly = usePlatformStore((s) => s.isChatOnly());

  // Chat collapsible state — open by default, syncs with route
  const isChatRoute = pathname.startsWith("/chat");
  const [chatOpen, setChatOpen] = useState(true);

  // Recipes collapsible state — syncs with route
  const isRecipesRoute = pathname.startsWith("/data-recipes");
  const [recipesOpen, setRecipesOpen] = useState(isRecipesRoute);

  // Sync collapsibles with route changes
  useEffect(() => {
    if (isRecipesRoute) setRecipesOpen(true);
    else setRecipesOpen(false);
  }, [isRecipesRoute]);

  useEffect(() => {
    if (isChatRoute) setChatOpen(true);
  }, [isChatRoute]);
  const recipeItems = useRecipeSidebarItems(recipesOpen);
  const activeRecipeId = pathname.startsWith("/data-recipes/")
    ? pathname.split("/data-recipes/")[1]
    : undefined;

  const { items: chatItems, canCompare } = useChatSidebarItems();
  const storeThreadId = useChatRuntimeStore((s) => s.activeThreadId);
  const activeThreadId = (search.thread as string | undefined) ?? storeThreadId ?? undefined;

  const chatDisabled = isTrainingRunning;

  // Suppress hover-collapse while a dropdown/popover is open.
  const dropdownOpenRef = useRef(false);

  async function handleDeleteThread(item: Parameters<typeof deleteChatItem>[0]) {
    await deleteChatItem(item, activeThreadId, (view) => {
      navigate({
        to: "/chat",
        search: { new: view.newThreadNonce },
      });
    });
  }

  return (
    <Sidebar
      collapsible="icon"
      variant="sidebar"
      onMouseEnter={() => { if (!pinned) setHovered(true); }}
      onMouseLeave={() => { if (!pinned && !dropdownOpenRef.current) setHovered(false); }}
    >
      <SidebarHeader className="group-data-[collapsible=icon]:px-0">
        <div
          className={cn(
            "flex items-center justify-between",
            "group-data-[collapsible=icon]:hidden",
          )}
        >
          <div className="flex items-center gap-0.5">
            <button
              type="button"
              onClick={() => router.history.back()}
              className="inline-flex h-7 w-7 items-center justify-center rounded-md text-sidebar-foreground/70 transition-colors hover:bg-sidebar-accent hover:text-sidebar-accent-foreground"
              title="Go back"
              aria-label="Go back"
            >
              <HugeiconsIcon icon={ArrowLeft02Icon} className="size-4" />
            </button>
            <button
              type="button"
              onClick={() => router.history.forward()}
              className="inline-flex h-7 w-7 items-center justify-center rounded-md text-sidebar-foreground/70 transition-colors hover:bg-sidebar-accent hover:text-sidebar-accent-foreground"
              title="Go forward"
              aria-label="Go forward"
            >
              <HugeiconsIcon icon={ArrowRight02Icon} className="size-4" />
            </button>
          </div>
          <button
            type="button"
            onClick={togglePinned}
            className="inline-flex h-7 w-7 items-center justify-center rounded-md text-sidebar-foreground/70 transition-colors hover:bg-sidebar-accent hover:text-sidebar-accent-foreground"
            title={pinned ? "Unpin sidebar" : "Pin sidebar"}
            aria-label={pinned ? "Unpin sidebar" : "Pin sidebar"}
          >
            <HugeiconsIcon
              icon={pinned ? PinOffIcon : PinIcon}
              className="size-4"
            />
          </button>
        </div>

        {/* Logo */}
        <SidebarMenu>
          <SidebarMenuItem>
            <SidebarMenuButton
              asChild
              className="data-[slot=sidebar-menu-button]:!p-1.5"
              tooltip="Unsloth"
            >
              <Link to={chatOnly ? "/chat" : "/studio"} className="select-none">
                {/* Collapsed: sticker icon */}
                <img
                  src="/sticker.png"
                  alt="Unsloth"
                  className="!size-7 group-data-[collapsible=icon]:block hidden"
                />
                {/* Expanded: full logo */}
                <img
                  src="/blacklogo.png"
                  alt="Unsloth"
                  className="h-7 w-auto dark:hidden group-data-[collapsible=icon]:hidden"
                />
                <img
                  src="/whitelogo.png"
                  alt="Unsloth"
                  className="hidden h-7 w-auto dark:block group-data-[collapsible=icon]:!hidden"
                />
                <span className="text-[10px] font-extrabold leading-none tracking-[0.12em] text-primary group-data-[collapsible=icon]:hidden">
                  BETA
                </span>
              </Link>
            </SidebarMenuButton>
          </SidebarMenuItem>
        </SidebarMenu>
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
                onClick={() => { if (!chatDisabled) navigate({ to: "/chat", search: { new: crypto.randomUUID() } }); }}
              />

              <NavItem
                icon={ZapIcon}
                label="Studio"
                active={pathname === "/studio" || pathname.startsWith("/studio/")}
                disabled={chatOnly}
                onClick={() => { if (!chatOnly) navigate({ to: "/studio" }); }}
              />

              <NavItem
                icon={ChefHatIcon}
                label="Recipes"
                active={isRecipesRoute}
                onClick={() => navigate({ to: "/data-recipes" })}
              />

              <NavItem
                icon={PackageIcon}
                label="Export"
                active={pathname === "/export" || pathname.startsWith("/export/")}
                disabled={chatOnly}
                onClick={() => { if (!chatOnly) navigate({ to: "/export" }); }}
              />
            </SidebarMenu>
          </SidebarGroupContent>
        </SidebarGroup>

        {/* Recents */}
        {chatItems.length > 0 && (
          <Collapsible open={isChatRoute ? chatOpen : true} onOpenChange={setChatOpen} asChild>
          <SidebarGroup className="group-data-[collapsible=icon]:hidden flex-1 overflow-hidden">
            <SidebarGroupLabel asChild>
              {isChatRoute ? (
                <CollapsibleTrigger className="cursor-pointer flex w-full items-center justify-between">
                  Recents
                  <ChevronDown className="size-3.5 transition-transform duration-200 data-[state=open]:rotate-0 [[data-state=closed]_&]:rotate-[-90deg]" />
                </CollapsibleTrigger>
              ) : (
                <span>Recents</span>
              )}
            </SidebarGroupLabel>
            <CollapsibleContent>
            <SidebarGroupContent className="overflow-y-auto">
              <SidebarMenu>
                {(isChatRoute ? chatItems : chatItems.slice(0, 1)).map((item) => (
                  <SidebarMenuItem key={item.id} className="group/recent-item relative">
                    <SidebarMenuButton
                      isActive={activeThreadId === item.id}
                      className=""
                      onClick={() =>
                        navigate({
                          to: "/chat",
                          search:
                            item.type === "single"
                              ? { thread: item.id }
                              : { compare: item.id },
                        })
                      }
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
            <DropdownMenu onOpenChange={(open) => {
              dropdownOpenRef.current = open;
              if (!open && !pinned) setHovered(false);
            }}>
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
