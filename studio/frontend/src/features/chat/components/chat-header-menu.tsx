// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuShortcut,
  DropdownMenuSub,
  DropdownMenuSubContent,
  DropdownMenuSubTrigger,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { useChatFavoritesStore } from "@/features/library/chats/favorites-store";
import { resumesThought } from "@/features/model-picker";
import { useShortcutLabel } from "@/features/settings";
import type { ShortcutId } from "@/features/settings";
import { useT } from "@/i18n";
import { StarPointedIcon } from "@/lib/hugeicons-derived";
import { isDownloadCancelled } from "@/lib/native-files";
import { toast } from "@/lib/toast";
import { cn } from "@/lib/utils";
import {
  Archive03Icon,
  BubbleChatTemporaryIcon,
  Cancel01Icon,
  CollapseIcon,
  Delete02Icon,
  Download01Icon,
  Edit03Icon,
  Folder02Icon,
  FolderExportIcon,
  LayerIcon,
  MoreHorizontalIcon,
  PinIcon,
  PinOffIcon,
  PlusSignIcon,
  Refresh01Icon,
  ViewIcon,
  ViewOffSlashIcon,
} from "@hugeicons/core-free-icons";
import { Copy01Icon } from "@/lib/copy-icon";
import { HugeiconsIcon, type IconSvgElement } from "@hugeicons/react";
import { ForkIcon } from "@/lib/fork-icon";
import { type ReactNode, useSyncExternalStore } from "react";
import { useChatRuntimeStore } from "../stores/chat-runtime-store";
import {
  type ActiveChatMenu,
  useActiveChatMenuStore,
} from "../stores/active-chat-menu-store";
import { requestThreadCompaction } from "../stores/compact-request-store";
import {
  liveThreadBranch,
  liveThreadRunning,
  subscribeLiveThreadViews,
} from "../utils/live-thread-head";
import {
  hasPreStreamRunReservation,
  subscribePreStreamRunReservations,
} from "../utils/pre-stream-run-reservation";
import {
  type ConversationExportFormat,
  chatExportOptions,
  exportConversationByFormat,
  getSidebarItemThreadIds,
  regenerateChatTitleWithToast,
} from "./chat-row-menu";
import { OpenChatFolderItem, UploadToChatFolderItem } from "./open-chat-folder-item";

const MENU = "library-actions-menu";
const ICON = "size-icon";
const LABEL = "px-3 pb-1 pt-2 font-normal text-muted-foreground";
/** The header's "…" buttons. Open reads aria-expanded: the trigger's tooltip overwrites data-state. */
export const CHAT_MENU_TRIGGER =
  "flex size-[calc(30px*var(--ui-space-scale,1))] cursor-pointer items-center justify-center rounded-[10px] text-nav-fg transition-colors hover:bg-nav-surface-hover hover:text-black focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring aria-expanded:bg-nav-surface-hover aria-expanded:text-black dark:hover:text-white dark:aria-expanded:text-white";
export const CHAT_MENU = MENU;
const MOVE_TO_LIST =
  "no-scrollbar -my-0.5 max-h-[calc(260px*var(--ui-space-scale,1))] overflow-y-auto overscroll-contain";

function Shortcut({ id }: { id: ShortcutId }) {
  const label = useShortcutLabel(id);
  return label ? <DropdownMenuShortcut>{label}</DropdownMenuShortcut> : null;
}

function Item({
  icon,
  glyph,
  children,
  onSelect,
  shortcut,
  destructive,
  disabled,
}: {
  icon?: IconSvgElement;
  glyph?: ReactNode;
  children: ReactNode;
  onSelect: () => void;
  shortcut?: ShortcutId;
  destructive?: boolean;
  disabled?: boolean;
}) {
  return (
    <DropdownMenuItem
      onSelect={onSelect}
      disabled={disabled}
      variant={destructive ? "destructive" : "default"}
    >
      {icon ? (
        <HugeiconsIcon icon={icon} strokeWidth={1.75} className={ICON} />
      ) : (
        glyph
      )}
      <span className="truncate">{children}</span>
      {shortcut ? <Shortcut id={shortcut} /> : null}
    </DropdownMenuItem>
  );
}

function TemporaryChatButton({
  temporary,
  onToggle,
}: {
  temporary: boolean;
  onToggle: () => void;
}) {
  const label = temporary
    ? "Turn off temporary chat"
    : "Turn on temporary chat";
  return (
    <Tooltip>
      <TooltipTrigger asChild={true}>
        <button
          type="button"
          onClick={onToggle}
          className={cn(
            "flex size-[calc(30px*var(--ui-space-scale,1))] cursor-pointer items-center justify-center rounded-[10px] transition-colors focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring",
            temporary
              ? "bg-wash-10 text-foreground hover:bg-wash-15"
              : "text-nav-fg hover:bg-nav-surface-hover hover:text-black dark:hover:text-white",
          )}
          aria-label={label}
          aria-pressed={temporary}
        >
          <HugeiconsIcon
            icon={BubbleChatTemporaryIcon}
            strokeWidth={1.75}
            className="size-icon"
          />
        </button>
      </TooltipTrigger>
      <TooltipContent side="bottom" sideOffset={6} className="tooltip-compact">
        {label}
      </TooltipContent>
    </Tooltip>
  );
}

function ChatMenuItems({ menu }: { menu: ActiveChatMenu }) {
  const t = useT();
  const preStreamRunActive = useSyncExternalStore(
    subscribePreStreamRunReservations,
    () => hasPreStreamRunReservation([menu.item.id]),
    () => false,
  );
  const liveRunActive = useSyncExternalStore(
    subscribeLiveThreadViews,
    () => liveThreadRunning(menu.item.id),
    () => false,
  );
  const busy = useChatRuntimeStore((state) =>
    state.modelLoading || Boolean(state.runningByThreadId[menu.item.id]),
  ) || preStreamRunActive || liveRunActive;
  const compact = () => {
    const runtime = useChatRuntimeStore.getState();
    if (
      menu.item.type !== "single" ||
      runtime.modelLoading ||
      runtime.runningByThreadId[menu.item.id] ||
      hasPreStreamRunReservation([menu.item.id]) ||
      liveThreadRunning(menu.item.id)
    ) return;
    if (!liveThreadBranch(menu.item.id)?.length) {
      toast.info(t("chatMenu.compactEmpty"));
      return;
    }
    if (!runtime.supportsTools || !resumesThought({
      loadedIsGguf: runtime.loadedIsGguf,
      loadedIsMlx: runtime.loadedIsMlx,
      activeGgufVariant: runtime.activeGgufVariant,
      activeNativePathToken: runtime.activeNativePathToken,
      checkpoint: runtime.params.checkpoint,
    })) {
      toast.info(t("chatMenu.compactUnsupported"));
      return;
    }
    requestThreadCompaction(menu.item.id);
    toast.info(t("chatMenu.compactQueued"));
  };
  const favorite = useChatFavoritesStore((state) =>
    state.chatIds.includes(menu.item.id),
  );
  const generating = useChatRuntimeStore((s) =>
    getSidebarItemThreadIds(menu.item).some((id) => Boolean(s.runningByThreadId[id])),
  );
  // As the Library exports one: each thread behind the chat, a comparison's two included.
  const exportAs = async (format: ConversationExportFormat) => {
    try {
      for (const id of getSidebarItemThreadIds(menu.item)) {
        await exportConversationByFormat(id, format);
      }
    } catch (error) {
      if (!isDownloadCancelled(error)) {
        toast.error(t("settings.data.exportFailed"));
      }
    }
  };
  return (
    <>
      <Item icon={Edit03Icon} onSelect={menu.rename} shortcut="renameChat">
        {t("common.rename")}
      </Item>
      <Item
        icon={Refresh01Icon}
        onSelect={() => void regenerateChatTitleWithToast(menu.item)}
        disabled={generating}
      >
        {t("library.menu.regenerateTitle")}
      </Item>
      <Item
        icon={menu.pinned ? PinOffIcon : PinIcon}
        onSelect={menu.togglePin}
        shortcut="togglePinChat"
      >
        {t(
          menu.pinned
            ? "settings.data.library.unpin"
            : "settings.data.library.pin",
        )}
      </Item>
      <Item
        glyph={
          <HugeiconsIcon
            icon={StarPointedIcon}
            strokeWidth={1.75}
            className={cn(ICON, favorite && "[&_path]:fill-current")}
          />
        }
        onSelect={() =>
          useChatFavoritesStore.getState().setChats([menu.item.id], !favorite)
        }
      >
        {t(
          favorite
            ? "library.menu.removeFromFavorites"
            : "library.menu.addToFavorites",
        )}
      </Item>
      <Item
        icon={menu.unread ? ViewIcon : ViewOffSlashIcon}
        onSelect={menu.toggleUnread}
        shortcut="markChatUnread"
      >
        {t(
          menu.unread
            ? "shell.selection.markRead"
            : "shell.selection.markUnread",
        )}
      </Item>
      <DropdownMenuSeparator className="mx-3" />
      <Item
        icon={CollapseIcon}
        onSelect={compact}
        disabled={menu.item.type !== "single" || busy}
      >
        {t("chatMenu.compact")}
      </Item>
      <Item
        glyph={<HugeiconsIcon icon={ForkIcon} strokeWidth={1.75} className={ICON} />}
        onSelect={menu.fork}
        disabled={!menu.canFork}
        shortcut="forkChat"
      >
        {t("library.chats.menu.fork")}
      </Item>
      <DropdownMenuSub>
        <DropdownMenuSubTrigger className="gap-2.5">
          <HugeiconsIcon
            icon={FolderExportIcon}
            strokeWidth={1.75}
            className={ICON}
          />
          {t("shell.sections.moveTo")}
        </DropdownMenuSubTrigger>
        <DropdownMenuSubContent
          className={cn(
            MENU,
            "max-h-[var(--radix-dropdown-menu-content-available-height)] w-56 overflow-y-auto",
          )}
        >
          <DropdownMenuLabel className={LABEL}>
            {t("shell.navigation.projects")}
          </DropdownMenuLabel>
          <Item icon={PlusSignIcon} onSelect={menu.newProject}>
            {t("library.chats.toolbar.newProject")}
          </Item>
          {menu.projects.length > 0 && (
            <div className={MOVE_TO_LIST}>
              {menu.projects.map((entry) => (
                <Item
                  key={entry.id}
                  icon={Folder02Icon}
                  onSelect={() => menu.moveToProject(entry.id)}
                >
                  {entry.name}
                </Item>
              ))}
            </div>
          )}
          {menu.project && (
            <Item icon={Cancel01Icon} onSelect={() => menu.moveToProject(null)}>
              {menu.project.name
                ? t("shell.sections.removeFrom", { name: menu.project.name })
                : t("shell.sections.removeFromProject")}
            </Item>
          )}
          <DropdownMenuSeparator className="mx-3" />
          <DropdownMenuLabel className={LABEL}>
            {t("shell.sections.sectionsHeading")}
          </DropdownMenuLabel>
          <Item icon={PlusSignIcon} onSelect={menu.newSection}>
            {t("shell.sections.newSection")}
          </Item>
          {menu.sections.length > 0 && (
            <div className={MOVE_TO_LIST}>
              {menu.sections.map((entry) => (
                <Item
                  key={entry.id}
                  icon={LayerIcon}
                  onSelect={() => menu.moveToSection(entry.id)}
                >
                  {entry.name}
                </Item>
              ))}
            </div>
          )}
          {menu.section && (
            <Item icon={Cancel01Icon} onSelect={() => menu.moveToSection(null)}>
              {t("shell.sections.removeFrom", { name: menu.section.name })}
            </Item>
          )}
        </DropdownMenuSubContent>
      </DropdownMenuSub>
      <DropdownMenuSub>
        <DropdownMenuSubTrigger className="gap-2.5">
          <HugeiconsIcon
            icon={Copy01Icon}
            strokeWidth={1.75}
            className={ICON}
          />
          {t("chatMenu.copy")}
        </DropdownMenuSubTrigger>
        <DropdownMenuSubContent className={cn(MENU, "w-56")}>
          <DropdownMenuItem onSelect={menu.copyMarkdown}>
            {t("settings.keyboardShortcuts.actions.copyChatAsMarkdown.label")}
            <Shortcut id="copyChatAsMarkdown" />
          </DropdownMenuItem>
          <DropdownMenuItem onSelect={menu.copySessionId}>
            {t("settings.keyboardShortcuts.actions.copySessionId.label")}
            <Shortcut id="copySessionId" />
          </DropdownMenuItem>
        </DropdownMenuSubContent>
      </DropdownMenuSub>
      <DropdownMenuSub>
        <DropdownMenuSubTrigger className="gap-2.5">
          <HugeiconsIcon
            icon={Download01Icon}
            strokeWidth={1.75}
            className={ICON}
          />
          {t("common.export")}
        </DropdownMenuSubTrigger>
        <DropdownMenuSubContent className={cn(MENU, "w-48")}>
          {chatExportOptions().map(({ label: name, format }) => (
            <DropdownMenuItem
              key={format}
              onSelect={() => void exportAs(format)}
            >
              {name}
            </DropdownMenuItem>
          ))}
        </DropdownMenuSubContent>
      </DropdownMenuSub>
      <OpenChatFolderItem item={menu.item} />
      <UploadToChatFolderItem item={menu.item} />
      <DropdownMenuSeparator className="mx-3" />
      <Item icon={Archive03Icon} onSelect={menu.archive} shortcut="archiveChat">
        {t("settings.data.library.archive")}
      </Item>
      <Item icon={Delete02Icon} onSelect={menu.remove} destructive={true}>
        {t("common.delete")}
      </Item>
    </>
  );
}

/** A saved chat's menu, or the temporary toggle on a new chat. Saved chats cannot turn temporary. */
export function ChatHeaderMenu({
  temporary,
  onToggleTemporary,
}: {
  temporary: boolean;
  onToggleTemporary: () => void;
}) {
  const t = useT();
  const menu = useActiveChatMenuStore((state) => state.menu);
  const label = t("chatMenu.more");
  if (temporary || !menu) {
    return (
      <TemporaryChatButton temporary={temporary} onToggle={onToggleTemporary} />
    );
  }
  return (
    <DropdownMenu>
      <Tooltip>
        <TooltipTrigger asChild={true}>
          <DropdownMenuTrigger asChild={true}>
            <button
              type="button"
              aria-label={label}
              className={CHAT_MENU_TRIGGER}
            >
              <HugeiconsIcon
                icon={MoreHorizontalIcon}
                strokeWidth={1.75}
                className="size-icon"
              />
            </button>
          </DropdownMenuTrigger>
        </TooltipTrigger>
        <TooltipContent
          side="bottom"
          sideOffset={6}
          className="tooltip-compact"
        >
          {label}
        </TooltipContent>
      </Tooltip>
      <DropdownMenuContent
        align="end"
        sideOffset={6}
        className={cn(MENU, "w-64")}
      >
        <ChatMenuItems menu={menu} />
      </DropdownMenuContent>
    </DropdownMenu>
  );
}
