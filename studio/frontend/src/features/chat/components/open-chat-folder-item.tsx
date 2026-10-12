// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { sandboxSessionIdFor } from "@/components/assistant-ui/sandbox-files";
import { revealSandbox } from "@/components/assistant-ui/sandbox-reveal";
import {
  MAX_SANDBOX_UPLOAD_BYTES,
  uploadToSandbox,
} from "@/components/assistant-ui/sandbox-upload";
import { DropdownMenuItem } from "@/components/ui/dropdown-menu";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { FolderOpenIcon, Upload01Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useT } from "@/i18n";
import { isTauri } from "@/lib/api-base";
import { openFilePicker } from "@/lib/open-file-picker";
import { toast } from "@/lib/toast";
import { useState, type ComponentProps, type ComponentType } from "react";
import type { SidebarItem } from "../hooks/use-chat-sidebar-items";
import { getSidebarItemThreadIds, sandboxSessionIdsHolding } from "./chat-row-menu";

/** A 404 means no folder yet (made on a tool's first write), not a failure. */
function reportRevealError(
  error: unknown,
  t: ReturnType<typeof useT>,
  kind: "chat" | "project",
): void {
  if ((error as { status?: number } | null)?.status === 404) {
    toast.info(t(`library.chats.folder.${kind}Missing` as const), {
      description: t("library.chats.folder.missingDescription"),
    });
    return;
  }
  toast.error(t(`library.chats.folder.${kind}Failed` as const), {
    description: error instanceof Error ? error.message : String(error),
  });
}

/**
 * "Open chat folder" (desktop app only). A compare chat outside a project spans two
 * sandboxes, so it gets no item.
 */
export function OpenChatFolderItem({
  item,
  Item = DropdownMenuItem,
}: {
  item: SidebarItem;
  Item?: ComponentType<ComponentProps<typeof DropdownMenuItem>>;
}) {
  const t = useT();
  const threadIds = getSidebarItemThreadIds(item);
  const sandboxSessionId =
    item.type === "single" || item.projectId
      ? sandboxSessionIdFor(threadIds[0] ?? item.id, item.projectId)
      : undefined;
  if (!sandboxSessionId) return null;
  if (!isTauri) return <OpenChatFolderUnavailableItem Item={Item} />;
  return (
    <Item
      title={t("library.chats.folder.openChatTitle")}
      onSelect={() => {
        void (async () => {
          try {
            // Read from history: a chat moved between projects keeps its old sandbox.
            const ids = threadIds.length > 0 ? threadIds : [item.id];
            const distinct = await sandboxSessionIdsHolding(ids);
            if (distinct.length > 1) {
              toast.error(t("library.chats.folder.manyFolders"), {
                description: t("library.chats.folder.manyFoldersDescription"),
              });
              return;
            }
            await revealSandbox(distinct[0] ?? sandboxSessionId);
          } catch (error) {
            reportRevealError(error, t, "chat");
          }
        })();
      }}
    >
      <HugeiconsIcon icon={FolderOpenIcon} strokeWidth={1.75} className="size-icon" />
      <span>{t("library.chats.folder.openChat")}</span>
    </Item>
  );
}

/**
 * "Upload to chat folder": files go to the folder the chat's next tool call runs in, without
 * entering the conversation. The bytes travel to the backend, so a browser session can use it too.
 */
export function UploadToChatFolderItem({
  item,
  Item = DropdownMenuItem,
}: {
  item: SidebarItem;
  Item?: ComponentType<ComponentProps<typeof DropdownMenuItem>>;
}) {
  const t = useT();
  const sandboxSessionId =
    item.type === "single" || item.projectId
      ? sandboxSessionIdFor(getSidebarItemThreadIds(item)[0] ?? item.id, item.projectId)
      : undefined;
  if (!sandboxSessionId) return null;
  return (
    <Item
      title={t("library.chats.folder.uploadTitle")}
      onSelect={() =>
        openFilePicker("*", (files) => void uploadToChatFolder(sandboxSessionId, files, t))
      }
    >
      <HugeiconsIcon icon={Upload01Icon} strokeWidth={1.75} className="size-icon" />
      <span>{t("library.chats.folder.upload")}</span>
    </Item>
  );
}

async function uploadToChatFolder(
  sessionId: string,
  files: File[],
  t: ReturnType<typeof useT>,
  overwrite = false,
): Promise<void> {
  const added: string[] = [];
  const taken: File[] = [];
  for (const file of files) {
    const failed = (description: string) =>
      toast.error(t("library.chats.folder.uploadFailed", { name: file.name }), { description });
    if (file.size > MAX_SANDBOX_UPLOAD_BYTES) {
      failed(t("library.toast.uploadTooLarge", { name: file.name }));
      continue;
    }
    try {
      added.push((await uploadToSandbox(sessionId, file, overwrite)).name);
    } catch (error) {
      if ((error as { status?: number } | null)?.status === 409) taken.push(file);
      else failed(error instanceof Error ? error.message : String(error));
    }
  }
  if (added.length > 0) {
    toast.success(
      added.length === 1
        ? t("library.chats.folder.uploaded", { name: added[0] })
        : t("library.chats.folder.uploadedMany", { count: added.length }),
      { description: t("library.chats.folder.uploadedDescription") },
    );
  }
  if (taken.length > 0) {
    toast.info(
      taken.length === 1
        ? t("library.chats.folder.uploadExists", { name: taken[0].name })
        : t("library.chats.folder.uploadExistsMany", { count: taken.length }),
      {
        action: {
          label: t("library.chats.folder.replace"),
          onClick: () => void uploadToChatFolder(sessionId, taken, t, true),
        },
      },
    );
  }
}

export function OpenProjectFolderItem({
  projectId,
  Item = DropdownMenuItem,
}: {
  projectId: string;
  Item?: ComponentType<ComponentProps<typeof DropdownMenuItem>>;
}) {
  const t = useT();
  const sandboxSessionId = sandboxSessionIdFor(undefined, projectId);
  if (!sandboxSessionId) return null;
  if (!isTauri) {
    return (
      <OpenChatFolderUnavailableItem
        Item={Item}
        label={t("library.chats.folder.openProject")}
        hint={t("library.chats.folder.projectHint")}
      />
    );
  }
  return (
    <Item
      title={t("library.chats.folder.openProjectTitle")}
      onSelect={() => {
        void revealSandbox(sandboxSessionId).catch((error: unknown) =>
          reportRevealError(error, t, "project"),
        );
      }}
    >
      <HugeiconsIcon icon={FolderOpenIcon} strokeWidth={1.75} className="size-icon" />
      <span>{t("library.chats.folder.openProject")}</span>
    </Item>
  );
}

/**
 * "Open chat folder" for a browser session, where the backend's file manager is not the user's.
 * Radix's `disabled` takes the row's pointer events away and a tooltip is blocked while the menu
 * owns the screen, so the row stays enabled, refuses the select itself, and drives a controlled
 * tooltip off a pointer-events-none anchor (as the MCP rows do). The reason is carried twice: that
 * tooltip opens on hover, which a screen reader never reaches and a touch device does not have, so
 * `title` describes the row and selecting it opens the hint rather than doing nothing.
 */
export function OpenChatFolderUnavailableItem({
  // The sidebar renders this row into its right-click menu too, which is a different Radix set.
  Item = DropdownMenuItem,
  label,
  hint: hintOverride,
}: {
  Item?: ComponentType<ComponentProps<typeof DropdownMenuItem>>;
  label?: string;
  hint?: string;
} = {}) {
  const t = useT();
  const [hintOpen, setHintOpen] = useState(false);
  const hint = hintOverride ?? t("library.chats.folder.chatHint");

  return (
    <Item
      aria-disabled={true}
      title={hint}
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
      <span>{label ?? t("library.chats.folder.openChat")}</span>
      <Tooltip open={hintOpen}>
        {/* Our wrapper, not the raw primitive: it registers the trigger element,
            without which the tooltip counts itself blocked by the open menu. */}
        <TooltipTrigger asChild={true}>
          <span
            aria-hidden={true}
            className="pointer-events-none absolute inset-y-0 right-0 w-0"
          />
        </TooltipTrigger>
        <TooltipContent side="right" className="max-w-[calc(220px*var(--ui-space-scale,1))]">
          {hint}
        </TooltipContent>
      </Tooltip>
    </Item>
  );
}
