// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  completeProgressiveMounts,
  revealProgressiveRow,
} from "@/components/assistant-ui/progressive-messages";
import { saveSandboxFile } from "@/components/assistant-ui/sandbox-download";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import { Tooltip, TooltipContent } from "@/components/ui/tooltip";
import {
  downloadFile,
  downloadUrl,
  isDownloadCancelled,
} from "@/lib/native-files";
import { toast } from "@/lib/toast";
import { useAuiState } from "@assistant-ui/react";
import {
  BrowserIcon,
  Download01Icon,
  File01Icon,
  FolderDownloadIcon,
  Image01Icon,
  SourceCodeIcon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { Tooltip as TooltipPrimitive } from "radix-ui";
import { useEffect, useMemo, useState } from "react";
import { create } from "zustand";
import { type ChatFile, collectChatFiles } from "../lib/chat-files";
import { CHAT_MENU_TRIGGER } from "./chat-header-menu";

const useChatFiles = create<{ files: ChatFile[] }>(() => ({ files: [] }));

const KIND_ICON = {
  code: SourceCodeIcon,
  html: BrowserIcon,
  image: Image01Icon,
  file: File01Icon,
} as const;

export function ChatFilesBridge() {
  const messages = useAuiState(({ thread }) => thread.messages);
  const files = useMemo(() => collectChatFiles(messages), [messages]);

  useEffect(() => {
    useChatFiles.setState({ files });
  }, [files]);
  useEffect(() => () => useChatFiles.setState({ files: [] }), []);

  return null;
}

async function saveChatFile(file: ChatFile): Promise<void> {
  if (file.sandbox) {
    await saveSandboxFile(file.sandbox.sessionId, file.sandbox.name);
  } else if (file.dataUrl) {
    await downloadUrl(file.dataUrl, file.name);
  } else if (file.text != null) {
    await downloadFile(
      file.text,
      file.name,
      file.kind === "html"
        ? "text/html;charset=utf-8"
        : "text/plain;charset=utf-8",
    );
  }
}

async function jumpToMessage(messageId: string): Promise<void> {
  const selector = `[data-message-id="${CSS.escape(messageId)}"]`;
  const viewports = [
    ...document.querySelectorAll<HTMLElement>(".aui-thread-viewport"),
  ].filter((viewport) => !viewport.closest("[inert]") && viewport.offsetParent);
  for (const viewport of viewports) {
    let target = viewport.querySelector<HTMLElement>(selector);
    if (!target && revealProgressiveRow(viewport, messageId)) {
      target = viewport.querySelector<HTMLElement>(selector);
    }
    if (!target) {
      await completeProgressiveMounts((candidate) => candidate === viewport);
      target = viewport.querySelector<HTMLElement>(selector);
    }
    if (!target) continue;
    const inset = Number.parseFloat(getComputedStyle(viewport).paddingTop) || 0;
    viewport.scrollBy({
      top:
        target.getBoundingClientRect().top -
        viewport.getBoundingClientRect().top -
        inset,
      behavior: "smooth",
    });
    return;
  }
}

function ChatFileRow({ file, onJump }: { file: ChatFile; onJump: () => void }) {
  return (
    <li className="flex items-center gap-1 rounded-md hover:bg-accent">
      <button
        type="button"
        onClick={onJump}
        className="flex min-w-0 flex-1 items-center gap-2 rounded-md px-2 py-1.5 text-left focus-visible:bg-accent focus-visible:outline-none"
        title="Show in chat"
      >
        <HugeiconsIcon
          icon={KIND_ICON[file.kind]}
          strokeWidth={1.75}
          className="size-4 shrink-0 text-muted-foreground"
        />
        <span className="truncate text-[13px]">{file.name}</span>
      </button>
      <button
        type="button"
        onClick={() =>
          saveChatFile(file).catch((error) => {
            if (!isDownloadCancelled(error)) {
              toast.error(`Could not save ${file.name}.`);
            }
          })
        }
        className="mr-1 flex size-7 shrink-0 items-center justify-center rounded-md text-muted-foreground hover:bg-background hover:text-foreground focus-visible:bg-background focus-visible:text-foreground focus-visible:outline-none"
        aria-label={`Download ${file.name}`}
      >
        <HugeiconsIcon icon={Download01Icon} className="size-4" />
      </button>
    </li>
  );
}

export function ChatFilesMenu() {
  const files = useChatFiles((s) => s.files);
  const [open, setOpen] = useState(false);

  if (files.length === 0) return null;

  return (
    <Popover open={open} onOpenChange={setOpen}>
      <Tooltip>
        <TooltipPrimitive.Trigger asChild={true}>
          <PopoverTrigger asChild={true}>
            <button
              type="button"
              className={CHAT_MENU_TRIGGER}
              aria-label="Files in this chat"
            >
              <HugeiconsIcon
                icon={FolderDownloadIcon}
                strokeWidth={1.75}
                className="size-icon"
              />
            </button>
          </PopoverTrigger>
        </TooltipPrimitive.Trigger>
        <TooltipContent
          side="bottom"
          sideOffset={6}
          className="tooltip-compact"
        >
          Files in this chat
        </TooltipContent>
      </Tooltip>
      <PopoverContent align="end" sideOffset={6} className="w-80 gap-1 p-1.5">
        <p className="px-2 pt-1 pb-1 text-xs font-medium text-muted-foreground">
          Files in this chat
        </p>
        <ul className="max-h-80 overflow-y-auto">
          {files.map((file) => (
            <ChatFileRow
              key={file.id}
              file={file}
              onJump={() => {
                setOpen(false);
                void jumpToMessage(file.messageId);
              }}
            />
          ))}
        </ul>
      </PopoverContent>
    </Popover>
  );
}
