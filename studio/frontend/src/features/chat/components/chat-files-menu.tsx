// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  listSandboxFiles,
  saveSandboxFile,
} from "@/components/assistant-ui/sandbox-download";
import { sandboxSessionIdFor } from "@/components/assistant-ui/sandbox-files";
import { sandboxUploadListeners } from "@/components/assistant-ui/sandbox-upload";
import { requestJumpToMessage } from "@/components/assistant-ui/message-jump";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import { Tooltip, TooltipContent } from "@/components/ui/tooltip";
import { useT } from "@/i18n";
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
import { useChatProjectScope } from "../chat-project-scope";
import {
  type ChatFile,
  type ChatFolder,
  createChatFilesCollector,
} from "../lib/chat-files";
import { useChatRuntimeStore } from "../stores/chat-runtime-store";
import { CHAT_MENU_TRIGGER } from "./chat-header-menu";

const useChatFiles = create<{ files: ChatFile[] }>(() => ({ files: [] }));

const KIND_ICON = {
  code: SourceCodeIcon,
  html: BrowserIcon,
  image: Image01Icon,
  file: File01Icon,
} as const;

const NO_FOLDERS: ChatFolder[] = [];

export function ChatFilesBridge() {
  const messages = useAuiState(({ thread }) => thread.messages);
  const running = useAuiState(({ thread }) => thread.isRunning);
  const remoteId = useAuiState(({ threadListItem }) => threadListItem.remoteId);
  const activeThreadId = useChatRuntimeStore((state) => state.activeThreadId);
  const projectId = useChatProjectScope();
  const threadId = remoteId ?? activeThreadId ?? undefined;
  // A chat moved into a project keeps its earlier files in its own thread folder.
  const sessionKey = [
    ...new Set([sandboxSessionIdFor(threadId, projectId), threadId]),
  ]
    .filter((id): id is string => Boolean(id))
    .join("\u0000");
  const [listed, setListed] = useState<{
    key: string;
    folders: ChatFolder[];
  } | null>(null);
  const [uploads, setUploads] = useState(0);

  useEffect(() => {
    const sessions = sessionKey.split("\u0000");
    const listener = (uploaded: string) => {
      if (sessions.includes(uploaded)) {
        setUploads((count) => count + 1);
      }
    };
    sandboxUploadListeners.add(listener);
    return () => {
      sandboxUploadListeners.delete(listener);
    };
  }, [sessionKey]);

  // Re-read after each run and upload: tools write to the folder without always naming the file.
  // biome-ignore lint/correctness/useExhaustiveDependencies: uploads only triggers a re-list.
  useEffect(() => {
    if (!sessionKey || running) {
      return;
    }
    let live = true;
    const sessions = sessionKey.split("\u0000");
    Promise.all(
      sessions.map((sessionId) =>
        listSandboxFiles(sessionId)
          .then((names) => ({ sessionId, names }))
          .catch(() => ({ sessionId, names: [] })),
      ),
    ).then((folders) => {
      if (live) {
        setListed({ key: sessionKey, folders });
      }
    });
    return () => {
      live = false;
    };
  }, [sessionKey, running, uploads]);

  const folders = listed?.key === sessionKey ? listed.folders : NO_FOLDERS;
  // The streaming reply changes every token: leave it out until it settles, and reuse the
  // result while the settled messages are the same objects.
  const [collect] = useState(createChatFilesCollector);
  const files = collect(running ? messages.slice(0, -1) : messages, folders);

  useEffect(() => {
    useChatFiles.setState({ files });
  }, [files]);
  useEffect(() => () => useChatFiles.setState({ files: [] }), []);

  return null;
}

async function saveChatFile(file: ChatFile): Promise<void> {
  if (file.sandbox) {
    await saveSandboxFile(file.sandbox.sessionId, file.sandbox.name, file.name);
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

function ChatFileRow({ file, onJump }: { file: ChatFile; onJump: () => void }) {
  const t = useT();
  const save = () =>
    saveChatFile(file).catch((error) => {
      if (!isDownloadCancelled(error)) {
        toast.error(t("chatFiles.saveFailed", { name: file.name }));
      }
    });
  return (
    <li className="flex items-center gap-1 rounded-md hover:bg-accent">
      <button
        type="button"
        onClick={file.messageId ? onJump : save}
        className="flex min-w-0 flex-1 items-center gap-2 rounded-md px-2 py-1.5 text-left focus-visible:bg-accent focus-visible:outline-none"
        title={file.messageId ? t("chatFiles.showInChat") : undefined}
      >
        <HugeiconsIcon
          icon={KIND_ICON[file.kind]}
          strokeWidth={1.75}
          className="size-4 shrink-0 text-muted-foreground"
        />
        <span className="truncate text-ui-13">{file.name}</span>
      </button>
      <button
        type="button"
        onClick={save}
        className="mr-1 flex size-7 shrink-0 items-center justify-center rounded-md text-muted-foreground hover:bg-background hover:text-foreground focus-visible:bg-background focus-visible:text-foreground focus-visible:outline-none"
        aria-label={t("chatFiles.download", { name: file.name })}
      >
        <HugeiconsIcon icon={Download01Icon} className="size-4" />
      </button>
    </li>
  );
}

function ChatFileSection({
  label,
  files,
  onJump,
}: {
  label: string;
  files: ChatFile[];
  onJump: (file: ChatFile) => void;
}) {
  if (files.length === 0) return null;
  return (
    <>
      <p className="px-2 pt-1 pb-1 text-xs font-medium text-muted-foreground">
        {label}
      </p>
      <ul>
        {files.map((file) => (
          <ChatFileRow key={file.id} file={file} onJump={() => onJump(file)} />
        ))}
      </ul>
    </>
  );
}

export function ChatFilesMenu() {
  const t = useT();
  const files = useChatFiles((s) => s.files);
  const [open, setOpen] = useState(false);
  // Files first: in a coding chat the snippets would bury what was actually made.
  const [documents, snippets] = useMemo(
    () => [
      files.filter((f) => f.kind !== "code"),
      files.filter((f) => f.kind === "code"),
    ],
    [files],
  );

  if (files.length === 0) return null;

  const jump = (file: ChatFile) => {
    setOpen(false);
    if (file.messageId) requestJumpToMessage(file.messageId);
  };
  const title = t("chatFiles.title");

  return (
    <Popover open={open} onOpenChange={setOpen}>
      <Tooltip>
        <TooltipPrimitive.Trigger asChild={true}>
          <PopoverTrigger asChild={true}>
            <button
              type="button"
              className={CHAT_MENU_TRIGGER}
              aria-label={title}
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
          {title}
        </TooltipContent>
      </Tooltip>
      <PopoverContent align="end" sideOffset={6} className="w-80 gap-1 p-1.5">
        <div className="max-h-80 overflow-y-auto">
          <ChatFileSection label={title} files={documents} onJump={jump} />
          <ChatFileSection
            label={t("chatFiles.code")}
            files={snippets}
            onJump={jump}
          />
        </div>
      </PopoverContent>
    </Popover>
  );
}
