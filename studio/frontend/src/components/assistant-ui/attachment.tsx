// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"use client";

// Avatar removed — caused circular crop on image thumbnails
import { TooltipIconButton } from "@/components/assistant-ui/tooltip-icon-button";
import {
  Dialog,
  DialogClose,
  DialogContent,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { cn } from "@/lib/utils";
import {
  AttachmentPrimitive,
  ComposerPrimitive,
  MessagePrimitive,
  useAui,
  useAuiState,
} from "@assistant-ui/react";
import { AudioWave01Icon, File02Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { PlusIcon, XIcon } from "lucide-react";
import { type FC, type PropsWithChildren, useEffect, useState } from "react";
import { useShallow } from "zustand/shallow";

const useFileSrc = (file: File | undefined): string | undefined => {
  const [objectUrl, setObjectUrl] = useState<string | undefined>(undefined);

  useEffect(() => {
    if (!file) {
      setObjectUrl(undefined);
      return;
    }
    const url = URL.createObjectURL(file);
    setObjectUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);

  return objectUrl;
};

const useAttachmentSrc = (): string | undefined => {
  const { file, src } = useAuiState(
    useShallow(({ attachment }): { file?: File; src?: string } => {
      if (attachment.type !== "image") {
        return {};
      }
      if (attachment.file) {
        return { file: attachment.file };
      }
      const src = attachment.content?.filter((c) => c.type === "image")[0]
        ?.image;
      if (!src) {
        return {};
      }
      return { src };
    }),
  );

  return useFileSrc(file) ?? src;
};

type AttachmentPreviewProps = {
  src: string;
};

const AttachmentPreview: FC<AttachmentPreviewProps> = ({ src }) => {
  const [isLoaded, setIsLoaded] = useState(false);
  return (
    <img
      src={src}
      alt="Preview"
      className={cn(
        "block h-auto max-h-[90dvh] w-auto max-w-[92vw] object-contain",
        isLoaded
          ? "aui-attachment-preview-image-loaded"
          : "aui-attachment-preview-image-loading invisible",
      )}
      onLoad={() => setIsLoaded(true)}
    />
  );
};

const AttachmentPreviewDialog: FC<PropsWithChildren> = ({ children }) => {
  const src = useAttachmentSrc();

  if (!src) {
    return children;
  }

  return (
    <Dialog>
      <DialogTrigger
        className="aui-attachment-preview-trigger cursor-pointer transition-colors hover:bg-accent/50"
        asChild={true}
      >
        {children}
      </DialogTrigger>
      {/* Chrome-free lightbox: the image floats on the dimmed backdrop with
          no dialog panel, and the close button sits in the screen corner. */}
      <DialogContent
        overlayClassName="bg-black/70"
        className="aui-attachment-preview-dialog-content top-0 left-0 grid h-dvh w-screen max-w-none translate-x-0 translate-y-0 place-items-center rounded-none border-0 bg-transparent p-0 shadow-none ring-0 sm:max-w-none [&>button]:fixed [&>button]:top-4 [&>button]:right-4 [&>button]:z-20 [&>button]:size-9 [&>button]:rounded-full [&>button]:bg-transparent [&>button]:text-white [&>button]:opacity-100 [&>button]:ring-0! [&>button]:hover:bg-white/25 [&>button]:hover:text-white [&_svg]:text-white"
      >
        <DialogTitle className="aui-sr-only sr-only">
          Image Attachment Preview
        </DialogTitle>
        {/* Clicking the backdrop (anywhere off the image) closes the preview. */}
        <DialogClose asChild={true}>
          <div aria-hidden="true" className="absolute inset-0" />
        </DialogClose>
        <div className="aui-attachment-preview pointer-events-none relative z-10 flex items-center justify-center">
          <span className="pointer-events-auto">
            <AttachmentPreview src={src} />
          </span>
        </div>
      </DialogContent>
    </Dialog>
  );
};

const AUDIO_ATTACHMENT_RE = /\.(wav|mp3|m4a|ogg|oga|flac|webm|mp4|aac)$/i;

const isAudioAttachment = (name: string | undefined, contentType: string) =>
  /^audio\//i.test(contentType) || AUDIO_ATTACHMENT_RE.test(name ?? "");

const AttachmentThumb: FC = () => {
  const src = useAttachmentSrc();
  const name = useAuiState(({ attachment }) => attachment.name);
  const contentType = useAuiState(
    ({ attachment }) =>
      (attachment as { file?: File }).file?.type ??
      (attachment as { contentType?: string }).contentType ??
      "",
  );

  if (src) {
    return (
      <img
        src={src}
        alt={name || "Attachment preview"}
        className="h-full w-full object-cover"
      />
    );
  }

  return (
    <div className="flex h-full w-full items-center justify-center">
      <HugeiconsIcon
        icon={isAudioAttachment(name, contentType) ? AudioWave01Icon : File02Icon}
        strokeWidth={2}
        className="size-6 text-muted-foreground"
      />
    </div>
  );
};

const AttachmentUI: FC = () => {
  const aui = useAui();
  const isComposer = aui.attachment.source === "composer";

  const isImage = useAuiState(({ attachment }) => attachment.type === "image");
  const name = useAuiState(({ attachment }) => attachment.name);
  const typeLabel = useAuiState(({ attachment }) => {
    const type = attachment.type;
    switch (type) {
      case "image":
        return "Image";
      case "document":
        return "Document";
      case "file":
        return isAudioAttachment(
          attachment.name,
          (attachment as { file?: File }).file?.type ?? "",
        )
          ? "Audio"
          : "File";
      default:
        throw new Error(`Unknown attachment type: ${type as string}`);
    }
  });
  // Filename in accessible name lets screen readers distinguish same-typed
  // attachments. Sighted users get it via the tooltip.
  const accessibleName = name
    ? `${typeLabel} attachment: ${name}`
    : `${typeLabel} attachment`;

  return (
    <Tooltip>
      <AttachmentPrimitive.Root
        className={cn(
          "aui-attachment-root relative",
          isImage &&
            "aui-attachment-root-composer only:[&>#attachment-tile]:size-16",
        )}
      >
        <AttachmentPreviewDialog>
          <TooltipTrigger asChild={true}>
            <button
              className={cn(
                "aui-attachment-tile size-14 cursor-pointer overflow-hidden rounded-[14px] border bg-muted transition-opacity hover:opacity-75",
                isComposer &&
                  "aui-attachment-tile-composer border-foreground/20",
              )}
              id="attachment-tile"
              aria-label={accessibleName}
              type="button"
            >
              <AttachmentThumb />
            </button>
          </TooltipTrigger>
        </AttachmentPreviewDialog>
        {isComposer && <AttachmentRemove />}
      </AttachmentPrimitive.Root>
      <TooltipContent side="top" className="tooltip-compact">
        <AttachmentPrimitive.Name />
      </TooltipContent>
    </Tooltip>
  );
};

const AttachmentRemove: FC = () => {
  return (
    <AttachmentPrimitive.Remove asChild={true}>
      <TooltipIconButton
        tooltip="Remove file"
        className="aui-attachment-tile-remove absolute top-1.5 right-1.5 size-3.5 rounded-full bg-white text-muted-foreground opacity-100 shadow-sm hover:bg-white! [&_svg]:text-black hover:[&_svg]:text-destructive"
        side="top"
      >
        <XIcon className="aui-attachment-remove-icon size-3 dark:stroke-[2.5px]" />
      </TooltipIconButton>
    </AttachmentPrimitive.Remove>
  );
};

export const UserMessageAttachments: FC = () => {
  return (
    <div className="aui-user-message-attachments-end col-span-full col-start-1 row-start-1 flex w-full flex-row justify-end gap-2">
      <MessagePrimitive.Attachments components={{ Attachment: AttachmentUI }} />
    </div>
  );
};

export const ComposerAttachments: FC = () => {
  return (
    <div className="aui-composer-attachments mb-2 flex w-full flex-row items-center gap-2 overflow-x-auto px-1.5 pt-0.5 pb-1 empty:hidden">
      <ComposerPrimitive.Attachments
        components={{ Attachment: AttachmentUI }}
      />
    </div>
  );
};

export const ComposerAddAttachment: FC = () => {
  return (
    <ComposerPrimitive.AddAttachment asChild={true}>
      <TooltipIconButton
        tooltip="Add Attachment"
        side="bottom"
        variant="ghost"
        size="icon"
        className="aui-composer-add-attachment size-8.5 rounded-full p-1 font-semibold text-xs hover:bg-muted-foreground/15 dark:hover:bg-muted-foreground/30"
        aria-label="Add Attachment"
      >
        <PlusIcon className="aui-attachment-add-icon size-5 stroke-[1.5px]" />
      </TooltipIconButton>
    </ComposerPrimitive.AddAttachment>
  );
};
