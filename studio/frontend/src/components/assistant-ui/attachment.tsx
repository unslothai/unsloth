


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
import {
  PASTED_TEXT_PREVIEW_MAX_CHARS,
  isPastedTextContent,
  isPastedTextFile,
  pastedTextContentBytes,
  pastedTextContentPreview,
  pastedTextPreview,
} from "@/features/chat";
import { formatBytes } from "@/features/hub";
import { cn } from "@/lib/utils";
import {
  AttachmentPrimitive,
  ComposerPrimitive,
  MessagePrimitive,
  useAui,
  useAuiState,
} from "@assistant-ui/react";
import {
  AudioWave01Icon,
  File02Icon,
  TextAlignLeft01Icon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { ChevronRightIcon, PlusIcon, XIcon } from "lucide-react";
import {
  type FC,
  type PropsWithChildren,
  useCallback,
  useEffect,
  useRef,
  useState,
} from "react";
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
        className="aui-attachment-preview-dialog-content top-0 left-0 grid h-dvh w-screen max-h-none max-w-none translate-x-0 translate-y-0 place-items-center overflow-hidden rounded-none border-0 bg-transparent p-0 shadow-none ring-0 sm:max-w-none [&>button]:fixed [&>button]:top-4 [&>button]:right-4 [&>button]:z-20 [&>button]:size-9 [&>button]:rounded-full [&>button]:bg-transparent [&>button]:text-white [&>button]:opacity-100 [&>button]:ring-0! [&>button]:hover:bg-white/25 [&>button]:hover:text-white [&_svg]:text-white"
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

type PastedTextAttachment = {
  readonly file?: File;
  readonly sentText?: string;
  readonly sentBytes?: number;
};

// Long pastes arrive as a synthetic .txt and render as a chip, not a tile.
// The selector only passes references along: the text can be megabytes, so
// nothing here may copy or scan it.
const usePastedTextAttachment = (): PastedTextAttachment | null => {
  return useAuiState(
    useShallow(({ attachment }): PastedTextAttachment | null => {
      if (attachment.type !== "document") return null;
      const file = (attachment as { file?: File }).file;
      const sentText = attachment.content?.flatMap((part) =>
        part.type === "text" ? [part.text] : [],
      )[0];
      const pasted = file
        ? isPastedTextFile(file)
        : isPastedTextContent(sentText);
      if (!pasted) return null;
      return { file, sentText, sentBytes: pastedTextContentBytes(sentText) };
    }),
  );
};

/** Only the composer inlines, and there the File is always still around. */
const readPastedText = async ({ file }: PastedTextAttachment): Promise<string> =>
  file ? await file.text() : "";

const readPastedTextPreview = async (
  attachment: PastedTextAttachment,
): Promise<{ text: string; remaining: number }> => {
  if (attachment.sentText !== undefined) {
    return pastedTextContentPreview(attachment.sentText);
  }
  return pastedTextPreview(await readPastedText(attachment));
};

const PastedTextPreviewDialog: FC<
  PropsWithChildren<{ name: string; attachment: PastedTextAttachment }>
> = ({ attachment, children, name }) => {
  const [open, setOpen] = useState(false);
  const [preview, setPreview] = useState<{
    text: string;
    remaining: number;
  } | null>(null);

  useEffect(() => {
    if (!open) return;
    let cancelled = false;
    // Laying out megabytes in one text node locks the page, so show an
    // opening. The attachment itself still holds everything.
    readPastedTextPreview(attachment)
      .then((value) => {
        if (!cancelled) setPreview(value);
      })
      .catch(() => {
        if (!cancelled) setPreview({ text: "", remaining: 0 });
      });
    return () => {
      cancelled = true;
    };
  }, [open, attachment]);

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogTrigger asChild={true}>{children}</DialogTrigger>
      <DialogContent className="aui-pasted-text-dialog flex max-h-[88dvh] w-[min(68rem,94vw)] max-w-none flex-col gap-3 overflow-hidden">
        <DialogTitle className="truncate pr-8 text-sm">{name}</DialogTitle>
        <pre className="aui-pasted-text-dialog-body max-h-[72dvh] overflow-auto whitespace-pre-wrap break-words rounded-lg border bg-muted/40 p-3 text-left font-mono text-xs leading-relaxed">
          {preview?.text ?? "Loading…"}
        </pre>
        {preview && preview.remaining > 0 ? (
          <p className="text-muted-foreground text-xs">
            {`First ${PASTED_TEXT_PREVIEW_MAX_CHARS.toLocaleString()} characters shown. ${preview.remaining.toLocaleString()} more were sent with the message.`}
          </p>
        ) : null}
      </DialogContent>
    </Dialog>
  );
};

const PastedTextAttachmentUI: FC<{
  attachment: PastedTextAttachment;
  isComposer: boolean;
  name: string;
}> = ({ attachment, isComposer, name }) => {
  const aui = useAui();
  const attachmentId = useAuiState(({ attachment: state }) => state.id);
  const [inlining, setInlining] = useState(false);
  const mountedRef = useRef(true);
  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
    };
  }, []);
  // Read off the header, never measured: the paste can be megabytes and this
  // runs while the thread is trying to paint.
  const bytes = attachment.file?.size ?? attachment.sentBytes;

  // Clicking the chip pours the text back into the composer.
  const showInTextField = useCallback(() => {
    if (inlining) return;
    setInlining(true);
    void readPastedText(attachment)
      .then((text) => {
        // Reading a big file is slow enough to outlive the send that cleared
        // the composer, which would leave the text behind as a stray draft.
        if (!mountedRef.current || text.length === 0) return;
        const composer = aui.composer();
        if (
          !composer
            .getState()
            .attachments.some((item) => item.id === attachmentId)
        ) {
          return;
        }
        const current = composer.getState().text;
        composer.setText(current.length > 0 ? `${current}\n\n${text}` : text);
        aui.attachment().remove();
      })
      .catch(() => undefined)
      .finally(() => {
        if (mountedRef.current) setInlining(false);
      });
  }, [attachment, attachmentId, aui, inlining]);

  const chip = (
    <button
      className={cn(
        // Borderless, and in dark mode a shade under the composer surface.
        "aui-pasted-text-chip group flex h-14 max-w-[15rem] min-w-0 cursor-pointer items-center gap-2.5 rounded-[14px] bg-muted px-3 text-left transition-colors hover:bg-muted-foreground/15 dark:bg-background dark:hover:bg-muted",
        // Keep the label clear of the remove button in the corner.
        isComposer && "aui-pasted-text-chip-composer pr-6",
      )}
      type="button"
      aria-label={
        isComposer
          ? `Pasted text: ${name}. Show in text field`
          : `Pasted text: ${name}. Show contents`
      }
      onClick={isComposer ? showInTextField : undefined}
    >
      <span className="flex size-8 shrink-0 items-center justify-center rounded-lg bg-foreground/10">
        <HugeiconsIcon
          icon={TextAlignLeft01Icon}
          strokeWidth={2}
          className="size-4 text-muted-foreground"
        />
      </span>
      <span className="flex min-w-0 flex-col">
        <span className="truncate font-medium text-xs">{name}</span>
        <span className="truncate text-ui-11 text-muted-foreground">
          {/* Hover swaps the size for the action. */}
          <span className={isComposer ? "group-hover:hidden" : undefined}>
            {bytes === undefined ? "Pasted text" : formatBytes(bytes)}
          </span>
          {isComposer ? (
            <span className="hidden items-center gap-0.5 underline underline-offset-2 group-hover:inline-flex">
              Show in text field
              <ChevronRightIcon className="size-3" />
            </span>
          ) : null}
        </span>
      </span>
    </button>
  );

  return (
    <AttachmentPrimitive.Root className="aui-attachment-root relative">
      {isComposer ? (
        chip
      ) : (
        <PastedTextPreviewDialog attachment={attachment} name={name}>
          {chip}
        </PastedTextPreviewDialog>
      )}
      {isComposer && <AttachmentRemove />}
    </AttachmentPrimitive.Root>
  );
};

const AttachmentUI: FC = () => {
  const aui = useAui();
  const isComposer = aui.attachment.source === "composer";
  const pastedText = usePastedTextAttachment();

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

  if (pastedText) {
    return (
      <PastedTextAttachmentUI
        attachment={pastedText}
        isComposer={isComposer}
        name={name ?? "Pasted text"}
      />
    );
  }

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
