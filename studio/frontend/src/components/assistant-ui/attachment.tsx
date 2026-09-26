// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"use client";

// Avatar removed — caused circular crop on image thumbnails
import { AttachmentPreviewDialog } from "@/components/assistant-ui/attachment-preview";
import { TooltipIconButton } from "@/components/assistant-ui/tooltip-icon-button";
import { Button } from "@/components/ui/button";
import { useAttachmentImageSrc } from "@/components/assistant-ui/use-attachment-source";
import {
  Dialog,
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
  ATTACHMENT_KIND_ICON_CLASS,
  ATTACHMENT_KIND_ICONS,
  PASTED_TEXT_PREVIEW_MAX_CHARS,
  attachmentFileKind,
  attachmentKindLabel,
  composerAttachmentsOverflow,
  isAudioAttachment,
  sentAttachmentLayout,
  type AttachmentFileKind,
  type SentAttachmentLayout,
  isPastedTextContent,
  isPastedTextFile,
  pastedTextContentBytes,
  pastedTextContentPreview,
  pastedTextPreview,
} from "@/features/chat";
import { formatBytes } from "@/features/hub";
import { useAppearanceCustomStore } from "@/features/settings";
import { cn } from "@/lib/utils";
import { useShallow } from "zustand/shallow";
import {
  AttachmentPrimitive,
  ComposerPrimitive,
  MessagePrimitive,
  useAui,
  useAuiState,
} from "@assistant-ui/react";
import {
  AudioWave01Icon,
  File01Icon,
  File02Icon,
  TextAlignLeft01Icon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { ChevronRightIcon, PlusIcon, XIcon } from "lucide-react";
import {
  createContext,
  type FC,
  type PropsWithChildren,
  type ReactNode,
  useCallback,
  useContext,
  useEffect,
  useLayoutEffect,
  useRef,
  useState,
} from "react";
import { ScrollPane } from "./scroll-pane";

const AttachmentThumb: FC = () => {
  const src = useAttachmentImageSrc();
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
        icon={
          isAudioAttachment(name, contentType) ? AudioWave01Icon : File02Icon
        }
        strokeWidth={2}
        className="size-6 text-muted-foreground"
      />
    </div>
  );
};

// Five cards to a row: each slot is a fifth of the row less its gap-2 gaps, and the card fills
// it. A narrow composer fits fewer rather than shrinking cards past their names.
const CARD_SLOT =
  "shrink-0 w-[calc((100%_-_var(--spacing)*8)/5)] min-w-[calc(7rem*var(--ui-space-scale,1))]";
const CARD_SIZE = "h-[calc(7rem*var(--ui-space-scale,1))] w-full";
const SENT_IMAGE_SIZE = "size-[calc(9rem*var(--ui-space-scale,1))]";
const SENT_IMAGE_SIZE_COMPACT = "size-[calc(5rem*var(--ui-space-scale,1))]";
const SENT_ROW_WIDTH = "w-[calc(18rem*var(--ui-space-scale,1))]";
// A hairline a shade off the surface, which the contrast setting can strengthen.
const CARD_EDGE =
  "border border-[color-mix(in_oklab,var(--foreground)_calc(12%*var(--contrast-edge-gain,1)),transparent)]";
const CARD_SURFACE =
  "bg-[color-mix(in_oklab,var(--foreground)_3%,transparent)] hover:bg-[color-mix(in_oklab,var(--foreground)_6%,transparent)]";

/** The attachment's name and kind, read off its header: the bytes are never touched. */
const useAttachmentKind = (): {
  name: string;
  kind: AttachmentFileKind;
} => {
  const name = useAuiState(({ attachment }) => attachment.name ?? "");
  const isImage = useAuiState(({ attachment }) => attachment.type === "image");
  const contentType = useAuiState(
    ({ attachment }) =>
      // An unknown type reads as "", which must not hide the stored one.
      (attachment as { file?: File }).file?.type ||
      (attachment as { contentType?: string }).contentType ||
      "",
  );
  return {
    name,
    kind: isImage ? "image" : attachmentFileKind(name, contentType),
  };
};

const AttachmentKindIcon: FC<{ kind: AttachmentFileKind; className?: string }> = ({
  kind,
  className,
}) => (
  <HugeiconsIcon
    icon={ATTACHMENT_KIND_ICONS[kind]}
    strokeWidth={1.75}
    className={cn("shrink-0", ATTACHMENT_KIND_ICON_CLASS[kind], className)}
  />
);

/** A card's face: a page in the middle and the file's kind and name along the bottom. */
const FileCardBody: FC<{
  name: string;
  kind: AttachmentFileKind;
  /** Replaces the page glyph, for a card with something better to say there. */
  center?: ReactNode;
  icon?: ReactNode;
}> = ({ name, kind, center, icon }) => (
  <span className="flex h-full w-full flex-col">
    <span className="flex min-h-0 flex-1 items-center justify-center px-3 text-muted-foreground">
      {center ?? (
        <HugeiconsIcon icon={File01Icon} strokeWidth={1.5} className="size-6" />
      )}
    </span>
    <span className="flex min-w-0 items-center gap-1.5 px-3 pb-2.5">
      {icon ?? <AttachmentKindIcon kind={kind} className="size-4" />}
      <span className="min-w-0 truncate text-ui-13 leading-ui-18 text-foreground">
        {name}
      </span>
    </span>
  </span>
);

/** An image fills its card; one with no preview left shows as a file card instead. */
const CardImageOrBody: FC<{ name: string; kind: AttachmentFileKind; src: string | undefined }> = ({
  name,
  kind,
  src,
}) => {
  if (src) {
    return (
      <img
        src={src}
        alt={name || "Attachment preview"}
        className="h-full w-full object-cover"
      />
    );
  }
  return <FileCardBody name={name} kind={kind} />;
};

type PastedTextAttachment = {
  readonly file?: File;
  readonly sentText?: string;
  readonly sentBytes?: number;
};

// Long pastes arrive as a synthetic .txt and render as a chip, not a tile. The selector only passes
// references along: the text can be megabytes, so nothing here may copy or scan it.
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
        <ScrollPane
          className="aui-pasted-text-dialog-body rounded-lg border bg-muted/40 p-3"
          scrollerClassName="max-h-[72dvh] overflow-auto whitespace-pre-wrap break-words text-left font-mono text-xs leading-relaxed"
        >
          {preview?.text ?? "Loading…"}
        </ScrollPane>
        {preview && preview.remaining > 0 ? (
          <p className="text-muted-foreground text-xs">
            {`First ${PASTED_TEXT_PREVIEW_MAX_CHARS.toLocaleString()} characters shown. ${preview.remaining.toLocaleString()} more were sent with the message.`}
          </p>
        ) : null}
      </DialogContent>
    </Dialog>
  );
};

/** How a pasted-text attachment is drawn: the composer's tile-height chip, a composer card, a
 *  sent message's list row, or a sent message's chip. */
type PastedTextVariant = "chip" | "card" | "row" | "pill";

const PastedTextAttachmentUI: FC<{
  attachment: PastedTextAttachment;
  isComposer: boolean;
  name: string;
  variant?: PastedTextVariant;
}> = ({ attachment, isComposer, name, variant = "chip" }) => {
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

  const sizeLabel = bytes === undefined ? "Pasted text" : formatBytes(bytes);
  const ariaLabel = isComposer
    ? `Pasted text: ${name}. Show in text field`
    : `Pasted text: ${name}. Show contents`;
  const textIcon = (className: string) => (
    <HugeiconsIcon
      icon={TextAlignLeft01Icon}
      strokeWidth={2}
      className={cn("shrink-0 text-muted-foreground", className)}
    />
  );
  const chip =
    variant === "card" ? (
      <button
        className={cn(
          "aui-pasted-text-card group flex cursor-pointer overflow-hidden rounded-[18px] text-left transition-colors",
          CARD_SIZE,
          CARD_EDGE,
          CARD_SURFACE,
        )}
        type="button"
        title={name}
        aria-label={ariaLabel}
        onClick={isComposer ? showInTextField : undefined}
      >
        <FileCardBody
          name={name}
          kind="text"
          icon={textIcon("size-4")}
          center={
            <span className="flex flex-col items-center gap-1 text-ui-11">
              {textIcon("size-6")}
              {/* Hover swaps the size for the action. */}
              <span className={isComposer ? "group-hover:hidden" : undefined}>
                {sizeLabel}
              </span>
              {isComposer ? (
                <span className="hidden items-center gap-0.5 underline underline-offset-2 group-hover:inline-flex">
                  Show in text field
                  <ChevronRightIcon className="size-3" />
                </span>
              ) : null}
            </span>
          }
        />
      </button>
    ) : variant === "row" ? (
      <button
        className={cn(
          "aui-pasted-text-row flex max-w-full cursor-pointer items-center gap-3 rounded-[18px] px-4 py-3 text-left transition-colors",
          SENT_ROW_WIDTH,
          CARD_EDGE,
          "hover:bg-[color-mix(in_oklab,var(--foreground)_5%,transparent)]",
        )}
        type="button"
        title={name}
        aria-label={ariaLabel}
      >
        {textIcon("size-7")}
        <span className="flex min-w-0 flex-col">
          <span className="truncate font-medium text-sm">{name}</span>
          <span className="truncate text-muted-foreground text-xs">
            {sizeLabel}
          </span>
        </span>
      </button>
    ) : variant === "pill" ? (
      <button
        className={cn(
          "aui-pasted-text-pill inline-flex h-9 max-w-[calc(16rem*var(--ui-space-scale,1))] cursor-pointer items-center gap-1.5 rounded-full px-3 text-sm transition-colors",
          CARD_EDGE,
          "hover:bg-[color-mix(in_oklab,var(--foreground)_5%,transparent)]",
        )}
        type="button"
        title={name}
        aria-label={ariaLabel}
      >
        {textIcon("size-4")}
        <span className="truncate">{name}</span>
      </button>
    ) : (
    <button
      className={cn(
        // Borderless, and in dark mode a shade under the composer surface.
        "aui-pasted-text-chip group flex h-14 max-w-[calc(15rem*var(--ui-space-scale,1))] min-w-0 cursor-pointer items-center gap-2.5 rounded-[14px] bg-muted px-3 text-left transition-colors hover:bg-muted-foreground/15 dark:bg-background dark:hover:bg-muted",
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
      <span className="flex size-8 shrink-0 items-center justify-center rounded-lg bg-[color-mix(in_oklab,var(--foreground)_calc(10%*var(--contrast-wash-gain,1)),transparent)]">
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
    <AttachmentPrimitive.Root
      className={cn(
        "aui-attachment-root relative",
        variant === "card" && cn("group/attachment-card", CARD_SLOT),
      )}
    >
      {isComposer ? (
        chip
      ) : (
        <PastedTextPreviewDialog attachment={attachment} name={name}>
          {chip}
        </PastedTextPreviewDialog>
      )}
      {isComposer &&
        (variant === "card" ? <AttachmentCardRemove /> : <AttachmentRemove />)}
    </AttachmentPrimitive.Root>
  );
};

const AttachmentUI: FC = () => {
  const aui = useAui();
  const isComposer = aui.attachment.source === "composer";
  const pastedText = usePastedTextAttachment();

  const isImage = useAuiState(({ attachment }) => attachment.type === "image");
  const attachmentId = useAuiState(({ attachment }) => attachment.id);
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
        key={attachmentId}
        attachment={pastedText}
        isComposer={isComposer}
        name={name ?? "Pasted text"}
      />
    );
  }

  return (
    <Tooltip>
      <AttachmentPrimitive.Root
        key={attachmentId}
        className={cn(
          "aui-attachment-root relative",
          isImage &&
            "aui-attachment-root-composer only:[&>#attachment-tile]:size-16",
        )}
      >
        <AttachmentPreviewDialog redactFromReload={isComposer}>
          <TooltipTrigger asChild={true}>
            <button
              className={cn(
                "aui-attachment-tile size-14 cursor-pointer overflow-hidden rounded-[14px] border bg-muted transition-opacity hover:opacity-75",
                isComposer &&
                  "aui-attachment-tile-composer border-[color-mix(in_oklab,var(--foreground)_calc(20%*var(--contrast-edge-gain,1)),transparent)]",
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
      <Button
        variant="ghost"
        size="icon"
        aria-label="Remove file"
        className="aui-attachment-tile-remove absolute top-1.5 right-1.5 size-3.5 rounded-full bg-white p-0 text-muted-foreground opacity-100 shadow-sm hover:bg-white! [&_svg]:text-black hover:[&_svg]:text-destructive"
      >
        <XIcon className="aui-attachment-remove-icon size-3 dark:stroke-[2.5px]" />
      </Button>
    </AttachmentPrimitive.Remove>
  );
};

/** The card's remove button: dark on light, shown while the card is hovered or focused, and
 *  always on a touch screen, which has no hover to reveal it. */
const AttachmentCardRemove: FC = () => {
  return (
    <AttachmentPrimitive.Remove asChild={true}>
      {/* No tooltip: the X says what it does, and a label popping over the next card is noise. */}
      <Button
        variant="ghost"
        size="icon"
        aria-label="Remove file"
        className="aui-attachment-card-remove absolute top-1.5 right-1.5 size-5 rounded-full bg-foreground p-0 opacity-0 shadow-sm transition-opacity hover:bg-foreground! focus-visible:opacity-100 group-hover/attachment-card:opacity-100 group-focus-within/attachment-card:opacity-100 [@media(pointer:coarse)]:opacity-100 [&_svg]:text-background"
      >
        <XIcon className="aui-attachment-remove-icon size-3 stroke-[2.5px]" />
      </Button>
    </AttachmentPrimitive.Remove>
  );
};

/** A file waiting in the composer, as a card: images fill it, everything else shows its kind
 *  and name. */
const ComposerAttachmentCard: FC = () => {
  const pastedText = usePastedTextAttachment();
  const src = useAttachmentImageSrc();
  const attachmentId = useAuiState(({ attachment }) => attachment.id);
  const { name, kind } = useAttachmentKind();

  if (pastedText) {
    return (
      <PastedTextAttachmentUI
        key={attachmentId}
        attachment={pastedText}
        isComposer={true}
        name={name || "Pasted text"}
        variant="card"
      />
    );
  }

  const label = attachmentKindLabel(kind, name);
  return (
    <AttachmentPrimitive.Root
      key={attachmentId}
      className={cn("aui-attachment-card group/attachment-card relative", CARD_SLOT)}
    >
      <AttachmentPreviewDialog redactFromReload={true}>
        <button
          className={cn(
            "aui-attachment-card-tile flex cursor-pointer overflow-hidden rounded-[18px] text-left transition-colors",
            CARD_SIZE,
            // An image fills the card edge to edge, so it needs no border or fill.
            !src && CARD_EDGE,
            !src && CARD_SURFACE,
          )}
          type="button"
          title={name}
          aria-label={name ? `${label} attachment: ${name}` : `${label} attachment`}
        >
          <CardImageOrBody name={name} kind={kind} src={src} />
        </button>
      </AttachmentPreviewDialog>
      <AttachmentCardRemove />
    </AttachmentPrimitive.Root>
  );
};

const SentAttachmentLayoutContext = createContext<SentAttachmentLayout>("list");

const NoAttachment: FC = () => null;

/** A sent image: a square thumbnail, smaller once the message collapses to chips. */
const SentImageTile: FC = () => {
  const layout = useContext(SentAttachmentLayoutContext);
  const attachmentId = useAuiState(({ attachment }) => attachment.id);
  const { name, kind } = useAttachmentKind();
  return (
    <AttachmentPrimitive.Root key={attachmentId} className="aui-attachment-root relative">
      <AttachmentPreviewDialog redactFromReload={false}>
        <button
          className={cn(
            "aui-attachment-image-tile block cursor-pointer overflow-hidden rounded-[18px] transition-opacity hover:opacity-85",
            layout === "list" ? SENT_IMAGE_SIZE : SENT_IMAGE_SIZE_COMPACT,
            CARD_EDGE,
          )}
          type="button"
          title={name}
          aria-label={name ? `Image attachment: ${name}` : "Image attachment"}
        >
          <SentImageThumb name={name} kind={kind} />
        </button>
      </AttachmentPreviewDialog>
    </AttachmentPrimitive.Root>
  );
};

const SentImageThumb: FC<{ name: string; kind: AttachmentFileKind }> = ({
  name,
  kind,
}) => {
  const src = useAttachmentImageSrc();
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
    <span className="flex h-full w-full items-center justify-center bg-muted">
      <AttachmentKindIcon kind={kind} className="size-6" />
    </span>
  );
};

/** A sent file: a row naming its kind, or a chip once the message collapses. */
const SentFileItem: FC = () => {
  const layout = useContext(SentAttachmentLayoutContext);
  const pastedText = usePastedTextAttachment();
  const attachmentId = useAuiState(({ attachment }) => attachment.id);
  const { name, kind } = useAttachmentKind();

  if (pastedText) {
    return (
      <PastedTextAttachmentUI
        key={attachmentId}
        attachment={pastedText}
        isComposer={false}
        name={name || "Pasted text"}
        variant={layout === "list" ? "row" : "pill"}
      />
    );
  }

  const label = attachmentKindLabel(kind, name);
  const accessibleName = name ? `${label} attachment: ${name}` : `${label} attachment`;
  return (
    <AttachmentPrimitive.Root
      key={attachmentId}
      className="aui-attachment-root relative max-w-full"
    >
      <AttachmentPreviewDialog redactFromReload={false}>
        {layout === "list" ? (
          <button
            className={cn(
              "aui-attachment-row flex max-w-full cursor-pointer items-center gap-3 rounded-[18px] px-4 py-3 text-left transition-colors",
              SENT_ROW_WIDTH,
              CARD_EDGE,
              "hover:bg-[color-mix(in_oklab,var(--foreground)_5%,transparent)]",
            )}
            type="button"
            title={name}
            aria-label={accessibleName}
          >
            <AttachmentKindIcon kind={kind} className="size-7" />
            <span className="flex min-w-0 flex-col">
              <span className="truncate font-medium text-sm">
                {name || label}
              </span>
              <span className="truncate text-muted-foreground text-xs">
                {label}
              </span>
            </span>
          </button>
        ) : (
          <button
            className={cn(
              "aui-attachment-chip inline-flex h-9 max-w-[calc(16rem*var(--ui-space-scale,1))] cursor-pointer items-center gap-1.5 rounded-full px-3 text-sm transition-colors",
              CARD_EDGE,
              "hover:bg-[color-mix(in_oklab,var(--foreground)_5%,transparent)]",
            )}
            type="button"
            title={name}
            aria-label={accessibleName}
          >
            <AttachmentKindIcon kind={kind} className="size-4" />
            <span className="truncate">{name || label}</span>
          </button>
        )}
      </AttachmentPreviewDialog>
    </AttachmentPrimitive.Root>
  );
};

// Module constants: the primitive re-renders every attachment when these change identity.
const SENT_IMAGE_COMPONENTS = {
  Image: SentImageTile,
  Document: NoAttachment,
  File: NoAttachment,
};
const SENT_FILE_COMPONENTS = {
  Image: NoAttachment,
  Document: SentFileItem,
  File: SentFileItem,
};
const COMPACT_COMPONENTS = { Attachment: AttachmentUI };
const CARD_COMPONENTS = { Attachment: ComposerAttachmentCard };

// Images lead as thumbnails and the other files follow as a list, as a sent message reads in
// most chat apps; the two passes pick each attachment by its type, so neither scans the other.
export const UserMessageAttachments: FC = () => {
  const setting = useAppearanceCustomStore(
    (s) => s.customization.sentAttachments,
  );
  const count = useAuiState(({ message }) =>
    message.role === "user" ? message.attachments.length : 0,
  );
  if (count === 0) return null;
  const layout = sentAttachmentLayout(setting, count);
  return (
    <SentAttachmentLayoutContext.Provider value={layout}>
      <div className="aui-user-message-attachments-end flex w-full flex-col items-end gap-2">
        <div className="aui-user-message-attachment-images flex max-w-full flex-row flex-wrap justify-end gap-2 empty:hidden">
          <MessagePrimitive.Attachments components={SENT_IMAGE_COMPONENTS} />
        </div>
        <div
          className={cn(
            "aui-user-message-attachment-files max-w-full empty:hidden",
            layout === "list"
              ? "flex flex-col items-end gap-2"
              : "flex flex-row flex-wrap justify-end gap-2",
          )}
        >
          <MessagePrimitive.Attachments components={SENT_FILE_COMPONENTS} />
        </div>
      </div>
    </SentAttachmentLayoutContext.Provider>
  );
};

// Cards wrap and the composer grows with them, up to two rows. Past that they collapse into one
// row that scrolls sideways with no scrollbar. The decision is written to the DOM as a data
// attribute, never to state: a resize or a new card re-lays out without re-rendering a card.
const ComposerAttachmentCards: FC = () => {
  const count = useAuiState(({ composer }) => composer.attachments.length);
  const ref = useRef<HTMLDivElement | null>(null);
  const previousCount = useRef(count);

  useLayoutEffect(() => {
    const el = ref.current;
    if (!el) return;
    const layout = () => {
      const card = el.firstElementChild as HTMLElement | null;
      const style = getComputedStyle(el);
      const width =
        el.clientWidth -
        Number.parseFloat(style.paddingLeft) -
        Number.parseFloat(style.paddingRight);
      const strip = composerAttachmentsOverflow(
        count,
        width,
        card?.getBoundingClientRect().width ?? 0,
        Number.parseFloat(style.columnGap) || 0,
      );
      const next = strip ? "strip" : "wrap";
      if (el.dataset.layout !== next) el.dataset.layout = next;
    };
    layout();
    // A card added to the strip lands past the right edge, so bring it into view.
    if (count > previousCount.current && el.dataset.layout === "strip") {
      el.scrollLeft = el.scrollWidth;
    }
    previousCount.current = count;
    const observer = new ResizeObserver(layout);
    observer.observe(el);
    // With no scrollbar, a mouse wheel is the only way along the strip for most mice.
    const onWheel = (event: WheelEvent) => {
      if (el.dataset.layout !== "strip") return;
      if (Math.abs(event.deltaY) <= Math.abs(event.deltaX)) return;
      const max = el.scrollWidth - el.clientWidth;
      const next = Math.min(max, Math.max(0, el.scrollLeft + event.deltaY));
      // At either end the page takes the wheel back.
      if (next === el.scrollLeft) return;
      event.preventDefault();
      el.scrollLeft = next;
    };
    el.addEventListener("wheel", onWheel, { passive: false });
    return () => {
      observer.disconnect();
      el.removeEventListener("wheel", onWheel);
    };
  }, [count]);

  return (
    <div
      ref={ref}
      data-reload-snapshot-sensitive
      className="aui-composer-attachments aui-composer-attachment-cards mb-4 flex w-full flex-row flex-wrap gap-2 px-1.5 pt-0.5 pb-1 empty:hidden data-[layout=strip]:flex-nowrap data-[layout=strip]:overflow-x-auto data-[layout=strip]:overscroll-x-contain"
    >
      <ComposerPrimitive.Attachments components={CARD_COMPONENTS} />
    </div>
  );
};

export const ComposerAttachments: FC = () => {
  const style = useAppearanceCustomStore(
    (s) => s.customization.composerAttachments,
  );
  if (style === "cards") return <ComposerAttachmentCards />;
  return (
    <div
      data-reload-snapshot-sensitive
      className="aui-composer-attachments mb-4 flex w-full flex-row items-center gap-2 overflow-x-auto px-1.5 pt-0.5 pb-1 empty:hidden"
    >
      <ComposerPrimitive.Attachments components={COMPACT_COMPONENTS} />
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
