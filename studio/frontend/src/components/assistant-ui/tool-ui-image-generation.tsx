// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"use client";

import { Button } from "@/components/ui/button";
import { FIND_SKIP_ATTRIBUTE } from "@/features/find-in-page";
import { cn } from "@/lib/utils";
import type { ToolCallMessagePartComponent } from "@assistant-ui/react";
import { Download01Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { downloadUrl, isDownloadCancelled } from "@/lib/native-files";
import { toast } from "@/lib/toast";
import { ImageIcon, PencilIcon, VideoIcon } from "lucide-react";
import type { CSSProperties, MouseEvent } from "react";
import { memo, useCallback, useEffect, useRef, useState } from "react";
import { useGeneratedImageOverlay } from "./generated-image-overlay-context";
import { downloadImagePart } from "./image";
import {
  isSandboxToolResult,
  type SandboxFile,
  sandboxFilePath,
} from "./sandbox-files";
import { isToolCallRunning, toolArgText } from "./tool-arg-text";
import {
  ToolFallbackContent,
  ToolFallbackRoot,
  ToolFallbackTrigger,
} from "./tool-fallback";
import { useSandboxImage } from "./use-sandbox-image";

/**
 * Renders the synthetic `_toolEvent` chunks emitted by
 * `_stream_openai_responses` when OpenAI's Responses-API `image_generation`
 * tool fires. The backend stashes the base64 image on `image_b64` of the
 * tool_end event (keeping the JSON small); the adapter repackages it into a
 * structured `result` (image_b64, image_mime e.g. "image/png", size? e.g.
 * "1024x1024", quality?, background?).
 * The `tool_start` carries the revised prompt as `args.prompt` plus
 * `args.kind: "image"`. Without this, ToolFallback would print the prompt as
 * JSON with an empty Result block (the "no image" symptom).
 *
 * `generate_image` (the image model loaded in Unsloth Studio) shares this card. Its result is the
 * sandbox wrapper, so the image is fetched from the chat's sandbox, and it has no Edit: that needs
 * an OpenAI image-generation call id. A failed call answers with an "Error: ..." string.
 */
interface ImageGenerationArgs {
  // Straight off the wire; the provider decides the JSON type, not this file.
  prompt?: unknown;
  kind?: string;
  openai_image_generation_call_id?: unknown;
  openai_response_id?: unknown;
  openai_reasoning_item?: unknown;
}

/**
 * Straight off the wire, like the args above. Only `image_b64` is type-guarded
 * before use; `size`, `quality` and `background` are copied verbatim from the
 * provider (external_provider.py:5883) and on the Gemini path so is `image_mime`
 * (:4481). Unsloth takes a user-set `base_url`, so none of them is ours.
 */
interface ImageGenerationResult {
  image_b64?: string;
  image_mime?: unknown;
  size?: unknown;
  quality?: unknown;
  background?: unknown;
  prompt?: unknown;
}

type GeneratedImagePart = {
  type: "image";
  image: string;
  filename?: string;
};

type SandboxMediaResult = {
  text: string;
  sessionId: string;
  images: unknown[];
  files?: SandboxFile[];
};

const sandboxMediaResult = (result: unknown): SandboxMediaResult | null =>
  isSandboxToolResult(result) ? (result as SandboxMediaResult) : null;

const toolErrorText = (result: unknown): string =>
  typeof result === "string" && result.startsWith("Error:") ? result : "";

const baseName = (name: string): string => name.split("/").pop() || name;

const CAPTION_COLLAPSED_LINES = 4;
const INLINE_IMAGE_MAX_WIDTH = 520;
const INLINE_IMAGE_MAX_HEIGHT = 620;

const extensionForMime = (mime: string): string => {
  switch (mime.toLowerCase()) {
    case "image/jpeg":
    case "image/jpg":
      return "jpg";
    case "image/webp":
      return "webp";
    case "image/gif":
      return "gif";
    case "image/svg+xml":
      return "svg";
    default:
      return "png";
  }
};

const imageFilenameFromPrompt = (prompt: string, mime: string): string => {
  const slug = prompt
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-|-$/g, "")
    .slice(0, 48);
  return `${slug || "generated-image"}.${extensionForMime(mime)}`;
};

const formatGeneratedImageLabel = (prompt: string): string => {
  if (!prompt) {
    return "Generated image";
  }
  return prompt.length > 80
    ? `Generated image: ${prompt.slice(0, 80)}…`
    : `Generated image: ${prompt}`;
};

// Takes text: `size?.match` guards nullish only, so `"size": 1024` reached `.match` on a number.
const parseImageSize = (
  size: string,
): { width: number; height: number } | null => {
  const match = size.match(/^(\d+)x(\d+)$/i);
  if (!match) return null;
  const width = Number(match[1]);
  const height = Number(match[2]);
  return width > 0 && height > 0 ? { width, height } : null;
};

const getInlineImageFrameWidth = ({
  width,
  height,
}: {
  width: number;
  height: number;
}): number =>
  Math.round(
    Math.min(
      width,
      INLINE_IMAGE_MAX_WIDTH,
      (INLINE_IMAGE_MAX_HEIGHT * width) / height,
    ),
  );

const loadingDots = Array.from({ length: 64 }, (_, index) => {
  const row = Math.floor(index / 8);
  const col = index % 8;
  return (
    <span
      key={index}
      className="generated-image-loading-dot"
      style={
        {
          "--dot-row": row,
          "--dot-col": col,
        } as CSSProperties
      }
    />
  );
});

/**
 * A generate_image or generate_video result: the file in the chat's sandbox, fetched with auth,
 * with Download and no Edit. The ref sits on the frame so the lazy fetch starts near the viewport.
 */
function SandboxMediaFigure({
  kind,
  url,
  filename,
  title,
  caption,
}: {
  kind: "image" | "video";
  url: string;
  filename: string;
  title: string;
  caption: string;
}) {
  const { ref, state } = useSandboxImage<HTMLDivElement>(url);
  const loaded = state.status === "loaded" ? state.url : null;
  const handleDownload = () => {
    if (!loaded) return;
    if (kind === "image") {
      downloadImagePart({ image: loaded, filename });
      return;
    }
    void downloadUrl(loaded, filename).catch((error) => {
      if (!isDownloadCancelled(error)) toast.error("Could not save video.");
    });
  };
  return (
    <figure
      className="m-0 flex max-w-full flex-col items-start gap-2 align-top"
      style={{ width: INLINE_IMAGE_MAX_WIDTH, maxWidth: "100%" }}
    >
      <div
        ref={ref}
        className="group/generated-image relative w-full overflow-hidden rounded-2xl align-top"
      >
        {state.status === "failed" ? (
          <p className="text-xs leading-5 text-muted-foreground">
            {kind === "image" ? "Image not available" : "Video not available"}
          </p>
        ) : !loaded ? (
          <div
            aria-busy="true"
            aria-label={kind === "image" ? "Loading image…" : "Loading video…"}
            className="aspect-square w-full rounded-2xl bg-muted/20"
          />
        ) : kind === "image" ? (
          <img
            src={loaded}
            alt={title}
            className="block h-auto max-h-[min(70dvh,620px)] max-w-full rounded-2xl object-contain"
          />
        ) : (
          <video
            src={loaded}
            controls={true}
            playsInline={true}
            preload="metadata"
            aria-label={title}
            className="block h-auto max-h-[min(70dvh,620px)] w-full rounded-2xl"
          />
        )}
        {loaded ? (
          <div
            {...{ [FIND_SKIP_ATTRIBUTE]: "" }}
            className="pointer-events-none absolute top-0 right-0 z-20 p-3 opacity-100 transition-opacity sm:opacity-0 sm:group-hover/generated-image:opacity-100 sm:group-focus-within/generated-image:opacity-100"
          >
            <Button
              type="button"
              variant="dark"
              size="icon-sm"
              className="pointer-events-auto rounded-full bg-black/70 text-white hover:bg-black/85"
              onClick={handleDownload}
              aria-label={`Download generated ${kind}`}
            >
              <HugeiconsIcon icon={Download01Icon} className="size-4" />
            </Button>
          </div>
        ) : null}
      </div>
      {caption ? (
        <figcaption className="line-clamp-4 w-full whitespace-pre-wrap break-words text-xs leading-5 text-muted-foreground">
          {caption}
        </figcaption>
      ) : null}
    </figure>
  );
}

function GeneratedImagePlaceholder({ label }: { label: string }) {
  return (
    <div
      className={cn(
        "generated-image-loading-card flex aspect-square w-[480px] max-w-full items-center justify-center rounded-2xl bg-muted/20 shadow-[0_0_12px_rgba(15,23,42,0.05),0_6px_18px_rgba(15,23,42,0.04)] dark:shadow-[0_0_12px_var(--background),0_6px_18px_var(--background)]",
      )}
      aria-busy="true"
      aria-label={label}
      aria-live="polite"
    >
      <span className="sr-only">{label}</span>
      <div className="generated-image-loading-wave" aria-hidden={true}>
        {loadingDots}
      </div>
    </div>
  );
}

const ImageGenerationToolUIImpl: ToolCallMessagePartComponent = ({
  args,
  result,
  status,
}) => {
  const { openOverlay } = useGeneratedImageOverlay();
  const parsedArgs = (args as ImageGenerationArgs) ?? {};
  const prompt = toolArgText(parsedArgs.prompt);
  const isRunning = isToolCallRunning(status);

  const isImageResult =
    !!result &&
    typeof result === "object" &&
    typeof (result as ImageGenerationResult).image_b64 === "string";
  const imageResult = isImageResult ? (result as ImageGenerationResult) : null;
  const localResult = isImageResult ? null : sandboxMediaResult(result);
  const localImageName = localResult?.images.find(
    (name): name is string => typeof name === "string" && name.length > 0,
  );
  const errorText = toolErrorText(result);
  // Coerced for the same reason `prompt` is: these reach `.match`,
  // `.toLowerCase` and `Array.join`, all of which throw on a non-string.
  const size = toolArgText(imageResult?.size);
  const quality = toolArgText(imageResult?.quality);
  const imageDimensions = parseImageSize(size);
  const imageFrameStyle: CSSProperties = {
    width: imageDimensions
      ? getInlineImageFrameWidth(imageDimensions)
      : INLINE_IMAGE_MAX_WIDTH,
    maxWidth: "100%",
  };
  const imageBoxStyle: CSSProperties | undefined = imageDimensions
    ? { aspectRatio: `${imageDimensions.width} / ${imageDimensions.height}` }
    : undefined;
  const mime = toolArgText(imageResult?.image_mime) || "image/png";
  const imageSrc = imageResult?.image_b64
    ? `data:${mime};base64,${imageResult.image_b64}`
    : null;
  const resultPrompt = toolArgText(imageResult?.prompt).trim();
  const imageTitle = resultPrompt || prompt.trim() || "Generated image";
  const captionPrompt = resultPrompt || prompt.trim();
  const promptLikelyNeedsExpansion = captionPrompt.length > 220;
  const imageMetadata = [size, quality, mime].filter(Boolean).join(" · ");
  const openaiImageGenerationCallId =
    typeof parsedArgs.openai_image_generation_call_id === "string"
      ? parsedArgs.openai_image_generation_call_id
      : undefined;
  const openaiResponseId =
    typeof parsedArgs.openai_response_id === "string"
      ? parsedArgs.openai_response_id
      : undefined;
  const imagePart: GeneratedImagePart | null = imageSrc
    ? {
        type: "image",
        image: imageSrc,
        filename: imageFilenameFromPrompt(prompt, mime),
      }
    : null;

  const [open, setOpen] = useState(true);
  const [expandedCaptionPrompt, setExpandedCaptionPrompt] = useState<
    string | null
  >(null);
  const [promptOverflow, setPromptOverflow] = useState<{
    prompt: string;
    canExpand: boolean;
  } | null>(null);
  const captionRef = useRef<HTMLDivElement | null>(null);
  const isPendingImage = !imagePart && !localImageName && isRunning;

  const promptOverflowMeasured = promptOverflow?.prompt === captionPrompt;
  const promptCanExpand = promptOverflowMeasured
    ? promptOverflow.canExpand
    : false;
  const promptExpanded = expandedCaptionPrompt === captionPrompt;

  const updatePromptOverflow = useCallback(() => {
    const captionElement = captionRef.current;
    if (!captionElement || !captionPrompt) {
      return;
    }
    const computedStyle = window.getComputedStyle(captionElement);
    const lineHeight = Number.parseFloat(computedStyle.lineHeight);
    const collapsedHeight =
      (Number.isFinite(lineHeight) ? lineHeight : 20) * CAPTION_COLLAPSED_LINES;
    const hasOverflow = captionElement.scrollHeight > collapsedHeight + 1;
    setPromptOverflow((current) =>
      current?.prompt === captionPrompt && current.canExpand === hasOverflow
        ? current
        : { prompt: captionPrompt, canExpand: hasOverflow },
    );
  }, [captionPrompt]);

  useEffect(() => {
    const captionElement = captionRef.current;
    if (!captionElement || !captionPrompt) {
      return;
    }
    const frame = window.requestAnimationFrame(updatePromptOverflow);
    const resizeObserver =
      typeof ResizeObserver === "undefined"
        ? null
        : new ResizeObserver(updatePromptOverflow);
    resizeObserver?.observe(captionElement);
    window.addEventListener("resize", updatePromptOverflow);
    return () => {
      window.cancelAnimationFrame(frame);
      resizeObserver?.disconnect();
      window.removeEventListener("resize", updatePromptOverflow);
    };
  }, [captionPrompt, updatePromptOverflow]);

  const shouldClampPrompt =
    (promptOverflowMeasured ? promptCanExpand : promptLikelyNeedsExpansion) &&
    !promptExpanded;

  const runningLabel = "Generating image…";
  const completedLabel = formatGeneratedImageLabel(prompt);

  const showPreview = () => {
    if (!imagePart) {
      return;
    }
    openOverlay({
      image: imagePart.image,
      title: imageTitle,
      metadata: imageMetadata,
      filename: imagePart.filename,
      openaiImageGenerationCallId,
      openaiResponseId,
      openaiReasoningItem: parsedArgs.openai_reasoning_item,
    });
  };

  const stopOverlayActionPropagation = (
    event: MouseEvent<HTMLButtonElement>,
  ) => {
    event.preventDefault();
    event.stopPropagation();
  };

  const handleDownload = (event: MouseEvent<HTMLButtonElement>) => {
    stopOverlayActionPropagation(event);
    if (imagePart) {
      downloadImagePart(imagePart);
    }
  };

  const handleEditClick = (event: MouseEvent<HTMLButtonElement>) => {
    stopOverlayActionPropagation(event);
    showPreview();
  };

  if (isPendingImage) {
    return (
      <div className="aui-tool-fallback-root w-full py-1">
        <GeneratedImagePlaceholder label={runningLabel} />
      </div>
    );
  }

  return (
    <ToolFallbackRoot open={open} onOpenChange={setOpen}>
      <ToolFallbackTrigger
        toolName={isRunning ? runningLabel : completedLabel}
        status={status}
        icon={ImageIcon}
      />
      <ToolFallbackContent>
        {errorText ? (
          <p className="text-xs leading-5 text-muted-foreground">{errorText}</p>
        ) : null}
        {localResult && localImageName ? (
          <SandboxMediaFigure
            kind="image"
            url={sandboxFilePath(localResult.sessionId, localImageName)}
            filename={baseName(localImageName)}
            title={imageTitle}
            caption={captionPrompt}
          />
        ) : null}
        {imagePart ? (
          <figure
            className="m-0 flex max-w-full flex-col items-start gap-2 align-top"
            style={imageFrameStyle}
          >
            <div
              className="group/generated-image relative w-full overflow-hidden rounded-2xl align-top"
              style={imageBoxStyle}
            >
              <button
                type="button"
                className={cn(
                  "block cursor-zoom-in rounded-2xl focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-inset focus-visible:ring-ring",
                  imageDimensions ? "size-full" : "max-w-full",
                )}
                onClick={showPreview}
                aria-label="Open generated image preview"
              >
                <img
                  src={imagePart.image}
                  alt={imageTitle}
                  width={imageDimensions?.width}
                  height={imageDimensions?.height}
                  className={cn(
                    "block rounded-2xl object-contain",
                    imageDimensions
                      ? "size-full"
                      : "h-auto max-h-[min(70dvh,620px)] max-w-full",
                  )}
                />
              </button>
              <div
                // Out of find-in-page's reach, like the response model badge: from `sm` up these
                // actions are transparent until the card is hovered, so "Edit" would be counted and
                // walked to under a highlight nobody can see. The index leaves opacity alone, since
                // every user message fades in and reading it would drop one mid-fade. Marked whole
                // rather than per breakpoint, which costs finding a button label on a phone.
                {...{ [FIND_SKIP_ATTRIBUTE]: "" }}
                className="pointer-events-none absolute inset-x-0 bottom-0 z-20 flex items-end justify-between gap-2 bg-gradient-to-t from-black/55 via-black/20 to-transparent p-3 opacity-100 transition-opacity sm:opacity-0 sm:group-hover/generated-image:opacity-100 sm:group-focus-within/generated-image:opacity-100"
              >
                <Button
                  type="button"
                  variant="dark"
                  size="sm"
                  className="pointer-events-auto h-8 rounded-full bg-black/70 text-white hover:bg-black/85"
                  onClick={handleEditClick}
                >
                  <PencilIcon className="size-3.5" />
                  Edit
                </Button>
                <Button
                  type="button"
                  variant="dark"
                  size="icon-sm"
                  className="pointer-events-auto rounded-full bg-black/70 text-white hover:bg-black/85"
                  onClick={handleDownload}
                  aria-label="Download generated image"
                >
                  <HugeiconsIcon icon={Download01Icon} className="size-4" />
                </Button>
              </div>
            </div>
            {captionPrompt ? (
              <figcaption className="w-full text-xs leading-5 text-muted-foreground">
                <div
                  ref={captionRef}
                  className={cn(
                    "whitespace-pre-wrap break-words",
                    shouldClampPrompt && "max-h-20 overflow-hidden",
                  )}
                >
                  {captionPrompt}
                </div>
                {promptCanExpand ? (
                  <button
                    type="button"
                    className="mt-2 inline-flex text-xs font-medium text-foreground/80 underline-offset-4 hover:text-foreground hover:underline focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
                    onClick={() =>
                      setExpandedCaptionPrompt((value) =>
                        value === captionPrompt ? null : captionPrompt,
                      )
                    }
                    aria-expanded={promptExpanded}
                  >
                    {promptExpanded ? "Show less" : "Show more"}
                  </button>
                ) : null}
              </figcaption>
            ) : null}
          </figure>
        ) : null}
      </ToolFallbackContent>
    </ToolFallbackRoot>
  );
};

export const ImageGenerationToolUI = memo(
  ImageGenerationToolUIImpl,
) as unknown as ToolCallMessagePartComponent;
ImageGenerationToolUI.displayName = "ImageGenerationToolUI";

const VIDEO_FILE = /\.(mp4|webm|mov)$/i;

/** `generate_video`: the .mp4 the video model loaded in Unsloth Studio wrote into the chat's sandbox. */
const VideoGenerationToolUIImpl: ToolCallMessagePartComponent = ({
  args,
  result,
  status,
}) => {
  const prompt = toolArgText((args as { prompt?: unknown })?.prompt).trim();
  const isRunning = isToolCallRunning(status);
  const localResult = sandboxMediaResult(result);
  const videoFile = localResult?.files?.find((file) =>
    VIDEO_FILE.test(file.name),
  );
  const [open, setOpen] = useState(true);
  const errorText = toolErrorText(result);
  const runningLabel = "Generating video…";

  if (!videoFile && isRunning) {
    return (
      <div className="aui-tool-fallback-root w-full py-1">
        <GeneratedImagePlaceholder label={runningLabel} />
      </div>
    );
  }
  const completedLabel = errorText
    ? "Video generation failed"
    : prompt
      ? `Generated video: ${prompt.length > 80 ? `${prompt.slice(0, 80)}…` : prompt}`
      : "Generated video";

  return (
    <ToolFallbackRoot open={open} onOpenChange={setOpen}>
      <ToolFallbackTrigger
        toolName={isRunning ? runningLabel : completedLabel}
        status={status}
        icon={VideoIcon}
      />
      <ToolFallbackContent>
        {errorText ? (
          <p className="text-xs leading-5 text-muted-foreground">{errorText}</p>
        ) : null}
        {localResult && videoFile ? (
          <SandboxMediaFigure
            kind="video"
            url={sandboxFilePath(localResult.sessionId, videoFile.name)}
            filename={baseName(videoFile.name)}
            title={prompt || "Generated video"}
            caption={prompt}
          />
        ) : null}
      </ToolFallbackContent>
    </ToolFallbackRoot>
  );
};

export const VideoGenerationToolUI = memo(
  VideoGenerationToolUIImpl,
) as unknown as ToolCallMessagePartComponent;
VideoGenerationToolUI.displayName = "VideoGenerationToolUI";
