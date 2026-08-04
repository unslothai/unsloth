// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"use client";

import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import type { ToolCallMessagePartComponent } from "@assistant-ui/react";
import { ImageIcon, PencilIcon } from "lucide-react";
import { Download01Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import type { CSSProperties, MouseEvent } from "react";
import { memo, useCallback, useEffect, useRef, useState } from "react";
import { useGeneratedImageOverlay } from "./generated-image-overlay-context";
import { downloadImagePart } from "./image";
import {
  ToolFallbackContent,
  ToolFallbackRoot,
  ToolFallbackTrigger,
} from "./tool-fallback";

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
 */
interface ImageGenerationArgs {
  prompt?: string;
  kind?: string;
  openai_image_generation_call_id?: unknown;
  openai_response_id?: unknown;
  openai_reasoning_item?: unknown;
}

interface ImageGenerationResult {
  image_b64?: string;
  image_mime?: string;
  size?: string;
  quality?: string;
  background?: string;
  prompt?: string;
}

type GeneratedImagePart = {
  type: "image";
  image: string;
  filename?: string;
};

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

const parseImageSize = (
  size?: string,
): { width: number; height: number } | null => {
  const match = size?.match(/^(\d+)x(\d+)$/i);
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

function GeneratedImagePlaceholder({ label }: { label: string }) {
  return (
    <div
      className={cn(
        "generated-image-loading-card flex aspect-square w-[480px] max-w-full items-center justify-center rounded-2xl bg-muted/20 shadow-[0_0_12px_rgba(15,23,42,0.05),0_6px_18px_rgba(15,23,42,0.04)] dark:shadow-[0_0_12px_rgba(0,0,0,0.18),0_6px_18px_rgba(0,0,0,0.12)]",
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
  const prompt = parsedArgs.prompt ?? "";
  const isRunning = status?.type === "running";

  const isImageResult =
    !!result &&
    typeof result === "object" &&
    typeof (result as ImageGenerationResult).image_b64 === "string";
  const imageResult = isImageResult ? (result as ImageGenerationResult) : null;
  const imageDimensions = parseImageSize(imageResult?.size);
  const imageFrameStyle: CSSProperties = {
    width: imageDimensions
      ? getInlineImageFrameWidth(imageDimensions)
      : INLINE_IMAGE_MAX_WIDTH,
    maxWidth: "100%",
  };
  const imageBoxStyle: CSSProperties | undefined = imageDimensions
    ? { aspectRatio: `${imageDimensions.width} / ${imageDimensions.height}` }
    : undefined;
  const mime = imageResult?.image_mime || "image/png";
  const imageSrc = imageResult?.image_b64
    ? `data:${mime};base64,${imageResult.image_b64}`
    : null;
  const imageTitle =
    imageResult?.prompt?.trim() || prompt.trim() || "Generated image";
  const captionPrompt = imageResult?.prompt?.trim() || prompt.trim();
  const promptLikelyNeedsExpansion = captionPrompt.length > 220;
  const imageMetadata = [imageResult?.size, imageResult?.quality, mime]
    .filter(Boolean)
    .join(" · ");
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
  const isPendingImage = !imagePart && status?.type === "running";

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
                      : "h-auto max-h-[min(70vh,620px)] max-w-full",
                  )}
                />
              </button>
              <div className="pointer-events-none absolute inset-x-0 bottom-0 z-20 flex items-end justify-between gap-2 bg-gradient-to-t from-black/55 via-black/20 to-transparent p-3 opacity-100 transition-opacity sm:opacity-0 sm:group-hover/generated-image:opacity-100 sm:group-focus-within/generated-image:opacity-100">
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
