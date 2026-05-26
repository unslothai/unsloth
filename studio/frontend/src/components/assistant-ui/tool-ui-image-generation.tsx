// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"use client";

import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import type { ToolCallMessagePartComponent } from "@assistant-ui/react";
import { DownloadIcon, ImageIcon, PencilIcon } from "lucide-react";
import type { CSSProperties, MouseEvent } from "react";
import { memo, useCallback, useEffect, useRef, useState } from "react";
import { useGeneratedImageOverlay } from "./generated-image-overlay-context";
import { Image, downloadImagePart } from "./image";
import {
  ToolFallbackContent,
  ToolFallbackRoot,
  ToolFallbackTrigger,
} from "./tool-fallback";

/**
 * Renders the synthetic `_toolEvent` chunks emitted by
 * `_stream_openai_responses` when OpenAI's Responses-API
 * `image_generation` tool fires. The backend stashes the base64
 * PNG/WebP/JPEG (the gpt-image backbone output) on an `image_b64`
 * field of the tool_end event so the JSON result stays small, and the
 * adapter repackages it into a structured `result` shape:
 *
 *   {
 *     image_b64: string,
 *     image_mime: string,        // e.g. "image/png"
 *     size?: string,             // "1024x1024" etc
 *     quality?: string,
 *     background?: string,
 *   }
 *
 * The corresponding `tool_start` carries the prompt as
 * `args.prompt` (after gpt-image's revision pass) plus `args.kind:
 * "image"`. Without this component the generic ToolFallback would
 * print the prompt as JSON args text with an empty Result block --
 * which is exactly the "no image" symptom users hit before this UI
 * landed.
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
    const rootFontSize = Number.parseFloat(
      window.getComputedStyle(document.documentElement).fontSize,
    );
    const collapsedHeight =
      (Number.isFinite(rootFontSize) ? rootFontSize : 16) * 4.75;
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
          <figure className="m-0 flex flex-col gap-2">
            <div className="group/generated-image relative aspect-square w-[480px] max-w-full overflow-hidden rounded-2xl bg-muted/25 shadow-lg shadow-foreground/5 dark:shadow-black/25">
              <img
                src={imagePart.image}
                alt=""
                aria-hidden={true}
                className="pointer-events-none absolute inset-0 size-full scale-110 object-cover opacity-25 blur-2xl saturate-125"
              />
              <div className="pointer-events-none absolute inset-0 bg-background/45" />
              <button
                type="button"
                className="relative z-10 block size-full cursor-zoom-in overflow-hidden rounded-2xl focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:ring-offset-background"
                onClick={showPreview}
                aria-label="Open generated image preview"
              >
                <Image.Preview
                  src={imagePart.image}
                  alt={imageTitle}
                  containerClassName="flex size-full min-h-0 items-center justify-center bg-transparent"
                  className="size-full object-contain"
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
                  <DownloadIcon className="size-4" />
                </Button>
              </div>
            </div>
            {captionPrompt ? (
              <figcaption className="max-w-[480px] text-xs leading-snug text-muted-foreground">
                <div
                  ref={captionRef}
                  className={cn(
                    "whitespace-pre-wrap break-words",
                    shouldClampPrompt && "max-h-[4.75rem] overflow-hidden",
                  )}
                >
                  {captionPrompt}
                </div>
                {promptCanExpand ? (
                  <button
                    type="button"
                    className="mt-1.5 inline-flex text-xs font-medium text-foreground/80 underline-offset-4 hover:text-foreground hover:underline focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:ring-offset-background"
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
