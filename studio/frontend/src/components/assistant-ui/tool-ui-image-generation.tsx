// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"use client";

import { type ToolCallMessagePartComponent, useAuiState } from "@assistant-ui/react";
import { ImageIcon, LoaderIcon } from "lucide-react";
import { memo, useEffect, useState } from "react";
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
}

interface ImageGenerationResult {
  image_b64?: string;
  image_mime?: string;
  size?: string;
  quality?: string;
  background?: string;
}

const ImageGenerationToolUIImpl: ToolCallMessagePartComponent = ({
  args,
  result,
  status,
}) => {
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

  // Collapse the card once the model has resumed streaming prose
  // after the image. Mirrors CodeExecutionToolUI so the inline image
  // doesn't collapse mid-stream and the user can click to re-expand.
  const hasText = useAuiState(({ message }) =>
    message.content.some(
      (p) =>
        p.type === "text" &&
        "text" in p &&
        (p as { text: string }).text.length > 0,
    ),
  );
  const [open, setOpen] = useState(true);
  useEffect(() => {
    if (isRunning) {
      setOpen(true);
    } else if (hasText && !imageSrc) {
      setOpen(false);
    }
  }, [isRunning, hasText, imageSrc]);

  const runningLabel = "Generating image…";
  const completedLabel = prompt
    ? prompt.length > 80
      ? `Generated image: ${prompt.slice(0, 80)}…`
      : `Generated image: ${prompt}`
    : "Generated image";

  return (
    <ToolFallbackRoot open={open} onOpenChange={setOpen}>
      <ToolFallbackTrigger
        toolName={isRunning ? runningLabel : completedLabel}
        status={status}
        icon={ImageIcon}
      />
      <ToolFallbackContent>
        {isRunning && !imageSrc ? (
          <div className="flex items-center gap-2 text-sm text-muted-foreground">
            <LoaderIcon className="size-3.5 animate-spin" />
            <span>{runningLabel}</span>
          </div>
        ) : imageSrc ? (
          <figure className="m-0 flex flex-col gap-1.5">
            <img
              src={imageSrc}
              alt={prompt || "Generated image"}
              className="max-w-full rounded-md border border-border/60"
            />
            {prompt ? (
              <figcaption className="text-xs leading-snug text-muted-foreground">
                {prompt}
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
