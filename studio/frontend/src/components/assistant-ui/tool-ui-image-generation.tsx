// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"use client";

import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogTitle,
} from "@/components/ui/dialog";
import { Textarea } from "@/components/ui/textarea";
import { useChatRuntimeStore } from "@/features/chat";
import { cn } from "@/lib/utils";
import { type ToolCallMessagePartComponent, useAui } from "@assistant-ui/react";
import { DownloadIcon, ImageIcon, LoaderIcon, PencilIcon } from "lucide-react";
import type { ComponentProps, MouseEvent } from "react";
import { memo, useState } from "react";
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
}

interface ImageGenerationResult {
  image_b64?: string;
  image_mime?: string;
  size?: string;
  quality?: string;
  background?: string;
}

function GeneratedImagePlaceholder({ label }: { label: string }) {
  return (
    <div
      className={cn(
        "relative flex aspect-square w-[480px] max-w-full overflow-hidden rounded-2xl border border-border/70",
        "bg-gradient-to-br from-primary/15 via-muted to-background",
      )}
      aria-busy="true"
      aria-label={label}
    >
      <div className="absolute inset-0 animate-pulse bg-[linear-gradient(110deg,transparent_0%,rgba(255,255,255,0.24)_42%,transparent_70%)]" />
      <div className="absolute inset-0 bg-[radial-gradient(circle,rgba(148,163,184,0.36)_1px,transparent_1px)] [background-size:22px_22px] opacity-45" />
      <div className="relative z-10 m-auto flex flex-col items-center gap-3 rounded-2xl border border-border/60 bg-background/70 px-5 py-4 shadow-sm backdrop-blur-md">
        <LoaderIcon className="size-5 animate-spin text-primary" />
        <span className="text-sm font-medium text-foreground/85">{label}</span>
        <span className="text-xs text-muted-foreground">
          Preparing a 480×480 preview
        </span>
      </div>
    </div>
  );
}

const ImageGenerationToolUIImpl: ToolCallMessagePartComponent = ({
  args,
  result,
  status,
}) => {
  const aui = useAui();
  const setImageToolsEnabled = useChatRuntimeStore(
    (s) => s.setImageToolsEnabled,
  );
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
  const imagePart = imageSrc
    ? { type: "image" as const, image: imageSrc }
    : null;

  const [open, setOpen] = useState(true);
  const [dialogOpen, setDialogOpen] = useState(false);
  const [editText, setEditText] = useState("");

  const runningLabel = "Generating image…";
  const completedLabel = prompt
    ? prompt.length > 80
      ? `Generated image: ${prompt.slice(0, 80)}…`
      : `Generated image: ${prompt}`
    : "Generated image";

  const stopActionClick = (event: MouseEvent<HTMLButtonElement>) => {
    event.preventDefault();
    event.stopPropagation();
  };

  const handleDownload = (event: MouseEvent<HTMLButtonElement>) => {
    stopActionClick(event);
    if (imagePart) {
      downloadImagePart(imagePart);
    }
  };

  const handleEditClick = (event: MouseEvent<HTMLButtonElement>) => {
    stopActionClick(event);
    setDialogOpen(true);
  };

  const handleSubmitEdit: NonNullable<ComponentProps<"form">["onSubmit"]> = (
    event,
  ) => {
    event.preventDefault();
    const trimmed = editText.trim();
    if (!trimmed) {
      return;
    }
    setImageToolsEnabled(true);
    aui.thread().append({
      role: "user",
      content: [
        {
          type: "text",
          text: `Use the previous generated image as the reference and apply this edit: ${trimmed}. Preserve everything else exactly.`,
        },
      ],
      createdAt: new Date(),
    } as never);
    setEditText("");
    setDialogOpen(false);
  };

  return (
    <ToolFallbackRoot open={open} onOpenChange={setOpen}>
      <ToolFallbackTrigger
        toolName={isRunning ? runningLabel : completedLabel}
        status={status}
        icon={ImageIcon}
      />
      <ToolFallbackContent>
        {isRunning && !imagePart ? (
          <GeneratedImagePlaceholder label={runningLabel} />
        ) : imagePart ? (
          <figure className="m-0 flex flex-col gap-2">
            <Dialog open={dialogOpen} onOpenChange={setDialogOpen}>
              <div className="group/generated-image relative aspect-square w-[480px] max-w-full overflow-hidden rounded-2xl border border-border/70 bg-muted/30 shadow-sm">
                <button
                  type="button"
                  className="block size-full cursor-zoom-in overflow-hidden rounded-2xl focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:ring-offset-background"
                  onClick={() => setDialogOpen(true)}
                  aria-label="Open generated image preview"
                >
                  <Image.Preview
                    src={imagePart.image}
                    alt={prompt || "Generated image"}
                    containerClassName="size-full min-h-0 bg-background"
                    className="size-full object-contain"
                  />
                </button>
                <div className="pointer-events-none absolute inset-x-0 bottom-0 flex items-end justify-between gap-2 bg-gradient-to-t from-black/55 via-black/20 to-transparent p-3 opacity-100 transition-opacity sm:opacity-0 sm:group-hover/generated-image:opacity-100 sm:group-focus-within/generated-image:opacity-100">
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
              <DialogContent className="flex max-h-[calc(100vh-2rem)] max-w-[1100px] grid-rows-none flex-col gap-4 rounded-3xl p-4 sm:max-w-[1100px]">
                <DialogTitle className="sr-only">
                  Generated image preview
                </DialogTitle>
                <DialogDescription className="sr-only">
                  Preview the generated image and describe follow-up edits.
                </DialogDescription>
                <div className="flex min-h-0 flex-1 items-center justify-center overflow-hidden rounded-2xl bg-muted/30">
                  <img
                    src={imagePart.image}
                    alt={prompt || "Generated image"}
                    className="max-h-[min(1100px,calc(100vh-12rem))] max-w-full object-contain"
                  />
                </div>
                <form
                  onSubmit={handleSubmitEdit}
                  className="flex flex-col gap-2"
                >
                  <Textarea
                    value={editText}
                    onChange={(event) => setEditText(event.target.value)}
                    placeholder="Describe edits…"
                    fieldSizing="fixed"
                    className="min-h-20 resize-none rounded-2xl bg-background/80"
                  />
                  <div className="flex items-center justify-between gap-2">
                    <Button
                      type="button"
                      variant="outline"
                      size="sm"
                      onClick={() => imagePart && downloadImagePart(imagePart)}
                    >
                      <DownloadIcon className="size-4" />
                      Download
                    </Button>
                    <Button type="submit" size="sm" disabled={!editText.trim()}>
                      Apply edit
                    </Button>
                  </div>
                </form>
              </DialogContent>
            </Dialog>
            {prompt ? (
              <figcaption className="max-w-[480px] text-xs leading-snug text-muted-foreground">
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
