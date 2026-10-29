// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type React from "react";
import { useCallback, useEffect, useRef, useState } from "react";
import { Delete02Icon, ImageCropIcon, PlusSignIcon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";

import { Button } from "@/components/ui/button";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import {
  readNativeAttachmentFile,
  registerNativeAttachmentPath,
  useNativeDropTarget,
} from "@/features/native-intents";
import { useImageCrop } from "@/hooks/use-image-crop";
import { GALLERY_ITEM_DRAG_TYPE } from "@/hooks/use-strip-reorder";
import { MAX_UPLOAD_IMAGE_SIDE, fitImageDataUrl } from "@/lib/image-fit";
import { cn } from "@/lib/utils";
import { toast } from "@/lib/toast";

const NATIVE_IMAGE_EXTS = ["jpg", "jpeg", "png", "webp", "gif"];

type GalleryImageSource = (id: string) => Promise<Blob | null>;
let gallerySource: GalleryImageSource | null = null;

/** Lets every ImageDropzone on the page accept a tile dragged from this page's gallery strip. */
export function useGalleryImageDropSource(source: GalleryImageSource) {
  useEffect(() => {
    gallerySource = source;
    return () => {
      if (gallerySource === source) gallerySource = null;
    };
  }, [source]);
}

function isGalleryDrag(event: React.DragEvent): boolean {
  return gallerySource !== null && event.dataTransfer.types.includes(GALLERY_ITEM_DRAG_TYPE);
}

/** Shared image picker that returns a data URL. */
export function ImageDropzone({
  value,
  onChange,
  label = "Click or drop an image",
  removeLabel = "Remove source image",
  className,
  maxSide = MAX_UPLOAD_IMAGE_SIDE,
  cropSubject,
}: {
  value: string | null;
  onChange: (dataUrl: string | null) => void;
  label?: string;
  removeLabel?: string;
  className?: string;
  /** Larger uploads are downscaled to this per side; null keeps them as chosen. */
  maxSide?: number | null;
  cropSubject?: string;
}) {
  const inputRef = useRef<HTMLInputElement | null>(null);
  const [dragging, setDragging] = useState(false);
  // Only the newest selection may update the shared field.
  const selection = useRef(0);
  // `onChange` is shared, so a read outliving this picker would land on whoever holds it.
  const mounted = useRef(true);
  useEffect(() => {
    mounted.current = true;
    return () => {
      mounted.current = false;
    };
  }, []);
  // Slots are keyed by index; a `value` this picker did not set invalidates a read in flight.
  const seen = useRef(value);
  useEffect(() => {
    if (seen.current === value) return;
    seen.current = value;
    selection.current += 1;
  }, [value]);

  const readFile = useCallback(
    (file: Blob | undefined | null) => {
      if (!file || !file.type.startsWith("image/")) {
        if (file) toast.error("Please choose an image file");
        return;
      }
      selection.current += 1;
      const claimed = selection.current;
      const reader = new FileReader();
      reader.onload = async () => {
        if (claimed !== selection.current) return;
        if (typeof reader.result !== "string") {
          onChange(null);
          return;
        }
        const dataUrl = maxSide ? await fitImageDataUrl(reader.result, maxSide) : reader.result;
        if (!mounted.current || claimed !== selection.current) return;
        onChange(dataUrl);
      };
      reader.onerror = () => toast.error("Could not read the image");
      reader.readAsDataURL(file);
    },
    [onChange, maxSide],
  );

  const readGalleryItem = useCallback(
    async (id: string) => {
      const source = gallerySource;
      if (!source) return;
      selection.current += 1;
      const claimed = selection.current;
      try {
        const blob = await source(id);
        if (!blob || !mounted.current || claimed !== selection.current) return;
        readFile(blob);
      } catch (error) {
        toast.error("Could not read the image", {
          description: error instanceof Error ? error.message : String(error),
        });
      }
    },
    [readFile],
  );

  // Tauri suppresses the webview drop event, so desktop drops arrive as paths
  // that the native side registers and reads for this picker.
  const readNativePath = useCallback(
    async (path: string | undefined) => {
      if (!path) return;
      if (!NATIVE_IMAGE_EXTS.includes(path.split(".").pop()?.toLowerCase() ?? "")) {
        toast.error("Drop a JPEG, PNG, WebP or GIF image", {
          description: "Other image formats can still be chosen with the picker.",
        });
        return;
      }
      selection.current += 1;
      const claimed = selection.current;
      try {
        const intent = await registerNativeAttachmentPath(path);
        const file = await readNativeAttachmentFile(intent.path.token);
        if (!mounted.current || claimed !== selection.current) return;
        const raw = `data:${file.mimeType};base64,${file.base64}`;
        const dataUrl = maxSide ? await fitImageDataUrl(raw, maxSide) : raw;
        if (!mounted.current || claimed !== selection.current) return;
        onChange(dataUrl);
      } catch (error) {
        toast.error("Could not read the image", {
          description: error instanceof Error ? error.message : String(error),
        });
      }
    },
    [onChange, maxSide],
  );

  const applyCrop = useCallback(
    (dataUrl: string) => {
      selection.current += 1;
      onChange(dataUrl);
    },
    [onChange],
  );
  const { openCrop, cropDialog } = useImageCrop(value, applyCrop, cropSubject ?? "");

  const nativeDropRef = useNativeDropTarget({
    onDrop: (paths) => void readNativePath(paths[0]),
    onDragOver: setDragging,
  });

  if (value) {
    return (
      <div
        data-reload-snapshot-sensitive
        className={cn("relative overflow-hidden rounded-lg border border-border", className)}
      >
        {/* Checkerboard behind the preview, so a transparent upload shows its alpha. */}
        <img
          src={value}
          alt="Source"
          className="max-h-44 w-full object-contain bg-muted/30"
          style={{
            backgroundImage:
              "repeating-conic-gradient(rgb(128 128 128 / 0.28) 0% 25%, transparent 0% 50%)",
            backgroundSize: "16px 16px",
          }}
        />
        {cropSubject && (
          <>
            <Tooltip>
              <TooltipTrigger asChild={true}>
                <Button
                  type="button"
                  variant="secondary"
                  size="icon"
                  aria-label={`Crop ${cropSubject.charAt(0).toLowerCase()}${cropSubject.slice(1)}`}
                  className="absolute right-10 top-1.5 size-7"
                  onClick={openCrop}
                >
                  <HugeiconsIcon icon={ImageCropIcon} className="size-3.5" />
                </Button>
              </TooltipTrigger>
              <TooltipContent>Crop</TooltipContent>
            </Tooltip>
            {cropDialog}
          </>
        )}
        <Tooltip>
          <TooltipTrigger asChild={true}>
            <Button
              type="button"
              variant="secondary"
              size="icon"
              aria-label={removeLabel}
              className="absolute right-1.5 top-1.5 size-7"
              onClick={() => {
                selection.current += 1;
                onChange(null);
                if (inputRef.current) inputRef.current.value = "";
              }}
            >
              <HugeiconsIcon icon={Delete02Icon} className="size-3.5" />
            </Button>
          </TooltipTrigger>
          <TooltipContent>Remove</TooltipContent>
        </Tooltip>
      </div>
    );
  }

  return (
    <button
      type="button"
      ref={nativeDropRef}
      onClick={() => inputRef.current?.click()}
      onDragOver={(e) => {
        e.preventDefault();
        // The strip offers copy or move; without this the cursor reads as a move.
        if (isGalleryDrag(e)) e.dataTransfer.dropEffect = "copy";
        setDragging(true);
      }}
      onDragLeave={() => setDragging(false)}
      onDrop={(e) => {
        e.preventDefault();
        setDragging(false);
        const galleryId = isGalleryDrag(e) ? e.dataTransfer.getData(GALLERY_ITEM_DRAG_TYPE) : "";
        if (galleryId) void readGalleryItem(galleryId);
        else readFile(e.dataTransfer.files?.[0]);
      }}
      className={cn(
        "flex h-28 w-full flex-col items-center justify-center gap-1 rounded-xl border border-dashed text-xs transition-colors",
        dragging
          ? "border-wash-30 bg-wash-3 dark:border-wash-60 dark:bg-wash-5 text-foreground"
          : "border-border text-muted-foreground hover:border-[color-mix(in_oklab,var(--foreground)_calc(30%*var(--contrast-edge-gain,1)),transparent)] hover:text-foreground",
        className,
      )}
    >
      <HugeiconsIcon icon={PlusSignIcon} className="size-5" />
      <span>{label}</span>
      <input
        ref={inputRef}
        type="file"
        accept="image/*"
        className="hidden"
        onChange={(e) => readFile(e.target.files?.[0])}
      />
    </button>
  );
}
