// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  type KeyboardEvent as ReactKeyboardEvent,
  type PointerEvent as ReactPointerEvent,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";

import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import {
  type CropPoint,
  type CropRect,
  type ImageSize,
  type StagedReferenceImage,
  clampCropRect,
  createCropImageLoadGate,
  createReferenceImageEditorActions,
  cropRectFromPoints,
  displayPointToSource,
  moveCropRect,
  rasterizeReferenceImageCrop,
  referenceImageDataUrlError,
} from "./reference-image-crop";

type CropHandle = "north-west" | "north-east" | "south-west" | "south-east";

type CropDrag =
  | { kind: "new"; start: CropPoint; startClient: CropPoint }
  | { kind: "move"; start: CropPoint; selection: CropRect }
  | { kind: "resize"; anchor: CropPoint };

function oppositeCropCorner(
  selection: CropRect,
  handle: CropHandle,
): CropPoint {
  const right = selection.x + selection.width;
  const bottom = selection.y + selection.height;
  switch (handle) {
    case "north-west":
      return { x: right, y: bottom };
    case "north-east":
      return { x: selection.x, y: bottom };
    case "south-west":
      return { x: right, y: selection.y };
    case "south-east":
      return { x: selection.x, y: selection.y };
  }
}

function arrowDelta(key: string, step: number): CropPoint | null {
  if (key === "ArrowLeft") return { x: -step, y: 0 };
  if (key === "ArrowRight") return { x: step, y: 0 };
  if (key === "ArrowUp") return { x: 0, y: -step };
  if (key === "ArrowDown") return { x: 0, y: step };
  return null;
}

/** A local freeform crop editor for MiniMax-H3 picture references. */
export function ReferenceImageEditor({
  open,
  picture,
  pictureNumber,
  onOpenChange,
  onApply,
}: {
  open: boolean;
  picture: StagedReferenceImage | null;
  pictureNumber: number;
  onOpenChange: (open: boolean) => void;
  onApply: (dataUrl: string, crop: CropRect | null) => void;
}) {
  const decodedImage = useRef<{
    dataUrl: string;
    image: HTMLImageElement;
  } | null>(null);
  const [imageLoadGate] = useState(createCropImageLoadGate);
  const cropDrag = useRef<CropDrag | null>(null);
  const [loadedDataUrl, setLoadedDataUrl] = useState<string | null>(null);
  const [loadedSourceSize, setLoadedSourceSize] = useState<ImageSize | null>(
    null,
  );
  const [loadedSelection, setLoadedSelection] = useState<CropRect | null>(null);
  const [loadError, setLoadError] = useState<{
    dataUrl: string;
    message: string;
  } | null>(null);
  const [applying, setApplying] = useState(false);
  const editorActions = useMemo(
    () => createReferenceImageEditorActions({ onApply, onOpenChange }),
    [onApply, onOpenChange],
  );
  const currentDataUrl = open ? (picture?.originalDataUrl ?? null) : null;
  const sourceSize = currentDataUrl === loadedDataUrl ? loadedSourceSize : null;
  const selection = currentDataUrl === loadedDataUrl ? loadedSelection : null;
  const visibleLoadError =
    currentDataUrl === loadError?.dataUrl ? loadError.message : null;

  useEffect(() => {
    if (!open || !picture) {
      decodedImage.current = null;
      cropDrag.current = null;
      return;
    }
    const dataUrl = picture.originalDataUrl;
    const loadClaim = imageLoadGate.begin(dataUrl);
    decodedImage.current = null;
    cropDrag.current = null;

    const image = new Image();
    image.onload = () => {
      if (!loadClaim.isCurrent()) return;
      const size = { width: image.naturalWidth, height: image.naturalHeight };
      if (size.width <= 0 || size.height <= 0) {
        setLoadError({
          dataUrl,
          message: "The picture has no readable pixels.",
        });
        return;
      }
      // Browser image dimensions and canvas drawing use the same corrected orientation.
      decodedImage.current = { dataUrl, image };
      setLoadError(null);
      setLoadedSourceSize(size);
      setLoadedSelection(
        picture.crop ? clampCropRect(picture.crop, size) : null,
      );
      setLoadedDataUrl(dataUrl);
    };
    image.onerror = () => {
      if (loadClaim.isCurrent()) {
        setLoadError({
          dataUrl,
          message: "Could not open this picture for cropping.",
        });
      }
    };
    image.src = dataUrl;
    return () => {
      loadClaim.cancel();
    };
  }, [imageLoadGate, open, picture]);

  const pointFromPointer = useCallback(
    (event: ReactPointerEvent<HTMLDivElement>): CropPoint | null => {
      if (!sourceSize) return null;
      const bounds = event.currentTarget.getBoundingClientRect();
      if (bounds.width <= 0 || bounds.height <= 0) return null;
      return displayPointToSource(
        { x: event.clientX - bounds.left, y: event.clientY - bounds.top },
        { width: bounds.width, height: bounds.height },
        sourceSize,
      );
    },
    [sourceSize],
  );

  const handlePointerDown = useCallback(
    (event: ReactPointerEvent<HTMLDivElement>) => {
      if (event.button !== 0) {
        return;
      }
      const point = pointFromPointer(event);
      if (!point || !sourceSize) return;
      event.preventDefault();
      try {
        event.currentTarget.setPointerCapture(event.pointerId);
      } catch {
        // Synthetic events and older webviews may not support pointer capture.
      }
      const target = event.target as HTMLElement;
      const handle = target.dataset.cropHandle as CropHandle | undefined;
      if (selection && handle) {
        cropDrag.current = {
          kind: "resize",
          anchor: oppositeCropCorner(selection, handle),
        };
        return;
      }
      if (selection && target.closest("[data-crop-selection]")) {
        cropDrag.current = {
          kind: "move",
          start: point,
          selection,
        };
        return;
      }
      cropDrag.current = {
        kind: "new",
        start: point,
        startClient: { x: event.clientX, y: event.clientY },
      };
    },
    [pointFromPointer, selection, sourceSize],
  );

  const handlePointerMove = useCallback(
    (event: ReactPointerEvent<HTMLDivElement>) => {
      const drag = cropDrag.current;
      const point = pointFromPointer(event);
      if (!drag || !point || !sourceSize) return;
      if (drag.kind === "move") {
        setLoadedSelection(
          moveCropRect(
            drag.selection,
            { x: point.x - drag.start.x, y: point.y - drag.start.y },
            sourceSize,
          ),
        );
        return;
      }
      if (
        drag.kind === "new" &&
        Math.hypot(
          event.clientX - drag.startClient.x,
          event.clientY - drag.startClient.y,
        ) < 4
      ) {
        return;
      }
      const anchor = drag.kind === "resize" ? drag.anchor : drag.start;
      setLoadedSelection(cropRectFromPoints(anchor, point, sourceSize));
    },
    [pointFromPointer, sourceSize],
  );

  const finishPointer = useCallback(() => {
    cropDrag.current = null;
  }, []);

  const handlePreviewKeyDown = useCallback(
    (event: ReactKeyboardEvent<HTMLDivElement>) => {
      const step = event.shiftKey ? 10 : 1;
      const delta = arrowDelta(event.key, step);
      if (!delta) return;
      event.preventDefault();
      if (!selection || !sourceSize) return;
      setLoadedSelection(moveCropRect(selection, delta, sourceSize));
    },
    [selection, sourceSize],
  );

  const apply = useCallback(() => {
    const decoded = decodedImage.current;
    if (
      !selection ||
      !sourceSize ||
      !picture ||
      !decoded ||
      decoded.dataUrl !== picture.originalDataUrl
    ) {
      return;
    }
    setApplying(true);
    setLoadError(null);
    try {
      const dataUrl = rasterizeReferenceImageCrop(
        decoded.image,
        selection,
        sourceSize,
      );
      const sizeError = referenceImageDataUrlError(dataUrl);
      if (sizeError) throw new Error(sizeError);
      editorActions.apply(dataUrl, selection);
    } catch (error) {
      setLoadError({
        dataUrl: picture.originalDataUrl,
        message:
          error instanceof Error
            ? error.message
            : "Could not crop this picture.",
      });
    } finally {
      setApplying(false);
    }
  }, [editorActions, picture, selection, sourceSize]);

  const validSelection =
    selection !== null && selection.width > 0 && selection.height > 0;

  return (
    <Dialog
      open={open}
      onOpenChange={(nextOpen) => {
        if (nextOpen) onOpenChange(true);
        else editorActions.cancel();
      }}
    >
      <DialogContent className="sm:max-w-3xl">
        <DialogHeader>
          <DialogTitle>Crop Picture {pictureNumber}</DialogTitle>
          <DialogDescription>
            Drag over the picture to choose what to keep. Move the selection or
            resize it from its corners. Choose Use full image to clear any crop.
          </DialogDescription>
        </DialogHeader>

        {picture && (
          <div className="grid gap-4">
            <div className="flex h-[min(50vh,30rem)] min-h-48 w-full items-center justify-center overflow-hidden rounded-xl bg-black/90">
              <div className="relative inline-block max-h-full max-w-full shrink-0">
                <img
                  src={picture.originalDataUrl}
                  alt={`Crop source ${pictureNumber}`}
                  draggable={false}
                  className="block max-h-[48vh] max-w-full select-none object-contain"
                />
                {sourceSize && (
                  <div
                    role="group"
                    tabIndex={0}
                    aria-label={`Crop selection for picture ${pictureNumber}. Drag to select an area. Drag the selection to move it, or drag a corner to resize it.`}
                    onPointerDown={handlePointerDown}
                    onPointerMove={handlePointerMove}
                    onPointerUp={finishPointer}
                    onPointerCancel={finishPointer}
                    onLostPointerCapture={finishPointer}
                    onKeyDown={handlePreviewKeyDown}
                    className="absolute inset-0 cursor-crosshair touch-none outline-none focus-visible:ring-2 focus-visible:ring-primary"
                  >
                    {selection &&
                      selection.width > 0 &&
                      selection.height > 0 && (
                        <div
                          data-crop-selection
                          className="absolute cursor-move border-2 border-white shadow-[0_0_0_9999px_rgba(0,0,0,0.48)]"
                          style={{
                            left: `${(selection.x / sourceSize.width) * 100}%`,
                            top: `${(selection.y / sourceSize.height) * 100}%`,
                            width: `${(selection.width / sourceSize.width) * 100}%`,
                            height: `${(selection.height / sourceSize.height) * 100}%`,
                          }}
                        >
                          {(
                            [
                              [
                                "north-west",
                                "-left-1.5 -top-1.5 cursor-nwse-resize",
                              ],
                              [
                                "north-east",
                                "-right-1.5 -top-1.5 cursor-nesw-resize",
                              ],
                              [
                                "south-west",
                                "-bottom-1.5 -left-1.5 cursor-nesw-resize",
                              ],
                              [
                                "south-east",
                                "-bottom-1.5 -right-1.5 cursor-nwse-resize",
                              ],
                            ] as const
                          ).map(([handle, position]) => (
                            <span
                              key={handle}
                              data-crop-handle={handle}
                              aria-hidden={true}
                              className={`absolute size-3 rounded-full border-2 border-white bg-primary shadow-sm ${position}`}
                            />
                          ))}
                        </div>
                      )}
                  </div>
                )}
              </div>
            </div>

            <p className="text-ui-11 text-muted-foreground">
              {selection
                ? "Drag inside the selection to move it, or drag a corner to resize it."
                : "No crop selected. Drag over the picture to create one."}
            </p>
            {visibleLoadError && (
              <p className="text-sm text-destructive">{visibleLoadError}</p>
            )}
          </div>
        )}

        <DialogFooter>
          {picture && (
            <Button
              type="button"
              variant="ghost"
              className="sm:mr-auto"
              onClick={() => editorActions.apply(picture.originalDataUrl, null)}
            >
              Use full image
            </Button>
          )}
          <Button
            type="button"
            variant="outline"
            onClick={editorActions.cancel}
          >
            Cancel
          </Button>
          <Button
            type="button"
            disabled={!validSelection || applying}
            onClick={apply}
          >
            {applying ? "Applying..." : "Apply crop"}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
