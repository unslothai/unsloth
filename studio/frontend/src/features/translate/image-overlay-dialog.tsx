// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Spinner } from "@/components/ui/spinner";
import { useT } from "@/i18n";
import { Download01Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useEffect, useRef, useState } from "react";
import {
  type ImageTextBlock,
  drawTranslatedBlocks,
  overlayFileName,
} from "./translate-image-overlay";
import { TranslateRunError, runTextListTranslation } from "./translate-runner";
import { type TranslatePrefs, translatePromptOptions } from "./translate-store";

export type ImageOverlaySource = {
  file: File;
  blocks: ImageTextBlock[];
};

function loadImage(url: string): Promise<HTMLImageElement> {
  const image = new Image();
  image.src = url;
  return image.decode().then(() => image);
}

export function ImageOverlayDialog({
  source,
  prefs,
  onOpenChange,
}: {
  source: ImageOverlaySource | null;
  prefs: TranslatePrefs;
  onOpenChange: (open: boolean) => void;
}) {
  return (
    <Dialog open={source !== null} onOpenChange={onOpenChange}>
      <DialogContent className="corner-squircle dialog-soft-surface gap-4 sm:max-w-3xl">
        {source && <ImageOverlay source={source} prefs={prefs} />}
      </DialogContent>
    </Dialog>
  );
}

function ImageOverlay({
  source,
  prefs,
}: {
  source: ImageOverlaySource;
  prefs: TranslatePrefs;
}) {
  const t = useT();
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [state, setState] = useState<
    { kind: "working" } | { kind: "done" } | { kind: "failed"; error: unknown }
  >({ kind: "working" });
  // Read once: the overlay shows the translation it was opened for.
  const [setup] = useState(() => ({
    prompt: translatePromptOptions(prefs),
    modelId: prefs.modelId,
    custom: prefs.customModel,
    target: prefs.target,
  }));

  useEffect(() => {
    const controller = new AbortController();
    const url = URL.createObjectURL(source.file);
    const { prompt } = setup;
    const run = async () => {
      if (!prompt) throw new TranslateRunError("unavailable");
      const [image, texts] = await Promise.all([
        loadImage(url),
        runTextListTranslation({
          texts: source.blocks.map((block) => block.text),
          prompt,
          modelId: setup.modelId,
          custom: setup.custom,
          signal: controller.signal,
          onPhase: () => {},
        }),
      ]);
      const canvas = canvasRef.current;
      const ctx = canvas?.getContext("2d", { willReadFrequently: true });
      if (!canvas || !ctx) return;
      canvas.width = image.naturalWidth;
      canvas.height = image.naturalHeight;
      ctx.drawImage(image, 0, 0);
      drawTranslatedBlocks(
        ctx,
        source.blocks,
        texts,
        getComputedStyle(canvas).fontFamily || "sans-serif",
      );
      setState({ kind: "done" });
    };
    run().catch((error: unknown) => {
      if (!controller.signal.aborted) setState({ kind: "failed", error });
    });
    return () => {
      controller.abort();
      URL.revokeObjectURL(url);
    };
  }, [source, setup]);

  const description =
    state.kind === "working"
      ? t("translate.imageOverlay.working")
      : state.kind === "done"
        ? t("translate.imageOverlay.description")
        : state.error instanceof TranslateRunError
          ? t(`translate.errors.${state.error.code}`)
          : state.error instanceof Error && state.error.message
            ? state.error.message
            : t("translate.imageOverlay.failed");

  const download = () => {
    canvasRef.current?.toBlob((blob) => {
      if (!blob) return;
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = overlayFileName(source.file.name, setup.target);
      link.click();
      window.setTimeout(() => URL.revokeObjectURL(url), 0);
    }, "image/png");
  };

  return (
    <>
      <DialogHeader>
        <DialogTitle className="text-ui-21">
          {t("translate.imageOverlay.title")}
        </DialogTitle>
        <DialogDescription>{description}</DialogDescription>
      </DialogHeader>
      <div className="relative grid min-h-24 place-items-center overflow-hidden rounded-lg bg-muted/40">
        <canvas
          ref={canvasRef}
          className={
            state.kind === "done"
              ? "block h-auto max-h-[65vh] w-auto max-w-full"
              : "hidden"
          }
        />
        {state.kind === "working" && <Spinner className="size-6" />}
      </div>
      <DialogFooter>
        <Button
          type="button"
          onClick={download}
          disabled={state.kind !== "done"}
        >
          <HugeiconsIcon icon={Download01Icon} className="size-4" />
          {t("translate.imageOverlay.download")}
        </Button>
      </DialogFooter>
    </>
  );
}
