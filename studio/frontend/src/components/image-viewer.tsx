// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Spinner } from "@/components/ui/spinner";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { AUTH_SESSION_CLEARED_EVENT } from "@/features/auth/session-events";
import { useLocale, useT } from "@/i18n";
import { downloadFile, isDownloadCancelled } from "@/lib/native-files";
import { openLink } from "@/lib/open-link";
import { toast } from "@/lib/toast";
import { cn } from "@/lib/utils";
import {
  ArrowLeft02Icon,
  ArrowRight02Icon,
  Cancel01Icon,
  Download01Icon,
  LinkSquare02Icon,
  MinusSignIcon,
  PlusSignIcon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon, type IconSvgElement } from "@hugeicons/react";
import { Dialog as DialogPrimitive } from "radix-ui";
import { useEffect, useLayoutEffect, useRef, useState } from "react";
import { create } from "zustand";

export type ViewerImage = {
  key: string;
  title: string;
  fileName: string | ((contentType: string) => string);
  load: () => Promise<Blob>;
  /** Shown as is when `load` fails, e.g. a remote image whose host doesn't allow CORS reads. */
  fallbackUrl?: string;
  source?: string;
};

type ImageViewerState = {
  images: ViewerImage[];
  index: number;
  open: boolean;
  show: (images: ViewerImage[], index?: number) => void;
  close: () => void;
  step: (delta: 1 | -1) => void;
};

export const useImageViewerStore = create<ImageViewerState>((set, get) => ({
  images: [],
  index: 0,
  open: false,
  show: (images, index = 0) =>
    images.length > 0 &&
    set({
      images,
      index: Math.min(Math.max(index, 0), images.length - 1),
      open: true,
    }),
  // The body unmounts on close, so the images (and the blobs their loaders hold) can go too.
  close: () => set({ images: [], index: 0, open: false }),
  step: (delta) => {
    const { images, index } = get();
    const next = index + delta;
    if (next >= 0 && next < images.length) set({ index: next });
  },
}));

// A sign-out leaves the app mounted: the viewer must not keep showing, or holding, the last account's images.
if (typeof window !== "undefined") {
  window.addEventListener(AUTH_SESSION_CLEARED_EVENT, () =>
    useImageViewerStore.setState({ images: [], index: 0, open: false }),
  );
}

export function openImageViewer(images: ViewerImage[], index = 0): void {
  useImageViewerStore.getState().show(images, index);
}

const ZOOM_STEPS = [
  0.1, 0.25, 0.33, 0.5, 0.67, 0.75, 0.9, 1, 1.25, 1.5, 2, 3, 4, 5,
];

const ROUND =
  "flex cursor-pointer items-center justify-center rounded-full bg-neutral-800 text-white shadow-lg transition-colors hover:bg-neutral-700 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-white/70 disabled:cursor-default disabled:opacity-40 disabled:hover:bg-neutral-800";

function RoundButton({
  label,
  icon,
  onClick,
  disabled,
  className,
}: {
  label: string;
  icon: IconSvgElement;
  onClick: () => void;
  disabled?: boolean;
  className?: string;
}) {
  return (
    <Tooltip>
      <TooltipTrigger asChild={true}>
        <button
          type="button"
          aria-label={label}
          onClick={onClick}
          disabled={disabled}
          className={cn(ROUND, className)}
        >
          <HugeiconsIcon icon={icon} strokeWidth={1.75} className="size-5" />
        </button>
      </TooltipTrigger>
      <TooltipContent side="bottom" className="tooltip-compact">
        {label}
      </TooltipContent>
    </Tooltip>
  );
}

type Loaded =
  | { key: string; url: string; blob: Blob | null }
  | { key: string; failed: true };

function useImageBlob(image: ViewerImage | undefined): Loaded | null {
  const [loaded, setLoaded] = useState<Loaded | null>(null);
  useEffect(() => {
    if (!image) return;
    let live = true;
    let url: string | null = null;
    image
      .load()
      .then(async (blob) => {
        if (!live) return;
        // A data: URL, so an opened SVG can't run as a page on Studio's origin.
        if (/^image\/svg\+xml\b/i.test(blob.type)) {
          const dataUrl = await new Promise<string>((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = () => resolve(String(reader.result));
            reader.onerror = () => reject(reader.error);
            reader.readAsDataURL(blob);
          });
          if (live) setLoaded({ key: image.key, url: dataUrl, blob });
          return;
        }
        url = URL.createObjectURL(blob);
        setLoaded({ key: image.key, url, blob });
      })
      .catch(
        () =>
          live &&
          setLoaded(
            image.fallbackUrl
              ? { key: image.key, url: image.fallbackUrl, blob: null }
              : { key: image.key, failed: true },
          ),
      );
    return () => {
      live = false;
      if (url) URL.revokeObjectURL(url);
    };
  }, [image]);
  return loaded && image && loaded.key === image.key ? loaded : null;
}

function ViewerBody() {
  const t = useT();
  const locale = useLocale();
  const images = useImageViewerStore((state) => state.images);
  const index = useImageViewerStore((state) => state.index);
  const { close, step } = useImageViewerStore.getState();
  const image = images[index];
  const loaded = useImageBlob(image);
  const stageRef = useRef<HTMLDivElement | null>(null);
  const [natural, setNatural] = useState<{
    key: string;
    width: number;
    height: number;
  } | null>(null);
  const [zoom, setZoom] = useState<{ key: string; value: number; fit: boolean } | null>(null);
  const size = natural && image && natural.key === image.key ? natural : null;

  const fitZoom = () => {
    const stage = stageRef.current;
    if (!stage || !size) return 1;
    return Math.min(
      1,
      stage.clientWidth / size.width,
      stage.clientHeight / size.height,
    );
  };
  useLayoutEffect(() => {
    if (size && image && zoom?.key !== image.key)
      setZoom({ key: image.key, value: fitZoom(), fit: true });
  });
  const fitted = Boolean(zoom?.fit && image && zoom.key === image.key);
  // biome-ignore lint/correctness/useExhaustiveDependencies: refit only when fit mode or the image's size changes
  useEffect(() => {
    const stage = stageRef.current;
    if (!fitted || !stage) return;
    const observer = new ResizeObserver(() =>
      setZoom((current) => {
        const value = fitZoom();
        return current?.fit && current.value !== value ? { ...current, value } : current;
      }),
    );
    observer.observe(stage);
    return () => observer.disconnect();
  }, [fitted, size]);
  const scale = zoom && image && zoom.key === image.key ? zoom.value : null;
  const setScale = (value: number, fit = false) =>
    image && setZoom({ key: image.key, value, fit });
  const zoomBy = (direction: 1 | -1) => {
    if (scale === null) return;
    const next =
      direction > 0
        ? ZOOM_STEPS.find((level) => level > scale + 0.001)
        : [...ZOOM_STEPS].reverse().find((level) => level < scale - 0.001);
    if (next !== undefined) setScale(next);
  };

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "ArrowLeft") step(-1);
      else if (event.key === "ArrowRight") step(1);
      else if (event.key === "+" || event.key === "=") zoomBy(1);
      else if (event.key === "-") zoomBy(-1);
      else if (event.key === "0") setScale(fitZoom(), true);
      else return;
      event.preventDefault();
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  });

  const download = () => {
    if (!loaded || "failed" in loaded || !loaded.blob || !image) return;
    const { blob } = loaded;
    const name = typeof image.fileName === "string" ? image.fileName : image.fileName(blob.type);
    void downloadFile(blob, name, blob.type).catch(
      (error) => {
        if (!isDownloadCancelled(error))
          toast.error(t("imageViewer.downloadFailed"));
      },
    );
  };
  const percent = new Intl.NumberFormat(locale, {
    style: "percent",
    maximumFractionDigits: 0,
  }).format(scale ?? 1);

  return (
    <>
      <DialogPrimitive.Title className="sr-only">
        {image?.title || t("imageViewer.title")}
      </DialogPrimitive.Title>
      <DialogPrimitive.Description className="sr-only">
        {t("imageViewer.description")}
      </DialogPrimitive.Description>
      <div
        ref={stageRef}
        className="absolute inset-x-[calc(6rem*var(--ui-space-scale,1))] top-[calc(5.5rem*var(--ui-space-scale,1))] bottom-[calc(6rem*var(--ui-space-scale,1))] flex overflow-auto"
        onClick={(event) => event.target === event.currentTarget && close()}
      >
        {loaded && "failed" in loaded ? (
          <p className="m-auto text-sm text-white/80">
            {t("imageViewer.failed")}
          </p>
        ) : loaded ? (
          <img
            key={loaded.key}
            src={loaded.url}
            alt={image?.title ?? ""}
            onLoad={(event) =>
              image &&
              setNatural({
                key: image.key,
                width: event.currentTarget.naturalWidth || 1,
                height: event.currentTarget.naturalHeight || 1,
              })
            }
            className={cn(
              "m-auto max-w-none rounded-lg shadow-2xl",
              scale === null && "invisible",
            )}
            style={
              size && scale !== null
                ? { width: size.width * scale, height: size.height * scale }
                : undefined
            }
          />
        ) : (
          <Spinner className="m-auto size-7 text-white" />
        )}
      </div>
      <div className="absolute top-4 right-4 flex items-center gap-3">
        {image?.source ? (
          <RoundButton
            label={t("imageViewer.openSource")}
            icon={LinkSquare02Icon}
            className="size-12"
            onClick={() => {
              const source = image.source;
              close();
              if (source) openLink(source);
            }}
          />
        ) : null}
        <RoundButton
          label={t("imageViewer.download")}
          icon={Download01Icon}
          className="size-12"
          disabled={!loaded || "failed" in loaded || !loaded.blob}
          onClick={download}
        />
        <RoundButton
          label={t("imageViewer.close")}
          icon={Cancel01Icon}
          className="size-12"
          onClick={close}
        />
      </div>
      {images.length > 1 ? (
        <>
          <RoundButton
            label={t("imageViewer.previous")}
            icon={ArrowLeft02Icon}
            disabled={index === 0}
            onClick={() => step(-1)}
            className="absolute top-1/2 left-4 size-12 -translate-y-1/2"
          />
          <RoundButton
            label={t("imageViewer.next")}
            icon={ArrowRight02Icon}
            disabled={index === images.length - 1}
            onClick={() => step(1)}
            className="absolute top-1/2 right-4 size-12 -translate-y-1/2"
          />
        </>
      ) : null}
      <div className="absolute bottom-5 left-1/2 flex -translate-x-1/2 items-center gap-1 rounded-full bg-neutral-800 p-1.5 text-white shadow-lg">
        <RoundButton
          label={t("imageViewer.zoomOut")}
          icon={MinusSignIcon}
          disabled={scale === null || scale <= (ZOOM_STEPS[0] ?? 0.1)}
          onClick={() => zoomBy(-1)}
          className="size-10 bg-neutral-700 shadow-none hover:bg-neutral-600 disabled:hover:bg-neutral-700"
        />
        <button
          type="button"
          aria-label={t("imageViewer.fit")}
          onClick={() => setScale(fitZoom(), true)}
          className="min-w-18 cursor-pointer text-center text-ui-15 tabular-nums"
        >
          {percent}
        </button>
        <RoundButton
          label={t("imageViewer.zoomIn")}
          icon={PlusSignIcon}
          disabled={
            scale === null || scale >= (ZOOM_STEPS[ZOOM_STEPS.length - 1] ?? 5)
          }
          onClick={() => zoomBy(1)}
          className="size-10 bg-neutral-700 shadow-none hover:bg-neutral-600 disabled:hover:bg-neutral-700"
        />
      </div>
    </>
  );
}

export function ImageViewer() {
  const open = useImageViewerStore((state) => state.open);
  return (
    <DialogPrimitive.Root
      open={open}
      onOpenChange={(next) => !next && useImageViewerStore.getState().close()}
    >
      <DialogPrimitive.Portal>
        {/* No backdrop blur: it would redraw with the streaming chat. */}
        <DialogPrimitive.Overlay className="data-open:animate-in data-closed:animate-out data-closed:fade-out-0 data-open:fade-in-0 fixed inset-0 z-50 bg-black/80 duration-150" />
        <DialogPrimitive.Content className="data-open:animate-in data-closed:animate-out data-closed:fade-out-0 data-open:fade-in-0 fixed inset-0 z-50 outline-none duration-150">
          {open ? <ViewerBody /> : null}
        </DialogPrimitive.Content>
      </DialogPrimitive.Portal>
    </DialogPrimitive.Root>
  );
}
