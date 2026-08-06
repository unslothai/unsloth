import { isTauri } from "@/lib/api-base";
import { useEffect, useRef, useState } from "react";
import { toast } from "@/lib/toast";
import { registerNativeAttachmentPath, registerNativeModelPath } from "./api";
import { classifyDropPaths, SUPPORTED_DROP_HINT } from "./drop-paths";
import { useNativeIntentStore } from "./store";
import type { NativeIntent } from "./types";

export type NativeModelDropState =
  | { status: "idle" }
  | { status: "valid"; action: "load" | "replace" | "chip" }
  | { status: "attach"; count: number }
  | { status: "invalid" };

interface NativeModelDropOptions {
  enabled?: boolean;
  attachmentScope?: string;
  nativePathLeasesSupported: boolean;
  hasActiveModel: boolean;
  isModelLoading: boolean;
  onAutoLoad?: (intent: NativeIntent) => Promise<void> | void;
  onAttach?: (intents: NativeIntent[]) => Promise<void> | void;
  onAttachImages?: (intents: NativeIntent[]) => Promise<void> | void;
}

function canAttachDocs(options: NativeModelDropOptions): boolean {
  return options.nativePathLeasesSupported && Boolean(options.onAttach);
}

function canAttachImages(options: NativeModelDropOptions): boolean {
  return Boolean(options.onAttachImages);
}

function canAutoLoadModel(options: NativeModelDropOptions): boolean {
  return (
    options.nativePathLeasesSupported &&
    !options.isModelLoading &&
    Boolean(options.onAutoLoad)
  );
}

function attachmentCount(dropped: ReturnType<typeof classifyDropPaths>): number {
  if (dropped.kind === "docs" || dropped.kind === "images") {
    return dropped.paths.length;
  }
  if (dropped.kind === "attach") {
    return dropped.docs.length + dropped.images.length;
  }
  return 0;
}

function dropStateForPaths(
  paths: string[],
  options: NativeModelDropOptions,
): NativeModelDropState {
  const dropped = classifyDropPaths(paths);
  if (dropped.kind === "none") return { status: "idle" };
  if (dropped.kind === "docs") {
    return canAttachDocs(options)
      ? { status: "attach", count: dropped.paths.length }
      : { status: "invalid" };
  }
  if (dropped.kind === "images") {
    return canAttachImages(options)
      ? { status: "attach", count: dropped.paths.length }
      : { status: "invalid" };
  }
  if (dropped.kind === "attach") {
    const docsSupported = dropped.docs.length === 0 || canAttachDocs(options);
    const imagesSupported =
      dropped.images.length === 0 || canAttachImages(options);
    return docsSupported && imagesSupported
      ? { status: "attach", count: attachmentCount(dropped) }
      : { status: "invalid" };
  }
  if (dropped.kind === "unsupported") return { status: "invalid" };
  if (!canAutoLoadModel(options)) {
    return { status: "valid", action: "chip" };
  }
  return {
    status: "valid",
    action: options.hasActiveModel ? "replace" : "load",
  };
}

async function registerDroppedAttachments(
  dropped: Extract<
    ReturnType<typeof classifyDropPaths>,
    { kind: "docs" | "images" | "attach" }
  >,
): Promise<{ docs: NativeIntent[]; images: NativeIntent[] }> {
  if (dropped.kind === "docs") {
    const docs = await Promise.all(
      dropped.paths.map(registerNativeAttachmentPath),
    );
    return { docs, images: [] };
  }
  if (dropped.kind === "images") {
    const images = await Promise.all(
      dropped.paths.map(registerNativeAttachmentPath),
    );
    return { docs: [], images };
  }
  const [docs, images] = await Promise.all([
    Promise.all(dropped.docs.map(registerNativeAttachmentPath)),
    Promise.all(dropped.images.map(registerNativeAttachmentPath)),
  ]);
  return { docs, images };
}

export function useNativeModelDrop(options: NativeModelDropOptions): NativeModelDropState {
  const { enabled = true } = options;
  const addIntent = useNativeIntentStore((state) => state.addIntent);
  const [dropState, setDropState] = useState<NativeModelDropState>({ status: "idle" });
  const optionsRef = useRef(options);
  optionsRef.current = options;

  useEffect(() => {
    if (!isTauri || !enabled) {
      setDropState({ status: "idle" });
      return;
    }
    let disposed = false;
    let unlisten: (() => void) | undefined;

    void import("@tauri-apps/api/window")
      .then(({ getCurrentWindow }) => getCurrentWindow().onDragDropEvent(async (event) => {
        const currentOptions = optionsRef.current;
        if (event.payload.type === "enter") {
          setDropState(dropStateForPaths(event.payload.paths, currentOptions));
          return;
        }
        if (event.payload.type === "leave") {
          setDropState({ status: "idle" });
          return;
        }
        if (event.payload.type !== "drop") return;
        setDropState({ status: "idle" });
        const dropped = classifyDropPaths(event.payload.paths);
        if (dropped.kind === "none") return;
        if (dropped.kind === "unsupported") {
          toast.error(SUPPORTED_DROP_HINT);
          return;
        }
        if (
          dropped.kind === "docs" ||
          dropped.kind === "images" ||
          dropped.kind === "attach"
        ) {
          const needsDocs =
            dropped.kind === "docs" ||
            (dropped.kind === "attach" && dropped.docs.length > 0);
          const needsImages =
            dropped.kind === "images" ||
            (dropped.kind === "attach" && dropped.images.length > 0);
          if (needsDocs && !canAttachDocs(currentOptions)) {
            toast.error("Attaching files needs the desktop backend", {
              description: "Retry once Studio has finished starting up.",
            });
            return;
          }
          if (needsImages && !canAttachImages(currentOptions)) {
            toast.error("Attaching images is unavailable right now", {
              description: "Retry once this chat is ready for attachments.",
            });
            return;
          }
          try {
            const { docs, images } = await registerDroppedAttachments(dropped);
            if (disposed) return;
            const latestOptions = optionsRef.current;
            const attachOptions =
              latestOptions.attachmentScope === currentOptions.attachmentScope
                ? latestOptions
                : currentOptions;
            if (docs.length > 0) {
              await attachOptions.onAttach?.(docs);
            }
            if (images.length > 0) {
              await attachOptions.onAttachImages?.(images);
            }
          } catch (error) {
            toast.error("Could not attach dropped files", {
              description: error instanceof Error ? error.message : String(error),
            });
          }
          return;
        }
        try {
          const intent = await registerNativeModelPath(dropped.path);
          if (disposed) return;
          if (!canAutoLoadModel(currentOptions)) {
            addIntent(intent);
            return;
          }
          try {
            await currentOptions.onAutoLoad?.(intent);
          } catch (error) {
            addIntent(intent);
            toast.error("Could not load dropped model", {
              description: error instanceof Error ? error.message : String(error),
            });
          }
        } catch (error) {
          toast.error("Could not use dropped model", {
            description: error instanceof Error ? error.message : String(error),
          });
        }
      }))
      .then((cleanup) => {
        if (disposed) {
          cleanup();
        } else {
          unlisten = cleanup;
        }
      })
      .catch(() => undefined);

    return () => {
      disposed = true;
      unlisten?.();
    };
  }, [addIntent, enabled]);

  return dropState;
}
