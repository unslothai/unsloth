import { isTauri } from "@/lib/api-base";
import { useEffect, useRef, useState } from "react";
import { toast } from "@/lib/toast";
import { registerNativeAttachmentPath, registerNativeModelPath } from "./api";
import { classifyDropPaths, SUPPORTED_DROP_HINT } from "./drop-paths";
import { nativeDropTargetAt } from "./native-drop-targets";
import { useNativeIntentStore } from "./store";
import type { NativeIntent } from "./types";

export type NativeModelDropState =
  | { status: "idle" }
  | { status: "valid"; action: "load" | "replace" | "chip" }
  | { status: "attach"; count: number; kind: "docs" | "images" | "audio" | "mixed" }
  | { status: "invalid" };

interface NativeModelDropOptions {
  enabled?: boolean;
  attachmentScope?: string;
  // Where a drop on this window belongs, for reporting a failure back to it.
  attachmentTargetKey?: string;
  nativePathLeasesSupported: boolean;
  hasActiveModel: boolean;
  isModelLoading: boolean;
  onAutoLoad?: (intent: NativeIntent) => Promise<void> | void;
  onAttach?: (intents: NativeIntent[]) => Promise<void> | void;
  onAttachImages?: (intents: NativeIntent[]) => Promise<void> | void;
  onAttachAudio?: (intents: NativeIntent[]) => Promise<void> | void;
}

function canAttachDocs(options: NativeModelDropOptions): boolean {
  return options.nativePathLeasesSupported && Boolean(options.onAttach);
}

function canAttachImages(options: NativeModelDropOptions): boolean {
  return Boolean(options.onAttachImages);
}

function canAttachAudio(options: NativeModelDropOptions): boolean {
  return Boolean(options.onAttachAudio);
}

function canAutoLoadModel(options: NativeModelDropOptions): boolean {
  return (
    options.nativePathLeasesSupported &&
    !options.isModelLoading &&
    Boolean(options.onAutoLoad)
  );
}

function attachmentCount(dropped: ReturnType<typeof classifyDropPaths>): number {
  if (
    dropped.kind === "docs" ||
    dropped.kind === "images" ||
    dropped.kind === "audio"
  ) {
    return dropped.paths.length;
  }
  if (dropped.kind === "attach") {
    return dropped.docs.length + dropped.images.length + dropped.audio.length;
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
      ? { status: "attach", count: dropped.paths.length, kind: "docs" }
      : { status: "invalid" };
  }
  if (dropped.kind === "images") {
    return canAttachImages(options)
      ? { status: "attach", count: dropped.paths.length, kind: "images" }
      : { status: "invalid" };
  }
  if (dropped.kind === "audio") {
    return canAttachAudio(options)
      ? { status: "attach", count: dropped.paths.length, kind: "audio" }
      : { status: "invalid" };
  }
  if (dropped.kind === "attach") {
    const docsSupported = dropped.docs.length === 0 || canAttachDocs(options);
    const imagesSupported =
      dropped.images.length === 0 || canAttachImages(options);
    const audioSupported = dropped.audio.length === 0 || canAttachAudio(options);
    return docsSupported && imagesSupported && audioSupported
      ? { status: "attach", count: attachmentCount(dropped), kind: "mixed" }
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

interface RegisteredDrop {
  docs: NativeIntent[];
  images: NativeIntent[];
  audio: NativeIntent[];
  docsFailed: number;
  imagesFailed: number;
  audioFailed: number;
  error?: Error;
}

// Per path, not all-or-nothing: one bad file in a batch used to discard every
// sibling that had already registered, leaving their leases to expire unused.
async function registerEach(paths: string[]) {
  const settled = await Promise.allSettled(
    paths.map(registerNativeAttachmentPath),
  );
  const intents = settled.flatMap((result) =>
    result.status === "fulfilled" ? [result.value] : [],
  );
  const rejection = settled.find((result) => result.status === "rejected");
  return {
    intents,
    failed: settled.length - intents.length,
    error:
      rejection && rejection.status === "rejected"
        ? rejection.reason instanceof Error
          ? rejection.reason
          : new Error(String(rejection.reason))
        : undefined,
  };
}

async function registerDroppedAttachments(
  dropped: Extract<
    ReturnType<typeof classifyDropPaths>,
    { kind: "docs" | "images" | "audio" | "attach" }
  >,
): Promise<RegisteredDrop> {
  const docPaths =
    dropped.kind === "docs"
      ? dropped.paths
      : dropped.kind === "attach"
        ? dropped.docs
        : [];
  const imagePaths =
    dropped.kind === "images"
      ? dropped.paths
      : dropped.kind === "attach"
        ? dropped.images
        : [];
  const audioPaths =
    dropped.kind === "audio"
      ? dropped.paths
      : dropped.kind === "attach"
        ? dropped.audio
        : [];
  const [docs, images, audio] = await Promise.all([
    registerEach(docPaths),
    registerEach(imagePaths),
    registerEach(audioPaths),
  ]);
  return {
    docs: docs.intents,
    images: images.intents,
    audio: audio.intents,
    docsFailed: docs.failed,
    imagesFailed: images.failed,
    audioFailed: audio.failed,
    error: docs.error ?? images.error ?? audio.error,
  };
}

function sameDropState(
  a: NativeModelDropState,
  b: NativeModelDropState,
): boolean {
  if (a.status !== b.status) return false;
  if (a.status === "valid" && b.status === "valid") return a.action === b.action;
  if (a.status === "attach" && b.status === "attach")
    return a.count === b.count && a.kind === b.kind;
  return true;
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
    // "over" carries no paths, so the ones announced on "enter" are what the
    // overlay keeps reading as the cursor moves across the window.
    let draggedPaths: string[] = [];
    // Tauri repeats "over" per cursor move, so a fresh object each time would
    // rerender ChatPage at drag frequency. Returning `prev` makes React bail out.
    const publish = (next: NativeModelDropState) =>
      setDropState((prev) => (sameDropState(prev, next) ? prev : next));

    void import("@tauri-apps/api/window")
      .then(({ getCurrentWindow }) => getCurrentWindow().onDragDropEvent(async (event) => {
        const currentOptions = optionsRef.current;
        if (event.payload.type === "leave") {
          draggedPaths = [];
          publish({ status: "idle" });
          return;
        }
        if (event.payload.type === "enter") {
          draggedPaths = event.payload.paths;
        }
        // A drop zone under the cursor owns this drop; leave it alone.
        if (nativeDropTargetAt(event.payload.position)) {
          publish({ status: "idle" });
          return;
        }
        if (event.payload.type !== "drop") {
          publish(dropStateForPaths(draggedPaths, currentOptions));
          return;
        }
        draggedPaths = [];
        publish({ status: "idle" });
        const dropped = classifyDropPaths(event.payload.paths);
        if (dropped.kind === "none") return;
        if (dropped.kind === "unsupported") {
          toast.error(SUPPORTED_DROP_HINT);
          return;
        }
        if (
          dropped.kind === "docs" ||
          dropped.kind === "images" ||
          dropped.kind === "audio" ||
          dropped.kind === "attach"
        ) {
          const needsDocs =
            dropped.kind === "docs" ||
            (dropped.kind === "attach" && dropped.docs.length > 0);
          const needsImages =
            dropped.kind === "images" ||
            (dropped.kind === "attach" && dropped.images.length > 0);
          const needsAudio =
            dropped.kind === "audio" ||
            (dropped.kind === "attach" && dropped.audio.length > 0);
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
          if (needsAudio && !canAttachAudio(currentOptions)) {
            toast.error("Attaching audio is unavailable right now", {
              description: "Retry once this chat is ready for attachments.",
            });
            return;
          }
          // Hold the send gate across registration too. Between the drop and the
          // intents reaching the queue there is nothing for the composer to see,
          // so an Enter in that window would send the text without the image.
          const store = useNativeIntentStore.getState();
          if (needsImages) store.beginImageDropRegistration();
          if (needsAudio) store.beginAudioDropRegistration();
          try {
            const registered = await registerDroppedAttachments(dropped);
            const latestOptions = optionsRef.current;
            // Both callbacks only enqueue against a target key, so a drop that
            // outlived this listener still reaches the chat it landed on.
            const attachOptions =
              !disposed &&
              latestOptions.attachmentScope === currentOptions.attachmentScope
                ? latestOptions
                : currentOptions;
            if (registered.docs.length > 0) {
              await attachOptions.onAttach?.(registered.docs);
            }
            if (registered.images.length > 0) {
              await attachOptions.onAttachImages?.(registered.images);
            }
            const failureKey = attachOptions.attachmentTargetKey;
            if (registered.imagesFailed > 0 && failureKey) {
              store.failImageDropRegistration(failureKey);
            }
            if (registered.audioFailed > 0 && failureKey) {
              store.failAudioDropRegistration(failureKey);
            }
            // A failed document cancels a send parked behind the image or audio
            // gate too, or the draft goes out with only what survived.
            if (registered.docsFailed > 0 && failureKey) {
              if (needsImages) store.failImageDropRegistration(failureKey);
              if (needsAudio) store.failAudioDropRegistration(failureKey);
            }
            if (registered.audio.length > 0) {
              await attachOptions.onAttachAudio?.(registered.audio);
            }
            if (
              registered.docsFailed +
                registered.imagesFailed +
                registered.audioFailed >
              0
            ) {
              toast.error("Could not attach dropped files", {
                description: registered.error?.message ?? "Some files were skipped.",
              });
            }
          } catch (error) {
            const failureKey = currentOptions.attachmentTargetKey;
            if (needsImages && failureKey) {
              store.failImageDropRegistration(failureKey);
            }
            if (needsAudio && failureKey) {
              store.failAudioDropRegistration(failureKey);
            }
            toast.error("Could not attach dropped files", {
              description: error instanceof Error ? error.message : String(error),
            });
          } finally {
            if (needsImages) store.endImageDropRegistration();
            if (needsAudio) store.endAudioDropRegistration();
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
