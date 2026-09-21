import { isTauri } from "@/lib/api-base";
import { toast } from "@/lib/toast";
import { useEffect, useRef, useState } from "react";
import { registerNativeAttachmentPath, registerNativeModelPath } from "./api";
import {
  SUPPORTED_DROP_HINT,
  classifyDropPaths,
  isComposerAttachmentName,
} from "./drop-paths";
import { nativeDropTargetAt } from "./native-drop-targets";
import { useNativeIntentStore } from "./store";
import type { NativeIntent } from "./types";

export type NativeModelDropState =
  | { status: "idle" }
  | { status: "valid"; action: "load" | "replace" | "chip" }
  | {
      status: "attach";
      count: number;
      kind: "docs" | "images" | "audio" | "video" | "mixed";
    }
  // `reason` explains a refusal the file types alone do not. Absent means the
  // files themselves are the problem.
  | { status: "invalid"; reason?: string };

interface NativeModelDropOptions {
  enabled?: boolean;
  attachmentScope?: string;
  // Where a drop on this window belongs, for reporting a failure back to it.
  attachmentTargetKey?: string;
  /** Set when this view takes no drops at all. Refuses every droppable payload
   * with this sentence instead of swallowing it, and loads nothing. */
  dropsUnsupportedReason?: string;
  nativePathLeasesSupported: boolean;
  hasActiveModel: boolean;
  isModelLoading: boolean;
  onAutoLoad?: (intent: NativeIntent) => Promise<void> | void;
  onAttach?: (intents: NativeIntent[]) => Promise<void> | void;
  onAttachImages?: (intents: NativeIntent[]) => Promise<void> | void;
  onAttachOpenDocuments?: (intents: NativeIntent[]) => Promise<void> | void;
  onAttachAudio?: (intents: NativeIntent[]) => Promise<void> | void;
  onAttachVideo?: (intents: NativeIntent[]) => Promise<void> | void;
}

function canAttachDocs(options: NativeModelDropOptions): boolean {
  return options.nativePathLeasesSupported && Boolean(options.onAttach);
}

function canAttachImages(options: NativeModelDropOptions): boolean {
  return Boolean(options.onAttachImages);
}

function canAttachOpenDocuments(options: NativeModelDropOptions): boolean {
  return Boolean(options.onAttachOpenDocuments);
}

function canAttachDocumentPaths(
  paths: string[],
  options: NativeModelDropOptions,
): boolean {
  return paths.every((path) =>
    isComposerAttachmentName(path)
      ? canAttachOpenDocuments(options)
      : canAttachDocs(options),
  );
}

function canAttachAudio(options: NativeModelDropOptions): boolean {
  return Boolean(options.onAttachAudio);
}

function canAttachVideo(options: NativeModelDropOptions): boolean {
  return Boolean(options.onAttachVideo);
}

function canAutoLoadModel(options: NativeModelDropOptions): boolean {
  return (
    options.nativePathLeasesSupported &&
    !options.isModelLoading &&
    Boolean(options.onAutoLoad)
  );
}

function attachmentCount(
  dropped: ReturnType<typeof classifyDropPaths>,
): number {
  if (
    dropped.kind === "docs" ||
    dropped.kind === "images" ||
    dropped.kind === "audio" ||
    dropped.kind === "video"
  ) {
    return dropped.paths.length;
  }
  if (dropped.kind === "attach") {
    return (
      dropped.docs.length +
      dropped.images.length +
      dropped.audio.length +
      dropped.video.length
    );
  }
  return 0;
}

/** Anything this handler would otherwise act on. "none" and "unsupported"
 * already have their own answers. */
function isActionableKind(
  dropped: ReturnType<typeof classifyDropPaths>,
): boolean {
  return dropped.kind !== "none" && dropped.kind !== "unsupported";
}

function dropStateForPaths(
  paths: string[],
  options: NativeModelDropOptions,
): NativeModelDropState {
  const dropped = classifyDropPaths(paths);
  if (dropped.kind === "none") return { status: "idle" };
  if (options.dropsUnsupportedReason && isActionableKind(dropped)) {
    return { status: "invalid", reason: options.dropsUnsupportedReason };
  }
  if (dropped.kind === "docs") {
    return canAttachDocumentPaths(dropped.paths, options)
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
  if (dropped.kind === "video") {
    return canAttachVideo(options)
      ? { status: "attach", count: dropped.paths.length, kind: "video" }
      : { status: "invalid" };
  }
  if (dropped.kind === "attach") {
    const docsSupported = canAttachDocumentPaths(dropped.docs, options);
    const imagesSupported =
      dropped.images.length === 0 || canAttachImages(options);
    const audioSupported =
      dropped.audio.length === 0 || canAttachAudio(options);
    const videoSupported =
      dropped.video.length === 0 || canAttachVideo(options);
    return docsSupported && imagesSupported && audioSupported && videoSupported
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
  composerDocuments: NativeIntent[];
  images: NativeIntent[];
  audio: NativeIntent[];
  video: NativeIntent[];
  docsFailed: number;
  imagesFailed: number;
  audioFailed: number;
  videoFailed: number;
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
    { kind: "docs" | "images" | "audio" | "video" | "attach" }
  >,
): Promise<RegisteredDrop> {
  const docPaths =
    dropped.kind === "docs"
      ? dropped.paths
      : dropped.kind === "attach"
        ? dropped.docs
        : [];
  const composerDocumentPaths = docPaths.filter(isComposerAttachmentName);
  const ragDocumentPaths = docPaths.filter(
    (path) => !isComposerAttachmentName(path),
  );
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
  const videoPaths =
    dropped.kind === "video"
      ? dropped.paths
      : dropped.kind === "attach"
        ? dropped.video
        : [];
  const [docs, composerDocuments, images, audio, video] = await Promise.all([
    registerEach(ragDocumentPaths),
    registerEach(composerDocumentPaths),
    registerEach(imagePaths),
    registerEach(audioPaths),
    registerEach(videoPaths),
  ]);
  return {
    docs: docs.intents,
    composerDocuments: composerDocuments.intents,
    images: images.intents,
    audio: audio.intents,
    video: video.intents,
    docsFailed: docs.failed + composerDocuments.failed,
    imagesFailed: images.failed,
    audioFailed: audio.failed,
    videoFailed: video.failed,
    error:
      docs.error ??
      composerDocuments.error ??
      images.error ??
      audio.error ??
      video.error,
  };
}

function sameDropState(
  a: NativeModelDropState,
  b: NativeModelDropState,
): boolean {
  if (a.status !== b.status) return false;
  if (a.status === "valid" && b.status === "valid")
    return a.action === b.action;
  if (a.status === "attach" && b.status === "attach")
    return a.count === b.count && a.kind === b.kind;
  if (a.status === "invalid" && b.status === "invalid")
    return a.reason === b.reason;
  return true;
}

export function useNativeModelDrop(
  options: NativeModelDropOptions,
): NativeModelDropState {
  const { enabled = true } = options;
  const addIntent = useNativeIntentStore((state) => state.addIntent);
  const [dropState, setDropState] = useState<NativeModelDropState>({
    status: "idle",
  });
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
      .then(({ getCurrentWindow }) =>
        getCurrentWindow().onDragDropEvent(async (event) => {
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
          // Before the model branch too: this view loads nothing, so a dropped
          // GGUF must not replace the active model behind it.
          if (
            currentOptions.dropsUnsupportedReason &&
            isActionableKind(dropped)
          ) {
            toast.error(currentOptions.dropsUnsupportedReason);
            return;
          }
          if (
            dropped.kind === "docs" ||
            dropped.kind === "images" ||
            dropped.kind === "audio" ||
            dropped.kind === "video" ||
            dropped.kind === "attach"
          ) {
            const needsImages =
              dropped.kind === "images" ||
              (dropped.kind === "attach" && dropped.images.length > 0);
            const documentPaths =
              dropped.kind === "docs"
                ? dropped.paths
                : dropped.kind === "attach"
                  ? dropped.docs
                  : [];
            const composerDocumentPaths = documentPaths.filter(
              isComposerAttachmentName,
            );
            const needsRagDocuments = documentPaths.some(
              (path) => !isComposerAttachmentName(path),
            );
            const needsComposerDocuments = composerDocumentPaths.length > 0;
            const needsComposerAttachments =
              needsImages || needsComposerDocuments;
            const needsAudio =
              dropped.kind === "audio" ||
              (dropped.kind === "attach" && dropped.audio.length > 0);
            const needsVideo =
              dropped.kind === "video" ||
              (dropped.kind === "attach" && dropped.video.length > 0);
            if (needsRagDocuments && !canAttachDocs(currentOptions)) {
              toast.error("Attaching files needs the desktop backend", {
                description: "Retry once Unsloth has finished starting up.",
              });
              return;
            }
            if (needsImages && !canAttachImages(currentOptions)) {
              toast.error("Attaching images is unavailable right now", {
                description: "Retry once this chat is ready for attachments.",
              });
              return;
            }
            if (
              needsComposerDocuments &&
              !canAttachOpenDocuments(currentOptions)
            ) {
              toast.error("Attaching files is unavailable right now", {
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
            if (needsVideo && !canAttachVideo(currentOptions)) {
              toast.error("Attaching video is unavailable right now", {
                description: "Retry once this chat is ready for attachments.",
              });
              return;
            }
            // Hold the send gate across registration too. Between the drop and the
            // intents reaching the queue there is nothing for the composer to see,
            // so an Enter in that window would send the text without the attachment.
            const store = useNativeIntentStore.getState();
            if (needsComposerAttachments) store.beginImageDropRegistration();
            if (needsAudio) store.beginAudioDropRegistration();
            if (needsVideo) store.beginVideoDropRegistration();
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
              // Documents first: a vision-less model throws on the image and
              // aborts the batch, which would discard them.
              const composerAttachments = [
                ...registered.composerDocuments,
                ...registered.images,
              ];
              if (registered.composerDocuments.length > 0) {
                await attachOptions.onAttachOpenDocuments?.(
                  composerAttachments,
                );
              } else if (registered.images.length > 0) {
                await attachOptions.onAttachImages?.(registered.images);
              }
              const failureKey = attachOptions.attachmentTargetKey;
              if (registered.imagesFailed > 0 && failureKey) {
                store.failImageDropRegistration(failureKey);
              }
              if (registered.audioFailed > 0 && failureKey) {
                store.failAudioDropRegistration(failureKey);
              }
              if (registered.videoFailed > 0 && failureKey) {
                store.failVideoDropRegistration(failureKey);
              }
              // A failed document cancels a send parked behind the attachment, audio
              // or video gates too, or the draft goes out with only what survived.
              if (registered.docsFailed > 0 && failureKey) {
                if (needsComposerAttachments) {
                  store.failImageDropRegistration(failureKey);
                }
                if (needsAudio) store.failAudioDropRegistration(failureKey);
                if (needsVideo) store.failVideoDropRegistration(failureKey);
              }
              if (registered.audio.length > 0) {
                await attachOptions.onAttachAudio?.(registered.audio);
              }
              if (registered.video.length > 0) {
                await attachOptions.onAttachVideo?.(registered.video);
              }
              if (
                registered.docsFailed +
                  registered.imagesFailed +
                  registered.audioFailed +
                  registered.videoFailed >
                0
              ) {
                toast.error("Could not attach dropped files", {
                  description:
                    registered.error?.message ?? "Some files were skipped.",
                });
              }
            } catch (error) {
              const failureKey = currentOptions.attachmentTargetKey;
              if (needsComposerAttachments && failureKey) {
                store.failImageDropRegistration(failureKey);
              }
              if (needsAudio && failureKey) {
                store.failAudioDropRegistration(failureKey);
              }
              if (needsVideo && failureKey) {
                store.failVideoDropRegistration(failureKey);
              }
              toast.error("Could not attach dropped files", {
                description:
                  error instanceof Error ? error.message : String(error),
              });
            } finally {
              if (needsComposerAttachments) store.endImageDropRegistration();
              if (needsAudio) store.endAudioDropRegistration();
              if (needsVideo) store.endVideoDropRegistration();
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
                description:
                  error instanceof Error ? error.message : String(error),
              });
            }
          } catch (error) {
            toast.error("Could not use dropped model", {
              description:
                error instanceof Error ? error.message : String(error),
            });
          }
        }),
      )
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
