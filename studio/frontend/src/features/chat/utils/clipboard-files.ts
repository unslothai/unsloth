


import { isTauri } from "@/lib/api-base";
import { MAX_AUDIO_SIZE } from "@/lib/audio-utils";
import {
  browserClipboardFiles,
  clipboardAdvertisesFiles,
  clipboardHasPlainText,
} from "./clipboard-payload.ts";

const MAX_NATIVE_IMAGE_DIMENSION = 8192;
const MAX_NATIVE_IMAGE_RGBA_BYTES = 64 * 1024 * 1024;
const MAX_CLIPBOARD_BYTES = MAX_AUDIO_SIZE;
const MAX_CLIPBOARD_NON_AUDIO_BYTES = 20 * 1024 * 1024;
const MAX_CLIPBOARD_FILES = 8;

type ClipboardPasteEvent = {
  readonly clipboardData: DataTransfer | null;
  readonly defaultPrevented: boolean;
  readonly isTrusted: boolean;
  preventDefault: () => void;
};
type NativeClipboardFile = {
  readonly name: string;
  readonly mimeType: string;
  readonly base64: string;
};

function validDimension(value: number): boolean {
  return (
    Number.isSafeInteger(value) &&
    value > 0 &&
    value <= MAX_NATIVE_IMAGE_DIMENSION
  );
}

function canvasPng(canvas: HTMLCanvasElement): Promise<Blob | null> {
  return new Promise((resolve) => canvas.toBlob(resolve, "image/png"));
}

function isLinuxDesktop(): boolean {
  if (typeof navigator === "undefined") return false;
  return `${navigator.platform} ${navigator.userAgent}`.toLowerCase().includes("linux");
}

async function readNativeClipboardFiles(): Promise<File[]> {
  const { invoke } = await import("@tauri-apps/api/core");
  const nativeFiles = await invoke<NativeClipboardFile[]>(
    "read_native_clipboard_files",
  );
  if (nativeFiles.length > MAX_CLIPBOARD_FILES) return [];

  let totalBytes = 0;
  const files: File[] = [];
  for (const file of nativeFiles) {
    if (file.base64.length === 0) continue;
    if (
      !file.name ||
      file.name.length > 255 ||
      file.name.includes("/") ||
      file.name.includes("\0")
    ) {
      return [];
    }
    const maxFileBytes = file.mimeType.startsWith("audio/")
      ? MAX_AUDIO_SIZE
      : MAX_CLIPBOARD_NON_AUDIO_BYTES;
    if (file.base64.length > Math.ceil((maxFileBytes * 4) / 3) + 4) {
      return [];
    }
    const binary = globalThis.atob(file.base64);
    const bytes = new Uint8Array(binary.length);
    for (let index = 0; index < binary.length; index += 1) {
      bytes[index] = binary.charCodeAt(index);
    }
    if (bytes.byteLength > maxFileBytes) return [];
    totalBytes += bytes.byteLength;
    if (totalBytes > MAX_CLIPBOARD_BYTES) return [];
    files.push(
      new File([bytes], file.name, {
        type: file.mimeType || "application/octet-stream",
        lastModified: Date.now(),
      }),
    );
  }
  return files;
}

async function readLinuxClipboardImage(): Promise<File | null> {
  const { invoke } = await import("@tauri-apps/api/core");
  const raw = await invoke<ArrayBuffer | Uint8Array>("read_native_clipboard_png");
  const png = Uint8Array.from(raw instanceof Uint8Array ? raw : new Uint8Array(raw));
  if (png.byteLength === 0 || png.byteLength > MAX_CLIPBOARD_NON_AUDIO_BYTES) {
    return null;
  }
  return new File([png], "pasted-image.png", {
    type: "image/png",
    lastModified: Date.now(),
  });
}

async function readNativeClipboardImage(): Promise<File | null> {
  let image: Awaited<ReturnType<
    typeof import("@tauri-apps/plugin-clipboard-manager").readImage
  >> | null = null;

  try {
    if (isLinuxDesktop()) return await readLinuxClipboardImage();
    const { readImage } = await import("@tauri-apps/plugin-clipboard-manager");
    image = await readImage();
    const { width, height } = await image.size();
    if (!validDimension(width) || !validDimension(height)) return null;

    const expectedRgbaBytes = width * height * 4;
    if (expectedRgbaBytes > MAX_NATIVE_IMAGE_RGBA_BYTES) return null;

    const rgba = await image.rgba();
    if (rgba.byteLength !== expectedRgbaBytes) return null;

    const canvas = document.createElement("canvas");
    canvas.width = width;
    canvas.height = height;
    try {
      const context = canvas.getContext("2d");
      if (!context) return null;
      const pixels = new Uint8ClampedArray(
        rgba.buffer as ArrayBuffer,
        rgba.byteOffset,
        rgba.byteLength,
      );
      context.putImageData(new ImageData(pixels, width, height), 0, 0);
      const blob = await canvasPng(canvas);
      if (!blob || blob.size === 0 || blob.size > MAX_CLIPBOARD_NON_AUDIO_BYTES) {
        return null;
      }
      return new File([blob], "pasted-image.png", {
        type: "image/png",
        lastModified: Date.now(),
      });
    } finally {
      canvas.width = 0;
      canvas.height = 0;
    }
  } catch {
    return null;
  } finally {
    if (image) {
      try {
        await image.close();
      } catch {
        // The native resource may already have been released after an invoke failure.
      }
    }
  }
}

function addClipboardFiles(
  files: readonly File[],
  addFiles: (files: readonly File[]) => void | Promise<void>,
  onError?: () => void,
): void {
  void Promise.resolve(addFiles(files)).catch(() => onError?.());
}

function addNativeClipboardFiles(
  addFiles: (files: readonly File[]) => void | Promise<void>,
  onError?: () => void,
): void {
  void (async () => {
    try {
      const files = await readNativeClipboardFiles();
      if (files.length > 0) return files;
    } catch {
      // The clipboard may contain image pixels instead of file paths.
    }
    const image = await readNativeClipboardImage();
    return image ? [image] : [];
  })().then((files) => {
    if (files.length > 0) addClipboardFiles(files, addFiles, onError);
    else onError?.();
  });
}

export function pasteClipboardFiles(
  event: ClipboardPasteEvent,
  addFiles: (files: readonly File[]) => void | Promise<void>,
  onError?: () => void,
): void {
  const { clipboardData } = event;
  if (clipboardData) {
    const browserFiles = browserClipboardFiles(clipboardData);
    if (browserFiles.length > 0) {
      event.preventDefault();
      addClipboardFiles(browserFiles, addFiles, onError);
      return;
    }
  }

  if (!isTauri || !event.isTrusted || event.defaultPrevented) return;
  if (!clipboardData) {
    addNativeClipboardFiles(addFiles, onError);
    return;
  }

  const advertisesFiles = clipboardAdvertisesFiles(clipboardData);
  if (!advertisesFiles && clipboardHasPlainText(clipboardData)) return;

  if (advertisesFiles) event.preventDefault();
  addNativeClipboardFiles(addFiles, onError);
}
