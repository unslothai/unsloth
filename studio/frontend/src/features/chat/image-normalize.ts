// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export const CHAT_IMAGE_MIMES = [
  "image/jpeg",
  "image/png",
  "image/webp",
  "image/gif",
  "image/heic",
  "image/heif",
  "image/avif",
  "image/bmp",
  "image/tiff",
];

const CONVERTED_IMAGE_TYPES: Record<string, string> = {
  heic: "image/jpeg",
  heif: "image/jpeg",
  avif: "image/png",
  bmp: "image/png",
  tif: "image/png",
  tiff: "image/png",
};
const CONVERTED_IMAGE_EXTENSIONS = Object.keys(CONVERTED_IMAGE_TYPES).map(
  (extension) => `.${extension}`,
);

export const CHAT_IMAGE_EXTENSIONS = [
  ".jpg,.jpeg,.png,.webp,.gif",
  ...CONVERTED_IMAGE_EXTENSIONS,
].join(",");
export const CHAT_IMAGE_ACCEPT = [
  ...CHAT_IMAGE_MIMES,
  ...CONVERTED_IMAGE_EXTENSIONS,
].join(",");

function imageKind(file: { name: string; type: string }): string | null {
  const mime = file.type.toLowerCase();
  if (CHAT_IMAGE_MIMES.includes(mime)) {
    return mime.slice("image/".length);
  }
  const extension = file.name.toLowerCase().split(".").pop() ?? "";
  return Object.hasOwn(CONVERTED_IMAGE_TYPES, extension) ? extension : null;
}

export function isChatImageFile(file: { name: string; type: string }): boolean {
  return imageKind(file) !== null;
}

export function convertedImageType(file: {
  name: string;
  type: string;
}): string | null {
  const kind = imageKind(file);
  return kind === null ? null : (CONVERTED_IMAGE_TYPES[kind] ?? null);
}

const MAX_CONVERTED_IMAGE_BYTES = 20 * 1024 * 1024;

export async function normalizeChatImage(file: File): Promise<File> {
  const type = convertedImageType(file);
  if (type === null) {
    return file;
  }
  const url = URL.createObjectURL(file);
  try {
    const image = new Image();
    image.src = url;
    try {
      await image.decode();
    } catch {
      throw new Error(
        `This app can't read ${file.name}. Convert it to JPEG or PNG and attach it again.`,
      );
    }
    const canvas = document.createElement("canvas");
    canvas.width = image.naturalWidth;
    canvas.height = image.naturalHeight;
    const context = canvas.getContext("2d");
    if (!context) {
      throw new Error(`Could not convert ${file.name}.`);
    }
    if (type === "image/jpeg") {
      context.fillStyle = "#fff";
      context.fillRect(0, 0, canvas.width, canvas.height);
    }
    context.drawImage(image, 0, 0);
    const blob = await new Promise<Blob | null>((resolve) =>
      canvas.toBlob(resolve, type, 0.92),
    );
    if (!blob) {
      throw new Error(`Could not convert ${file.name}.`);
    }
    if (blob.size > MAX_CONVERTED_IMAGE_BYTES) {
      throw new Error(`${file.name} is over 20 MB once converted.`);
    }
    const extension = type === "image/jpeg" ? ".jpg" : ".png";
    return new File([blob], file.name.replace(/\.[^.]*$/, "") + extension, {
      type,
      lastModified: file.lastModified,
    });
  } finally {
    URL.revokeObjectURL(url);
  }
}
