// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { toast } from "@/lib/toast";

const MAX_AVATAR_BYTES = 20 * 1024 * 1024;

/** Upload types the server takes, with their file extensions (the first one names a download). */
export const AVATAR_TYPES: Record<string, string[]> = {
  "image/png": ["png"],
  "image/jpeg": ["jpg", "jpeg"],
  "image/webp": ["webp"],
  "image/gif": ["gif"],
};

export const AVATAR_ACCEPT = Object.keys(AVATAR_TYPES).join(",");
export const AVATAR_EXTENSIONS = Object.values(AVATAR_TYPES)
  .flat()
  .map((extension) => `.${extension}`)
  .join(",");

/** False, with a toast, for an avatar over the server's upload limit. */
export function avatarFits(file: File, t: (key: "characters.editor.pictureTooLarge") => string): boolean {
  if (file.size <= MAX_AVATAR_BYTES) return true;
  toast.error(t("characters.editor.pictureTooLarge"));
  return false;
}

/** An avatar change waiting for the save: a new file, a removal, or none. */
export type AvatarChoice = File | "remove" | null;

/** Applies a pending avatar change once the record's fields are saved. */
export async function applyAvatarChoice<R extends { id: string; hasAvatar: boolean }>(
  saved: R,
  choice: AvatarChoice,
  upload: (id: string, file: File) => Promise<R>,
  remove: (id: string) => Promise<R>,
): Promise<R> {
  if (choice instanceof File) return upload(saved.id, choice);
  if (choice === "remove" && saved.hasAvatar) return remove(saved.id);
  return saved;
}
