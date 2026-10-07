// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

const LIST_SEPARATOR = /[\n,]/;

/** One path per line or comma separated. */
export function splitComponentFileList(raw: string | null | undefined): string[] {
  if (!raw) {
    return [];
  }
  return raw
    .split(LIST_SEPARATOR)
    .map((part) => part.trim())
    .filter((part) => part.length > 0);
}

/** Only a single-file / GGUF transformer takes these, so plan and load requests stay identical. */
export function componentFileFields(
  kind: string | null | undefined,
  textEncoderFiles: string | readonly string[] | undefined,
  vaeFile: string | undefined,
): { text_encoder_file?: string[]; vae_file?: string } {
  if (kind !== "gguf" && kind !== "single_file") {
    return {};
  }
  const encoders =
    typeof textEncoderFiles === "string"
      ? splitComponentFileList(textEncoderFiles)
      : (textEncoderFiles ?? []).map((f) => f.trim()).filter((f) => f.length > 0);
  const vae = vaeFile?.trim();
  return {
    ...(encoders.length > 0 ? { text_encoder_file: encoders } : {}),
    ...(vae ? { vae_file: vae } : {}),
  };
}
