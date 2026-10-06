// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

const LIST_SEPARATOR = /[\n,]/;

/** Split the "Text encoder file(s)" field: one path per line or comma separated, trimmed, blanks dropped. */
export function splitComponentFileList(raw: string | null | undefined): string[] {
  if (!raw) {
    return [];
  }
  return raw
    .split(LIST_SEPARATOR)
    .map((part) => part.trim())
    .filter((part) => part.length > 0);
}

/** The separate text-encoder / VAE fields a load or download plan sends. The backend accepts them
 *  only for a single-file or GGUF transformer, so any other kind (and empty values) sends nothing,
 *  which keeps the plan and the load requests identical. */
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
