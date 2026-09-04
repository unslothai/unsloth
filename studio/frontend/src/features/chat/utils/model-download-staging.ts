// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** A pick is a GGUF: HF variant, native file, or a direct local .gguf. */
export function hasGgufSource(x: {
  ggufVariant?: string;
  nativePathToken?: string;
  isGguf?: boolean;
}): boolean {
  return (
    x.ggufVariant != null || x.nativePathToken != null || x.isGguf === true
  );
}

/** A local-disk model id: Unix absolute, relative, tilde, Windows drive or UNC. Shared so the
 *  loader and the hub-repo predicate classify ids identically. */
export function isLocalModelPath(id: string): boolean {
  return /^(\/|\.{1,2}[\\/]|~[\\/]|[A-Za-z]:[\\/]|\\\\)/.test(id);
}

/** An uncached HF hub repo we can download as a full snapshot (non-GGUF safetensors / MLX).
 *  Excludes GGUF sources, local paths, native files, LoRA and external provider models so none
 *  are mis-routed into a snapshot. */
export function isDownloadableHubRepo(x: {
  id: string;
  source?: string;
  isLora?: boolean;
  ggufVariant?: string;
  nativePathToken?: string;
  isGguf?: boolean;
}): boolean {
  return (
    x.source === "hub" &&
    !hasGgufSource(x) &&
    x.isLora !== true &&
    x.nativePathToken == null &&
    !isLocalModelPath(x.id)
  );
}

/** A pick the Hub download manager must fetch first: an uncached snapshot repo, or a Hub GGUF
 *  whose quant is not on disk. Everything else loads directly, so a caller that cannot prove a
 *  Hub pick is missing must leave `source` unset. */
export function wantsDownloadManagerStaging(x: {
  id: string;
  source?: string;
  isLora?: boolean;
  ggufVariant?: string;
  nativePathToken?: string;
  isGguf?: boolean;
  isDownloaded?: boolean;
}): boolean {
  if (x.isDownloaded) return false;
  return isDownloadableHubRepo(x) || (x.source === "hub" && hasGgufSource(x));
}
