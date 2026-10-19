// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { explicitFamily } from "../model-picker/components/model-selector/family-override.ts";

export interface RememberedImageModel {
  repoId: string;
  kind: "gguf" | "single_file" | "pipeline";
  filename?: string;
  // An opaque pipeline reloads only under the family it was loaded with.
  familyOverride?: string;
  // Supplied text-encoder / VAE files: status names only basenames, so a recall must carry the paths.
  textEncoderFiles?: string[];
  vaeFile?: string;
}

const KEY = "unsloth:images:last-model";

// Retired full copies of the gated FLUX.3 repos. They stay on the Hub for existing links, but Studio's copy of those
// files lives in the -FP8 repos, so a pick naming one loads the base it copies (as the backend redirects it), and the
// remembered model matches the status the load reports. The root model.safetensors is that base's bf16 transformer.
const RETIRED_FULL_COPIES: Record<string, string> = {
  "unsloth/flux-3-image-klein": "black-forest-labs/flux-3-image-klein",
  "unsloth/flux-3-image-klein-base": "black-forest-labs/flux-3-image-klein-base",
};

export function redirectRetiredImagePick<
  T extends {
    repoId: string;
    kind: RememberedImageModel["kind"];
    filename?: string;
    displayRepoId?: string;
    textEncoderFiles?: readonly string[];
    vaeFile?: string;
  },
>(pick: T): T {
  // A downloaded copy arrives as its snapshot path with the retired id as the display id.
  const base =
    RETIRED_FULL_COPIES[pick.repoId.trim().toLowerCase()] ??
    RETIRED_FULL_COPIES[(pick.displayRepoId ?? "").trim().toLowerCase()];
  if (!base) return pick;
  const { displayRepoId: _display, ...rest } = pick;
  if (pick.kind === "pipeline") return { ...rest, repoId: base } as T;
  // Supplied encoder / VAE files make it a different build, which the backend leaves on the copy too.
  const components = (pick.textEncoderFiles?.length ?? 0) > 0 || Boolean(pick.vaeFile);
  if (pick.kind === "single_file" && pick.filename === "model.safetensors" && !components) {
    const { filename: _filename, textEncoderFiles: _te, vaeFile: _vae, ...model } = rest;
    return { ...model, repoId: base, kind: "pipeline" } as T;
  }
  return pick;
}

export function readImageModel(): RememberedImageModel | null {
  try {
    const value = JSON.parse(localStorage.getItem(KEY) ?? "null");
    if (!value || typeof value.repoId !== "string" || !value.repoId.trim())
      return null;
    if (!["gguf", "single_file", "pipeline"].includes(value.kind)) return null;
    if (
      value.kind !== "pipeline" &&
      (typeof value.filename !== "string" || !value.filename)
    )
      return null;
    const family = explicitFamily(value.familyOverride);
    const encoders = Array.isArray(value.textEncoderFiles)
      ? value.textEncoderFiles.filter(
          (f: unknown): f is string => typeof f === "string" && f.length > 0,
        )
      : [];
    return redirectRetiredImagePick({
      repoId: value.repoId,
      kind: value.kind,
      ...(typeof value.filename === "string"
        ? { filename: value.filename }
        : {}),
      ...(family ? { familyOverride: family } : {}),
      ...(encoders.length > 0 ? { textEncoderFiles: encoders } : {}),
      ...(typeof value.vaeFile === "string" && value.vaeFile
        ? { vaeFile: value.vaeFile }
        : {}),
    } as RememberedImageModel);
  } catch {
    return null;
  }
}

export function rememberImageModel(model: RememberedImageModel): void {
  try {
    localStorage.setItem(KEY, JSON.stringify(model));
  } catch {
    // storage can be unavailable in private browsing
  }
}

export function matchesRememberedModel(
  model: RememberedImageModel,
  status: {
    loaded: boolean;
    repo_id?: string | null;
    model_kind?: string | null;
    gguf_filename?: string | null;
  },
): boolean {
  return (
    status.loaded &&
    status.repo_id === model.repoId &&
    status.model_kind === model.kind &&
    (model.kind === "pipeline" || status.gguf_filename === model.filename)
  );
}

const basename = (path: string): string => path.split(/[\\/]/).pop() ?? path;

/** Whether the resident build's component files (basenames, from status) are the remembered paths. */
export function componentFilesMatch(
  model: Pick<RememberedImageModel, "textEncoderFiles" | "vaeFile">,
  componentFiles: Record<string, string> | null | undefined,
): boolean {
  const remembered = [...(model.textEncoderFiles ?? []), ...(model.vaeFile ? [model.vaeFile] : [])]
    .map(basename)
    .sort();
  const resident = Object.values(componentFiles ?? {}).sort();
  return (
    remembered.length === resident.length &&
    remembered.every((name, i) => name === resident[i])
  );
}
