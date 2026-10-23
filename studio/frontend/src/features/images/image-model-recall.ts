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

// ``.../models--unsloth--flux-3-image-klein/snapshots/<sha>`` -> the repo id it is a snapshot of ("" otherwise).
function retiredSnapshotRepo(path: string): string {
  const match = /(?:^|[\\/])models--unsloth--(flux-3-image-klein(?:-base)?)[\\/]snapshots[\\/][^\\/]+[\\/]?$/i.exec(
    path.trim(),
  );
  return match ? `unsloth/${match[1].toLowerCase()}` : "";
}

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
  // A downloaded copy arrives as its snapshot path with the retired id as the display id; a model remembered by an
  // older build is that snapshot path alone.
  const base =
    RETIRED_FULL_COPIES[pick.repoId.trim().toLowerCase()] ??
    RETIRED_FULL_COPIES[(pick.displayRepoId ?? "").trim().toLowerCase()] ??
    RETIRED_FULL_COPIES[retiredSnapshotRepo(pick.repoId)];
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

export type ImageModelKey = Pick<RememberedImageModel, "repoId" | "kind" | "filename">;
export type ImageComponentFiles = Pick<RememberedImageModel, "textEncoderFiles" | "vaeFile">;

// The encoder / VAE paths each model last loaded with, most recent last, so a pick can refill the Advanced fields.
const COMPONENTS_KEY = "unsloth:images:component-files";
const COMPONENTS_CAP = 50;

type ComponentEntry = ImageComponentFiles & { key: string };

const componentsKey = ({ repoId, kind, filename }: ImageModelKey): string =>
  [repoId, kind, kind === "pipeline" ? "" : (filename ?? "")].join("\n");

function cleanComponentFiles(value: unknown): ImageComponentFiles {
  const raw = (value ?? {}) as { textEncoderFiles?: unknown; vaeFile?: unknown };
  const encoders = Array.isArray(raw.textEncoderFiles)
    ? raw.textEncoderFiles.filter((f): f is string => typeof f === "string" && f.trim().length > 0)
    : [];
  const vae = typeof raw.vaeFile === "string" ? raw.vaeFile.trim() : "";
  return {
    ...(encoders.length > 0 ? { textEncoderFiles: encoders } : {}),
    ...(vae ? { vaeFile: vae } : {}),
  };
}

function readComponentEntries(): ComponentEntry[] {
  try {
    const value = JSON.parse(localStorage.getItem(COMPONENTS_KEY) ?? "[]");
    return Array.isArray(value)
      ? value.filter((e): e is ComponentEntry => Boolean(e) && typeof e.key === "string")
      : [];
  } catch {
    return [];
  }
}

export function sameComponentFiles(a: ImageComponentFiles, b: ImageComponentFiles): boolean {
  const x = cleanComponentFiles(a);
  const y = cleanComponentFiles(b);
  return (x.textEncoderFiles ?? []).join("\n") === (y.textEncoderFiles ?? []).join("\n") && x.vaeFile === y.vaeFile;
}

export function readImageComponents(model: ImageModelKey): ImageComponentFiles | null {
  const key = componentsKey(model);
  const entry = readComponentEntries().find((e) => e.key === key);
  const files = cleanComponentFiles(entry);
  return files.textEncoderFiles || files.vaeFile ? files : null;
}

const sameEncoders = (a: ImageComponentFiles, b: ImageComponentFiles): boolean =>
  (cleanComponentFiles(a).textEncoderFiles ?? []).join("\n") === (cleanComponentFiles(b).textEncoderFiles ?? []).join("\n");
const sameVae = (a: ImageComponentFiles, b: ImageComponentFiles): boolean =>
  cleanComponentFiles(a).vaeFile === cleanComponentFiles(b).vaeFile;

/** What a load of `model` sends from the Advanced fields, the encoder and VAE field each on its own: a field still
 *  holding `owner`'s value (what was filled in or loaded last) is not the user's edit, so it gives way to the model's
 *  own saved file, and no entry sends none. `owner` is the new owner (an edited field keeps its old one, so it stays an
 *  edit until a load takes it); `replaceEncoders` / `replaceVae` say which fields to refill. */
export function componentFilesForModel(
  typed: ImageComponentFiles,
  owner: ImageComponentFiles,
  model: ImageModelKey,
): { files: ImageComponentFiles; owner: ImageComponentFiles; replaceEncoders: boolean; replaceVae: boolean } {
  const replaceEncoders = sameEncoders(typed, owner);
  const replaceVae = sameVae(typed, owner);
  if (!replaceEncoders && !replaceVae) return { files: typed, owner, replaceEncoders, replaceVae };
  const saved = readImageComponents(model) ?? {};
  const pick = (encoders: ImageComponentFiles, vae: ImageComponentFiles) =>
    cleanComponentFiles({ textEncoderFiles: encoders.textEncoderFiles, vaeFile: vae.vaeFile });
  return {
    files: pick(replaceEncoders ? saved : typed, replaceVae ? saved : typed),
    owner: pick(replaceEncoders ? saved : owner, replaceVae ? saved : owner),
    replaceEncoders,
    replaceVae,
  };
}

/** The files a recalled model loads with: its per-model entry (which keeps what a native load strips from the
 *  request), else the last-model slot's, which an older build or an evicted entry leaves as the only record. */
export function recalledComponentFiles(model: RememberedImageModel): ImageComponentFiles {
  return readImageComponents(model) ?? cleanComponentFiles(model);
}

/** The remembered model's files on page open, copying the last-model slot's into the map when it has no entry (an
 *  older build kept them only there), so a pick of that model with the fields untouched keeps them. */
export function seedImageComponents(model: RememberedImageModel): ImageComponentFiles {
  const saved = readImageComponents(model);
  if (saved) return saved;
  const slot = cleanComponentFiles(model);
  if (slot.textEncoderFiles || slot.vaeFile) rememberImageComponents(model, slot);
  return slot;
}

/** Forget a model's saved files when its resident build (status ``component_files``, basenames) uses others, e.g. a
 *  reload by another client without them, so a later recall or pick does not bring back what no longer engages. A
 *  native (sd_cpp) build never takes supplied files, so there the entry records the request and stays. */
export function forgetStaleImageComponents(
  model: ImageModelKey,
  status: { engine?: string | null; component_files?: Record<string, string> | null },
): void {
  if (status.engine === "sd_cpp") return;
  const saved = readImageComponents(model);
  if (saved && !componentFilesMatch(saved, status.component_files)) rememberImageComponents(model, {});
}

/** Forget the resident model's own saved files when its build does not use them, on every settled loaded status: the
 *  page may remember another model, or the resident build may change after the first status, and a pick or Reapply of
 *  the resident model would otherwise send paths the status shows it no longer loads. The last-model slot loses them
 *  too when it names the resident model, since recall falls back to it: returns that slot without its files when it
 *  dropped them, else null. */
export function forgetStaleResidentComponents(status: {
  loaded: boolean;
  repo_id?: string | null;
  model_kind?: string | null;
  gguf_filename?: string | null;
  engine?: string | null;
  component_files?: Record<string, string> | null;
}): RememberedImageModel | null {
  const kind = status.model_kind;
  if (!status.loaded || !status.repo_id) return null;
  let resident: ImageModelKey;
  if (kind === "pipeline") resident = { repoId: status.repo_id, kind };
  else if ((kind === "gguf" || kind === "single_file") && status.gguf_filename)
    resident = { repoId: status.repo_id, kind, filename: status.gguf_filename };
  else return null;
  forgetStaleImageComponents(resident, status);
  if (status.engine === "sd_cpp") return null;
  const slot = readImageModel();
  if (!slot || componentsKey(slot) !== componentsKey(resident)) return null;
  const files = cleanComponentFiles(slot);
  if ((!files.textEncoderFiles && !files.vaeFile) || componentFilesMatch(files, status.component_files)) return null;
  const { textEncoderFiles: _te, vaeFile: _vae, ...bare } = slot;
  rememberImageModel(bare);
  return bare;
}

/** On the first loaded status: forget the remembered model's saved files when that same model is resident on a
 *  build that does not use them (another client reloaded it without them), so the page does not reload them as
 *  untouched fields. The last-model slot loses them too, since seed and recall fall back to it. Returns the model
 *  without its files when they were dropped, else null. */
export function reconcileRememberedComponents<T extends ImageModelKey & ImageComponentFiles>(
  model: T,
  status: {
    loaded: boolean;
    repo_id?: string | null;
    model_kind?: string | null;
    gguf_filename?: string | null;
    engine?: string | null;
    component_files?: Record<string, string> | null;
  },
): Omit<T, "textEncoderFiles" | "vaeFile"> | null {
  if (!status.loaded || !matchesRememberedModel(model, status) || !readImageComponents(model)) return null;
  forgetStaleImageComponents(model, status);
  if (readImageComponents(model) !== null) return null;
  const slot = readImageModel();
  if (slot && componentsKey(slot) === componentsKey(model)) {
    const { textEncoderFiles: _te, vaeFile: _vae, ...bare } = slot;
    rememberImageModel(bare);
  }
  const { textEncoderFiles: _te, vaeFile: _vae, ...rest } = model;
  return rest;
}

/** Record a finished load's files: a load this page started stores what it asked for (`fieldFiles`, which a native
 *  load strips from the request), and returns them. A load it only resumed (re-entered mid-load, so `lastLoad` is the
 *  resident descriptor) carries no request, so the saved entry is kept if the build engages it and forgotten if not. */
export function rememberLoadedImageComponents(
  load: ImageModelKey & { fieldFiles?: ImageComponentFiles },
  status: { engine?: string | null; component_files?: Record<string, string> | null },
): ImageComponentFiles | null {
  if (!load.fieldFiles) {
    forgetStaleImageComponents(load, status);
    return null;
  }
  rememberImageComponents(load, load.fieldFiles);
  return load.fieldFiles;
}

/** Who owns the fields after a load finished, field by field: that load, only while the field shows its file. A load that never touched
 *  them (an adapter deploy, a recalled generation) leaves them with the model that filled them, and so does a load of
 *  a `kind` that ignores the fields (a pipeline), whose empty request would otherwise match a field the user cleared. */
export function componentFieldsOwnerAfterLoad(
  shown: ImageComponentFiles,
  loaded: ImageComponentFiles,
  owner: ImageComponentFiles,
  kind: RememberedImageModel["kind"],
): ImageComponentFiles {
  if (kind !== "gguf" && kind !== "single_file") return cleanComponentFiles(owner);
  return cleanComponentFiles({
    textEncoderFiles: (sameEncoders(shown, loaded) ? loaded : owner).textEncoderFiles,
    vaeFile: (sameVae(shown, loaded) ? loaded : owner).vaeFile,
  });
}

/** Store the paths a load succeeded with; a load with none forgets the model's entry. */
export function rememberImageComponents(model: ImageModelKey, files: ImageComponentFiles): void {
  const key = componentsKey(model);
  const clean = cleanComponentFiles(files);
  const entries = readComponentEntries().filter((e) => e.key !== key);
  if (clean.textEncoderFiles || clean.vaeFile) entries.push({ key, ...clean });
  try {
    localStorage.setItem(COMPONENTS_KEY, JSON.stringify(entries.slice(-COMPONENTS_CAP)));
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
