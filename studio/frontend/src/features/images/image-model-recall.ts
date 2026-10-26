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

/** A retired copy pick (its id or a snapshot): the backend loads it from disk when complete, else through its base. */
export function isRetiredCopySnapshot(pick: { repoId: string; displayRepoId?: string }): boolean {
  return Boolean(retiredCopyOf(pick));
}

function retiredCopyOf(pick: { repoId: string; displayRepoId?: string }): string {
  const id = pick.repoId.trim().toLowerCase();
  const display = (pick.displayRepoId ?? "").trim().toLowerCase();
  if (RETIRED_FULL_COPIES[id]) return id;
  return RETIRED_FULL_COPIES[display] ? display : retiredSnapshotRepo(pick.repoId);
}

/** A retired snapshot that loaded through its base, as that base (else the pick), so Reapply and recall match the status. */
export function retiredSnapshotAsLoaded<T extends RememberedImageModel & { displayRepoId?: string; baseRepo?: string }>(
  pick: T,
  status: Parameters<typeof matchesRememberedModel>[1],
): T {
  if (!isRetiredCopySnapshot(pick) || matchesRememberedModel(pick, status)) return pick;
  // A complete snapshot's root model.safetensors pick loads as the snapshot's own pipeline.
  const { filename: _filename, baseRepo: _base, ...asPipeline } = { ...pick, kind: "pipeline" as const };
  if (matchesRememberedModel(asPipeline as T, status)) return asPipeline as T;
  const copy = retiredCopyOf(pick);
  // A redirect loads another checkpoint, whose base the backend infers (handleLoad drops it the same way).
  const { baseRepo: _baseRepo, ...rest } = pick;
  const asBase = redirectRetiredImagePick({ ...rest, repoId: copy } as T);
  return asBase.repoId !== copy && matchesRememberedModel(asBase, status) ? asBase : pick;
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
  const base = RETIRED_FULL_COPIES[pick.repoId.trim().toLowerCase()];
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
    return {
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
    } as RememberedImageModel;
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

// Encoder / VAE paths each model last loaded with, most recent last.
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

/** Advanced-field files a load of `model` sends: a field still holding `owner`'s value is not a user edit, so the model's saved file replaces it. */
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

/** A recalled model's files: its per-model entry, else the last-model slot's. */
export function recalledComponentFiles(model: RememberedImageModel): ImageComponentFiles {
  return readImageComponents(model) ?? cleanComponentFiles(model);
}

/** Seed the remembered model's files on page open, migrating the last-model slot into the map. */
export function seedImageComponents(model: RememberedImageModel): ImageComponentFiles {
  const saved = readImageComponents(model);
  if (saved) return saved;
  const slot = cleanComponentFiles(model);
  if (slot.textEncoderFiles || slot.vaeFile) rememberImageComponents(model, slot);
  return slot;
}

/** Forget a model's saved files when its resident build uses others; native (sd_cpp) builds keep the entry. */
export function forgetStaleImageComponents(
  model: ImageModelKey,
  status: { engine?: string | null; component_files?: Record<string, string> | null },
): void {
  if (status.engine === "sd_cpp") return;
  const saved = readImageComponents(model);
  if (saved && !componentFilesMatch(saved, status.component_files))
    rememberImageComponents(model, engagedComponentFiles(saved, status.component_files));
}

/** The part of `files` the resident build still engages, encoders and VAE judged separately. */
function engagedComponentFiles(
  files: ImageComponentFiles,
  componentFiles: Record<string, string> | null | undefined,
): ImageComponentFiles {
  const { vae, ...encoders } = componentFiles ?? {};
  const clean = cleanComponentFiles(files);
  return cleanComponentFiles({
    textEncoderFiles: componentFilesMatch({ textEncoderFiles: clean.textEncoderFiles }, encoders)
      ? clean.textEncoderFiles
      : undefined,
    vaeFile: componentFilesMatch({ vaeFile: clean.vaeFile }, vae ? { vae } : {}) ? clean.vaeFile : undefined,
  });
}

/** Forget the resident model's saved files (and the slot's, if it names that model) when its build does not use them. */
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
  const next = { ...bare, ...engagedComponentFiles(files, status.component_files) };
  rememberImageModel(next);
  return next;
}

/** First loaded status: drop the remembered model's files its resident build no longer uses; returns the trimmed model or null. */
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
): (Omit<T, "textEncoderFiles" | "vaeFile"> & ImageComponentFiles) | null {
  const saved = status.loaded && matchesRememberedModel(model, status) ? readImageComponents(model) : null;
  if (!saved) return null;
  forgetStaleImageComponents(model, status);
  // What the build still engages (one part when a reload changed only the other), or nothing.
  const kept = readImageComponents(model) ?? {};
  if (sameComponentFiles(kept, saved)) return null;
  const slot = readImageModel();
  if (slot && componentsKey(slot) === componentsKey(model)) {
    const { textEncoderFiles: _te, vaeFile: _vae, ...bare } = slot;
    rememberImageModel({ ...bare, ...engagedComponentFiles(slot, status.component_files) });
  }
  const { textEncoderFiles: _te, vaeFile: _vae, ...rest } = model;
  return { ...rest, ...kept };
}

/** Record a finished load's files: a page-started load stores its request; a resumed one keeps or forgets the entry by the build. */
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

/** Per-field owner after a load: that load only while the field shows its file; field-ignoring loads (pipelines) leave ownership. */
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

/** Fields after a load recovered on reopen: untouched fields take the recovered files, edits stay. */
export function recoveredComponentFields(
  shown: ImageComponentFiles,
  owner: ImageComponentFiles,
  loaded: ImageComponentFiles,
  kind: RememberedImageModel["kind"],
): { shown: ImageComponentFiles; replaceEncoders: boolean; replaceVae: boolean } {
  if (kind !== "gguf" && kind !== "single_file") return { shown, replaceEncoders: false, replaceVae: false };
  const replaceEncoders = sameEncoders(shown, owner);
  const replaceVae = sameVae(shown, owner);
  return {
    shown: cleanComponentFiles({
      textEncoderFiles: (replaceEncoders ? loaded : shown).textEncoderFiles,
      vaeFile: (replaceVae ? loaded : shown).vaeFile,
    }),
    replaceEncoders,
    replaceVae,
  };
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

// The pending load and its requested paths: the backend load outlives a page reload and status gives only basenames.
const PENDING_KEY = "unsloth:images:pending-load";

export type PendingImageLoad = RememberedImageModel & { displayRepoId?: string; baseRepo?: string; fieldFiles?: ImageComponentFiles };

export function rememberPendingImageLoad(load: PendingImageLoad): void {
  try {
    localStorage.setItem(PENDING_KEY, JSON.stringify(load));
  } catch {
    // storage can be unavailable in private browsing
  }
}

export function clearPendingImageLoad(): void {
  try {
    localStorage.removeItem(PENDING_KEY);
  } catch {
    // storage can be unavailable in private browsing
  }
}

function readPendingImageLoad(): PendingImageLoad | null {
  try {
    const value = JSON.parse(localStorage.getItem(PENDING_KEY) ?? "null");
    if (!value || typeof value.repoId !== "string" || !value.repoId) return null;
    if (!["gguf", "single_file", "pipeline"].includes(value.kind)) return null;
    return { ...value, fieldFiles: cleanComponentFiles(value.fieldFiles) };
  } catch {
    return null;
  }
}

/** The descriptor a finished load reconciles with, consuming the pending one if the resident build is that load. */
export function finishedLoadTarget<T extends ImageModelKey & { fieldFiles?: ImageComponentFiles }>(
  current: T | null,
  status: Parameters<typeof matchesRememberedModel>[1] & Parameters<typeof forgetStaleImageComponents>[1],
): T | PendingImageLoad | null {
  const stored = readPendingImageLoad();
  clearPendingImageLoad();
  if (current?.fieldFiles) return current;
  // A retired snapshot is published as what it loaded as (its base, or its own pipeline for a root pick).
  const pending = stored ? retiredSnapshotAsLoaded(stored, status) : stored;
  if (
    pending &&
    matchesRememberedModel(pending, status) &&
    (status.engine === "sd_cpp" || componentFilesMatch(pending.fieldFiles ?? {}, status.component_files))
  )
    return pending;
  return current;
}
