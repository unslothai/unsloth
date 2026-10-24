// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Unzip, UnzipInflate } from "fflate";

// Match the backend's manifest and library limits.
export const MAX_REWARD_BYTES = 64 * 1024;
const MAX_REWARD_FILES = 500;
export const MAX_REWARD_ZIP_BYTES = MAX_REWARD_BYTES * MAX_REWARD_FILES;

export interface RewardImportFile {
  path: string;
  markdown: string;
}

export class RewardImportError extends Error {
  readonly code:
    | "importInvalidZip"
    | "importZipEmpty"
    | "importTooLarge"
    | "importDuplicate"
    | "importConflict";
  readonly detail: string;

  constructor(code: RewardImportError["code"], detail = "") {
    super(code);
    this.code = code;
    this.detail = detail;
  }
}

export function readRewardZip(bytes: Uint8Array): RewardImportFile[] {
  if (bytes.length > MAX_REWARD_ZIP_BYTES)
    throw new RewardImportError("importTooLarge");
  const files: RewardImportFile[] = [];
  const paths = new Set<string>();
  let pending = 0;
  try {
    const unzip = new Unzip((file) => {
      if (
        file.name.startsWith("__MACOSX/") ||
        !/(^|\/)REWARD\.md$/.test(file.name)
      )
        return;
      if (paths.has(file.name))
        throw new RewardImportError("importDuplicate", file.name);
      paths.add(file.name);
      if (
        paths.size > MAX_REWARD_FILES ||
        (file.originalSize ?? 0) > MAX_REWARD_BYTES
      )
        throw new RewardImportError("importTooLarge");
      const decoder = new TextDecoder("utf-8", { fatal: true });
      let size = 0;
      let markdown = "";
      pending++;
      file.ondata = (error, chunk, final) => {
        if (error) throw error;
        size += chunk.length;
        if (size > MAX_REWARD_BYTES) {
          file.terminate();
          throw new RewardImportError("importTooLarge");
        }
        markdown += decoder.decode(chunk, { stream: !final });
        if (final) {
          files.push({ path: file.name, markdown });
          pending--;
        }
      };
      file.start();
    });
    unzip.register(UnzipInflate);
    // Bound the amount inflated in one push, including archives with false sizes.
    for (let offset = 0; offset < bytes.length; offset += 1024) {
      unzip.push(
        bytes.subarray(offset, offset + 1024),
        offset + 1024 >= bytes.length,
      );
    }
    if (pending) throw new RewardImportError("importInvalidZip");
  } catch (error) {
    if (error instanceof RewardImportError) throw error;
    throw new RewardImportError("importInvalidZip");
  }
  if (!files.length) throw new RewardImportError("importZipEmpty");
  return files;
}

export function checkRewardImportNames(
  names: string[],
  existing: string[],
  overwrite: boolean,
): void {
  const duplicate = names.find((name, i) => names.indexOf(name) !== i);
  if (duplicate) throw new RewardImportError("importDuplicate", duplicate);
  if (!overwrite) {
    const conflicts = names.filter((name) => existing.includes(name));
    if (conflicts.length)
      throw new RewardImportError("importConflict", conflicts.join(", "));
  }
}
