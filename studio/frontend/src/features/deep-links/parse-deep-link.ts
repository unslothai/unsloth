// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

const MAX_REPO_ID_SEGMENT_LENGTH = 96;
const MAX_GGUF_FILE_LENGTH = 512;
const REPO_SEGMENT = /^[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?$/;
function hasControlCharacters(value: string): boolean {
  return [...value].some((character) => {
    const codePoint = character.codePointAt(0) ?? 0;
    return codePoint <= 0x1f || codePoint === 0x7f;
  });
}

export interface UnslothDeepLinkIntent {
  model: string;
  file?: string;
}

function isValidRepoSegment(segment: string): boolean {
  return (
    segment.length <= MAX_REPO_ID_SEGMENT_LENGTH &&
    REPO_SEGMENT.test(segment) &&
    !segment.includes("--") &&
    !segment.includes("..")
  );
}

function isValidGgufFile(file: string): boolean {
  if (
    file.length === 0 ||
    file.length > MAX_GGUF_FILE_LENGTH ||
    file !== file.trim() ||
    hasControlCharacters(file) ||
    file.includes("\\") ||
    file.startsWith("/") ||
    !file.toLowerCase().endsWith(".gguf")
  ) {
    return false;
  }
  return file
    .split("/")
    .every((segment) => segment !== "" && segment !== "." && segment !== "..");
}

export function parseUnslothDeepLink(
  rawUrl: string,
): UnslothDeepLinkIntent | null {
  const queryIndex = rawUrl.indexOf("?");
  const target = queryIndex === -1 ? rawUrl : rawUrl.slice(0, queryIndex);
  if (
    target !== "unsloth://open_from_hf" &&
    target !== "unsloth://open_from_hf/"
  ) {
    return null;
  }

  let url: URL;
  try {
    url = new URL(rawUrl);
  } catch {
    return null;
  }

  if (
    url.protocol !== "unsloth:" ||
    url.hostname !== "open_from_hf" ||
    (url.pathname !== "" && url.pathname !== "/") ||
    url.username !== "" ||
    url.password !== "" ||
    url.port !== "" ||
    url.hash !== ""
  ) {
    return null;
  }

  const keys = [...url.searchParams.keys()];
  if (
    keys.length < 1 ||
    keys.length > 2 ||
    !keys.includes("model") ||
    new Set(keys).size !== keys.length ||
    keys.some((key) => key !== "model" && key !== "file")
  ) {
    return null;
  }

  const model = url.searchParams.get("model") ?? "";
  const segments = model.split("/");
  if (
    model.endsWith(".git") ||
    segments.length !== 2 ||
    !segments.every(isValidRepoSegment)
  ) {
    return null;
  }

  const file = url.searchParams.get("file");
  if (file !== null && !isValidGgufFile(file)) return null;

  return file === null ? { model } : { model, file };
}
