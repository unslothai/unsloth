// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

const DOWNLOAD_LINE = /^Downloading (.+):\s+(.+)$/;
const KNOWN_SIZE = /^(\d+(?:\.\d+)?)% \(.+\) at .+\/s$/;
const UNKNOWN_SIZE = /^.+ downloaded at .+\/s$/;

export function parseBackendDownloadProgress(line: string) {
  const match = DOWNLOAD_LINE.exec(line.trim());
  if (!match) {
    return null;
  }

  const [, file, detail] = match;
  const knownSize = KNOWN_SIZE.exec(detail);
  if (knownSize) {
    const percent = Number(knownSize[1]);
    if (percent > 100) {
      return null;
    }
    return { file, detail, percent };
  }
  if (UNKNOWN_SIZE.test(detail)) {
    return { file, detail, percent: null };
  }
  return null;
}
