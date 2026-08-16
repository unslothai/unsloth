// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

const UPLOADED_DATASET_HASH_PREFIX_RE = /^[0-9a-f]{32}_(.+)$/i;

export function datasetDisplayName(value: string): string {
  const parts = value.replaceAll("\\", "/").split("/").filter(Boolean);
  const parquetFilesIndex = parts.lastIndexOf("parquet-files");
  if (parquetFilesIndex > 0) {
    return parts[parquetFilesIndex - 1];
  }
  const basename = parts[parts.length - 1] ?? value;
  return basename.replace(UPLOADED_DATASET_HASH_PREFIX_RE, "$1");
}
