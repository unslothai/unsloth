// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

const PATH_SEPARATOR_RE = /[\\/]/;
const UPLOADED_DATASET_HASH_PREFIX_RE = /^[0-9a-f]{32}_(.+)$/i;

export function datasetDisplayName(value: string): string {
  const leaf = value.split(PATH_SEPARATOR_RE).pop() ?? value;
  return leaf.replace(UPLOADED_DATASET_HASH_PREFIX_RE, "$1");
}
