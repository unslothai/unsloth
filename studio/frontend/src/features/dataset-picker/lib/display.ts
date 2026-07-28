// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { pathDisplayName } from "@/components/resource-picker/path-display-name";

const UPLOADED_DATASET_HASH_PREFIX_RE = /^[0-9a-f]{32}_(.+)$/i;

export function datasetDisplayName(value: string): string {
  return pathDisplayName(value).replace(UPLOADED_DATASET_HASH_PREFIX_RE, "$1");
}
