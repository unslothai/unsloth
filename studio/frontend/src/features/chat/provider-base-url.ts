// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

const TRAILING_SLASHES = /\/+$/;
const ENDPOINT_PATH = /\/(?:chat\/completions|completions|responses|models)$/;

// A full endpoint pasted as the base URL would otherwise become .../chat/completions/models.
export function stripEndpointPath(pathname: string): string {
  return pathname.replace(TRAILING_SLASHES, "").replace(ENDPOINT_PATH, "");
}
