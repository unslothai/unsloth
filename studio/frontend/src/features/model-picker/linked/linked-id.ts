// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** `@name/<id>`: served by a linked instance through this server's /v1. */
export function isLinkedModelId(value: string | null | undefined): boolean {
  return typeof value === "string" && value.startsWith("@");
}

/** A path on a linked instance, reached through this server so its key never leaves it. */
export function linkedProxyPath(instanceId: string, path: string): string {
  return `/api/linked-instances/${encodeURIComponent(instanceId)}/proxy${path}`;
}
