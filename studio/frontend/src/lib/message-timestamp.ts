// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** Importers may need a synthetic ordering key when no send time is known. */
export function messageTimestamp(message: {
  createdAt?: Date;
  metadata?: { custom?: Record<string, unknown> };
}): number | undefined {
  if (message.metadata?.custom?.createdAtEstimated === true) return undefined;
  const time = message.createdAt?.getTime();
  return time !== undefined && Number.isFinite(time) ? time : undefined;
}
