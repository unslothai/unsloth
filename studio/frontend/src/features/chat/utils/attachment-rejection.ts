// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** Marker for an attachment rejection whose reason the adapter has already shown.
 *
 * The attachment adapters toast the reason they refuse a file and then reject, so
 * the rejection carries no new information for the caller. A caller that treats it
 * as an unexplained failure adds a SECOND toast on top -- and the paste path's
 * generic one ("the clipboard item is unsupported, unreadable, or exceeds its size
 * limit") contradicts the first, telling someone who turned Vision off that their
 * screenshot is broken. Tagging the rejection lets a caller skip its own message
 * for exactly the failures that already have one, while still reporting the rest. */
const ALREADY_TOASTED = "AttachmentRejectionAlreadyToasted";

export function attachmentRejectionAlreadyToasted(message: string): Error {
  const error = new Error(message);
  error.name = ALREADY_TOASTED;
  return error;
}

export function isAttachmentRejectionAlreadyToasted(error: unknown): boolean {
  return error instanceof Error && error.name === ALREADY_TOASTED;
}
