// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Only current shapes (34 chars after hf_, or hf_oauth_) are checked while typing; action-time
// validation still accepts legacy shapes.
const COMPLETE_HF_TOKEN = /^hf_(?:[A-Za-z0-9]{34}|oauth_[A-Za-z0-9]{30,})$/;

export function isCompleteHfTokenShape(token: string): boolean {
  return COMPLETE_HF_TOKEN.test(token);
}
