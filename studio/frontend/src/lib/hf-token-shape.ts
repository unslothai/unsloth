// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Current user access tokens contain 34 characters after the hf_ prefix; OAuth access tokens
// (Sign in with Hugging Face) carry hf_oauth_ and a longer body. Only these are checked while
// typing; action-time validation still accepts legacy shapes without spending quota on every
// keystroke.
const COMPLETE_HF_TOKEN = /^hf_(?:[A-Za-z0-9]{34}|oauth_[A-Za-z0-9]{30,})$/;

/** Whether *token* is a whole token worth asking the Hub about while the user types. */
export function isCompleteHfTokenShape(token: string): boolean {
  return COMPLETE_HF_TOKEN.test(token);
}
