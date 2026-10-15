// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export function preserveThinkingDefaultFromLoad(resp: {
  supports_preserve_thinking?: boolean | null;
  preserve_thinking_default?: boolean | null;
}): boolean {
  return Boolean(
    resp.supports_preserve_thinking && resp.preserve_thinking_default,
  );
}
