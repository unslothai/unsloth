// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** A port/wire colour that stays readable on both themes (see `.flows-editor` in index.css). */
export function portInk(color: string): string {
  return `color-mix(in oklab, ${color} var(--flows-port-mix, 100%), black)`;
}
