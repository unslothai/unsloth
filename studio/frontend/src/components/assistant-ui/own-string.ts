// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export const ownString = (text: string): string =>
  text.length < 16 ? text : (" " + text).slice(1);
