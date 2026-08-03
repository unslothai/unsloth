// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { ToolScaffoldFile } from "../../types";

export function updateToolScaffoldRow(
  rows: ToolScaffoldFile[] | undefined,
  index: number,
  next: ToolScaffoldFile,
): ToolScaffoldFile[] {
  const current = Array.isArray(rows) ? rows : [];
  return current.map((file, fileIndex) => (fileIndex === index ? next : file));
}

export function removeToolScaffoldRow(
  rows: ToolScaffoldFile[] | undefined,
  index: number,
): ToolScaffoldFile[] {
  const current = Array.isArray(rows) ? rows : [];
  return current.filter((_, fileIndex) => fileIndex !== index);
}

export function addToolScaffoldRow(
  rows: ToolScaffoldFile[] | undefined,
): ToolScaffoldFile[] {
  const current = Array.isArray(rows) ? rows : [];
  return [...current, { path: "", content: "" }];
}
