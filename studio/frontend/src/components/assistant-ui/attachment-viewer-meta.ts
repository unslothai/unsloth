// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { formatBytes } from "@/features/hub";

/** The zoom steps a page, grid or source file offers, as the Library's viewer does. */
export const ATTACHMENT_PAGE_SCALES = [0.5, 0.75, 1, 1.25, 1.5, 2];

/** The attachment's type as the Library's subtitle names it: its extension, or failing that a
 *  MIME subtype. */
function typeLabel(name: string, contentType: string | undefined): string | null {
  const dot = name.lastIndexOf(".");
  if (dot > 0 && dot < name.length - 1) return name.slice(dot + 1).toUpperCase();
  const subtype = contentType?.split("/")[1]?.split(";")[0];
  return subtype ? subtype.toUpperCase() : null;
}

/** The Library viewer's subtitle for an attachment: type, size, and anything the body adds. */
export function attachmentViewerMeta(
  source: { name: string; contentType: string | undefined },
  bytes: number | null | undefined,
  ...extra: Array<string | null | false | undefined>
): string {
  return [typeLabel(source.name, source.contentType), bytes ? formatBytes(bytes) : null, ...extra]
    .filter(Boolean)
    .join(" · ");
}
