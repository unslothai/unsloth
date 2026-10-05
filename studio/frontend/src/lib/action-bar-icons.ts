// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { IconSvgElement } from "@hugeicons/react";

// Message action bar glyphs, rescaled about the centre so they match Refresh01Icon's optical size
// at the same stroke. Hugeicons (MIT).

const path = (d: string, key: string) =>
  ["path", { d, stroke: "currentColor", strokeLinecap: "round", strokeLinejoin: "round", key }] as const;

// copy-01 at 0.9x: the stock one fills the whole canvas.
export const CopyActionIcon = [
  path(
    "M9.3 14.7C9.3 12.1544 9.3 10.8817 10.0908 10.0908C10.8817 9.3 12.1544 9.3 14.7 9.3L15.6 9.3C18.1456 9.3 19.4183 9.3 20.2092 10.0908C21 10.8817 21 12.1544 21 14.7V15.6C21 18.1456 21 19.4183 20.2092 20.2092C19.4183 21 18.1456 21 15.6 21H14.7C12.1544 21 10.8817 21 10.0908 20.2092C9.3 19.4183 9.3 18.1456 9.3 15.6L9.3 14.7Z",
    "0",
  ),
  path(
    "M16.4999 9.3C16.4977 6.6386 16.4575 5.2601 15.6828 4.3162C15.5332 4.1339 15.3661 3.9668 15.1838 3.8172C14.1881 3 12.7088 3 9.75 3C6.7913 3 5.3119 3 4.3162 3.8172C4.1339 3.9668 3.9668 4.1339 3.8172 4.3162C3 5.3119 3 6.7913 3 9.75C3 12.7088 3 14.1881 3.8172 15.1838C3.9668 15.3661 4.1339 15.5332 4.3162 15.6828C5.2601 16.4575 6.6386 16.4977 9.3 16.4999",
    "1",
  ),
] as unknown as IconSvgElement;

// arrow-right-02 at 1.35x: the stock one draws small and thin.
export const ContinueArrowIcon = [
  path("M20.775 12L2.55 12", "0"),
  path("M13.35 20.1C13.35 20.1 21.45 14.1345 21.45 12C21.45 9.8654 13.35 3.9 13.35 3.9", "1"),
] as unknown as IconSvgElement;

// Plain straight chevrons for the branch picker.
export const BranchPrevIcon = [path("M15 6L9 12L15 18", "0")] as unknown as IconSvgElement;

export const BranchNextIcon = [path("M9 6L15 12L9 18", "0")] as unknown as IconSvgElement;
