// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { SVGProps } from "react";

// The composer's expand / collapse corners, shared so every expand control draws the same mark.
export function ExpandToggleIcon({
  expanded,
  ...props
}: { expanded: boolean } & SVGProps<SVGSVGElement>) {
  return (
    <svg
      viewBox="0 0 20 20"
      fill="none"
      stroke="currentColor"
      strokeWidth={2.25}
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden={true}
      {...props}
    >
      <path d={expanded ? "M13 2v5h5M2 13h5v5" : "M11 4h5v5M4 11v5h5"} />
    </svg>
  );
}
