// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { FC } from "react";

/** Keyline Icons `arrow-left-right-3` (MIT, github.com/keyline-icons/keyline-icons), flipped vertically. */
export const SwapArrowsIcon: FC<{ className?: string }> = ({ className }) => (
  <svg
    className={className}
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth={2}
    strokeLinecap="round"
    strokeLinejoin="round"
    xmlns="http://www.w3.org/2000/svg"
    focusable="false"
    aria-hidden={true}
  >
    <path
      transform="matrix(1 0 0 -1 0 24)"
      d="M11 17L21.2604 17M17.1053 12L21.8633 16.6759C22.0456 16.8549 22.0456 17.1451 21.8633 17.3241L17.1053 22M13 7L2.7396 7M6.8947 12L2.1367 7.3241C1.9544 7.1451 1.9544 6.8549 2.1367 6.6759L6.8947 2"
    />
  </svg>
);
