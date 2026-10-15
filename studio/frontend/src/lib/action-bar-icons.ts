// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { IconSvgElement } from "@hugeicons/react";

// Rescaled about the centre (stroke unchanged) to match the stock icons. Hugeicons (MIT).

const path = (d: string, key: string) =>
  ["path", { d, stroke: "currentColor", strokeLinecap: "round", strokeLinejoin: "round", key }] as const;

// volume-02 at 1.1x: the stock one draws short beside Copy.
export const ReadAloudIcon = [
  path(
    "M14.2 15.0948V8.9051C14.2 5.4454 14.2 3.7155 13.1822 3.285C12.1643 2.8545 10.9663 4.0777 8.5706 6.5241C7.3298 7.791 6.6219 8.0716 4.8566 8.0716C3.3128 8.0716 2.5409 8.0716 1.9864 8.4499C0.8354 9.2352 1.0094 10.7702 1.0094 12C1.0094 13.2298 0.8354 14.7647 1.9864 15.5501C2.5409 15.9284 3.3128 15.9284 4.8566 15.9284C6.6219 15.9284 7.3298 16.209 8.5706 17.4759C10.9663 19.9223 12.1643 21.1455 13.1822 20.715C14.2 20.2844 14.2 18.5546 14.2 15.0948Z",
    "0",
  ),
  path(
    "M17.5 8.7C18.1879 9.6016 18.6 10.7497 18.6 12C18.6 13.2503 18.1879 14.3983 17.5 15.3",
    "1",
  ),
  path(
    "M20.8 6.5C22.1759 8.0027 23 9.9163 23 12C23 14.0837 22.1759 15.9973 20.8 17.5",
    "2",
  ),
] as unknown as IconSvgElement;

// arrow-right-02 at 1.25x: the stock one draws small and thin.
export const ContinueArrowIcon = [
  path("M20.125 12L3.25 12", "0"),
  path("M13.25 19.5C13.25 19.5 20.75 13.9764 20.75 12C20.75 10.0235 13.25 4.5 13.25 4.5", "1"),
] as unknown as IconSvgElement;

// edit-03 at 0.92x: its diagonal reads larger than the other icons.
export const EditResponseIcon = [
  path(
    "M4.4393 15.9645L3.72 20.28L8.0356 19.5607C8.785 19.4359 9.4767 19.08 10.0139 18.5427L19.7462 8.8102C20.4579 8.0985 20.4579 6.9445 19.7461 6.2328L17.7672 4.2538C17.0554 3.5421 15.9014 3.5421 15.1895 4.2538L5.4573 13.9863C4.9201 14.5235 4.5642 15.2151 4.4393 15.9645Z",
    "0",
  ),
  path("M13.84 6.48L17.52 10.16", "1"),
] as unknown as IconSvgElement;

export const BranchPrevIcon = [path("M15 6L9 12L15 18", "0")] as unknown as IconSvgElement;

export const BranchNextIcon = [path("M9 6L15 12L9 18", "0")] as unknown as IconSvgElement;
