// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { HugeiconsIcon, type IconSvgElement } from "@hugeicons/react";

// Hugeicons shield-check's outline with Hugeicons "sparkle" (stroke-rounded) at 55% in the middle.
export const SparklesIcon: IconSvgElement = [
  [
    "path",
    {
      d: "M20.9922 11.1833V8.28029C20.9922 6.64029 20.9922 5.82028 20.5881 5.28529C20.184 4.75029 19.2703 4.49056 17.4429 3.9711C16.1944 3.6162 15.0938 3.18863 14.2145 2.79829C13.0156 2.2661 12.4161 2 11.9922 2C11.5682 2 10.9688 2.2661 9.7699 2.79829C8.89057 3.18863 7.79002 3.61619 6.54152 3.9711C4.71411 4.49056 3.80041 4.75029 3.3963 5.28529C2.99219 5.82028 2.99219 6.64029 2.99219 8.28029V11.1833C2.99219 16.8085 8.05496 20.1835 10.5861 21.5194C11.1932 21.8398 11.4968 22 11.9922 22C12.4876 22 12.7911 21.8398 13.3982 21.5194C15.9294 20.1835 20.9922 16.8085 20.9922 11.1833Z",
      stroke: "currentColor",
      strokeLinecap: "round",
      strokeWidth: "1.5",
      key: "0",
    },
  ],
  [
    "path",
    {
      d: "M11.1903 9.3268C11.6192 8.5423 11.8337 8.15 11.9977 8.15C12.1617 8.15 12.3761 8.5423 12.8051 9.3268C13.2466 10.1343 13.8366 10.7312 14.6669 11.186C15.4528 11.6164 15.8457 11.8316 15.8457 11.9958C15.8457 12.1599 15.4525 12.3754 14.6661 12.8062C13.8594 13.2481 13.2631 13.8388 12.8088 14.67C12.3789 15.4566 12.1639 15.85 11.9999 15.85C11.8359 15.85 11.6207 15.4564 11.1903 14.6692C10.7489 13.8616 10.1589 13.2647 9.3285 12.81C8.54 12.3781 8.1457 12.1622 8.1457 11.998C8.1457 11.8338 8.54 11.6179 9.3285 11.186C10.1589 10.7312 10.7489 10.1343 11.1903 9.3268Z",
      stroke: "currentColor",
      strokeLinecap: "round",
      strokeLinejoin: "round",
      strokeWidth: "1.5",
      key: "1",
    },
  ],
];

export function SparklesGlyph({
  className,
  strokeWidth,
}: {
  className?: string;
  strokeWidth?: number;
}) {
  return (
    <HugeiconsIcon
      icon={SparklesIcon}
      className={className}
      strokeWidth={strokeWidth}
    />
  );
}
