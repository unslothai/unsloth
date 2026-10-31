// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { HugeiconsIcon, type IconSvgElement } from "@hugeicons/react";

// Hugeicons "shield-check" outline with "sparkle" at 55%, raised 2%.
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
      d: "M11.1903 8.8468C11.6192 8.0623 11.8337 7.67 11.9977 7.67C12.1617 7.67 12.3761 8.0623 12.8051 8.8468C13.2466 9.6543 13.8366 10.2512 14.6669 10.706C15.4528 11.1364 15.8457 11.3516 15.8457 11.5158C15.8457 11.6799 15.4525 11.8954 14.6661 12.3262C13.8594 12.7681 13.2631 13.3588 12.8088 14.19C12.3789 14.9766 12.1639 15.37 11.9999 15.37C11.8359 15.37 11.6207 14.9764 11.1903 14.1892C10.7489 13.3816 10.1589 12.7847 9.3285 12.33C8.54 11.8981 8.1457 11.6822 8.1457 11.518C8.1457 11.3538 8.54 11.1379 9.3285 10.706C10.1589 10.2512 10.7489 9.6543 11.1903 8.8468Z",
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
