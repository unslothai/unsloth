// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { HugeiconsIcon, type IconSvgElement } from "@hugeicons/react";

// Hugeicons "Shield Alert" (stroke-rounded), newer than the pinned free-icons package.
// https://hugeicons.com/icon/shield-alert
export const ShieldAlertIcon: IconSvgElement = [
  [
    "path",
    {
      d: "M11.9922 8L11.9922 12",
      stroke: "currentColor",
      strokeLinecap: "round",
      strokeLinejoin: "round",
      strokeWidth: "1.5",
      key: "0",
    },
  ],
  [
    "path",
    {
      d: "M12.1172 15.75L11.9922 15.75M12.2422 15.75C12.2422 15.8881 12.1303 16 11.9922 16C11.8541 16 11.7422 15.8881 11.7422 15.75C11.7422 15.6119 11.8541 15.5 11.9922 15.5C12.1303 15.5 12.2422 15.6119 12.2422 15.75Z",
      stroke: "currentColor",
      strokeLinecap: "round",
      strokeLinejoin: "round",
      strokeWidth: "1.5",
      key: "1",
    },
  ],
  [
    "path",
    {
      d: "M20.9922 11.1835V8.28041C20.9922 6.64041 20.9922 5.82041 20.5881 5.28541C20.184 4.75042 19.2703 4.49068 17.4429 3.97122C16.1944 3.61632 15.0938 3.18875 14.2145 2.79841C13.0156 2.26622 12.4161 2.00012 11.9922 2.00012C11.5682 2.00012 10.9688 2.26622 9.7699 2.79841C8.89057 3.18875 7.79002 3.61632 6.54152 3.97122C4.71411 4.49068 3.80041 4.75042 3.3963 5.28541C2.99219 5.82041 2.99219 6.64041 2.99219 8.28041V11.1835C2.99219 16.8086 8.05496 20.1836 10.5861 21.5195C11.1932 21.8399 11.4968 22.0001 11.9922 22.0001C12.4876 22.0001 12.7911 21.8399 13.3982 21.5195C15.9294 20.1836 20.9922 16.8086 20.9922 11.1835Z",
      stroke: "currentColor",
      strokeLinecap: "round",
      strokeWidth: "1.5",
      key: "2",
    },
  ],
];

/** Lucide-compatible wrapper, for the permission-mode option list. */
export function ShieldAlertGlyph({
  className,
  strokeWidth,
}: {
  className?: string;
  strokeWidth?: number;
}) {
  return (
    <HugeiconsIcon
      icon={ShieldAlertIcon}
      className={className}
      strokeWidth={strokeWidth}
    />
  );
}
