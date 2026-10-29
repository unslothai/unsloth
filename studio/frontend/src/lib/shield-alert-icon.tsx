// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { ShieldAlert } from "lucide-react";

export function ShieldAlertGlyph({
  className,
  strokeWidth,
}: {
  className?: string;
  strokeWidth?: number;
}) {
  return <ShieldAlert className={className} strokeWidth={strokeWidth} />;
}
