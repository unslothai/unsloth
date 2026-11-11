// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Fragment, type ReactNode } from "react";

/** Plain roleplay prose for previews: *actions* in italics, everything else as typed. */
export function RoleplayText({ text, className }: { text: string; className?: string }) {
  const parts: ReactNode[] = [];
  const pattern = /\*([^*\n]+)\*/g;
  let last = 0;
  let match: RegExpExecArray | null;
  let key = 0;
  while ((match = pattern.exec(text))) {
    if (match.index > last) parts.push(<Fragment key={key++}>{text.slice(last, match.index)}</Fragment>);
    parts.push(
      <em key={key++} className="text-muted-foreground">
        {match[1]}
      </em>,
    );
    last = match.index + match[0].length;
  }
  if (last < text.length) parts.push(<Fragment key={key++}>{text.slice(last)}</Fragment>);
  return <span className={className}>{parts}</span>;
}
