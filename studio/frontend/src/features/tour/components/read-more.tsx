// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export function ReadMore({ href = "#" }: { href?: string }) {
  return (
    <a
      href={href}
      target="_blank"
      rel="noopener noreferrer"
      onClick={(e) => {
        if (href === "#") e.preventDefault();
      }}
      className="text-emerald-600 underline underline-offset-2 hover:text-emerald-700"
    >
      Read more
    </a>
  );
}
