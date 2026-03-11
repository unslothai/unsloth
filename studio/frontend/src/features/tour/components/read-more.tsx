// SPDX-License-Identifier: AGPL-3.0-only - See /studio/LICENSE.AGPL-3.0
// Copyright © 2025 Unsloth AI

export function ReadMore({ href = "#" }: { href?: string }) {
  return (
    <a
      href={href}
      onClick={(e) => {
        if (href === "#") e.preventDefault();
      }}
      className="text-emerald-600 underline underline-offset-2 hover:text-emerald-700"
    >
      Read more
    </a>
  );
}
