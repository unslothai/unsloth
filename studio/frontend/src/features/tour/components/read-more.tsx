// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

const EXTERNAL_URL_RE = /^https?:\/\//;

export function ReadMore({ href = "#" }: { href?: string }) {
  const isExternal = EXTERNAL_URL_RE.test(href);
  return (
    <a
      href={href}
      target={isExternal ? "_blank" : undefined}
      rel={isExternal ? "noopener noreferrer" : undefined}
      onClick={(e) => {
        if (href === "#") e.preventDefault();
      }}
      className="text-control-accent underline underline-offset-2 hover:text-control-accent/80"
    >
      Read more
    </a>
  );
}
