// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { ReactNode } from "react";

export function SettingsSection({
  title,
  description,
  children,
}: {
  title: string;
  description?: ReactNode;
  children: ReactNode;
}) {
  return (
    <section data-settings-label={title} className="flex flex-col">
      <div className="mb-1 flex flex-col gap-0.5">
        <h2 className="text-base font-semibold font-heading text-foreground">
          {title}
        </h2>
        {description ? (
          <p className="text-xs text-muted-foreground leading-relaxed">
            {description}
          </p>
        ) : null}
      </div>
      {/* No per-row dividers: rows inside a titled section are related.
          SettingsGroupDivider separates unrelated clusters. */}
      <div className="flex flex-col">{children}</div>
    </section>
  );
}

/** Divider between unrelated clusters of rows inside one section. */
export function SettingsGroupDivider() {
  return <div className="my-1 border-t border-border/60" />;
}
