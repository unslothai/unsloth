// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { ReactNode, Ref } from "react";

export function SettingsSection({
  title,
  description,
  children,
  ref,
  hideHeading = false,
  action,
}: {
  title: string;
  /** Control shown beside the heading, e.g. Refresh. */
  action?: ReactNode;
  hideHeading?: boolean;
  description?: ReactNode;
  children: ReactNode;
  ref?: Ref<HTMLElement>;
}) {
  return (
    <section ref={ref} data-settings-label={title} className="flex flex-col">
      {hideHeading ? null : (
        <div className="mb-1 flex flex-col gap-0.5">
          {action ? (
            <div className="flex items-center justify-between gap-3">
              <h2 className="settings-heading text-base font-semibold font-heading">
                {title}
              </h2>
              {action}
            </div>
          ) : (
            <h2 className="settings-heading text-base font-semibold font-heading">
              {title}
            </h2>
          )}
          {description ? (
            <p className="text-xs text-muted-foreground leading-relaxed">
              {description}
            </p>
          ) : null}
        </div>
      )}
      <div className="flex flex-col">{children}</div>
    </section>
  );
}

export function SettingsGroupDivider() {
  return <div className="my-1 border-t border-border/60" />;
}
