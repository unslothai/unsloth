// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { ReactNode, Ref } from "react";

import { cn } from "@/lib/utils";

export function SettingsSection({
  title,
  description,
  children,
  ref,
  hideHeading = false,
  action,
  danger = false,
}: {
  title: string;
  /** Control shown beside the heading, e.g. Refresh. */
  action?: ReactNode;
  hideHeading?: boolean;
  /** Danger zone: a divider above the heading sets it apart from the sections before it. */
  danger?: boolean;
  description?: ReactNode;
  children: ReactNode;
  ref?: Ref<HTMLElement>;
}) {
  return (
    <section
      ref={ref}
      data-settings-label={title}
      className={cn(
        "flex flex-col",
        danger && "border-t border-border/60 pt-6",
      )}
    >
      {hideHeading ? null : (
        <div className="mb-1 flex flex-col gap-0.5">
          {action ? (
            <div
              className={cn(
                "flex justify-between gap-3",
                description ? "items-start" : "items-center",
              )}
            >
              <h2 className="settings-heading text-base font-semibold font-heading">
                {title}
              </h2>
              {description ? (
                // Top edge on the h2's cap top (text-base at leading 1.5; Hellix: 0.372em).
                <div className="mt-[calc(var(--text-base)*0.372)] shrink-0">
                  {action}
                </div>
              ) : (
                action
              )}
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
