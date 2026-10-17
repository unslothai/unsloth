// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { cn } from "@/lib/utils";
import { HugeiconsIcon, type IconSvgElement } from "@hugeicons/react";
import type { CSSProperties, ReactNode } from "react";

export function SectionBox({
  title,
  description,
  icon,
  chipTint,
  titleAction,
  compactTitleAction = false,
  titleBadge,
  children,
  dataTour,
}: {
  title?: string;
  titleBadge?: ReactNode;
  description?: string;
  icon?: IconSvgElement;
  chipTint?: string;
  titleAction?: ReactNode;
  compactTitleAction?: boolean;
  children: ReactNode;
  dataTour?: string;
}) {
  return (
    <section
      data-tour={dataTour}
      className="@container/train-card elevated-card flex flex-col gap-7 bg-card p-5 pb-7"
    >
      {title && (
        <div
          className={cn(
            "flex gap-3",
            compactTitleAction
              ? "items-center justify-between"
              : "flex-col items-stretch @md/train-card:flex-row @md/train-card:items-center @md/train-card:justify-between",
          )}
        >
          <div className="flex min-w-0 items-center gap-3">
            {icon && (
              <span
                className="train-section-chip inline-flex size-9 shrink-0 items-center justify-center rounded-full"
                style={
                  chipTint
                    ? ({ "--chip-tint": chipTint } as CSSProperties)
                    : undefined
                }
              >
                <HugeiconsIcon
                  icon={icon}
                  strokeWidth={1.5}
                  className="size-[calc(18px*var(--ui-space-scale,1))]"
                />
              </span>
            )}
            <div className="min-w-0">
              <h2 className="flex select-none items-center gap-1.5 text-ui-13p5 font-semibold leading-ui-18 tracking-[-0.012em] text-foreground">
                {title}
                {titleBadge}
              </h2>
              {description && (
                <p className="text-ui-11p5 leading-ui-15 text-muted-foreground/85">
                  {description}
                </p>
              )}
            </div>
          </div>
          {titleAction && (
            <div
              className={
                compactTitleAction
                  ? "shrink-0"
                  : "w-full min-w-0 @md/train-card:w-auto @md/train-card:shrink-0"
              }
            >
              {titleAction}
            </div>
          )}
        </div>
      )}
      <div className="@container/train-section min-w-0">{children}</div>
    </section>
  );
}
