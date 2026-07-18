// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { ReactNode } from "react";

import { cn } from "@/lib/utils";

type PageHeadingProps = {
  title: ReactNode;
  subtitle?: ReactNode;
  className?: string;
  onTitleClick?: () => void;
};

const TITLE_CLASS =
  "text-[30px] font-semibold leading-[1.04] tracking-[-0.028em] text-foreground sm:text-[34px]";

export function PageHeading({
  title,
  subtitle,
  className,
  onTitleClick,
}: PageHeadingProps) {
  return (
    <div className={cn("page-title-halo min-w-0", className)}>
      {onTitleClick ? (
        <button
          type="button"
          onClick={onTitleClick}
          className={cn(
            TITLE_CLASS,
            "cursor-pointer rounded-md text-left outline-none transition-opacity hover:opacity-80 focus-visible:ring-1 focus-visible:ring-ring",
          )}
        >
          {title}
        </button>
      ) : (
        <h1 className={TITLE_CLASS}>{title}</h1>
      )}
      {subtitle ? (
        <p className="mt-2 text-[13px] leading-[19px] text-muted-foreground">
          {subtitle}
        </p>
      ) : null}
    </div>
  );
}
