// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { ReactNode } from "react";

import { cn } from "@/lib/utils";

type PageHeadingProps = {
  title: ReactNode;
  subtitle?: ReactNode;
  className?: string;
};

export function PageHeading({ title, subtitle, className }: PageHeadingProps) {
  return (
    <div className={cn("page-title-halo min-w-0", className)}>
      <h1 className="text-[30px] font-semibold leading-[1.04] tracking-[-0.028em] text-foreground sm:text-[34px]">
        {title}
      </h1>
      {subtitle ? (
        <p className="mt-2 text-[13px] leading-[19px] text-muted-foreground">
          {subtitle}
        </p>
      ) : null}
    </div>
  );
}
