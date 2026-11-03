// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { cn } from "@/lib/utils";
import type { ReactElement, ReactNode } from "react";

export interface ExportSummaryRow {
  label: string;
  value: ReactNode;
  mono?: boolean;
}

export function ExportSummaryCard({
  ready,
  rows,
  steps,
  cta,
}: {
  ready: boolean;
  rows: readonly ExportSummaryRow[];
  steps: readonly string[];
  cta?: ReactNode;
}): ReactElement {
  return (
    <aside
      className={cn(
        "elevated-card elevated-card-dark-border flex flex-col gap-7 bg-[color-mix(in_oklab,var(--foreground)_calc(1.2%*var(--contrast-wash-gain,1)),transparent)] p-6",
        "dark:bg-[rgb(255_255_255_/_calc(0.018*var(--contrast-wash-gain,1)))]",
      )}
    >
      <header className="flex items-center justify-between gap-3">
        <h2 className="text-ui-11 font-medium tracking-nav text-muted-foreground">
          Export summary
        </h2>
        <span
          className={cn(
            "inline-flex h-5 items-center rounded-full px-2 text-ui-10 font-medium tracking-nav",
            ready
              ? "bg-[color-mix(in_oklab,var(--foreground)_calc(6%*var(--contrast-wash-gain,1)),transparent)] text-foreground/90 dark:bg-[rgb(255_255_255_/_calc(0.08*var(--contrast-wash-gain,1)))]"
              : "bg-[color-mix(in_oklab,var(--foreground)_calc(3%*var(--contrast-wash-gain,1)),transparent)] text-muted-foreground/70 dark:bg-[rgb(255_255_255_/_calc(0.04*var(--contrast-wash-gain,1)))]",
          )}
        >
          {ready ? "Ready" : "Not ready"}
        </span>
      </header>

      <section className="flex flex-col gap-3">
        {rows.map((row) => (
          <div
            key={row.label}
            className="flex items-baseline justify-between gap-3"
          >
            <span className="shrink-0 text-ui-11p5 text-muted-foreground/85">
              {row.label}
            </span>
            <span
              className={cn(
                "min-w-0 truncate text-ui-12p5 text-foreground/90",
                row.mono && "font-mono",
              )}
              title={typeof row.value === "string" ? row.value : undefined}
            >
              {row.value}
            </span>
          </div>
        ))}
      </section>

      <section className="flex flex-col gap-3">
        <p className="text-ui-11 font-medium tracking-nav text-muted-foreground">
          Quick guide
        </p>
        <ol className="flex flex-col gap-2.5">
          {steps.map((step, i) => (
            <li
              key={step}
              className="flex items-start gap-2.5 text-ui-11p5 leading-ui-15 text-muted-foreground"
            >
              <span className="flex size-4 shrink-0 items-center justify-center rounded-full bg-[color-mix(in_oklab,var(--foreground)_calc(8%*var(--contrast-wash-gain,1)),transparent)] text-ui-10 font-semibold text-foreground/80">
                {i + 1}
              </span>
              {step}
            </li>
          ))}
        </ol>
      </section>

      {cta && (
        <>
          <div className="-mx-6 h-px bg-[color-mix(in_oklab,var(--foreground)_calc(7%*var(--contrast-wash-gain,1)),transparent)] dark:bg-[rgb(255_255_255_/_calc(0.06*var(--contrast-wash-gain,1)))]" />
          <div className="-mt-2">{cta}</div>
        </>
      )}
    </aside>
  );
}
