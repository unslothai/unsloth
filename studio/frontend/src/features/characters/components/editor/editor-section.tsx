// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { cn } from "@/lib/utils";
import { ArrowDown01Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useState, type ReactNode } from "react";

export function EditorCard({ children }: { children: ReactNode }) {
  return <section className="flex flex-col gap-4 rounded-[24px] bg-muted/45 p-5 dark:bg-card">{children}</section>;
}

export function EditorField({
  label,
  hint,
  htmlFor,
  error,
  children,
}: {
  label: string;
  hint?: string;
  htmlFor?: string;
  error?: string;
  children: ReactNode;
}) {
  return (
    <div className="flex flex-col gap-1.5">
      <label htmlFor={htmlFor} className="text-ui-14 font-semibold text-foreground">
        {label}
      </label>
      {hint ? <p className="-mt-0.5 text-ui-13 text-muted-foreground">{hint}</p> : null}
      {children}
      {error ? <p className="text-ui-13 text-destructive">{error}</p> : null}
    </div>
  );
}

export function EditorSection({
  title,
  badge,
  defaultOpen = false,
  children,
}: {
  title: string;
  badge?: string;
  defaultOpen?: boolean;
  children: ReactNode;
}) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <section className="rounded-[24px] bg-muted/45 dark:bg-card">
      <button
        type="button"
        aria-expanded={open}
        onClick={() => setOpen((value) => !value)}
        className="flex w-full cursor-pointer items-center gap-3 px-5 py-4 text-left"
      >
        <span className="min-w-0 flex-1 text-ui-15 font-semibold text-foreground">{title}</span>
        {badge ? <span className="text-ui-12 text-muted-foreground">{badge}</span> : null}
        <HugeiconsIcon
          icon={ArrowDown01Icon}
          strokeWidth={1.75}
          className={cn("size-4 text-muted-foreground transition-transform", open && "rotate-180")}
        />
      </button>
      {open ? <div className="flex flex-col gap-4 px-5 pb-5">{children}</div> : null}
    </section>
  );
}
