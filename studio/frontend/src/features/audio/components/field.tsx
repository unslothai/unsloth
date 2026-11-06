// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { cn } from "@/lib/utils";
import type { MouseEvent, ReactNode } from "react";

const CONTROL = "button, input, select, textarea, a[href], [role='switch']";

/** `htmlFor` still names the control; a click on the title text is cancelled, one on a nested control is not. */
export function FieldLabel({
  htmlFor,
  className,
  children,
}: {
  htmlFor: string;
  className?: string;
  children: ReactNode;
}) {
  const onClick = (event: MouseEvent<HTMLLabelElement>) => {
    if (!(event.target as Element).closest(CONTROL)) event.preventDefault();
  };
  return (
    // biome-ignore lint/a11y/useKeyWithClickEvents: the handler only cancels the label's own activation.
    <label
      htmlFor={htmlFor}
      onClick={onClick}
      className={cn("cursor-default!", className)}
    >
      {children}
    </label>
  );
}

export function FieldTag({ children }: { children: ReactNode }) {
  return (
    <span className="ml-1.5 text-ui-11p5 font-normal text-muted-foreground/70">
      {children}
    </span>
  );
}

export function Field({
  label,
  optional = false,
  hint,
  htmlFor,
  children,
}: {
  label: string;
  optional?: boolean;
  hint?: string;
  htmlFor: string;
  children: ReactNode;
}) {
  return (
    <div className="grid gap-1.5">
      <FieldLabel
        className="text-ui-13 font-medium text-foreground"
        htmlFor={htmlFor}
      >
        {label}
        {optional ? <FieldTag>Optional</FieldTag> : null}
      </FieldLabel>
      {children}
      {hint ? (
        <p className="text-ui-11p5 leading-snug text-muted-foreground">
          {hint}
        </p>
      ) : null}
    </div>
  );
}
