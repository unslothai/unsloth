// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Textarea } from "@/components/ui/textarea";
import { type ComponentProps, startTransition, useState } from "react";

type DeferredTextareaProps = Omit<
  ComponentProps<typeof Textarea>,
  "value" | "onChange"
> & {
  value: string;
  onValueChange: (value: string) => void;
};

/** Echoes keys itself and hands them to the owner as a transition, so a slow owner render yields to the next key (#15221). */
export function DeferredTextarea({
  value,
  onValueChange,
  onBlur,
  ...props
}: DeferredTextareaProps) {
  const [draft, setDraft] = useState(value);
  const [seen, setSeen] = useState(value);
  // A value the owner set itself (recall, caption, buttons) replaces the draft; our own echo already matches.
  if (value !== seen) {
    setSeen(value);
    if (value !== draft) {
      setDraft(value);
    }
  }
  return (
    <Textarea
      {...props}
      value={draft}
      onChange={(e) => {
        const next = e.target.value;
        setDraft(next);
        startTransition(() => onValueChange(next));
      }}
      onBlur={(e) => {
        // Blur is a discrete event, so this lands before a Generate click reads a still-pending transition.
        if (draft !== value) {
          onValueChange(draft);
        }
        onBlur?.(e);
      }}
    />
  );
}
