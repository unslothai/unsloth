// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";

export interface PromptExample {
  label: string;
  text: string;
}

export function PromptExamples({
  label,
  examples,
  empty,
  hint,
  onPick,
  disabled = false,
}: {
  label: string;
  examples: readonly PromptExample[];
  empty: boolean;
  hint?: string;
  onPick: (text: string) => void;
  disabled?: boolean;
}) {
  if (!empty) {
    return hint ? (
      <p className="text-ui-12/normal text-muted-foreground">{hint}</p>
    ) : null;
  }
  return (
    <fieldset
      className="m-0 flex min-w-0 flex-wrap items-center gap-1 border-0 p-0"
      aria-label={label}
    >
      <span className="me-0.5 text-ui-11p5 text-muted-foreground">Try</span>
      {examples.map((example) => (
        <Button
          key={example.label}
          type="button"
          variant="muted"
          size="sm"
          disabled={disabled}
          className="h-[calc(24px*var(--ui-space-scale,1))] px-2.5 text-ui-11p5 font-normal"
          title={example.text}
          onClick={() => onPick(example.text)}
        >
          {example.label}
        </Button>
      ))}
    </fieldset>
  );
}
