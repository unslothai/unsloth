// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { QWEN3_LANGUAGE_NAMES } from "../clone-policy";

// Radix Select cannot hold an empty value, so "no language" travels as this.
const AUTO = "__auto__";

/** A language by name, or none (Auto). A saved value outside the list stays selectable. */
export function LanguageSelect({
  id,
  label,
  value,
  onChange,
  emptyLabel = "Auto",
  hint,
  disabled,
}: {
  id: string;
  label: string;
  value: string;
  onChange: (value: string) => void;
  emptyLabel?: string;
  hint?: string;
  disabled?: boolean;
}) {
  const names: string[] = [...QWEN3_LANGUAGE_NAMES];
  if (value && !names.includes(value)) names.push(value);
  return (
    <div className="grid gap-1.5">
      <label className="text-ui-13 font-medium text-foreground" htmlFor={id}>
        {label}
      </label>
      <Select
        value={value || AUTO}
        onValueChange={(next) => onChange(next === AUTO ? "" : next)}
        disabled={disabled}
      >
        <SelectTrigger id={id} size="sm" className="w-full">
          <SelectValue />
        </SelectTrigger>
        <SelectContent>
          <SelectItem value={AUTO}>{emptyLabel}</SelectItem>
          {names.map((name) => (
            <SelectItem key={name} value={name}>
              {name}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
      {hint ? (
        <p className="text-ui-11p5 leading-snug text-muted-foreground">
          {hint}
        </p>
      ) : null}
    </div>
  );
}
