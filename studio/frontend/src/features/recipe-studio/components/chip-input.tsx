// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import { useUiSpaceScale } from "@/hooks/use-ui-space-scale";
import { Cancel01Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import {
  type KeyboardEvent,
  type ReactElement,
  useEffect,
  useId,
  useMemo,
  useRef,
  useState,
} from "react";

type ChipInputProps = {
  values: string[];
  onAdd: (value: string) => void;
  onRemove: (index: number) => void;
  placeholder?: string;
  suggestions?: string[];
};

export function ChipInput({
  values,
  onAdd,
  onRemove,
  placeholder = "Type and press Enter",
  suggestions,
}: ChipInputProps): ReactElement {
  const [draft, setDraft] = useState("");
  const [isWrapped, setIsWrapped] = useState(false);
  const uiSpaceScale = useUiSpaceScale();
  const containerRef = useRef<HTMLDivElement | null>(null);
  const listId = useId();
  const suggestionSet = useMemo(
    () => new Set((suggestions ?? []).map((value) => value.trim())),
    [suggestions],
  );

  useEffect(() => {
    const element = containerRef.current;
    if (!element) {
      return;
    }
    // min-h-9 follows the UI font size, so a one-line field is 27px at the
    // 12px setting and 46px at 20px. A fixed cutoff calls the tall one wrapped
    // before a single chip has moved.
    const syncWrapped = () => {
      setIsWrapped(element.clientHeight > 44 * uiSpaceScale);
    };
    syncWrapped();
    const observer = new ResizeObserver(syncWrapped);
    observer.observe(element);
    return () => observer.disconnect();
  }, [values.length, draft, uiSpaceScale]);

  function addValue(rawValue: string, allowAny: boolean): void {
    const trimmed = rawValue.trim();
    if (!trimmed) {
      return;
    }
    if (!allowAny && !suggestionSet.has(trimmed)) {
      return;
    }
    onAdd(trimmed);
    setDraft("");
  }

  const handleKeyDown = (event: KeyboardEvent<HTMLInputElement>) => {
    if (event.key === "Enter") {
      event.preventDefault();
      addValue(draft, true);
    }
    if (event.key === "Backspace" && !draft && values.length > 0) {
      onRemove(values.length - 1);
    }
  };

  function handleChange(nextDraft: string): void {
    setDraft(nextDraft);
    if (suggestionSet.has(nextDraft.trim())) {
      addValue(nextDraft, false);
    }
  }

  return (
    <div
      ref={containerRef}
      className={`bg-input/30 border-input dark:border-transparent focus-within:border-ring dark:focus-within:border-transparent dark:focus-within:bg-[rgb(255_255_255_/_calc(0.09*var(--contrast-wash-gain,1)))] flex min-h-9 flex-wrap items-center gap-1.5 border bg-clip-padding px-1.5 py-1.5 text-sm transition-colors ${isWrapped ? "corner-squircle rounded-xl" : "rounded-4xl"}`}
    >
      {values.map((value, index) => (
        <span
          key={`${value}-${index}`}
          className="bg-muted-foreground/10 text-foreground flex h-[calc(--spacing(5.5))] w-fit items-center justify-center gap-1 rounded-4xl pr-0 pl-2 text-xs font-medium whitespace-nowrap"
        >
          {value}
          <Button
            type="button"
            variant="ghost"
            size="icon-xs"
            className="-ml-1 opacity-50 hover:opacity-100"
            onClick={() => onRemove(index)}
          >
            <HugeiconsIcon
              icon={Cancel01Icon}
              strokeWidth={2}
              className="pointer-events-none"
            />
          </Button>
        </span>
      ))}
      <input
        className="nodrag min-w-16 flex-1 bg-transparent text-sm outline-none placeholder:text-muted-foreground"
        placeholder={values.length === 0 ? placeholder : ""}
        value={draft}
        list={suggestions && suggestions.length > 0 ? listId : undefined}
        onChange={(event) => handleChange(event.target.value)}
        onBlur={() => addValue(draft, false)}
        onKeyDown={handleKeyDown}
      />
      {suggestions && suggestions.length > 0 && (
        <datalist id={listId}>
          {suggestions.map((value) => (
            <option key={value} value={value} />
          ))}
        </datalist>
      )}
    </div>
  );
}
