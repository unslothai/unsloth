// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  type SkillRecord,
  refreshSkillsCatalog,
  useSkillsCatalog,
} from "@/features/chat";
import type {
  Unstable_DirectiveFormatter,
  Unstable_DirectiveSegment,
} from "@assistant-ui/core";
import {
  ComposerPrimitive,
  unstable_useMentionAdapter,
} from "@assistant-ui/react";
import { BookOpen01Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import {
  type KeyboardEvent,
  type MutableRefObject,
  type ReactElement,
  type RefObject,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";

function enabled(records: readonly SkillRecord[]): readonly SkillRecord[] {
  return records.filter(
    (skill) => skill.valid && !skill.shadowed && skill.enabled,
  );
}

const MENTION_PATTERN = /(^|\s)@([a-z0-9][a-z0-9-]{0,63})/gim;

const skillMentionFormatter: Unstable_DirectiveFormatter = {
  serialize: (item) => `@${item.label}`,
  parse(text) {
    const segments: Unstable_DirectiveSegment[] = [];
    let lastIndex = 0;

    for (const match of text.matchAll(MENTION_PATTERN)) {
      const whitespace = match[1] ?? "";
      const mentionStart = match.index + whitespace.length;
      if (mentionStart > lastIndex) {
        segments.push({
          kind: "text",
          text: text.slice(lastIndex, mentionStart),
        });
      }
      const label = match[2] ?? "";
      segments.push({
        kind: "mention",
        type: "skill",
        label,
        id: label,
      });
      lastIndex = match.index + match[0].length;
    }

    if (lastIndex < text.length) {
      segments.push({ kind: "text", text: text.slice(lastIndex) });
    }
    return segments;
  },
};

function useHighlightedItemScroll(root: HTMLElement | null) {
  useEffect(() => {
    if (!root) return;
    const scroll = () => {
      root
        .querySelector<HTMLElement>("[data-highlighted]")
        ?.scrollIntoView({ block: "nearest" });
    };
    const observer = new MutationObserver(scroll);
    observer.observe(root, {
      subtree: true,
      attributes: true,
      attributeFilter: ["data-highlighted"],
    });
    scroll();
    return () => observer.disconnect();
  }, [root]);
}

// Mounted inside the results list, so it exists exactly while the popover is
// open; reports whether Enter would pick a row rather than send the message.
function MentionEnterSignal({
  onChange,
  active,
}: {
  onChange?: (consumesEnter: boolean) => void;
  active: boolean;
}): null {
  useEffect(() => {
    if (!onChange) return;
    onChange(active);
    return () => onChange(false);
  }, [active, onChange]);
  return null;
}

export function SkillMentionPopover({
  enabled: mentionsEnabled,
  onConsumesEnterChange,
}: {
  enabled: boolean;
  onConsumesEnterChange?: (consumesEnter: boolean) => void;
}): ReactElement | null {
  const { skills } = useSkillsCatalog();
  const [listElement, setListElement] = useState<HTMLDivElement | null>(null);
  const available = mentionsEnabled ? enabled(skills) : [];
  const items = useMemo(
    () =>
      available.map((skill) => ({
        id: skill.name,
        type: "skill",
        label: skill.name,
        description: skill.description,
      })),
    [available],
  );
  const mention = unstable_useMentionAdapter({
    items,
    includeModelContextTools: false,
    formatter: skillMentionFormatter,
  });
  useHighlightedItemScroll(listElement);
  if (items.length === 0) return null;

  return (
    <ComposerPrimitive.Unstable_TriggerPopover
      char="@"
      adapter={mention.adapter}
      aria-label="Agent Skills"
      className="data-open:animate-in data-closed:animate-out data-closed:fade-out-0 data-open:fade-in-0 data-closed:zoom-out-95 data-open:zoom-in-95 absolute bottom-[calc(100%+8px)] left-3 z-40 w-[min(360px,calc(100%-24px))] overflow-hidden rounded-lg border border-border/60 bg-popover p-1 text-popover-foreground shadow-border duration-100"
    >
      <ComposerPrimitive.Unstable_TriggerPopover.Directive
        {...mention.directive}
      />
      <ComposerPrimitive.Unstable_TriggerPopoverItems>
        {(results) => (
          <>
            <MentionEnterSignal
              onChange={onConsumesEnterChange}
              active={results.length > 0}
            />
            <div className="px-2 py-1.5 text-xs font-medium text-muted-foreground">
              Agent Skills
            </div>
            <div ref={setListElement} className="max-h-64 overflow-y-auto">
              {results.map((item, index) => (
                <ComposerPrimitive.Unstable_TriggerPopoverItem
                  key={item.id}
                  item={item}
                  index={index}
                  className="flex w-full items-start gap-2.5 rounded-[11px] px-3 py-2 text-left outline-none transition-colors hover:bg-accent hover:text-accent-foreground data-highlighted:bg-accent data-highlighted:text-accent-foreground"
                >
                  <HugeiconsIcon
                    icon={BookOpen01Icon}
                    strokeWidth={1.75}
                    className="mt-0.5 size-4 shrink-0 text-primary"
                  />
                  <span className="min-w-0 flex-1">
                    <span className="block truncate text-sm font-medium">
                      {item.label}
                    </span>
                    {item.description ? (
                      <span className="line-clamp-2 max-h-8 overflow-hidden break-words text-xs leading-4 text-muted-foreground">
                        {item.description}
                      </span>
                    ) : null}
                  </span>
                </ComposerPrimitive.Unstable_TriggerPopoverItem>
              ))}
            </div>
          </>
        )}
      </ComposerPrimitive.Unstable_TriggerPopoverItems>
    </ComposerPrimitive.Unstable_TriggerPopover>
  );
}

type MentionRange = { start: number; end: number; query: string } | null;

function mentionAtCaret(text: string, caret: number): MentionRange {
  const prefix = text.slice(0, caret);
  const match = /(?:^|\s)@([a-z0-9-]*)$/i.exec(prefix);
  if (!match) return null;
  const at = prefix.lastIndexOf("@");
  return { start: at, end: caret, query: match[1] ?? "" };
}

export function useTextareaSkillMentions({
  text,
  setText,
  inputRef,
  composingRef,
  enabled: mentionsEnabled,
}: {
  text: string;
  setText: (text: string) => void;
  inputRef: RefObject<HTMLTextAreaElement | null>;
  composingRef: MutableRefObject<boolean>;
  enabled: boolean;
}) {
  const { skills } = useSkillsCatalog();
  const [range, setRange] = useState<MentionRange>(null);
  const [highlighted, setHighlighted] = useState(0);
  const results = useMemo(() => {
    if (!range) return [];
    const query = range.query.toLowerCase();
    if (!mentionsEnabled) return [];
    return enabled(skills).filter(
      (skill) =>
        skill.name.toLowerCase().includes(query) ||
        skill.description.toLowerCase().includes(query),
    );
  }, [mentionsEnabled, range, skills]);
  useEffect(() => {
    if (
      !mentionsEnabled ||
      (range &&
        (text.length < range.end ||
          !text.slice(range.start, range.end).startsWith("@")))
    ) {
      setRange(null);
    }
  }, [mentionsEnabled, range, text]);

  const resultsRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!range) return;
    resultsRef.current
      ?.querySelector<HTMLElement>(`[data-mention-index="${highlighted}"]`)
      ?.scrollIntoView({ block: "nearest" });
  }, [highlighted, range]);

  const update = useCallback(
    (nextText: string, caret: number) => {
      if (composingRef.current) {
        setRange(null);
        return;
      }
      const next = mentionAtCaret(nextText, caret);
      // A fresh @ re-reads the folders, so a skill written since page load is offered.
      if (next?.query === "") refreshSkillsCatalog();
      setRange(next);
      setHighlighted(0);
    },
    [composingRef],
  );

  const insert = useCallback(
    (skill: SkillRecord) => {
      if (!range) return;
      const directive = skillMentionFormatter.serialize({
        id: skill.name,
        type: "skill",
        label: skill.name,
      });
      const next = `${text.slice(0, range.start)}${directive} ${text.slice(range.end)}`;
      const caret = range.start + directive.length + 1;
      setText(next);
      setRange(null);
      requestAnimationFrame(() => {
        inputRef.current?.focus();
        inputRef.current?.setSelectionRange(caret, caret);
      });
    },
    [inputRef, range, setText, text],
  );

  const onKeyDown = useCallback(
    (event: KeyboardEvent<HTMLTextAreaElement>): boolean => {
      if (!range || composingRef.current || results.length === 0) return false;
      if (event.key === "ArrowDown" || event.key === "ArrowUp") {
        event.preventDefault();
        const direction = event.key === "ArrowDown" ? 1 : -1;
        setHighlighted(
          (index) => (index + direction + results.length) % results.length,
        );
        return true;
      }
      if ((event.key === "Enter" && !event.shiftKey) || event.key === "Tab") {
        event.preventDefault();
        insert(results[highlighted] ?? results[0]);
        return true;
      }
      if (event.key === "Escape") {
        event.preventDefault();
        setRange(null);
        return true;
      }
      return false;
    },
    [composingRef, highlighted, insert, range, results],
  );

  const popover =
    range && results.length > 0 ? (
      <div className="animate-in fade-in-0 zoom-in-95 absolute bottom-[calc(100%+8px)] left-3 z-40 w-[min(360px,calc(100%-24px))] overflow-hidden rounded-lg border border-border/60 bg-popover p-1 text-popover-foreground shadow-border duration-100">
        <div className="px-2 py-1.5 text-xs font-medium text-muted-foreground">
          Agent Skills
        </div>
        <div ref={resultsRef} className="max-h-64 overflow-y-auto">
          {results.map((skill, index) => (
            <button
              key={`${skill.source}:${skill.name}`}
              type="button"
              data-mention-index={index}
              data-highlighted={index === highlighted}
              className="flex w-full items-start gap-2.5 rounded-[11px] px-3 py-2 text-left outline-none transition-colors hover:bg-accent hover:text-accent-foreground data-[highlighted=true]:bg-accent data-[highlighted=true]:text-accent-foreground"
              onMouseDown={(event) => event.preventDefault()}
              onClick={() => insert(skill)}
              onMouseEnter={() => setHighlighted(index)}
            >
              <HugeiconsIcon
                icon={BookOpen01Icon}
                strokeWidth={1.75}
                className="mt-0.5 size-4 shrink-0 text-primary"
              />
              <span className="min-w-0 flex-1">
                <span className="block truncate text-sm font-medium">
                  {skill.name}
                </span>
                <span className="line-clamp-2 max-h-8 overflow-hidden break-words text-xs leading-4 text-muted-foreground">
                  {skill.description}
                </span>
              </span>
            </button>
          ))}
        </div>
      </div>
    ) : null;

  return { update, onKeyDown, popover, close: () => setRange(null) };
}
