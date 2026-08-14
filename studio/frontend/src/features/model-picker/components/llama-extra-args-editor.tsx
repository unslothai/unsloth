// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import { Input } from "@/components/ui/input";
import {
  Popover,
  PopoverAnchor,
  PopoverContent,
} from "@/components/ui/popover";
import { ChevronDownStandardIcon } from "@/lib/chevron-icons";
import { cn } from "@/lib/utils";
import { Cancel01Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import {
  type KeyboardEvent,
  useEffect,
  useId,
  useMemo,
  useRef,
  useState,
} from "react";
import {
  cachedAuthoritativeLlamaServerArguments,
  cachedLlamaServerManagedPolicy,
  fetchLlamaServerArguments,
} from "../api/llama-server-arguments";
import {
  type LlamaExtraArgsCompletion,
  type LlamaServerArgument,
  applyLlamaExtraArgsCompletion,
  completeLlamaExtraArgs,
  countLlamaExtraArgFlags,
  diagnoseLlamaExtraArgs,
  formatLlamaExtraArgs,
  llamaExtraArgRows,
  llamaExtraArgsCatalogBlocksPersistence,
  llamaServerArgumentGroupLabel,
  llamaServerArgumentTakesValue,
  llamaServerDiagnosticCatalog,
  moveLlamaExtraArgsSelection,
  parseLlamaExtraArgs,
  replaceLlamaExtraArgRow,
  replaceLlamaExtraArgRowFlag,
} from "../model-config/llama-extra-args";

const LISTBOX_ROLE = { role: "listbox" as const };

export function LlamaExtraArgsEditor({
  value,
  onChange,
  onBlockingChange,
}: {
  value: string[] | undefined;
  onChange: (tokens: string[]) => void;
  onBlockingChange: (blocked: boolean) => void;
}) {
  const [open, setOpen] = useState(() => (value?.length ?? 0) > 0);
  const [catalog, setCatalog] = useState<LlamaServerArgument[] | null>(null);
  const [catalogState, setCatalogState] = useState<
    "loading" | "available" | "unavailable"
  >("loading");
  const [catalogAuthoritative, setCatalogAuthoritative] = useState(false);
  const [draft, setDraft] = useState("");
  const [draftError, setDraftError] = useState<string | null>(null);
  const [completions, setCompletions] = useState<LlamaExtraArgsCompletion[]>(
    [],
  );
  const [completionSide, setCompletionSide] = useState<"top" | "bottom">(
    "bottom",
  );
  const [selected, setSelected] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);
  const optionRefs = useRef<Array<HTMLButtonElement | null>>([]);
  const completionListId = useId();

  useEffect(() => {
    let cancelled = false;
    void fetchLlamaServerArguments()
      .then((response) => {
        if (cancelled) return;
        setCatalog(llamaServerDiagnosticCatalog(response));
        setCatalogAuthoritative(response.authoritative);
        setCatalogState(response.authoritative ? "available" : "unavailable");
      })
      .catch(() => {
        if (!cancelled) {
          const authoritative = cachedAuthoritativeLlamaServerArguments();
          const managedPolicy = cachedLlamaServerManagedPolicy();
          setCatalog(
            authoritative
              ? llamaServerDiagnosticCatalog(authoritative)
              : managedPolicy
                ? llamaServerDiagnosticCatalog({
                    arguments: [],
                    ...managedPolicy,
                  })
                : null,
          );
          // A cached catalog is useful diagnostic context, but an outage cannot
          // prove that the installed binary still accepts it.
          setCatalogAuthoritative(false);
          setCatalogState("unavailable");
        }
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const tokens = useMemo(() => value ?? [], [value]);
  const rows = useMemo(
    () => llamaExtraArgRows(tokens, catalog ?? []),
    [tokens, catalog],
  );
  const diagnostics = useMemo(
    () =>
      diagnoseLlamaExtraArgs(
        formatLlamaExtraArgs(tokens),
        catalog,
        catalogAuthoritative,
      ),
    [tokens, catalog, catalogAuthoritative],
  );
  const catalogUnavailableBlocked = llamaExtraArgsCatalogBlocksPersistence(
    tokens,
    catalogState === "available",
    catalogAuthoritative,
  );
  const blocked =
    draftError !== null ||
    catalogUnavailableBlocked ||
    diagnostics.some((diagnostic) => diagnostic.severity === "error");
  const errors = diagnostics.filter(
    (diagnostic) => diagnostic.severity === "error",
  );
  const warnings = diagnostics.filter(
    (diagnostic) => diagnostic.severity === "warning",
  );
  useEffect(() => {
    onBlockingChange(blocked);
  }, [blocked, onBlockingChange]);

  useEffect(() => {
    optionRefs.current[selected]?.scrollIntoView({ block: "nearest" });
  }, [selected]);

  const refreshCompletions = (nextDraft: string, caret: number) => {
    const next = catalog
      ? completeLlamaExtraArgs(nextDraft, caret, catalog)
      : [];
    if (next.length > 0 && completions.length === 0) {
      const rect = inputRef.current?.getBoundingClientRect();
      if (rect) {
        const spaceAbove = rect.top;
        const spaceBelow = window.innerHeight - rect.bottom;
        setCompletionSide(spaceBelow >= spaceAbove ? "bottom" : "top");
      }
    }
    setCompletions(next);
    setSelected(0);
  };

  const changeDraft = (nextDraft: string, caret?: number) => {
    setDraft(nextDraft);
    setDraftError(null);
    refreshCompletions(nextDraft, caret ?? nextDraft.length);
  };

  const commitDraft = (nextDraft = draft): boolean => {
    const parsed = parseLlamaExtraArgs(nextDraft);
    if (parsed.error) {
      setDraftError(parsed.error.message);
      return false;
    }
    if (parsed.tokens.length === 0) return false;
    onChange([...tokens, ...parsed.tokens]);
    setDraft("");
    setDraftError(null);
    setCompletions([]);
    return true;
  };

  const accept = (completion: LlamaExtraArgsCompletion) => {
    const applied = applyLlamaExtraArgsCompletion(draft, completion);
    const needsValue =
      completion.kind === "flag" &&
      llamaServerArgumentTakesValue(completion.argument);
    if (needsValue) {
      changeDraft(applied.text, applied.caret);
    } else {
      commitDraft(applied.text);
    }
    requestAnimationFrame(() => {
      inputRef.current?.focus({ preventScroll: true });
      if (needsValue) {
        inputRef.current?.setSelectionRange(applied.caret, applied.caret);
      }
    });
  };

  const onKeyDown = (event: KeyboardEvent<HTMLInputElement>) => {
    if (
      completions.length > 0 &&
      (event.key === "ArrowDown" || event.key === "ArrowUp")
    ) {
      event.preventDefault();
      setSelected((current) =>
        moveLlamaExtraArgsSelection(
          current,
          event.key === "ArrowDown" ? "next" : "previous",
          completions.length,
        ),
      );
      return;
    }
    if (
      completions.length > 0 &&
      (event.key === "Tab" || event.key === "Enter")
    ) {
      event.preventDefault();
      accept(completions[selected]);
      return;
    }
    if (event.key === "Enter") {
      event.preventDefault();
      commitDraft();
      return;
    }
    if (event.key === "Escape" && completions.length > 0) {
      event.preventDefault();
      setCompletions([]);
    }
  };

  const selectedArgument = completions[selected]?.argument;
  const selectedGroup = selectedArgument
    ? llamaServerArgumentGroupLabel(selectedArgument)
    : null;
  const selectedDescription = selectedArgument?.description
    .replace(/\s+\(env:[^)]*\)\s*$/i, "")
    .replace(/\[([^\]]+)\]\([^)]+\)/g, "$1")
    .trim();
  const flagCount = countLlamaExtraArgFlags(value);
  const configured = tokens.length > 0;
  return (
    <Collapsible
      open={open}
      onOpenChange={setOpen}
      className="rounded-xl border border-border/60"
    >
      <CollapsibleTrigger className="flex min-h-10 w-full items-center justify-between gap-3 px-3 text-start">
        <span className="min-w-0 truncate text-ui-13 font-medium text-nav-fg">
          llama.cpp arguments
        </span>
        <span className="flex shrink-0 items-center gap-2 text-ui-12 tabular-nums text-muted-foreground">
          {configured
            ? `${flagCount} ${flagCount === 1 ? "flag" : "flags"}`
            : "None"}
          <HugeiconsIcon
            icon={ChevronDownStandardIcon}
            className={cn(
              "size-3.5 motion-safe:transition-transform",
              open && "rotate-180",
            )}
            strokeWidth={1.75}
          />
        </span>
      </CollapsibleTrigger>
      <CollapsibleContent>
        <div className="space-y-2 border-t border-border/60 p-3">
          {rows.length > 0 ? (
            <div className="space-y-1.5">
              {rows.map((row) => (
                <div
                  key={row.start}
                  className="flex min-w-0 items-center gap-1.5"
                >
                  <div
                    className={cn(
                      "grid min-h-9 min-w-0 flex-1 items-center gap-2 rounded-lg border border-border bg-background px-2.5 dark:border-transparent dark:bg-white/[0.06]",
                      row.valueExpected
                        ? "grid-cols-[minmax(0,1fr)_1px_minmax(5rem,1fr)]"
                        : "grid-cols-1",
                    )}
                  >
                    <Input
                      value={row.flag}
                      title={row.flag}
                      spellCheck={false}
                      autoComplete="off"
                      aria-label={`Argument ${row.flag}`}
                      className="h-7 min-w-0 cursor-text rounded-none border-0 bg-transparent px-2 font-mono text-ui-12 font-medium shadow-none hover:bg-muted/25 focus-visible:bg-muted/40 focus-visible:ring-0 dark:bg-transparent dark:hover:bg-white/[0.04] dark:focus-visible:bg-white/[0.06]"
                      onChange={(event) =>
                        onChange(
                          replaceLlamaExtraArgRowFlag(
                            tokens,
                            row,
                            event.target.value,
                          ),
                        )
                      }
                    />
                    {row.valueExpected ? (
                      <>
                        <span
                          aria-hidden="true"
                          className="h-5 bg-border dark:bg-white/15"
                        />
                        <Input
                          value={row.value ?? ""}
                          title={row.value ?? ""}
                          spellCheck={false}
                          aria-label={`Value for ${row.flag}`}
                          placeholder={row.argument?.value_hint ?? "value"}
                          className="h-7 min-w-0 cursor-text rounded-none border-0 bg-transparent px-2 font-mono text-ui-12 shadow-none hover:bg-muted/25 focus-visible:bg-muted/40 focus-visible:ring-0 dark:bg-transparent dark:hover:bg-white/[0.04] dark:focus-visible:bg-white/[0.06]"
                          onChange={(event) =>
                            onChange(
                              replaceLlamaExtraArgRow(
                                tokens,
                                row,
                                event.target.value,
                              ),
                            )
                          }
                        />
                      </>
                    ) : null}
                  </div>
                  <Button
                    type="button"
                    variant="ghost"
                    size="icon-sm"
                    aria-label={`Remove ${row.flag}`}
                    className="rounded-lg text-muted-foreground hover:text-destructive"
                    onClick={() =>
                      onChange([
                        ...tokens.slice(0, row.start),
                        ...tokens.slice(row.end),
                      ])
                    }
                  >
                    <HugeiconsIcon
                      icon={Cancel01Icon}
                      className="size-3.5"
                      strokeWidth={2}
                    />
                  </Button>
                </div>
              ))}
            </div>
          ) : null}

          <div className="flex items-center gap-1.5">
            <Popover open={completions.length > 0}>
              <PopoverAnchor asChild={true}>
                <Input
                  ref={inputRef}
                  value={draft}
                  spellCheck={false}
                  autoComplete="off"
                  role="combobox"
                  aria-label="Add llama.cpp argument"
                  aria-controls={
                    completions.length > 0 ? completionListId : undefined
                  }
                  aria-expanded={completions.length > 0}
                  aria-activedescendant={
                    completions.length > 0
                      ? `${completionListId}-${selected}`
                      : undefined
                  }
                  aria-invalid={draftError ? true : undefined}
                  placeholder="Add argument"
                  className="h-9 rounded-lg font-mono text-ui-12"
                  onChange={(event) =>
                    changeDraft(
                      event.target.value,
                      event.target.selectionStart ?? event.target.value.length,
                    )
                  }
                  onClick={(event) =>
                    refreshCompletions(
                      draft,
                      event.currentTarget.selectionStart ?? draft.length,
                    )
                  }
                  onKeyDown={onKeyDown}
                  onBlur={() => setCompletions([])}
                />
              </PopoverAnchor>
              <PopoverContent
                align="start"
                side={completionSide}
                sideOffset={4}
                avoidCollisions={false}
                onOpenAutoFocus={(event) => event.preventDefault()}
                onCloseAutoFocus={(event) => event.preventDefault()}
                className="menu-soft-surface w-[min(18rem,calc(100vw-2rem))] gap-0 border-0 p-1 ring-0"
              >
                <div
                  {...LISTBOX_ROLE}
                  id={completionListId}
                  tabIndex={-1}
                  aria-label="llama.cpp argument suggestions"
                  className="max-h-44 overflow-y-auto overscroll-contain"
                >
                  {completions.map((completion, index) => (
                    <button
                      {...{
                        role: "option" as const,
                        "aria-selected": index === selected,
                      }}
                      key={`${completion.kind}:${completion.label}`}
                      ref={(element) => {
                        optionRefs.current[index] = element;
                      }}
                      id={`${completionListId}-${index}`}
                      type="button"
                      tabIndex={-1}
                      onMouseDown={(event) => event.preventDefault()}
                      onMouseEnter={() => setSelected(index)}
                      onClick={() => accept(completion)}
                      className={cn(
                        "flex min-h-8 w-full items-center justify-between gap-3 rounded-md px-2.5 py-1.5 text-start",
                        index === selected
                          ? "bg-accent text-accent-foreground"
                          : "hover:bg-accent/60",
                      )}
                    >
                      <code className="truncate text-ui-12 font-semibold">
                        {completion.label}
                      </code>
                      <span className="max-w-32 truncate text-ui-10 text-muted-foreground">
                        {llamaServerArgumentGroupLabel(completion.argument)}
                      </span>
                    </button>
                  ))}
                </div>
                {selectedDescription || selectedGroup ? (
                  <p className="line-clamp-3 break-words border-t border-border/60 px-2.5 py-2 text-pretty text-ui-10 leading-snug text-muted-foreground">
                    {selectedGroup}
                    {selectedDescription ? ` · ${selectedDescription}` : ""}
                  </p>
                ) : null}
              </PopoverContent>
            </Popover>
            <Button
              type="button"
              variant="secondary"
              size="sm"
              className="rounded-lg"
              disabled={draft.trim().length === 0}
              onClick={() => commitDraft()}
            >
              Add
            </Button>
          </div>

          {draftError ? (
            <p className="text-ui-11 text-destructive">{draftError}</p>
          ) : catalogUnavailableBlocked ? (
            <p className="text-ui-11 text-destructive">
              Argument policy unavailable. Remove custom arguments or retry
              before saving.
            </p>
          ) : catalogState === "unavailable" ? (
            <p className="text-ui-11 text-muted-foreground">
              Autocomplete unavailable.
            </p>
          ) : null}
          {errors.length > 0 ? (
            <ul className="space-y-1" aria-live="polite">
              {errors.map((diagnostic, index) => (
                <li
                  key={`${diagnostic.kind}:${diagnostic.tokenIndex ?? "all"}:${index}`}
                  className="text-ui-11 leading-snug text-destructive"
                >
                  {diagnostic.message}
                </li>
              ))}
            </ul>
          ) : null}
          {warnings.length > 0 ? (
            <details className="group text-ui-11">
              <summary className="flex cursor-pointer list-none items-center gap-1.5 text-muted-foreground outline-none hover:text-foreground focus-visible:text-foreground [&::-webkit-details-marker]:hidden">
                <span className="size-1.5 rounded-full bg-amber-500" />
                <span>
                  {warnings.length}{" "}
                  {warnings.length === 1 ? "warning" : "warnings"}
                </span>
                <HugeiconsIcon
                  icon={ChevronDownStandardIcon}
                  className="ml-auto size-3 motion-safe:transition-transform group-open:rotate-180"
                  strokeWidth={1.75}
                />
              </summary>
              <ul className="mt-2 max-h-28 space-y-1.5 overflow-y-auto rounded-lg bg-muted/35 p-2 text-amber-700 dark:text-amber-300">
                {warnings.map((diagnostic, index) => (
                  <li
                    key={`${diagnostic.kind}:${diagnostic.tokenIndex ?? "all"}:${index}`}
                    className="leading-snug"
                  >
                    {diagnostic.message}
                  </li>
                ))}
              </ul>
            </details>
          ) : null}
          <p
            title="Inherited LLAMA_ARG_* environment variables are ignored."
            className="truncate text-ui-10 text-muted-foreground"
          >
            LLAMA_ARG_* env vars are ignored.
          </p>
        </div>
      </CollapsibleContent>
    </Collapsible>
  );
}
