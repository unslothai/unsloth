// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { useT } from "@/i18n";
import { cn } from "@/lib/utils";
import { PlusSignIcon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useState } from "react";
import { MAX_ALTERNATE_GREETINGS } from "../../draft";
import { Card } from "./editor-section";

/** The main greeting and its alternates, one tab each. */
export function GreetingsEditor({
  firstMes,
  alternates,
  onFirstMes,
  onAlternates,
}: {
  firstMes: string;
  alternates: string[];
  onFirstMes: (text: string) => void;
  onAlternates: (alternates: string[]) => void;
}) {
  const t = useT();
  const [current, setCurrent] = useState(0);
  const greetings = [firstMes, ...alternates];

  function setText(index: number, text: string) {
    if (index === 0) onFirstMes(text);
    else onAlternates(alternates.map((g, i) => (i === index - 1 ? text : g)));
  }

  return (
    <Card>
      <div className="flex flex-col gap-1">
        <h2 className="text-ui-15 font-semibold text-foreground">{t("characters.editor.greetings")}</h2>
        <p className="text-ui-13 text-muted-foreground">{t("characters.editor.greetingsHint")}</p>
      </div>
      <div className="flex flex-wrap items-center gap-1.5" role="tablist" aria-label={t("characters.editor.greetings")}>
        {greetings.map((_, index) => (
          <button
            key={index}
            type="button"
            role="tab"
            aria-selected={current === index}
            onClick={() => setCurrent(index)}
            className={cn(
              "h-8 cursor-pointer rounded-full px-3.5 text-ui-13 font-medium transition-colors",
              current === index ? "bg-foreground text-background" : "bg-muted text-muted-foreground hover:text-foreground dark:bg-card",
            )}
          >
            {index === 0 ? t("characters.editor.mainGreeting") : t("characters.editor.alternateGreeting", { number: index })}
          </button>
        ))}
        <Button
          type="button"
          variant="ghost"
          size="sm"
          className="rounded-full"
          disabled={alternates.length >= MAX_ALTERNATE_GREETINGS}
          onClick={() => {
            onAlternates([...alternates, ""]);
            setCurrent(greetings.length);
          }}
        >
          <HugeiconsIcon icon={PlusSignIcon} strokeWidth={1.75} className="size-icon" />
          {t("characters.editor.addGreeting")}
        </Button>
      </div>
      <div className="relative">
        <Textarea
          value={greetings[current] ?? ""}
          placeholder={t("characters.editor.greetingPlaceholder")}
          aria-label={t("characters.editor.greetings")}
          className="max-h-[55vh] min-h-36 font-sans text-ui-15 leading-relaxed"
          onChange={(event) => setText(current, event.target.value)}
        />
        {current > 0 ? (
          <Button
            type="button"
            variant="ghost"
            size="sm"
            className="mt-2"
            onClick={() => {
              onAlternates(alternates.filter((_, i) => i !== current - 1));
              setCurrent(Math.max(0, current - 1));
            }}
          >
            {t("characters.editor.removeGreeting")}
          </Button>
        ) : null}
      </div>
    </Card>
  );
}
