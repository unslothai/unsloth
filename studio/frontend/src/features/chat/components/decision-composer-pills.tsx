// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * The three decision controls in the chat composer.
 *
 * A decision-only GGUF cannot chat, so the composer's text is the state and
 * these pills carry the question: its type, its instructions, and the answer
 * space (criteria). Everything is edited in place, inside the composer surface,
 * with no dialog and no second border.
 */

import {
  ChoiceCriteria,
  NoulCriteria,
  ScoreCriteria,
} from "@/components/decision/decision-parts";
import { AutoGrowTextarea } from "@/components/decision/auto-grow-textarea";
import {
  DecisionCriteriaIcon,
  DecisionInstructionsIcon,
  DecisionQuestionIcon,
} from "@/components/decision/decision-icons";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
// eslint-disable-next-line no-restricted-imports -- Leaf module: request types, no React.
import type { DecisionType } from "@/features/settings/lib/decision-request";
import { type TranslationKey, useT } from "@/i18n";
import { ChevronDownIcon, Tick02Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { type ReactElement, useEffect, useRef } from "react";
import { resolveDecisionQuestionTypes } from "../lib/decision-capabilities";
import {
  useDecisionConfigStore,
} from "../stores/decision-config-store";
import { useChatRuntimeStore } from "../stores/chat-runtime-store";

const TYPE_LABEL: Record<DecisionType, TranslationKey> = {
  noul: "decisions.typeBoolean",
  choice: "decisions.typeChoice",
  score: "decisions.typeScore",
};

function Pill({
  icon,
  title,
  value,
  active,
  onClick,
}: {
  icon: ReactElement;
  title: string;
  value?: string;
  active?: boolean;
  onClick: () => void;
}): ReactElement {
  return (
    <button
      type="button"
      onClick={onClick}
      aria-expanded={active}
      data-keep-label=""
      data-active={active ? "true" : "false"}
      className="composer-pill-btn"
      data-pill-label={title}
    >
      {icon}
      <span>{title}</span>
      {value ? (
        <span className="max-w-28 truncate text-muted-foreground">{value}</span>
      ) : null}
      <HugeiconsIcon
        icon={ChevronDownIcon}
        strokeWidth={1.75}
        className="size-3 opacity-60"
      />
    </button>
  );
}

/** The three pills. They render nothing for a non-decision model. */
export function DecisionComposerPills(): ReactElement | null {
  const t = useT();
  const isDecision = useChatRuntimeStore((s) => s.loadedIsDecision);
  const decisionType = useChatRuntimeStore((s) => s.loadedDecisionType);
  const question = useDecisionConfigStore((s) => s.question);
  const openPanel = useDecisionConfigStore((s) => s.openPanel);
  const togglePanel = useDecisionConfigStore((s) => s.togglePanel);
  const update = useDecisionConfigStore((s) => s.update);

  const supported = resolveDecisionQuestionTypes({ isDecision, decisionType });

  useEffect(() => {
    if (isDecision && supported.length > 0 && !supported.includes(question.type)) {
      update({ type: supported[0] });
    }
  }, [isDecision, supported, question.type, update]);

  if (!isDecision || supported.length === 0) return null;

  return (
    <>
      <DropdownMenu>
        <DropdownMenuTrigger asChild={true}>
          <button
            type="button"
            data-keep-label=""
            className="composer-pill-btn"
            data-pill-label={t("decisions.questionPill")}
          >
            <DecisionQuestionIcon className="size-[calc(16px*var(--ui-space-scale,1))]" />
            <span>{t(TYPE_LABEL[question.type])}</span>
            <HugeiconsIcon
              icon={ChevronDownIcon}
              strokeWidth={1.75}
              className="size-3 opacity-60"
            />
          </button>
        </DropdownMenuTrigger>
        <DropdownMenuContent
          side="bottom"
          align="start"
          sideOffset={6}
          className="w-48 rounded-xl border border-border/60 bg-background p-1 shadow-lg dark:bg-card"
        >
          {supported.map((type) => (
            <DropdownMenuItem
              key={type}
              onSelect={() => update({ type })}
              className="justify-between gap-6"
            >
              <span>{t(TYPE_LABEL[type])}</span>
              {type === question.type ? (
                <HugeiconsIcon
                  icon={Tick02Icon}
                  strokeWidth={1.75}
                  className="size-3.5 text-foreground"
                />
              ) : null}
            </DropdownMenuItem>
          ))}
        </DropdownMenuContent>
      </DropdownMenu>

      <Pill
        icon={
          <DecisionInstructionsIcon className="size-[calc(16px*var(--ui-space-scale,1))]" />
        }
        title={t("decisions.instructionsPill")}
        active={openPanel === "instructions"}
        onClick={() => togglePanel("instructions")}
      />

      <Pill
        icon={
          <DecisionCriteriaIcon className="size-[calc(16px*var(--ui-space-scale,1))]" />
        }
        title={t("decisions.criteriaPill")}
        active={openPanel === "criteria"}
        onClick={() => togglePanel("criteria")}
      />
    </>
  );
}

/**
 * The inline editor for whichever pill is open.
 *
 * `placement` decides the side: under the prompt box in a new chat, above it in
 * a conversation, so the history never jumps when a drawer opens.
 */
export function DecisionComposerPanel({
  placement = "bottom",
}: {
  placement?: "top" | "bottom";
}): ReactElement | null {
  const t = useT();
  const isDecision = useChatRuntimeStore((s) => s.loadedIsDecision);
  const question = useDecisionConfigStore((s) => s.question);
  const openPanel = useDecisionConfigStore((s) => s.openPanel);
  const setPanel = useDecisionConfigStore((s) => s.setPanel);
  const update = useDecisionConfigStore((s) => s.update);
  const panelRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (!isDecision || openPanel === "none") return;
    const handlePointerDown = (event: PointerEvent) => {
      const target = event.target as Element | null;
      if (!target) return;
      // Keep the drawer open when interacting with the composer, the drawer itself,
      // or a portaled dropdown menu (e.g. switching question type).
      const root = panelRef.current?.closest("[data-decision-drawer]");
      if (root?.contains(target)) return;
      if (target.closest("[data-radix-popper-content-wrapper]")) return;
      setPanel("none");
    };
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") setPanel("none");
    };
    document.addEventListener("pointerdown", handlePointerDown);
    document.addEventListener("keydown", handleKeyDown);
    return () => {
      document.removeEventListener("pointerdown", handlePointerDown);
      document.removeEventListener("keydown", handleKeyDown);
    };
  }, [isDecision, openPanel, setPanel]);

  if (!isDecision || openPanel === "none") return null;

  const isCriteria = openPanel === "criteria";
  const panelLabel = isCriteria
    ? t("decisions.criteriaPill")
    : t("decisions.instructionsPill");

  // Keep Instructions and Boolean Criteria at the exact same compact height:
  // NoulCriteria = 22px label+gap + 52px textarea = 74px.
  const COMPACT_HEIGHT = 74;

  return (
    <div
      ref={panelRef}
      data-decision-panel={openPanel}
      data-placement={placement}
      role="region"
      aria-label={panelLabel}
    >
      <span aria-hidden="true" className="decision-neck" />
      <span aria-hidden="true" className="decision-fillet decision-fillet-left" />
      <span aria-hidden="true" className="decision-fillet decision-fillet-right" />
      <div className={placement === "bottom" ? "px-3 pb-2.5 pt-0.5" : "px-3 pt-2.5 pb-0.5"}>
        <div className="max-h-[32dvh] w-full overflow-x-hidden overflow-y-auto overscroll-contain rounded-lg bg-muted/40 p-1.5 dark:bg-card">
          {isCriteria ? (
            question.type === "noul" ? (
              <NoulCriteria
                draft={question.draft}
                update={(patch) => update({ draft: patch })}
                idPrefix={question.id}
                minHeight={52}
              />
            ) : question.type === "choice" ? (
              <ChoiceCriteria
                draft={question.draft}
                update={(patch) => update({ draft: patch })}
              />
            ) : (
              <div className="flex flex-col gap-1.5">
                <span className="px-0.5 text-ui-11 text-muted-foreground">
                  {t("decisions.criteriaScoreHelp")}
                </span>
                <ScoreCriteria
                  draft={question.draft}
                  update={(patch) => update({ draft: patch })}
                />
              </div>
            )
          ) : (
            <AutoGrowTextarea
              autoFocus={true}
              minHeight={COMPACT_HEIGHT}
              value={question.draft.instructions}
              onChange={(event) =>
                update({ draft: { instructions: event.target.value } })
              }
              placeholder={t("decisions.instructionsPlaceholder")}
              aria-label={t("decisions.instructionsPill")}
              className="max-h-[26dvh] border-0 bg-transparent px-1.5 py-1 focus-visible:border-0"
            />
          )}
        </div>
      </div>
    </div>
  );
}
