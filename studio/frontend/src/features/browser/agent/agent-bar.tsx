// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import { Spinner } from "@/components/ui/spinner";
import { useT } from "@/i18n";
import { useCallback, useState } from "react";
import { useBrowserStore } from "../store";
import { type AgentStep, IDLE_AGENT, answerQuestion, stopBrowserAgent, useBrowserAgentStore } from "./agent-store";
import { shareAgentTab } from "./tab-sharing";

const NAMED_STEPS = new Set<AgentStep["kind"]>(["open", "find", "click", "type", "select", "press"]);

export function AgentBar({ activeTabId }: { activeTabId: string | null }) {
  const t = useT();
  const chat = useBrowserStore((state) => state.chat);
  const { tabId, shared, working, step, runKey, stoppedRun, question } = useBrowserAgentStore((state) => state.chats[chat] ?? IDLE_AGENT);
  const target = useBrowserStore((state) => state.tabs.find((tab) => tab.id === tabId && tab.chat === chat));
  // The whole run, not just each action: Stop between calls stops the next one before it starts.
  const running = working || (!!runKey && stoppedRun !== runKey);
  const [readId, setReadId] = useState<number | null>(null);
  const script = question?.ask === "script";
  const targetName = target?.customTitle || target?.title || target?.displayUrl || "";
  const sharedLabel = t("browser.agent.sharedTarget", { title: targetName });
  const doing =
    working && step && (step.detail || !NAMED_STEPS.has(step.kind))
      ? t(`browser.agent.step.${step.kind}`, { detail: step.detail })
      : t("browser.agent.working");
  if (!question && !(running && tabId === activeTabId) && !(shared && target)) return null;
  return (
    <div
      role={question ? "alertdialog" : "status"}
      aria-live="polite"
      className="flex shrink-0 flex-wrap items-center gap-2 border-t border-border/70 bg-wash-5 px-3 py-1.5 text-ui-12"
    >
      {!question && running ? <Spinner className="size-3.5 text-muted-foreground" /> : null}
      <div className="min-w-0 flex-[1_1_12rem]">
        <p className="truncate font-medium leading-tight text-foreground">
          {question ? t(`browser.agent.ask.${question.ask}`, { host: question.host, name: question.name }) : running ? doing : sharedLabel}
        </p>
        {shared && target && (running || question) ? <p className="truncate text-muted-foreground">{sharedLabel}</p> : null}
        {question && script ? (
          <ScriptPreview key={question.id} code={question.detail} onRead={() => setReadId(question.id)} />
        ) : question?.detail ? (
          <p className="truncate text-ui-11 leading-tight text-muted-foreground" title={question.detail}>
            {question.detail}
          </p>
        ) : null}
      </div>
      <div className="ml-auto flex flex-wrap items-center justify-end gap-2">
        {running || question ? (
          <Button type="button" size="xs" variant="outline" onClick={() => stopBrowserAgent(chat)}>
            {t("browser.agent.stop")}
          </Button>
        ) : null}
        {question ? (
          <>
            <span aria-hidden={true} className="h-4 w-px bg-border" />
            <Button type="button" size="xs" variant="ghost" onClick={() => answerQuestion(question.id, false)}>
              {t("browser.agent.deny")}
            </Button>
            <Button
              type="button"
              size="xs"
              disabled={script && readId !== question.id}
              onClick={() => answerQuestion(question.id, true)}
            >
              {t("browser.agent.allow")}
            </Button>
            {question.always ? (
              <Button type="button" size="xs" variant="outline" onClick={() => answerQuestion(question.id, true, true)}>
                {t("browser.agentSettings.alwaysAllow")}
              </Button>
            ) : null}
          </>
        ) : null}
        {shared && target ? (
          <Button type="button" size="xs" variant="outline" onClick={() => shareAgentTab(chat, null)}>
            {t("browser.tabMenu.stopSharing")}
          </Button>
        ) : null}
      </div>
    </div>
  );
}

/** `onRead` once the script's end has been on screen: code past the fold can't be approved unseen. */
function ScriptPreview({ code, onRead }: { code: string; onRead: () => void }) {
  const check = useCallback(
    (el: HTMLElement | null) => {
      if (el && el.scrollTop + el.clientHeight >= el.scrollHeight - 2) onRead();
    },
    [onRead],
  );
  return (
    <pre
      ref={check}
      onScroll={(event) => check(event.currentTarget)}
      className="mt-1 max-h-40 overflow-auto whitespace-pre-wrap break-all rounded-lg bg-muted px-2 py-1 font-mono text-ui-11 text-foreground"
    >
      {code}
    </pre>
  );
}
