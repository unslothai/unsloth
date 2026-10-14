// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import { Spinner } from "@/components/ui/spinner";
import { useT } from "@/i18n";
import { useCallback, useState } from "react";
import { answerQuestion, stopBrowserAgent, useBrowserAgentStore } from "./agent-store";

/** Above the agent's tab while it works: what it is doing, its questions, and Stop. */
export function AgentBar({ activeTabId }: { activeTabId: string | null }) {
  const t = useT();
  const tabId = useBrowserAgentStore((state) => state.tabId);
  // The whole run, not just each action: Stop between calls stops the next one before it starts.
  const running = useBrowserAgentStore((state) => state.working || (!!state.runKey && state.stoppedRun !== state.runKey));
  const question = useBrowserAgentStore((state) => state.question);
  const [readId, setReadId] = useState<number | null>(null);
  const script = question?.ask === "script";
  if (!question && !(running && tabId === activeTabId)) return null;
  return (
    <div
      role={question ? "alertdialog" : "status"}
      aria-live="polite"
      className="flex shrink-0 items-center gap-2 border-t border-border/70 bg-primary/5 px-3 py-1.5 text-ui-12"
    >
      {question ? null : <Spinner className="size-3.5 text-primary" />}
      <div className="min-w-0 flex-1">
        <p className="truncate font-medium text-foreground">
          {question ? t(`browser.agent.ask.${question.ask}`, { host: question.host, name: question.name }) : t("browser.agent.working")}
        </p>
        {question && script ? (
          <ScriptPreview key={question.id} code={question.detail} onRead={() => setReadId(question.id)} />
        ) : question?.detail ? (
          <p className="truncate text-muted-foreground" title={question.detail}>
            {question.detail}
          </p>
        ) : null}
      </div>
      {question ? (
        <>
          <Button type="button" size="xs" variant="ghost" onClick={() => answerQuestion(question.id, false)}>
            {t("browser.agent.deny")}
          </Button>
          {script ? null : (
            <Button type="button" size="xs" variant="outline" onClick={() => answerQuestion(question.id, true, true)}>
              {t("browser.agentSettings.alwaysAllow")}
            </Button>
          )}
          <Button
            type="button"
            size="xs"
            disabled={script && readId !== question.id}
            onClick={() => answerQuestion(question.id, true)}
          >
            {t("browser.agent.allow")}
          </Button>
        </>
      ) : null}
      <Button type="button" size="xs" variant="outline" onClick={stopBrowserAgent}>
        {t("browser.agent.stop")}
      </Button>
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
      className="mt-1 max-h-40 overflow-auto whitespace-pre-wrap break-all rounded-md bg-muted px-2 py-1 font-mono text-ui-11 text-foreground"
    >
      {code}
    </pre>
  );
}
