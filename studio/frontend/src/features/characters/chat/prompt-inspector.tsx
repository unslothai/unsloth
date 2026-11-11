// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { useT } from "@/i18n";
import { copyToClipboard } from "@/lib/copy-to-clipboard";
import { useEffect, useState } from "react";
import { prepareCharacterRun, textOf, type CharacterRun } from "../chat-runtime";
import { estimateTokens } from "../prompt/lorebook";
import { useAskBridge } from "./ask-bridge";

/** Shows what the next reply's model call carries: the system block by section, the tail of the chat, and active lore. */
export function PromptInspectorDialog({ open, threadId, onClose }: { open: boolean; threadId: string; onClose: () => void }) {
  const t = useT();
  const [run, setRun] = useState<CharacterRun | null>(null);
  const [copied, setCopied] = useState(false);

  useEffect(() => {
    if (!open) return;
    let live = true;
    setRun(null);
    const bridge = useAskBridge.getState();
    const messages = bridge.threadId === threadId && bridge.messages ? bridge.messages() : [];
    void prepareCharacterRun({ threadId, messages }).then((next) => {
      if (live) setRun(next);
    });
    return () => {
      live = false;
    };
  }, [open, threadId]);

  const tail = run ? run.messages.slice(-4) : [];
  return (
    <Dialog open={open} onOpenChange={(next) => !next && onClose()}>
      <DialogContent className="flex max-h-[min(860px,90dvh)] flex-col gap-4 sm:max-w-3xl">
        <DialogHeader>
          <DialogTitle>{t("characters.panel.promptTitle")}</DialogTitle>
          <DialogDescription>
            {run ? t("characters.panel.promptHint", { name: run.replyMeta.speakerName }) : " "}
          </DialogDescription>
        </DialogHeader>
        {run ? (
          <div className="-mx-1 flex min-h-0 flex-1 flex-col gap-4 overflow-y-auto px-1" data-testid="prompt-inspector">
            {run.built.sections.map((section) => (
              <section key={section.label} className="flex flex-col gap-1.5">
                <h3 className="flex items-baseline justify-between text-ui-13 font-semibold text-foreground">
                  {section.label}
                  <span className="text-ui-12 font-normal text-muted-foreground">~{estimateTokens(section.text)} tok</span>
                </h3>
                <pre className="max-h-64 overflow-y-auto whitespace-pre-wrap rounded-xl bg-muted px-3.5 py-3 font-mono text-ui-12 leading-relaxed text-foreground/90 dark:bg-card">
                  {section.text}
                </pre>
              </section>
            ))}
            <section className="flex flex-col gap-1.5">
              <h3 className="text-ui-13 font-semibold text-foreground">{t("characters.panel.loreInUse")}</h3>
              {run.built.lore.fired.length ? (
                <div className="flex flex-wrap gap-1.5">
                  {run.built.lore.fired.map((entry) => (
                    <span key={`${entry.id}-${entry.label}`} className="rounded-full bg-muted px-2.5 py-0.5 text-ui-12 dark:bg-card">
                      {entry.label}
                    </span>
                  ))}
                </div>
              ) : (
                <p className="text-ui-13 text-muted-foreground">{t("characters.panel.noLore")}</p>
              )}
            </section>
            {tail.length ? (
              <section className="flex flex-col gap-1.5">
                {tail.map((message, index) => (
                  <pre
                    key={`${message.id}-${index}`}
                    className="whitespace-pre-wrap rounded-xl border border-border/70 px-3.5 py-2.5 font-mono text-ui-12 leading-relaxed text-foreground/85"
                  >
                    <span className="mb-1 block text-ui-11 font-semibold uppercase tracking-wider text-muted-foreground">{message.role}</span>
                    {textOf(message)}
                  </pre>
                ))}
                <p className="text-ui-12 text-muted-foreground">stop: {run.stop.map((s) => JSON.stringify(s)).join(", ")}</p>
              </section>
            ) : null}
          </div>
        ) : (
          <div className="h-40 animate-pulse rounded-xl bg-muted" />
        )}
        {run ? (
          <div className="flex justify-end">
            <Button
              variant="outline"
              size="sm"
              onClick={() => {
                void copyToClipboard(run.system).then(() => {
                  setCopied(true);
                  window.setTimeout(() => setCopied(false), 1500);
                });
              }}
            >
              {copied ? t("characters.panel.copied") : t("characters.panel.copy")}
            </Button>
          </div>
        ) : null}
      </DialogContent>
    </Dialog>
  );
}
