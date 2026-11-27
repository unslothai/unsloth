// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { errorMessage } from "@/lib/error-message";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { useT, type TranslationKey } from "@/i18n";
import { estimateTokenCount } from "@/lib/estimate-tokens";
import { toast } from "@/lib/toast";
import { useCopyFeedback } from "@/hooks/use-copy-feedback";
import { useEffect, useState } from "react";
import { prepareCharacterRun, type CharacterRun } from "../chat-runtime";
import { textOf } from "../prompt/round";
import { useAskBridge } from "./ask-bridge";

const ROLE_LABEL: Record<string, TranslationKey> = {
  system: "characters.panel.roleSystem",
  user: "characters.panel.roleUser",
  assistant: "characters.panel.roleAssistant",
};

type Inspected =
  | { status: "loading" }
  | { status: "ready"; run: CharacterRun; project: string }
  | { status: "error"; message: string };

// Loaded when the dialog opens: the chat adapter imports this feature, so a static import would be a cycle.
async function projectBlock(threadId: string): Promise<string> {
  const { projectInstructionsFor } = await import("@/features/chat/api/chat-adapter");
  return projectInstructionsFor(threadId);
}

/** Shows what the next reply's model call carries: the system block by section, the tail of the chat, and active lore. */
export function PromptInspectorDialog({ open, threadId, onClose }: { open: boolean; threadId: string; onClose: () => void }) {
  const t = useT();
  const [inspected, setInspected] = useState<Inspected>({ status: "loading" });
  const { copied, copy } = useCopyFeedback();

  useEffect(() => {
    if (!open) return;
    let live = true;
    setInspected({ status: "loading" });
    const bridge = useAskBridge.getState();
    const messages = bridge.threadId === threadId && bridge.messages ? bridge.messages() : [];
    Promise.all([prepareCharacterRun({ threadId, messages }), projectBlock(threadId)]).then(
      ([next, project]) => {
        if (!live) return;
        setInspected(
          next ? { status: "ready", run: next, project } : { status: "error", message: t("characters.panel.promptUnavailable") },
        );
      },
      (error: unknown) => {
        if (live) setInspected({ status: "error", message: errorMessage(error) });
      },
    );
    return () => {
      live = false;
    };
  }, [open, threadId, t]);

  const run = inspected.status === "ready" ? inspected.run : null;
  // The send puts the project's instructions and AGENTS.md ahead of the character prompt.
  const project = inspected.status === "ready" ? inspected.project : "";
  const sections = run ? [...(project ? [{ label: "Project", text: project }] : []), ...run.built.sections] : [];

  // The last few messages, reaching back far enough to show every note placed at a depth.
  const firstNoted = run ? run.messages.findIndex((message) => run.notedIds.has(message.id)) : -1;
  const from = run ? Math.max(0, Math.min(run.messages.length - 4, firstNoted < 0 ? Infinity : firstNoted)) : 0;
  const tail = run ? run.messages.slice(from) : [];
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
            {sections.map((section) => (
              <section key={section.label} className="flex flex-col gap-1.5">
                <h3 className="flex items-baseline justify-between text-ui-13 font-semibold text-foreground">
                  {section.label}
                  <span className="text-ui-12 font-normal text-muted-foreground">
                    {t("characters.panel.approxTokens", { count: estimateTokenCount(section.text) ?? 0 })}
                  </span>
                </h3>
                <pre className="max-h-64 overflow-y-auto whitespace-pre-wrap rounded-xl bg-muted px-3.5 py-3 font-mono text-ui-12 leading-relaxed text-foreground/90 dark:bg-card">
                  {section.text}
                </pre>
              </section>
            ))}
            <section className="flex flex-col gap-1.5">
              <h3 className="text-ui-13 font-semibold text-foreground">{t("characters.panel.lorebookInUse")}</h3>
              {run.built.lore.fired.length ? (
                <div className="flex flex-wrap gap-1.5">
                  {run.built.lore.fired.map((entry) => (
                    <span key={`${entry.id}-${entry.label}`} className="rounded-full bg-muted px-2.5 py-0.5 text-ui-12 dark:bg-card">
                      {entry.label}
                    </span>
                  ))}
                </div>
              ) : (
                <p className="text-ui-13 text-muted-foreground">{t("characters.panel.noLorebook")}</p>
              )}
            </section>
            {tail.length ? (
              <section className="flex flex-col gap-1.5">
                {from > 0 ? (
                  <p className="text-ui-12 text-muted-foreground">{t("characters.panel.earlierMessages", { count: from })}</p>
                ) : null}
                {tail.map((message, index) => (
                  <pre
                    key={`${message.id}-${index}`}
                    className="whitespace-pre-wrap rounded-xl border border-border/70 px-3.5 py-2.5 font-mono text-ui-12 leading-relaxed text-foreground/85"
                  >
                    <span className="mb-1 block text-ui-11 font-semibold uppercase tracking-wider text-muted-foreground">
                      {ROLE_LABEL[message.role] ? t(ROLE_LABEL[message.role]) : message.role}
                    </span>
                    {textOf(message)}
                  </pre>
                ))}
                <p className="text-ui-12 text-muted-foreground">
                  {t("characters.panel.stopStrings", { list: run.stop.map((s) => JSON.stringify(s)).join(", ") })}
                </p>
              </section>
            ) : null}
          </div>
        ) : inspected.status === "error" ? (
          <p className="rounded-xl bg-muted px-3.5 py-3 text-ui-13 text-muted-foreground dark:bg-card" role="alert">
            {inspected.message}
          </p>
        ) : (
          <div className="h-40 animate-pulse rounded-xl bg-muted" />
        )}
        {run ? (
          <div className="flex justify-end">
            <Button
              variant="outline"
              size="sm"
              onClick={() => {
                void copy([project, run.system.trim()].filter(Boolean).join("\n\n")).then((ok) => {
                  if (!ok) toast.error(t("characters.panel.copyFailed"));
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
