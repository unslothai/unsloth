// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Spinner } from "@/components/ui/spinner";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { useOnlineStatus } from "@/features/hub/hooks/use-online-status";
import { useSelectedModelMetadata } from "@/features/hub/hooks/use-selected-model-metadata";
import { confirmExternalLink } from "@/features/hub/stores/external-link-confirm";
import { useHfTokenStore } from "@/features/hub/stores/hf-token-store";
import { useHfEndpoint } from "@/lib/hf-endpoint";
import { cn } from "@/lib/utils";
import { LinkSquare02Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { type MouseEvent, useRef, useState } from "react";
import { ChatTemplateEditorDialog } from "../chat-template-editor-dialog";
import { localModelInfoFacts } from "./local-model-facts";
import {
  type ModelInfoFact,
  metaFromHfResult,
  modelInfoFacts,
} from "./model-info-facts";
import { useLocalModelMeta } from "./use-local-model-meta";

type InfoRow = Pick<ModelInfoFact, "label" | "value" | "detail"> & {
  key: string;
};

function FactRow({
  fact,
  onActivate,
}: {
  fact: InfoRow;
  /** Makes the row a button. Used by the chat-template row, which opens the template. */
  onActivate?: (event: MouseEvent<HTMLButtonElement>) => void;
}) {
  const value = (
    <span
      className={cn(
        fact.key === "license"
          ? "inline-flex min-h-6 items-center rounded-full border border-border/60 bg-muted/40 px-2 text-ui-11 font-medium text-foreground break-all"
          : "text-ui-12 text-foreground",
        onActivate && "underline decoration-dotted underline-offset-2",
      )}
    >
      {fact.value}
    </span>
  );

  const inner = (
    <div className="flex items-baseline justify-between gap-3 py-1.5">
      <span className="shrink-0 text-ui-12 text-muted-foreground">
        {fact.label}
      </span>
      {value}
    </div>
  );

  // A button, not a click handler on the div, so the row is reachable by keyboard.
  const row = onActivate ? (
    <button
      type="button"
      onClick={onActivate}
      className="block w-full cursor-pointer text-left"
    >
      {inner}
    </button>
  ) : (
    inner
  );

  if (!fact.detail) return row;
  return (
    <Tooltip delayDuration={0}>
      <TooltipTrigger
        asChild={true}
        tabIndex={0}
        className="rounded-sm outline-none focus-visible:ring-2 focus-visible:ring-ring"
      >
        {row}
      </TooltipTrigger>
      <TooltipContent side="left" className="max-w-[280px]">
        {fact.detail}
      </TooltipContent>
    </Tooltip>
  );
}

export function ModelInfoDialog({
  repoId,
  variant,
  hasLocalGguf = false,
  open,
  onOpenChange,
  onCloseAutoFocus,
}: {
  repoId: string;
  /** Quant to read the header of, when the row names one. */
  variant?: string | null;
  hasLocalGguf?: boolean;
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onCloseAutoFocus?: (event: Event) => void;
}) {
  const online = useOnlineStatus();
  const hfToken = useHfTokenStore((s) => s.token);
  const hfEndpoint = useHfEndpoint();
  const [templateOpen, setTemplateOpen] = useState(false);
  const templateTriggerRef = useRef<HTMLButtonElement>(null);

  // Only fetches while the dialog is open, and shares one in-flight request per repo with
  // the Hub through hf-cache, so opening this for a model the catalog already showed is free.
  const { result, error } = useSelectedModelMetadata(open ? repoId : null, {
    accessToken: hfToken || undefined,
    enabled: open,
    online,
  });

  // Validation does remote preflights too; only probe known local GGUFs.
  const localMeta = useLocalModelMeta(open ? repoId : null, {
    variant,
    hfToken: hfToken || undefined,
    enabled: open && hasLocalGguf,
  });

  const meta = metaFromHfResult(result);
  const facts = meta ? modelInfoFacts(meta) : [];
  const localFacts = localMeta ? localModelInfoFacts(localMeta) : [];
  const template = localMeta?.chatTemplate?.trim()
    ? localMeta.chatTemplate
    : null;

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent
        className="max-w-[460px]"
        onCloseAutoFocus={onCloseAutoFocus}
      >
        <DialogHeader>
          <DialogTitle className="truncate">{repoId}</DialogTitle>
          <DialogDescription className="sr-only">
            Model details.
          </DialogDescription>
        </DialogHeader>

        {/* Read from disk, so it renders on its own terms. The offline and error states
            below are scoped to the Hub section alone. */}
        {localFacts.length > 0 && (
          <div className="divide-y divide-border/50">
            {localFacts.map((fact) => (
              <FactRow
                key={fact.key}
                fact={fact}
                onActivate={
                  fact.key === "chatTemplate" && template
                    ? (event) => {
                        templateTriggerRef.current = event.currentTarget;
                        setTemplateOpen(true);
                      }
                    : undefined
                }
              />
            ))}
          </div>
        )}

        {online ? (
          error ? (
            <p
              className={cn(
                "text-ui-12 text-muted-foreground",
                localFacts.length > 0 ? "pt-1" : "py-6 text-center",
              )}
            >
              Could not reach Hugging Face for this model. It may be private,
              renamed, or removed.
            </p>
          ) : result ? (
            <>
              <div className="divide-y divide-border/50">
                {facts.map((fact) => (
                  <FactRow key={fact.key} fact={fact} />
                ))}
              </div>
              {/* Href kept so middle-click and copy-link behave, but the click routes through
                the same confirmation gate every other external link in the app uses. */}
              <a
                href={`${hfEndpoint}/${repoId}`}
                target="_blank"
                rel="noopener noreferrer"
                onClick={(event) => {
                  event.stopPropagation();
                  if (confirmExternalLink(`${hfEndpoint}/${repoId}`)) {
                    event.preventDefault();
                  }
                }}
                className="mt-1 inline-flex items-center gap-1.5 text-ui-12 text-muted-foreground transition-colors hover:text-foreground"
              >
                <HugeiconsIcon
                  icon={LinkSquare02Icon}
                  strokeWidth={1.75}
                  className="size-3.5"
                />
                View on Hugging Face
              </a>
            </>
          ) : (
            <div className="flex items-center justify-center gap-2 py-6 text-ui-12 text-muted-foreground">
              <Spinner className="size-4" />
              Loading details…
            </div>
          )
        ) : (
          <p
            className={cn(
              "text-ui-12 text-muted-foreground",
              localFacts.length > 0 ? "pt-1" : "py-6 text-center",
            )}
          >
            {localFacts.length > 0
              ? "Licence and dates need a connection."
              : "Connect to the internet to look up model details."}
          </p>
        )}
      </DialogContent>

      {/* Already in hand from the header read, so viewing it costs no fetch. Read-only:
          this panel reports on the model, it does not configure it. */}
      {template && (
        <ChatTemplateEditorDialog
          open={templateOpen}
          onOpenChange={setTemplateOpen}
          onCloseAutoFocus={(event) => {
            event.preventDefault();
            templateTriggerRef.current?.focus();
          }}
          value={null}
          defaultTemplate={template}
          defaultLoading={false}
          readOnly={true}
          description={`The chat template embedded in ${repoId}.`}
          onSave={() => {}}
        />
      )}
    </Dialog>
  );
}
