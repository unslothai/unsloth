// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// "Model info" for a picker row (issue #11017): what a model is, and whether its licence
// actually lets you use it, without leaving the chat for the Hub catalog.
//
// Rendering only. Which rows are worth showing is decided by `modelInfoFacts`, which is pure
// and covered by tests/picker-model-info-facts.test.ts.

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
import type { LicenseOpenness } from "./license-openness";
import {
  type ModelInfoFact,
  metaFromHfResult,
  modelInfoFacts,
} from "./model-info-facts";

// Green for "you may use this", amber for "there is a condition", red for "you may not".
// Amber rather than green for the community licences is the whole point of the panel.
const OPENNESS_TONE: Record<LicenseOpenness, string> = {
  open: "border-emerald-500/30 bg-emerald-500/8 text-emerald-700 dark:text-emerald-300",
  restricted:
    "border-amber-500/30 bg-amber-500/8 text-amber-700 dark:text-amber-300",
  proprietary: "border-red-500/30 bg-red-500/8 text-red-700 dark:text-red-300",
  unknown: "border-border/60 bg-muted/40 text-muted-foreground",
};

const OPENNESS_LABEL: Record<LicenseOpenness, string> = {
  open: "Open source",
  restricted: "Open weights, with conditions",
  proprietary: "Not redistributable",
  unknown: "Licence unclear",
};

function FactRow({ fact }: { fact: ModelInfoFact }) {
  const value =
    fact.key === "license" && fact.openness ? (
      <span
        className={cn(
          "inline-flex h-6 shrink-0 items-center rounded-full border px-2 text-ui-11 font-medium leading-none",
          OPENNESS_TONE[fact.openness],
        )}
      >
        {fact.value}
      </span>
    ) : (
      <span className="text-ui-12 text-foreground">{fact.value}</span>
    );

  const row = (
    <div className="flex items-baseline justify-between gap-3 py-1.5">
      <span className="shrink-0 text-ui-12 text-muted-foreground">
        {fact.label}
      </span>
      {value}
    </div>
  );

  if (!fact.detail) return row;
  return (
    <Tooltip delayDuration={0}>
      <TooltipTrigger asChild={true}>{row}</TooltipTrigger>
      <TooltipContent side="left" className="max-w-[280px]">
        {fact.detail}
      </TooltipContent>
    </Tooltip>
  );
}

export function ModelInfoDialog({
  repoId,
  open,
  onOpenChange,
}: {
  repoId: string;
  open: boolean;
  onOpenChange: (open: boolean) => void;
}) {
  const online = useOnlineStatus();
  const hfToken = useHfTokenStore((s) => s.token);
  const hfEndpoint = useHfEndpoint();

  // Only fetches while the dialog is open, and shares one in-flight request per repo with
  // the Hub through hf-cache, so opening this for a model the catalog already showed is free.
  const { result, error } = useSelectedModelMetadata(open ? repoId : null, {
    accessToken: hfToken || undefined,
    enabled: open,
    online,
  });

  const meta = metaFromHfResult(result);
  const facts = meta ? modelInfoFacts(meta) : [];
  const license = facts.find((f) => f.key === "license");

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-[460px]">
        <DialogHeader>
          <DialogTitle className="truncate">{repoId}</DialogTitle>
          <DialogDescription>
            {license?.openness
              ? OPENNESS_LABEL[license.openness]
              : "Details from Hugging Face."}
          </DialogDescription>
        </DialogHeader>

        {/* Offline is not an error: the panel simply has nothing to report, and saying so is
            more useful than a failure the user cannot act on. */}
        {online ? (
          error ? (
            <p className="py-6 text-center text-ui-12 text-muted-foreground">
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
              {license?.detail && (
                <p className="pt-1 text-ui-11 leading-relaxed text-muted-foreground">
                  {license.detail}
                </p>
              )}
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
          <p className="py-6 text-center text-ui-12 text-muted-foreground">
            Connect to the internet to look up model details.
          </p>
        )}
      </DialogContent>
    </Dialog>
  );
}
