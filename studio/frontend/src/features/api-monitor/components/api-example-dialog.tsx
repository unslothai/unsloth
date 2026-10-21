// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { createCodePlugin } from "@/components/assistant-ui/code-plugin";
import {
  unslothDarkTheme,
  unslothLightTheme,
} from "@/components/assistant-ui/code-themes";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { authFetch } from "@/features/auth";
import { copyToClipboard } from "@/lib/copy-to-clipboard";
import { Tick02Icon } from "@/lib/tick-icon";
import { Copy01Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { type ReactElement, useEffect, useMemo, useRef, useState } from "react";
import { Streamdown } from "streamdown";
import {
  type ApiSamplingDefaults,
  buildApiSamplingExample,
} from "../api-sampling-example";

// The Settings usage examples' highlighter (usage-examples.tsx HighlightedCode), so both read alike.
const SHIKI_THEMES = [unslothLightTheme, unslothDarkTheme] as [
  typeof unslothLightTheme,
  typeof unslothDarkTheme,
];
const codePlugin = createCodePlugin({ themes: SHIKI_THEMES });

/** A request for one served model, with the sampling the server fills in for omitted fields. */
export function ApiExampleDialog({
  open,
  onOpenChange,
  baseUrl,
  model,
  samplingModelId,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  baseUrl: string;
  /** What the client sends as `model`, quant included. */
  model: string;
  /** The checkpoint the server resolves sampling for. */
  samplingModelId: string;
}): ReactElement {
  const [sampling, setSampling] = useState<ApiSamplingDefaults | null>(null);
  const [copied, setCopied] = useState(false);
  const timer = useRef<number | undefined>(undefined);

  useEffect(() => {
    if (!open) return;
    let cancelled = false;
    setSampling(null);
    authFetch(
      `/api/inference/sampling-defaults?model=${encodeURIComponent(samplingModelId)}`,
    )
      .then((res) => (res.ok ? res.json() : {}))
      .then((body: ApiSamplingDefaults) => {
        if (!cancelled) setSampling(body);
      })
      // An older backend has no route: the example still runs, it just carries no values.
      .catch(() => {
        if (!cancelled) setSampling({});
      });
    return () => {
      cancelled = true;
    };
  }, [open, samplingModelId]);

  useEffect(() => () => window.clearTimeout(timer.current), []);

  const snippet = buildApiSamplingExample({
    baseUrl,
    model,
    sampling: sampling ?? {},
  });
  const markdown = useMemo(() => `\`\`\`python\n${snippet}\n\`\`\``, [snippet]);

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="api-rounding corner-squircle dialog-soft-surface gap-5 sm:max-w-3xl">
        <DialogHeader>
          <DialogTitle>API example</DialogTitle>
          <DialogDescription>
            A request to <span className="font-mono">{model}</span> with the
            sampling it gets when a request leaves these out. Send your own
            values to override them.
          </DialogDescription>
        </DialogHeader>
        <div className="relative min-w-0">
          <div className="corner-squircle max-h-[60dvh] overflow-auto scroll-rounded rounded-xl bg-[color-mix(in_oklab,var(--foreground)_calc(5%*var(--contrast-wash-gain,1)),transparent)] py-3 pr-11 pl-3 text-ui-12 leading-[1.6] dark:bg-[rgb(255_255_255_/_calc(0.05*var(--contrast-wash-gain,1)))] [&_[data-streamdown=code-block-header]]:hidden [&_[data-streamdown=code-block]]:!my-0 [&_[data-streamdown=code-block]]:!border-0 [&_[data-streamdown=code-block]]:!bg-transparent [&_[data-streamdown=code-block]]:!p-0 [&_code]:!text-ui-12 [&_pre]:!m-0 [&_pre]:!border-0 [&_pre]:!bg-transparent [&_pre]:!p-0 [&_pre]:!text-ui-12 [&_pre]:!leading-[1.6]">
            {sampling ? (
              <Streamdown
                mode="static"
                plugins={{ code: codePlugin }}
                controls={{ code: false }}
                shikiTheme={SHIKI_THEMES}
              >
                {markdown}
              </Streamdown>
            ) : (
              <span className="font-mono text-muted-foreground">Loading…</span>
            )}
          </div>
          <Button
            type="button"
            variant="ghost"
            size="sm"
            aria-label="Copy example request"
            title={copied ? "Copied" : "Copy example request"}
            disabled={!sampling}
            onClick={async () => {
              if (await copyToClipboard(snippet)) {
                setCopied(true);
                window.clearTimeout(timer.current);
                timer.current = window.setTimeout(() => setCopied(false), 1800);
              }
            }}
            className="absolute top-2 right-2 size-7 rounded-full p-0"
          >
            <HugeiconsIcon
              icon={copied ? Tick02Icon : Copy01Icon}
              strokeWidth={1.75}
              className="size-3.5"
            />
          </Button>
        </div>
      </DialogContent>
    </Dialog>
  );
}
