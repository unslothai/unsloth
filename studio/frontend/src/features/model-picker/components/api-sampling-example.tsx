// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import { authFetch } from "@/features/auth";
import { copyToClipboard } from "@/lib/copy-to-clipboard";
import { Tick02Icon } from "@/lib/tick-icon";
import { Copy01Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import {
  type ReactElement,
  createContext,
  useEffect,
  useRef,
  useState,
} from "react";
import {
  type ApiSamplingDefaults,
  apiExampleModelName,
  buildApiSamplingExample,
} from "../model-config/api-sampling-example";

/** Set by the API page: Run settings opened there also shows an example request. */
export const ApiExampleContext = createContext<{ baseUrl: string } | null>(
  null,
);

export function ApiSamplingExample({
  baseUrl,
  modelId,
  ggufVariant,
}: {
  baseUrl: string;
  modelId: string;
  ggufVariant: string | null;
}): ReactElement {
  const [sampling, setSampling] = useState<ApiSamplingDefaults | null>(null);
  const [copied, setCopied] = useState(false);
  const timer = useRef<number | undefined>(undefined);

  useEffect(() => {
    let cancelled = false;
    setSampling(null);
    authFetch(
      `/api/inference/sampling-defaults?model=${encodeURIComponent(modelId)}`,
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
  }, [modelId]);

  useEffect(() => () => window.clearTimeout(timer.current), []);

  const snippet = buildApiSamplingExample({
    baseUrl,
    model: apiExampleModelName(modelId, ggufVariant),
    sampling: sampling ?? {},
  });

  return (
    <section className="mt-5 flex min-w-0 flex-col gap-2 border-t border-border pt-5">
      <div className="flex flex-col gap-0.5">
        <h3 className="text-ui-13 font-medium text-foreground">API example</h3>
        <p className="text-ui-11 text-muted-foreground">
          The sampling a request gets when it leaves these out. Send your own
          values to override them.
        </p>
      </div>
      <div className="relative min-w-0">
        <pre className="max-h-56 overflow-auto scroll-rounded rounded-lg bg-muted/40 py-2.5 pr-10 pl-3 font-mono text-ui-11 leading-[1.6] text-foreground dark:bg-[rgb(255_255_255_/_calc(0.05*var(--contrast-wash-gain,1)))]">
          {sampling ? snippet : "Loading…"}
        </pre>
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
          className="absolute top-1.5 right-1.5 size-7 rounded-full p-0"
        >
          <HugeiconsIcon
            icon={copied ? Tick02Icon : Copy01Icon}
            strokeWidth={1.75}
            className="size-3.5"
          />
        </Button>
      </div>
    </section>
  );
}
