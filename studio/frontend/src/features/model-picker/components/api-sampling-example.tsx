// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { createCodePlugin } from "@/components/assistant-ui/code-plugin";
import {
  unslothDarkTheme,
  unslothLightTheme,
} from "@/components/assistant-ui/code-themes";
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
  useMemo,
  useRef,
  useState,
} from "react";
import { Streamdown } from "streamdown";
import {
  type ApiSamplingDefaults,
  apiExampleModelName,
  buildApiSamplingExample,
} from "../model-config/api-sampling-example";

// The Settings usage examples' highlighter (usage-examples.tsx HighlightedCode), so both read alike.
const SHIKI_THEMES = [unslothLightTheme, unslothDarkTheme] as [
  typeof unslothLightTheme,
  typeof unslothDarkTheme,
];
const codePlugin = createCodePlugin({ themes: SHIKI_THEMES });

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
  const markdown = useMemo(() => `\`\`\`python\n${snippet}\n\`\`\``, [snippet]);

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
        <div className="max-h-56 overflow-auto scroll-rounded rounded-lg bg-muted/40 py-2.5 pr-10 pl-3 text-ui-11 leading-[1.6] dark:bg-[rgb(255_255_255_/_calc(0.05*var(--contrast-wash-gain,1)))] [&_[data-streamdown=code-block-header]]:hidden [&_[data-streamdown=code-block]]:!my-0 [&_[data-streamdown=code-block]]:!border-0 [&_[data-streamdown=code-block]]:!bg-transparent [&_[data-streamdown=code-block]]:!p-0 [&_code]:!text-ui-11 [&_pre]:!m-0 [&_pre]:!border-0 [&_pre]:!bg-transparent [&_pre]:!p-0 [&_pre]:!text-ui-11 [&_pre]:!leading-[1.6]">
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
