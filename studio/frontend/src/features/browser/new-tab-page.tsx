// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useT } from "@/i18n";
import { Search01Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useState } from "react";
import { SEARCH_ENGINES, hostOf, resolveAddress } from "./address";
import { useBrowserPrefsStore } from "./prefs-store";
import { useBrowserStore } from "./store";

const SUGGESTED = [
  { title: "Unsloth", url: "https://unsloth.ai" },
  { title: "Unsloth Docs", url: "https://docs.unsloth.ai" },
  { title: "Unsloth on GitHub", url: "https://github.com/unslothai/unsloth" },
  { title: "Hugging Face", url: "https://huggingface.co/unsloth" },
  { title: "Wikipedia", url: "https://en.wikipedia.org" },
  { title: "Hacker News", url: "https://news.ycombinator.com" },
];

function SiteIcon({ url }: { url: string }) {
  const [failed, setFailed] = useState(false);
  const host = hostOf(url);
  if (failed) {
    return (
      <span className="flex size-6 items-center justify-center rounded-md bg-muted text-xs font-semibold uppercase text-muted-foreground">
        {host.charAt(0)}
      </span>
    );
  }
  return (
    <img
      src={`${new URL(url).origin}/favicon.ico`}
      alt=""
      referrerPolicy="no-referrer"
      onError={() => setFailed(true)}
      className="size-6 rounded-md object-contain"
    />
  );
}

export function NewTabPage({ tabId }: { tabId: string }) {
  const t = useT();
  const engine = useBrowserPrefsStore((state) => state.searchEngine);
  const navigate = useBrowserStore((state) => state.navigate);
  const [query, setQuery] = useState("");

  const go = (url: string | null) => {
    if (url) navigate(tabId, { url });
  };

  return (
    <div className="size-full overflow-auto">
      <div className="mx-auto flex w-full max-w-xl flex-col gap-8 px-6 pb-10 pt-[12vh]">
        <form
          onSubmit={(event) => {
            event.preventDefault();
            go(resolveAddress(query, engine));
          }}
          className="flex h-12 items-center gap-2.5 rounded-full border border-border/80 bg-background px-4 shadow-xs focus-within:border-ring focus-within:ring-2 focus-within:ring-ring/20"
        >
          <HugeiconsIcon icon={Search01Icon} strokeWidth={1.75} className="size-4.5 shrink-0 text-muted-foreground" />
          <input
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder={t("browser.searchPlaceholder", { engine: SEARCH_ENGINES[engine].label })}
            aria-label={t("browser.searchPlaceholder", { engine: SEARCH_ENGINES[engine].label })}
            spellCheck={false}
            className="min-w-0 flex-1 bg-transparent text-ui-15 outline-none placeholder:text-muted-foreground"
          />
        </form>
        <section className="flex flex-col gap-3">
          <h2 className="px-1 text-ui-13 font-medium text-muted-foreground">{t("browser.suggested")}</h2>
          <div className="grid grid-cols-2 gap-2 sm:grid-cols-3">
            {SUGGESTED.map((site) => (
              <button
                key={site.url}
                type="button"
                onClick={() => go(site.url)}
                className="flex min-w-0 cursor-pointer items-center gap-2.5 rounded-xl border border-border/60 bg-card px-3 py-2.5 text-start transition-colors hover:bg-muted/60 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
              >
                <SiteIcon url={site.url} />
                <span className="min-w-0 flex-1">
                  <span className="block truncate text-ui-13p5 font-medium text-foreground">{site.title}</span>
                  <span className="block truncate text-ui-12 text-muted-foreground">{hostOf(site.url)}</span>
                </span>
              </button>
            ))}
          </div>
        </section>
      </div>
    </div>
  );
}
