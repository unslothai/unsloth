// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Spinner } from "@/components/ui/spinner";
import { cn } from "@/lib/utils";
import { code } from "@streamdown/code";
import { math } from "@streamdown/math";
import { mermaid } from "@streamdown/mermaid";
import { useEffect, useState } from "react";
import { Streamdown } from "streamdown";
import { fetchReadme, stripFrontmatter } from "../lib/hf-readme";

const PLUGINS = { code, math, mermaid } as const;

const PROSE = cn(
  "max-w-none text-[13.5px] leading-[1.7] text-foreground/85",
  "[&_h1]:text-[18px] [&_h1]:font-semibold [&_h1]:tracking-tight [&_h1]:mt-2 [&_h1]:mb-3",
  "[&_h2]:text-[15.5px] [&_h2]:font-semibold [&_h2]:tracking-tight [&_h2]:mt-5 [&_h2]:mb-2",
  "[&_h3]:text-[14px] [&_h3]:font-semibold [&_h3]:mt-4 [&_h3]:mb-1.5",
  "[&_p]:my-2.5 [&_ul]:my-2 [&_ol]:my-2 [&_li]:my-0.5",
  "[&_a]:text-primary [&_a:hover]:underline",
  "[&_code]:rounded-md [&_code]:bg-muted/60 [&_code]:px-1 [&_code]:py-0.5 [&_code]:text-[12px] [&_code]:font-mono",
  "[&_pre]:rounded-[12px] [&_pre]:border [&_pre]:border-border/60 [&_pre]:bg-muted/40 [&_pre]:p-3 [&_pre]:text-[12px] [&_pre]:overflow-x-auto",
  "[&_pre_code]:bg-transparent [&_pre_code]:p-0",
  "[&_blockquote]:border-l-2 [&_blockquote]:border-border/60 [&_blockquote]:pl-3 [&_blockquote]:text-muted-foreground",
  "[&_table]:my-3 [&_table]:text-[12.5px]",
  "[&_th]:px-2 [&_th]:py-1.5 [&_th]:text-left [&_th]:font-semibold [&_th]:border-b [&_th]:border-border/60",
  "[&_td]:px-2 [&_td]:py-1.5 [&_td]:border-b [&_td]:border-border/60/40",
  "[&_img]:rounded-[10px] [&_img]:my-2 [&_img]:max-w-full",
  "[&_hr]:my-4 [&_hr]:border-border/60/40",
);

export function ModelReadme({
  repoId,
  kind = "model",
}: {
  repoId: string;
  kind?: "model" | "dataset";
}) {
  const [body, setBody] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let canceled = false;
    setLoading(true);
    setError(null);
    setBody(null);
    fetchReadme(repoId, kind)
      .then((markdown) => {
        if (canceled) return;
        if (markdown == null) {
          setError("No model card available.");
          return;
        }
        const { body } = stripFrontmatter(markdown);
        setBody(body.trim());
      })
      .catch((err) => {
        if (canceled) return;
        setError(
          err instanceof Error ? err.message : "Failed to load model card",
        );
      })
      .finally(() => {
        if (!canceled) setLoading(false);
      });
    return () => {
      canceled = true;
    };
  }, [repoId, kind]);

  if (loading) {
    return (
      <div className="flex items-center gap-2 rounded-[14px] border border-border/60 bg-background/80 px-4 py-3 text-[12.5px] text-muted-foreground">
        <Spinner className="size-3.5" />
        Loading model card…
      </div>
    );
  }

  if (error) {
    return (
      <div className="rounded-[14px] border border-border/60 bg-background/80 px-4 py-3 text-[12.5px] text-muted-foreground">
        {error}
      </div>
    );
  }

  if (!body) {
    return (
      <div className="rounded-[14px] border border-border/60 bg-background/80 px-4 py-3 text-[12.5px] text-muted-foreground">
        This repository has no README.
      </div>
    );
  }

  return (
    <div
      className={cn(
        "rounded-[14px] border border-border/60 bg-background/80 px-5 py-4",
        PROSE,
      )}
    >
      <Streamdown mode="static" plugins={PLUGINS} controls={false}>
        {body}
      </Streamdown>
    </div>
  );
}
