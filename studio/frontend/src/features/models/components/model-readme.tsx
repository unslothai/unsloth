// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Spinner } from "@/components/ui/spinner";
import { cn } from "@/lib/utils";
import { useHfTokenStore } from "@/stores/hf-token-store";
import { useEffect, useMemo, useState } from "react";
import type { ComponentProps } from "react";
import { Streamdown } from "streamdown";
import {
  createReadmeUrlTransform,
  fetchReadme,
  readmeBaseUrl,
  stripChromeHeadings,
  stripFrontmatter,
} from "../lib/hf-readme";

type ReadmePlugins = NonNullable<ComponentProps<typeof Streamdown>["plugins"]>;

const README_ALLOWED_TAGS: NonNullable<
  ComponentProps<typeof Streamdown>["allowedTags"]
> = {
  audio: ["src", "controls", "preload", "loop", "muted"],
  video: [
    "src",
    "controls",
    "preload",
    "loop",
    "muted",
    "poster",
    "width",
    "height",
    "playsinline",
  ],
  source: ["src", "type", "media"],
  track: ["src", "kind", "srclang", "label", "default"],
};

const READMEX_NEEDS_MATH = /\$\$|\\\(|\\\[/;
const READMEX_NEEDS_MERMAID = /```mermaid\b/;

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
  "[&_td]:px-2 [&_td]:py-1.5 [&_td]:border-b [&_td]:border-border/40",
  "[&_img]:rounded-[10px] [&_img]:my-2 [&_img]:max-w-full",
  "[&_audio]:my-1 [&_audio]:max-w-full [&_audio]:h-9 [&_audio]:align-middle",
  "[&_video]:my-2 [&_video]:max-w-full [&_video]:rounded-[10px]",
  "[&_hr]:my-4 [&_hr]:border-border/40",
);

async function loadPlugins(needs: {
  math: boolean;
  mermaid: boolean;
}): Promise<ReadmePlugins> {
  const [codeModule, mathModule, mermaidModule] = await Promise.all([
    import("@streamdown/code"),
    needs.math ? import("@streamdown/math") : Promise.resolve(null),
    needs.mermaid ? import("@streamdown/mermaid") : Promise.resolve(null),
  ]);
  const plugins: ReadmePlugins = { code: codeModule.code };
  if (mathModule) plugins.math = mathModule.math;
  if (mermaidModule) plugins.mermaid = mermaidModule.mermaid;
  return plugins;
}

export function ModelReadme({
  repoId,
  kind = "model",
}: {
  repoId: string;
  kind?: "model" | "dataset";
}) {
  const hfToken = useHfTokenStore((s) => s.token);
  const [body, setBody] = useState<string | null>(null);
  const [baseUrl, setBaseUrl] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [plugins, setPlugins] = useState<ReadmePlugins | null>(null);

  const urlTransform = useMemo(
    () => (baseUrl ? createReadmeUrlTransform(baseUrl) : undefined),
    [baseUrl],
  );

  useEffect(() => {
    let canceled = false;
    setLoading(true);
    setError(null);
    setBody(null);
    setBaseUrl(null);
    setPlugins(null);
    fetchReadme(repoId, kind, hfToken || null)
      .then((fetched) => {
        if (canceled) return;
        if (fetched == null) {
          setError("No model card available.");
          return;
        }
        const { body } = stripFrontmatter(fetched.markdown);
        const cleaned = stripChromeHeadings(body).trim();
        setBaseUrl(readmeBaseUrl(repoId, kind, fetched.branch));
        setBody(cleaned);

        void loadPlugins({
          math: READMEX_NEEDS_MATH.test(cleaned),
          mermaid: READMEX_NEEDS_MERMAID.test(cleaned),
        }).then((next) => {
          if (!canceled) setPlugins(next);
        });
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
  }, [repoId, kind, hfToken]);

  if (loading) {
    return (
      <div className="flex items-center gap-2 text-[12.5px] text-muted-foreground">
        <Spinner className="size-3.5" />
        Loading {kind === "dataset" ? "dataset" : "model"} card…
      </div>
    );
  }

  if (error) {
    return (
      <p className="text-[12.5px] text-muted-foreground">{error}</p>
    );
  }

  if (!body) {
    return (
      <p className="text-[12.5px] text-muted-foreground">
        This repository has no README.
      </p>
    );
  }

  return (
    <div className={PROSE}>
      <Streamdown
        mode="static"
        plugins={plugins ?? undefined}
        controls={false}
        allowedTags={README_ALLOWED_TAGS}
        urlTransform={urlTransform}
      >
        {body}
      </Streamdown>
    </div>
  );
}
