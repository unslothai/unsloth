// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import { Copy01Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useEffect, useRef, useState, type ReactElement } from "react";
import { copyToClipboard } from "@/lib/copy-to-clipboard";

export function VideoGenerationApiDocs({
  baseUrl,
}: {
  baseUrl: string;
}): ReactElement {
  const [copied, setCopied] = useState(false);
  const timerRef = useRef<number | undefined>(undefined);
  const authorization = ["Authorization:", "Bearer", "$UNSLOTH_API_KEY"].join(
    " ",
  );
  const example = [
    `API_BASE="${baseUrl}"`,
    'export UNSLOTH_API_KEY="sk-unsloth-YOUR_KEY"',
    "",
    'curl -sS -X POST "$API_BASE/videos" \\',
    `  -H "${authorization}" \\`,
    '  -H "Content-Type: application/json" \\',
    `  -d '{"prompt":"A slow cinematic shot of a red fox in falling snow","seconds":"4"}'`,
    "",
    '# Poll GET "$API_BASE/videos/<video_id>" until status is "completed".',
    'curl -sS "$API_BASE/videos/<video_id>" \\',
    `  -H "${authorization}"`,
    "",
    "# Download the completed MP4.",
    'curl -sS "$API_BASE/videos/<video_id>/content" \\',
    `  -H "${authorization}" \\`,
    "  -o generated.mp4",
  ].join("\n");

  useEffect(
    () => () => {
      if (timerRef.current !== undefined) window.clearTimeout(timerRef.current);
    },
    [],
  );

  const copyExample = async () => {
    if (!(await copyToClipboard(example))) return;
    setCopied(true);
    if (timerRef.current !== undefined) window.clearTimeout(timerRef.current);
    timerRef.current = window.setTimeout(() => setCopied(false), 1800);
  };

  return (
    <details className="overflow-hidden rounded-xl border border-border/60 bg-card">
      <summary className="flex cursor-pointer flex-wrap items-center gap-x-3 gap-y-1 px-4 py-3 text-sm font-semibold text-foreground marker:text-muted-foreground">
        Video generation API
        <span className="font-mono text-ui-11 font-normal text-muted-foreground">
          POST /v1/videos
        </span>
      </summary>
      <div className="border-t border-border/60 px-4 py-4">
        <p className="mb-3 text-sm text-muted-foreground">
          Create a video asynchronously, poll its job, then download the MP4.
          Authenticate with a Studio API key in the Authorization header and
          load a video model first.
          <code className="ml-1 font-mono text-ui-11 text-foreground">
            model
          </code>{" "}
          is optional when the desired model is already active.
        </p>
        <div className="relative overflow-hidden rounded-lg border border-border/60 bg-muted/30">
          <Button
            type="button"
            variant="outline"
            size="sm"
            onClick={() => void copyExample()}
            className="absolute right-2 top-2 z-10 h-7 gap-1.5 bg-background px-2 text-ui-11"
            aria-label={copied ? "Copied example" : "Copy video API example"}
          >
            <HugeiconsIcon icon={Copy01Icon} className="size-3.5" />
            {copied ? "Copied" : "Copy"}
          </Button>
          <pre className="overflow-x-auto whitespace-pre-wrap break-words p-3 pr-20 font-mono text-ui-11 leading-relaxed text-foreground">
            <code>{example}</code>
          </pre>
        </div>
        <dl className="mt-4 grid gap-x-6 gap-y-3 text-ui-11 sm:grid-cols-2">
          <div>
            <dt className="font-mono font-medium text-foreground">
              POST /v1/videos
            </dt>
            <dd className="mt-1 text-muted-foreground">
              JSON fields: prompt (required), model, seconds, and size
              (&quot;widthxheight&quot;). Omit seconds and size to use the
              loaded model&apos;s defaults. An optional input_reference accepts
              an image upload or base64 data URL.
            </dd>
          </div>
          <div>
            <dt className="font-mono font-medium text-foreground">
              {"GET /v1/videos/{video_id}"}
            </dt>
            <dd className="mt-1 text-muted-foreground">
              Poll the returned job id. Status is queued, in_progress,
              completed, or failed.
            </dd>
          </div>
          <div>
            <dt className="font-mono font-medium text-foreground">
              {"GET /v1/videos/{video_id}/content"}
            </dt>
            <dd className="mt-1 text-muted-foreground">
              Download the completed video as an MP4.
            </dd>
          </div>
          <div>
            <dt className="font-mono font-medium text-foreground">
              {"GET /v1/videos · DELETE /v1/videos/{video_id}"}
            </dt>
            <dd className="mt-1 text-muted-foreground">
              List jobs or delete a job and its generated clip.
            </dd>
          </div>
        </dl>
      </div>
    </details>
  );
}
