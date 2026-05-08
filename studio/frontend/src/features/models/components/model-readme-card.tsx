// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { ModelReadme } from "./model-readme";

export function ModelReadmeCard({
  repoId,
  title,
}: {
  repoId: string | null;
  title: string | null;
}) {
  return (
    <section className="corner-squircle rounded-[30px] border border-border/60 bg-card/70 p-5 shadow-[0_18px_38px_-32px_rgba(15,23,42,0.28)]">
      <div className="mb-4 space-y-1">
        <h2 className="text-[18px] font-semibold tracking-tight text-foreground">
          Model card
        </h2>
        <p className="text-[13px] leading-6 text-muted-foreground">
          {title
            ? `Repository notes and usage guidance for ${title}.`
            : "Select a repository-backed model to view its model card."}
        </p>
      </div>

      {repoId ? (
        <ModelReadme repoId={repoId} />
      ) : (
        <div className="corner-squircle rounded-[22px] border border-border/60 bg-background/80 px-4 py-4 text-[13px] leading-6 text-muted-foreground">
          Local folders without a linked Hugging Face repository do not expose a
          remote model card.
        </div>
      )}
    </section>
  );
}
