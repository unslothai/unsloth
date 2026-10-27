// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Spinner } from "@/components/ui/spinner";
import { ChevronDownStandardIcon } from "@/lib/chevron-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { type ReactElement, useState } from "react";
import { type NotebookEntry, listNotebooks } from "../api";
import { Search01Icon } from "../editor/icon-data";
import {
  downloadNotebook,
  notebookActions,
  visibleNotebooks,
} from "./notebooks";

const SHOWN = 30;

export function NotebooksSection({
  onUseTemplate,
  onImport,
}: {
  onUseTemplate: (exampleId: string) => void;
  onImport: () => void;
}): ReactElement {
  const [open, setOpen] = useState(false);
  const [entries, setEntries] = useState<NotebookEntry[] | null>(null);
  const [failed, setFailed] = useState(false);
  const [query, setQuery] = useState("");

  const toggle = () => {
    setOpen((v) => !v);
    if (entries === null && !failed) {
      listNotebooks()
        .then(setEntries)
        .catch(() => setFailed(true));
    }
  };
  const shown = visibleNotebooks(entries ?? [], query).slice(0, SHOWN);

  return (
    <section className="flex flex-col gap-3" data-testid="flows-notebooks">
      <button
        type="button"
        onClick={toggle}
        className="flex items-center gap-1 self-start font-semibold text-foreground text-sm transition hover:text-foreground/80"
        aria-expanded={open}
      >
        Unsloth notebooks
        <HugeiconsIcon
          icon={ChevronDownStandardIcon}
          className={`size-4 transition-transform ${open ? "rotate-180" : ""}`}
        />
      </button>
      {open && (
        <div className="flex flex-col gap-2">
          <p className="text-muted-foreground text-xs">
            Fine-tuning notebooks from unslothai/notebooks. Use one as a flow,
            or download the .ipynb and import it: Studio reads its settings and
            never runs its code.
          </p>
          <div className="relative">
            <HugeiconsIcon
              icon={Search01Icon}
              size={14}
              className="pointer-events-none absolute top-1/2 left-3 -translate-y-1/2 text-muted-foreground"
            />
            <Input
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Search notebooks"
              className="h-8 pl-8"
            />
          </div>
          {failed ? (
            <p className="text-muted-foreground text-sm">
              Could not load the notebook list.
            </p>
          ) : entries === null ? (
            <p className="flex items-center gap-2 text-muted-foreground text-sm">
              <Spinner /> Loading…
            </p>
          ) : (
            <ul className="divide-y divide-border/60 rounded-lg border border-border/60 bg-card">
              {shown.map((entry) => {
                const actions = notebookActions(entry);
                return (
                  <li
                    key={entry.id}
                    className="flex flex-wrap items-center gap-2 px-3 py-2"
                  >
                    <span className="flex-1 truncate text-foreground text-sm">
                      {entry.title}
                    </span>
                    <Badge variant="outline" className="text-muted-foreground">
                      {actions.importable ? entry.category : "Colab only"}
                    </Badge>
                    {actions.template ? (
                      <Button
                        size="sm"
                        onClick={() =>
                          actions.template && onUseTemplate(actions.template)
                        }
                      >
                        Use as flow
                      </Button>
                    ) : (
                      actions.importable && (
                        <>
                          <Button size="sm" variant="outline" asChild={true}>
                            <a
                              href={actions.download}
                              target="_blank"
                              rel="noreferrer"
                              onClick={(event) => {
                                event.preventDefault();
                                void downloadNotebook(
                                  actions.download,
                                  entry.notebook_file,
                                );
                              }}
                            >
                              Download .ipynb
                            </a>
                          </Button>
                          <Button
                            size="sm"
                            variant="outline"
                            onClick={onImport}
                          >
                            Import
                          </Button>
                        </>
                      )
                    )}
                    <Button size="sm" variant="ghost" asChild={true}>
                      <a href={actions.colab} target="_blank" rel="noreferrer">
                        Open in Colab
                      </a>
                    </Button>
                  </li>
                );
              })}
            </ul>
          )}
        </div>
      )}
    </section>
  );
}
