// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  UPDATE_NOTES_BULLET_CLASS,
  UPDATE_NOTES_EXPANDED_SCROLL_CLASS,
  UPDATE_NOTES_FOOTER_CLASS,
  UPDATE_NOTES_ITEM_CLASS,
  UPDATE_NOTES_LEAD_CLASS,
  UPDATE_NOTES_LINK_CLASS,
  UPDATE_NOTES_ROOT_CLASS,
  UPDATE_NOTES_SURFACE_CLASS,
} from "@/components/update/update-notes-layout";
import { useLlamaUpdateChangelog } from "@/hooks/use-llama-update-changelog";
import type { ReactElement, ReactNode } from "react";

const LINK_CLASS =
  "font-medium text-foreground underline decoration-foreground/30 underline-offset-2 hover:decoration-foreground/70";

function Message({
  children,
  action,
}: {
  children: ReactNode;
  action?: ReactNode;
}): ReactElement {
  return (
    <div className="flex items-center justify-between gap-2 px-1 py-2">
      <p className="text-ui-11 text-muted-foreground">{children}</p>
      {action}
    </div>
  );
}

export function LlamaUpdateChangelogPanel({
  installedTag,
  latestTag,
}: {
  installedTag: string;
  latestTag: string;
}): ReactElement {
  const { state, changelog, retry } = useLlamaUpdateChangelog({
    enabled: true,
    installedTag,
    latestTag,
  });

  const releaseLink = changelog?.releaseUrl ? (
    <a
      href={changelog.releaseUrl}
      target="_blank"
      rel="noopener noreferrer"
      className={UPDATE_NOTES_LINK_CLASS}
      data-testid="llama-update-release-link"
    >
      Open release
    </a>
  ) : null;

  return (
    <div
      className={UPDATE_NOTES_ROOT_CLASS}
      data-testid="llama-update-changelog-panel"
      data-changelog-state={state}
    >
      <div className={UPDATE_NOTES_SURFACE_CLASS}>
        {state === "loading" || state === "idle" ? (
          <Message>Loading new changes...</Message>
        ) : state === "error" ? (
          <Message
            action={
              <button
                type="button"
                onClick={retry}
                className={`shrink-0 text-ui-11 ${LINK_CLASS}`}
                data-testid="llama-update-changelog-retry"
              >
                Retry
              </button>
            }
          >
            Could not compare these releases.
          </Message>
        ) : changelog && changelog.changes.length > 0 ? (
          <ul
            // biome-ignore lint/a11y/noNoninteractiveTabindex: keyboard-scrollable region
            tabIndex={0}
            aria-label={`New llama.cpp changes from ${installedTag} to ${latestTag}`}
            className={`${UPDATE_NOTES_EXPANDED_SCROLL_CLASS} space-y-1`}
            data-testid="llama-update-changelog-list"
          >
            {changelog.changes.map((change, index) => (
              <li
                key={`${index}-${change.summary}`}
                className={UPDATE_NOTES_ITEM_CLASS}
              >
                <span aria-hidden="true" className={UPDATE_NOTES_BULLET_CLASS}>
                  &bull;
                </span>
                <span className="min-w-0">
                  <span className={UPDATE_NOTES_LEAD_CLASS}>
                    {change.summary}
                  </span>
                  {change.links.length > 0 ? (
                    <span>
                      {" ("}
                      {change.links.map((link, linkIndex) => (
                        <span key={link.url}>
                          {linkIndex > 0 ? ", " : ""}
                          <a
                            href={link.url}
                            target="_blank"
                            rel="noopener noreferrer"
                            className={LINK_CLASS}
                          >
                            {link.label}
                          </a>
                        </span>
                      ))}
                      {")"}
                    </span>
                  ) : null}
                </span>
              </li>
            ))}
            {changelog.truncated ? (
              <li className="pl-3 text-ui-10 text-muted-foreground/70">
                Showing {changelog.changes.length} of {changelog.totalChanges}{" "}
                changes
              </li>
            ) : null}
          </ul>
        ) : (
          <Message>No new carried changes are listed for this build.</Message>
        )}
      </div>
      {state === "ready" && releaseLink ? (
        <div className={UPDATE_NOTES_FOOTER_CLASS}>{releaseLink}</div>
      ) : null}
    </div>
  );
}
