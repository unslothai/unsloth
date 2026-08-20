


import { MarkdownPreview } from "@/components/markdown/markdown-preview";
import { useReleaseNotes } from "@/hooks/use-release-notes";
import { resolveReleaseBodyLinks } from "@/lib/release-body-links";
import { releaseNotesPreview } from "@/lib/release-notes-preview";
import { cn } from "@/lib/utils";
import {
  type ReactElement,
  type ReactNode,
  useEffect,
  useMemo,
  useRef,
} from "react";

interface ReleaseNotesPanelProps {
  // Version being offered. Carried through the lookup, not looked up by.
  version: string;
  // Collapsed previews the top bullets; expanded scrolls the full notes.
  open: boolean;
  releaseNotesUrl?: string | null;
  className?: string;
}

const NOTES_LINK_CLASS =
  "shrink-0 whitespace-nowrap text-ui-11 font-medium text-foreground underline underline-offset-2";

function NotesMessage({
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

function NotesLink({
  href,
  isRelease,
}: {
  href: string;
  isRelease: boolean;
}): ReactElement {
  return (
    <a
      href={href}
      target="_blank"
      rel="noopener noreferrer"
      className={NOTES_LINK_CLASS}
      data-testid="update-release-notes-link"
    >
      {isRelease ? "Open release" : "Open changelog"}
    </a>
  );
}

export function ReleaseNotesPanel({
  version,
  open,
  releaseNotesUrl = null,
  className,
}: ReleaseNotesPanelProps): ReactElement | null {
  // Fetched with the popup: the collapsed preview needs the notes too.
  const { state, notes, retry } = useReleaseNotes({ version, enabled: true });
  const scrollRef = useRef<HTMLElement | null>(null);

  // No fallback body: the updater's `notes` is latest.json's static download
  // blurb, the same install boilerplate the backend now strips.
  const source = notes?.matched ? notes.markdown : null;
  // Notes target the repository, so relative links must point back at it.
  const markdown = useMemo(
    () => (source === null ? null : resolveReleaseBodyLinks(source)),
    [source],
  );

  // Notes that are only a code block or a table preview as nothing.
  const preview = useMemo(
    () => (markdown === null ? null : releaseNotesPreview(markdown)),
    [markdown],
  );

  // Start at the top on expand, and again once async notes land.
  useEffect(() => {
    if (open && markdown && scrollRef.current) {
      scrollRef.current.scrollTop = 0;
    }
  }, [open, markdown]);

  // The page the notes are on wins, then the caller's URL, then the API's.
  const notesUrl = notes?.htmlUrl ?? releaseNotesUrl ?? notes?.releaseNotesUrl;
  const link = notesUrl ? (
    <NotesLink href={notesUrl} isRelease={notesUrl === notes?.htmlUrl} />
  ) : null;

  // Nothing previewable yet or ever: keep the collapsed popup compact.
  if (
    !open &&
    (!markdown ||
      state === "loading" ||
      state === "idle" ||
      preview?.items.length === 0)
  ) {
    return null;
  }

  return (
    <div
      // Clipped, not just capped: this is the one part of the card allowed to
      // give up height, so its content must not paint over the buttons below.
      className={cn("mt-3 flex min-h-0 flex-col overflow-hidden", className)}
      data-testid="update-release-notes-panel"
      data-notes-state={state}
      data-notes-version={version}
      data-notes-tag={notes?.tag ?? undefined}
      data-notes-open={open}
    >
      {/* borderless fill, lighter than the card in dark mode */}
      <div className="flex min-h-0 flex-col rounded-[14px] bg-muted/40 px-3 py-1 dark:bg-white/[0.06]">
        {markdown ? (
          open ? (
            <section
              ref={scrollRef}
              // biome-ignore lint/a11y/noNoninteractiveTabindex: keyboard-scrollable region
              tabIndex={0}
              aria-label={`Release notes for ${notes?.tag ?? `version ${version}`}`}
              // Long notes scroll here instead of pushing the buttons off screen.
              className="hover-scrollbar max-h-64 min-h-0 flex-1 overflow-y-auto overscroll-contain py-3 pr-1"
              data-testid="update-release-notes-scroll"
            >
              <MarkdownPreview
                markdown={markdown}
                // Streamdown ships headings at mt-6 and code at text-sm, and
                // clears max-width on descendants, so rescale and re-cap both.
                className="max-h-none overflow-visible border-0 bg-transparent p-0 text-ui-11 [&_[data-streamdown=link-safety-modal]>*]:max-w-md [&_img]:h-auto [&_img]:max-w-full [&>*:first-child]:mt-0 [&>*>*:first-child]:mt-0 [&_code]:text-[0.92em] [&_h1]:mt-4 [&_h1]:font-heading [&_h1]:text-ui-13 [&_h2]:mt-4 [&_h2]:font-heading [&_h2]:text-ui-13 [&_h3]:mt-4 [&_h3]:font-heading [&_h3]:text-ui-11 [&_pre]:text-[0.92em]"
              />
              {notes?.truncated ? (
                <p className="mt-2 text-ui-10 text-muted-foreground/80">
                  Notes truncated. See the full release notes.
                </p>
              ) : null}
            </section>
          ) : (
            <ReleaseNotesSummary preview={preview} />
          )
        ) : (
          <NotesStatus
            state={state}
            release={notes?.tag ?? null}
            link={link}
            retry={retry}
          />
        )}
      </div>
      {open && markdown && link ? (
        <div className="mt-2 flex justify-end px-1">{link}</div>
      ) : null}
    </div>
  );
}

/** Collapsed view: the first few bullets, one line each where possible. */
function ReleaseNotesSummary({
  preview,
}: {
  preview: ReturnType<typeof releaseNotesPreview> | null;
}): ReactElement | null {
  if (preview === null || preview.items.length === 0) {
    return null;
  }
  const { items, remaining } = preview;

  return (
    <ul
      // Scrolls like the expanded notes: in a short window this list is
      // taller than its slot, and unscrolled it paints over the buttons.
      // biome-ignore lint/a11y/noNoninteractiveTabindex: keyboard-scrollable region
      tabIndex={0}
      // Unversioned: the notes are the release's, not the offered version's.
      aria-label="Release notes summary"
      className="hover-scrollbar min-h-0 flex-1 space-y-1 overflow-y-auto overscroll-contain py-2 pr-1"
      data-testid="update-release-notes-summary"
    >
      {items.map((item, index) => (
        <li
          // Two releases can carry the same bullet text, so index is the key.
          key={`${index}-${item.lead}`}
          className="flex gap-1.5 text-ui-11 leading-snug text-muted-foreground"
        >
          <span aria-hidden="true" className="text-muted-foreground/60">
            &bull;
          </span>
          <span className="line-clamp-2 min-w-0">
            {/* lead sentence carries the change */}
            <span className="font-medium text-foreground">{item.lead}</span>
            {item.rest ? <span> {item.rest}</span> : null}
          </span>
        </li>
      ))}
      {remaining > 0 ? (
        <li className="pl-3 text-ui-10 text-muted-foreground/70">
          +{remaining} more
        </li>
      ) : null}
    </ul>
  );
}

function NotesStatus({
  state,
  release,
  link,
  retry,
}: {
  state: ReturnType<typeof useReleaseNotes>["state"];
  // Tag of the release that was found, when one was.
  release: string | null;
  link: ReactNode;
  retry: () => void;
}): ReactElement {
  if (state === "loading" || state === "idle") {
    return <NotesMessage>Loading release notes...</NotesMessage>;
  }

  if (state === "error") {
    return (
      <NotesMessage
        action={
          // The release page may be reachable when the lookup is not.
          <span className="flex shrink-0 items-center gap-3">
            <button
              type="button"
              onClick={retry}
              className={NOTES_LINK_CLASS}
              data-testid="update-release-notes-retry"
            >
              Retry
            </button>
            {link}
          </span>
        }
      >
        Could not load release notes.
      </NotesMessage>
    );
  }

  // The release published no notes: link out rather than show the generated list.
  return (
    <NotesMessage action={link}>
      {release
        ? `No release notes published for ${release} yet.`
        : "No release notes published yet."}
    </NotesMessage>
  );
}
