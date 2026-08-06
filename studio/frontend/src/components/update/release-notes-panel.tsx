


import { MarkdownPreview } from "@/components/markdown/markdown-preview";
import { useReleaseNotes } from "@/hooks/use-release-notes";
import { resolveChangelogLinks } from "@/lib/changelog-links";
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
  // Notes are looked up for this exact version only.
  version: string;
  // Collapsed previews the top bullets; expanded scrolls the full notes.
  open: boolean;
  // Desktop updater's body, used only if CHANGELOG.md has no section here.
  fallbackMarkdown?: string | null;
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

function ChangelogLink({ href }: { href: string }): ReactElement {
  return (
    <a
      href={href}
      target="_blank"
      rel="noopener noreferrer"
      className={NOTES_LINK_CLASS}
      data-testid="update-release-notes-link"
    >
      Open changelog
    </a>
  );
}

export function ReleaseNotesPanel({
  version,
  open,
  fallbackMarkdown = null,
  releaseNotesUrl = null,
  className,
}: ReleaseNotesPanelProps): ReactElement | null {
  // Fetched with the popup: the collapsed preview needs the notes too.
  const { state, notes, retry } = useReleaseNotes({ version, enabled: true });
  const scrollRef = useRef<HTMLElement | null>(null);

  // The fallback stands in for "no section in the changelog", which the hook
  // reports as ready. An error is retryable, and the desktop fallback is the
  // updater's static blurb, so taking it there would hide Retry until cache expiry.
  const source = notes?.matched
    ? notes.markdown
    : state === "error"
      ? null
      : (fallbackMarkdown ?? null);
  // Notes target the repository, so relative links must point back at it.
  const markdown = useMemo(
    () => (source === null ? null : resolveChangelogLinks(source)),
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

  // Caller's URL wins: the API returns only the generic changelog, while the
  // desktop banner passes this version's release page.
  const notesUrl = releaseNotesUrl ?? notes?.releaseNotesUrl;
  const link = notesUrl ? <ChangelogLink href={notesUrl} /> : null;

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
      className={cn("mt-3 flex min-h-0 flex-col", className)}
      data-testid="update-release-notes-panel"
      data-notes-state={state}
      data-notes-version={version}
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
              aria-label={`Release notes for version ${version}`}
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
                  Notes truncated. See the full changelog.
                </p>
              ) : null}
            </section>
          ) : (
            <ReleaseNotesSummary preview={preview} />
          )
        ) : (
          <NotesStatus
            state={state}
            version={version}
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
      className="space-y-1 py-2 pr-1"
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
  version,
  link,
  retry,
}: {
  state: ReturnType<typeof useReleaseNotes>["state"];
  version: string;
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
          // The changelog page may be reachable when the lookup is not.
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

  // Matched nothing: link out rather than show another release's notes.
  return (
    <NotesMessage action={link}>
      No release notes published for {version} yet.
    </NotesMessage>
  );
}
