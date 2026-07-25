# Changelog

Release notes for Unsloth and Unsloth Studio.

Unsloth Studio reads this file to show release notes inside the "New Unsloth
version" update popup. Edit it here and the popup picks the change up on the
next update check, with no release or rebuild required.

## Format

Every release is a level-2 heading whose first token is the version, optionally
followed by a date:

```md
## 2026.7.6 - 2026-07-22
```

`## [2026.7.6] - 2026-07-22` and `## v2026.7.6` also work. Everything under a
heading, up to the next level-2 heading, is that release's notes and renders as
Markdown in the popup.

Notes are matched to one exact version. When Studio offers an update to
`2026.7.6` it renders the `2026.7.6` section and nothing else. If that section
is missing, the popup links out to the online changelog rather than showing
notes from an unrelated release, so a new version needs its own section here
before its notes can appear.

Keep the newest release at the top. Lead each bullet with the change itself:
the collapsed popup highlights the first sentence and dims the rest.
`## Unreleased` is ignored by the popup, so it is safe to stage notes there and
rename the heading at release time.

<!-- Add new releases directly below this line. -->

## Unreleased

## 2026.7.5

### What's Changed

- The update popup previews release notes inline. The collapsed popup lists the
  top changes, and "Show release notes" expands the full notes in a scrollable
  panel without leaving Studio.
- Release notes come from `CHANGELOG.md` in the Unsloth repository, matched to
  the exact version being offered, so the popup never shows notes from an
  older, unrelated release.
