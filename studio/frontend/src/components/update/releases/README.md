# Release notes for What's New

One `notes.json` per release tag, newest first in `index.ts`. Draft one from the published
GitHub release, then edit it:

```bash
node scripts/draft-release-notes.mjs v0.1.923-beta
```

## Fields

- `title`: what the release is about, in a few words ("Translate", "Computer use and a new model picker").
- `date`: the publish date, `YYYY-MM-DD` (UTC).
- `features`: one or two cards with artwork. Each has `image`, `alt`, `title` and a one-sentence `summary`.
  Artwork lives in `public/whats-new/<tag>/` as a 1200×500 WebP (12:5), so it also loads offline.
- `changes`: up to four `[label, sentence]` rows under "Also in this release".
- `more`: how many other changes the release page lists. The dialog links it as "Plus N more changes".

## What makes the cut

- Something users can do now that they could not before, a limit that is gone, or real time saved
  on a common task. A new setting or one more edge case is not enough.
- A fix earns a row only when it cost people their work or blocked a core flow.
- Fewer is fine. Stop when the qualifying changes run out.

## How to write it

- A short label and one sentence of at most 20 words. Say what the user can now do.
- Use the names people see in the app. Leave out protocols, PR numbers and internals.
- Mark beta features "(beta)". No hype, no exclamation marks, no em dashes.
- Artwork: flat backgrounds, no mascots. Zoom a real Studio screen on the one element that matters,
  or set the model name and two or three facts as a card.
