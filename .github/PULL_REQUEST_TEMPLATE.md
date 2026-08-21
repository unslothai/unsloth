## What this changes

<!-- One or two sentences. What moves, and why. -->

## Motivation

<!-- Link the issue if there is one. If this is a performance change, say what the cost was
     and where you measured it. -->

## Testing

<!-- What you ran, and what it said. Numbers rather than adjectives where you have them. -->

---

### If this touches the Studio UI

Studio changes are expected to leave the interface behaving the same way it did before.
Tick whichever applies, and delete the rest.

- [ ] No user-visible change. The UI behaves identically.
- [ ] A deliberate UI difference, justified by a dramatic performance improvement. The
      difference and its cost to the user are described above, with the measurement.
- [ ] The difference exists only off screen. Everything inside the viewport is unchanged.

If you ticked the third box, confirm that selection, clipboard, native find-in-page,
printing and scroll geometry all still cover the whole conversation rather than the visible
window. Those are whole-document operations and the off-screen exemption does not reach
them.

See [CONTRIBUTING.md](../CONTRIBUTING.md) for the full rule.
