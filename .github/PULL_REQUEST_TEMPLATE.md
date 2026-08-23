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

**Whichever box you ticked, run the parity suite and paste its verdict**, including the
number of action pairs compared and the concurrent null control's score on the same run. A
parity result without its null beside it is not evidence, because it gives no floor to judge
a difference against.

```
parity verdict:
pairs compared:
null control:
```

Two things the verdict does not tell you on its own, so say which one you are claiming:

- The structural digest covers thread structure. It is blind to the sidebar and to computed
  layout, and it never reads geometry or custom properties. A change confined to either will
  pass it while being invisible to it.
- Visible-region parity is the check that supports the off-screen exemption. It is the one to
  use for a windowed or deferred change, since a whole-document digest will report
  differences by design and prove nothing.

See [CONTRIBUTING.md](https://github.com/unslothai/unsloth/blob/main/CONTRIBUTING.md) for the
full rule.
