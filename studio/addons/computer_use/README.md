# unsloth-computer-use

The optional computer use add-on for [Unsloth](https://github.com/unslothai/unsloth): it lets a
model see this computer's screen and use its mouse and keyboard.

It is not part of the `unsloth` package. Unsloth installs it into its own folder when the owner
presses **Install** under Settings > Computer use, and removes it again from the same page. It
does nothing on its own: Unsloth loads its modules, and every action goes through Unsloth's
approval and app rules.

## Releasing

Unsloth installs the exact version named by `ADDON_VERSION` in
`studio/backend/core/inference/computer_use/__init__.py`, so publish this package before the
Unsloth release that names it:

1. Raise `version` in `pyproject.toml` and `ADDON_VERSION` together.
2. Build and upload the wheel from this folder: `python -m build --wheel`, then `twine upload dist/*`.
3. Put the wheel's sha256 in `ADDON_SHA256` in
   `studio/backend/core/inference/computer_use/addon.py`. Unsloth downloads nothing while it is
   empty, and refuses a wheel that does not match it.

A source checkout copies this folder instead of downloading.
