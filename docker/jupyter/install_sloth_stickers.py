#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-Present the Unsloth team. See /studio/LICENSE.AGPL-3.0

"""Install the Unsloth Studio sloth stickers for the JupyterLab login screen.

The branded login page (login.html) shows a different sloth sticker on each
visit, the same curated set Studio offers as profile avatars. The PNGs live in
the Studio frontend (`studio/frontend/public/Sloth emojis/`), which is present
in the studio image after install.sh runs. This copies the curated subset into
jupyter_server's static dir as `sloth/01.png .. sloth/20.png` so the template
can reference stable, space-free, auth-free URLs via `static_url(...)`.

Usage:
    install_sloth_stickers.py --src "<Sloth emojis dir>" --dest "<static>/sloth"

Fail-soft: a missing source file is skipped (login.html's onerror falls back to
the Unsloth logo), and the script still exits 0 as long as at least one sticker
was installed. Stdlib only.
"""

import argparse
import os
import shutil
import sys

# Curated, in display order -> NN.png. Mirrors Studio's SLOTH_AVATARS: the square,
# low-whitespace stickers that frame cleanly. Synced by hand; missing names skipped.
CURATED = [
    "large sloth yay.png",
    "large sloth heart.png",
    "large sloth wave.png",
    "large sloth thumbs.png",
    "large sloth cheeky.png",
    "large sloth glasses.png",
    "large sloth fire.png",
    "large sloth drink.png",
    "large sloth sad.png",
    "Large sloth Question mark.png",
    "sloth shy large.png",
    "sloth shock large.png",
    "sloth sir large.png",
    "sloth huglove large.png",
    "sloth headphones.png",
    "sloth pc square.png",
    "sloth on phone.png",
    "sloth magnify final.png",
    "Sloth loca pc.png",
    "UnSloth GPU Front square.png",
]


def main() -> int:
    parser = argparse.ArgumentParser(description = __doc__)
    parser.add_argument("--src", required = True, help = "Studio 'Sloth emojis' dir")
    parser.add_argument("--dest", required = True, help = "output dir (static/sloth)")
    args = parser.parse_args()

    os.makedirs(args.dest, exist_ok = True)
    installed = 0
    for index, name in enumerate(CURATED, start = 1):
        source = os.path.join(args.src, name)
        target = os.path.join(args.dest, "%02d.png" % index)
        if not os.path.isfile(source):
            print("  skip (missing): %s" % name)
            continue
        try:
            shutil.copyfile(source, target)
            installed += 1
        except OSError as error:
            print("  skip (%s): %s" % (error, name))

    print("installed %d/%d sloth stickers into %s" % (installed, len(CURATED), args.dest))
    # Non-fatal, but an empty copy usually means a wrong --src, so signal it.
    return 0 if installed else 1


if __name__ == "__main__":
    sys.exit(main())
