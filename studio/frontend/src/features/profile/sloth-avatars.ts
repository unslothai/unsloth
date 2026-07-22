// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Curated list of sloth emoji stickers offered as profile pictures.
//
// The full `public/Sloth emojis` folder has ~38 PNGs, but many are non-square
// or carry heavy whitespace on one or more edges, which crops badly inside the
// round avatar frame. This list is the subset that is (a) effectively square
// (aspect ratio within ~10% of 1:1) and (b) low-whitespace on every edge, so
// each one fills the avatar circle cleanly. Exact duplicates are de-duped.
//
// Paths are relative to the public folder; resolve with `publicAssetUrl(...)`
// before using as an <img> src so spaces and subpath deploys are handled.
export const SLOTH_AVATARS: readonly string[] = [
  "Sloth emojis/large sloth yay.png",
  "Sloth emojis/large sloth heart.png",
  "Sloth emojis/large sloth wave.png",
  "Sloth emojis/large sloth thumbs.png",
  "Sloth emojis/large sloth cheeky.png",
  "Sloth emojis/large sloth glasses.png",
  "Sloth emojis/large sloth fire.png",
  "Sloth emojis/large sloth drink.png",
  "Sloth emojis/large sloth sad.png",
  "Sloth emojis/Large sloth Question mark.png",
  "Sloth emojis/sloth shy large.png",
  "Sloth emojis/sloth shock large.png",
  "Sloth emojis/sloth sir large.png",
  "Sloth emojis/sloth huglove large.png",
  "Sloth emojis/sloth headphones.png",
  "Sloth emojis/sloth pc square.png",
  "Sloth emojis/sloth on phone.png",
  "Sloth emojis/sloth magnify final.png",
  "Sloth emojis/Sloth loca pc.png",
  "Sloth emojis/UnSloth GPU Front square.png",
  "Sloth emojis/UnSloth Sparkling large.png",
];
