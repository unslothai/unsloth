# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Calibration prompts for the NVFP4 build passes (GPTQ Hessians and activation-scale baking).

Deliberately DISJOINT from the gate's evaluation suite: a quantisation calibrated on the prompts it
is then scored on measures how well it memorised them. ``GATE_SUITE_PROMPTS`` is a frozen copy of
that suite so a test can assert the disjointness.
"""

from __future__ import annotations

# Copied verbatim from ``prequant_accuracy_gate.PROMPT_SUITE``, not imported: that script is not
# always beside this one, and a test compares the copy.
GATE_SUITE_PROMPTS: tuple = (
    (
        "a photoreal close-up portrait of an older fisherman, weathered skin, sharp "
        "catchlights in the eyes, natural window light, 85mm lens"
    ),
    'a storefront sign that says "UNSLOTH" in bold clean letters, daytime, sharp focus',
    (
        "a close-up of two human hands carefully folding an origami crane, "
        "anatomically correct fingers, soft studio light"
    ),
    (
        "extreme close-up of a woven wicker basket next to coarse tweed fabric and "
        "cracked dry earth, intricate high-frequency detail"
    ),
    (
        "a sweeping cinematic landscape at golden hour showing a winding cobblestone "
        "road that climbs through terraced vineyards toward a distant medieval hilltop "
        "village, warm amber light raking across the stone walls, a lone cyclist in a red "
        "jacket pedaling uphill, cypress trees casting long shadows, scattered wildflowers "
        "in the foreground, soft volumetric haze in the valley below, ultra detailed, "
        "rich color grading, shot on a full frame camera with a wide angle lens"
    ),
    "a single red apple on a plain white background, centered, soft even light",
    (
        "a hyperdetailed steampunk pocket watch with exposed brass gears, dramatic "
        "rim lighting, dark background, high contrast"
    ),
)

CALIBRATION_PROMPTS: tuple = (
    "a studio headshot of a young woman with freckles, softbox key light, shallow depth of field",
    "a candid photograph of a grandmother laughing in a kitchen, warm tungsten light, grain",
    "a stage portrait of a violinist mid performance, hard spotlight, deep black surroundings",
    "a group of four coworkers standing in an open plan office, overhead fluorescent light",
    "a macro photograph of moss and lichen covering wet granite after rain",
    "an overhead shot of a market stall piled with spices in open sacks, saturated colour",
    "a tangle of bicycle chains and gears in a repair shop, oily metal, cluttered background",
    "a dense pine forest canopy seen from directly above at midday",
    "an empty concrete skate park under an overcast sky, no people, muted grey tones",
    "a minimalist studio still life of three white ceramic bowls on a pale grey seamless",
    "a calm lake at dawn with a flat mirror surface and low mist, almost no detail",
    "a plain blue swimming pool photographed from above, one lane rope, midday sun",
    "a neon lit alley in heavy rain at night, reflections on wet asphalt, deep shadows",
    "a welder at work in a dark workshop, sparks lighting the scene, everything else black",
    "a lighthouse beam cutting through fog on a moonless night, seen from the cliffs",
    "a campfire on a beach after sunset with silhouetted figures around it",
    "a vintage enamel road sign reading DETOUR bolted to a rusted post, blue sky behind",
    "a chalkboard menu in a cafe listing coffee prices in neat handwriting",
    "the cover of a paperback novel titled MERIDIAN on a wooden table, top down",
    "a scoreboard in an empty stadium showing 3 to 2 in the final minute",
    "a brutalist concrete library interior with tall narrow windows and long shadows",
    "an aerial view of a highway interchange at rush hour, long exposure light trails",
    "a narrow spiral staircase photographed straight down from the top floor",
    "a greenhouse full of tomato plants with condensation on the glass panes",
    "a sprinter leaving the blocks, motion blur on the limbs, sharp on the face",
    "a taxi driving through a snowstorm at dusk, headlights diffused by falling snow",
    "waves breaking over a harbour wall during a gale, spray frozen mid air",
    "a hot air balloon rising through low cloud at sunrise, seen from the ground",
    "a cast iron pan of fried eggs on a gas hob, steam rising, close up",
    "a sleeping tabby cat curled on a radiator cover, afternoon light through blinds",
    "a disassembled mechanical keyboard with its keycaps laid out in rows on a desk",
    "a single dandelion seed head backlit against a dark background, every filament visible",
)


def prompts() -> tuple:
    """The calibration prompts, as a function so the builder reads this module and a
    ``--calib-prompts`` file the same way."""
    return CALIBRATION_PROMPTS
