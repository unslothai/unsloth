# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Calibration prompts for the NVFP4 build passes (GPTQ Hessians and activation-scale baking).

These prompts are the CALIBRATION set. They are deliberately DISJOINT from the evaluation suite the
prequant accuracy gate renders, because a quantisation calibrated on the prompts it is then scored
on measures how well it memorised them, not how well it renders. That is the whole reason this file
exists rather than reusing the gate's seven cases, and ``GATE_SUITE_PROMPTS`` below is a frozen copy
of that suite so the disjointness can be asserted by a test instead of maintained by hand.

Chosen for COVERAGE of what the DiT's activations do, not for beauty: faces and skin, dense
high-frequency texture, flat low-frequency fields, extreme contrast, deep night scenes, rendered
text, crowds, macro detail, motion blur, and several deliberately plain compositions. The Hessian a
layer accumulates is the second moment of its inputs, so the calibration set has to reach the
activation ranges a real prompt reaches; a set of thirty-two variations on one landscape would
produce a confident correction for a model nobody runs that way.

Thirty-two: enough that a 3072-wide Hessian over four sampled steps of each is well conditioned
(measured: the Cholesky factors at the first damping rung), few enough that the pass stays inside an
hour on the largest image DiT.
"""

from __future__ import annotations

# The gate's evaluation prompts, copied verbatim from ``prequant_accuracy_gate.PROMPT_SUITE``. Not
# imported: the gate is a research script that is not always beside this one, and a copy that a test
# compares against catches the only failure that matters here, which is a calibration prompt
# drifting INTO the evaluation set.
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
    # faces and skin, across ages, lighting and framing
    "a studio headshot of a young woman with freckles, softbox key light, shallow depth of field",
    "a candid photograph of a grandmother laughing in a kitchen, warm tungsten light, grain",
    "a stage portrait of a violinist mid performance, hard spotlight, deep black surroundings",
    "a group of four coworkers standing in an open plan office, overhead fluorescent light",
    # dense high-frequency texture
    "a macro photograph of moss and lichen covering wet granite after rain",
    "an overhead shot of a market stall piled with spices in open sacks, saturated colour",
    "a tangle of bicycle chains and gears in a repair shop, oily metal, cluttered background",
    "a dense pine forest canopy seen from directly above at midday",
    # flat, low-frequency fields
    "an empty concrete skate park under an overcast sky, no people, muted grey tones",
    "a minimalist studio still life of three white ceramic bowls on a pale grey seamless",
    "a calm lake at dawn with a flat mirror surface and low mist, almost no detail",
    "a plain blue swimming pool photographed from above, one lane rope, midday sun",
    # extreme contrast and night
    "a neon lit alley in heavy rain at night, reflections on wet asphalt, deep shadows",
    "a welder at work in a dark workshop, sparks lighting the scene, everything else black",
    "a lighthouse beam cutting through fog on a moonless night, seen from the cliffs",
    "a campfire on a beach after sunset with silhouetted figures around it",
    # rendered text and graphic structure
    "a vintage enamel road sign reading DETOUR bolted to a rusted post, blue sky behind",
    "a chalkboard menu in a cafe listing coffee prices in neat handwriting",
    "the cover of a paperback novel titled MERIDIAN on a wooden table, top down",
    "a scoreboard in an empty stadium showing 3 to 2 in the final minute",
    # architecture, perspective and scale
    "a brutalist concrete library interior with tall narrow windows and long shadows",
    "an aerial view of a highway interchange at rush hour, long exposure light trails",
    "a narrow spiral staircase photographed straight down from the top floor",
    "a greenhouse full of tomato plants with condensation on the glass panes",
    # motion, weather and atmosphere
    "a sprinter leaving the blocks, motion blur on the limbs, sharp on the face",
    "a taxi driving through a snowstorm at dusk, headlights diffused by falling snow",
    "waves breaking over a harbour wall during a gale, spray frozen mid air",
    "a hot air balloon rising through low cloud at sunrise, seen from the ground",
    # objects, food and animals
    "a cast iron pan of fried eggs on a gas hob, steam rising, close up",
    "a sleeping tabby cat curled on a radiator cover, afternoon light through blinds",
    "a disassembled mechanical keyboard with its keycaps laid out in rows on a desk",
    "a single dandelion seed head backlit against a dark background, every filament visible",
)


def prompts() -> tuple:
    """The calibration prompts. A function so a file passed to ``--calib-prompts`` and this module
    are read the same way by the builder."""
    return CALIBRATION_PROMPTS
