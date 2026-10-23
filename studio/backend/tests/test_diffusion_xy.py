# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""X/Y grid on /images/generate: cell expansion, validation, the labelled grid image, and the route, renderer faked."""

from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from core.inference import diffusion_xy
from models.inference import MAX_XY_CELLS, DiffusionGenerateRequest

_TESTS_DIR = str(Path(__file__).resolve().parent)
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

from test_diffusion_routes import client  # noqa: E402,F401  (fixture)


def _req(**body) -> DiffusionGenerateRequest:
    return DiffusionGenerateRequest.model_validate({"prompt": "a red sloth", **body})


def test_expand_is_row_major_with_one_shared_seed():
    plan = diffusion_xy.expand(
        _req(
            xy = {
                "x": {"field": "steps", "values": [4, 8, 12]},
                "y": {"field": "guidance", "values": [0, 1.5]},
            }
        ),
        rng = random.Random(0),
    )
    assert (plan.nx, plan.ny) == (3, 2)
    assert [(c.request.steps, c.request.guidance) for c in plan.cells] == [
        (4, 0.0),
        (8, 0.0),
        (12, 0.0),
        (4, 1.5),
        (8, 1.5),
        (12, 1.5),
    ]
    assert len({c.request.seed for c in plan.cells}) == 1
    assert all(c.request.xy is None and c.request.enhance_prompt is False for c in plan.cells)
    assert plan.x_labels == ["Steps: 4", "Steps: 8", "Steps: 12"]
    assert plan.y_labels == ["CFG: 0", "CFG: 1.5"]


def test_expand_keeps_unswept_fields_unset_so_model_defaults_still_apply():
    plan = diffusion_xy.expand(_req(xy = {"x": {"field": "steps", "values": [4, 8]}}))
    fields = plan.cells[0].request.model_fields_set
    assert "steps" in fields and "guidance" not in fields and "width" not in fields
    assert plan.y_labels == [""]


def test_seed_axis_sets_each_cell_seed():
    plan = diffusion_xy.expand(_req(seed = 1, xy = {"x": {"field": "seed", "values": [7, 8]}}))
    assert [c.request.seed for c in plan.cells] == [7, 8]


def test_sampler_axis_uses_display_labels():
    plan = diffusion_xy.expand(
        _req(xy = {"x": {"field": "sampler", "values": ["default", "euler", "dpmpp_2m"]}})
    )
    assert [c.request.sampler for c in plan.cells] == ["default", "euler", "dpmpp_2m"]
    assert plan.x_labels[0] == "Default" and plan.x_labels[1] == "Euler"


def test_prompt_sr_replaces_the_first_value_in_prompt_and_negative():
    plan = diffusion_xy.expand(
        _req(
            negative_prompt = "red blur",
            xy = {"x": {"field": "prompt_sr", "values": ["red", "blue"]}},
        )
    )
    assert [c.request.prompt for c in plan.cells] == ["a red sloth", "a blue sloth"]
    assert plan.cells[1].request.negative_prompt == "blue blur"
    with pytest.raises(ValueError, match = "not in the prompt"):
        diffusion_xy.expand(_req(xy = {"x": {"field": "prompt_sr", "values": ["green", "blue"]}}))


def test_lora_weight_axis_targets_one_entry_and_zero_drops_it():
    plan = diffusion_xy.expand(
        _req(
            loras = [{"id": "a", "weight": 1.0}, {"id": "b", "weight": 0.5}],
            xy = {"x": {"field": "lora_weight", "index": 1, "values": [0, 1]}},
        )
    )
    assert [[(l.id, l.weight) for l in c.request.loras] for c in plan.cells] == [
        [("a", 1.0)],
        [("a", 1.0), ("b", 1.0)],
    ]
    # Adapter-free cells render last.
    assert plan.order == [1, 0]


def test_out_of_range_value_names_the_cell():
    with pytest.raises(ValueError, match = r"xy cell \(Steps: 200\): steps"):
        diffusion_xy.expand(_req(xy = {"x": {"field": "steps", "values": [4, 200]}}))


@pytest.mark.parametrize(
    "body, match",
    [
        ({"xy": {"x": {"field": "width", "values": [512]}}}, "field"),
        (
            {
                "xy": {
                    "x": {"field": "steps", "values": list(range(1, 10))},
                    "y": {"field": "guidance", "values": list(range(8))},
                }
            },
            "over the limit",
        ),
        (
            {
                "xy": {
                    "x": {"field": "steps", "values": [4]},
                    "y": {"field": "steps", "values": [8]},
                }
            },
            "both vary",
        ),
        ({"batch_size": 2, "xy": {"x": {"field": "steps", "values": [4]}}}, "batch size 1"),
        ({"seeds": [1, 2], "xy": {"x": {"field": "steps", "values": [4]}}}, "batch size 1"),
        ({"xy": {"x": {"field": "lora_weight", "values": [1]}}}, "index"),
        ({"xy": {"x": {"field": "lora_weight", "index": 0, "values": [1]}}}, "no entry in loras"),
        ({"xy": {"x": {"field": "prompt_sr", "values": ["", "x"]}}}, "cannot be empty"),
    ],
)
def test_spec_is_validated(body, match):
    with pytest.raises(ValidationError, match = match):
        _req(**body)


def test_cap_matches_the_images_page_compare_grid():
    assert MAX_XY_CELLS == 64
    _req(
        xy = {
            "x": {"field": "steps", "values": list(range(1, 9))},
            "y": {"field": "seed", "values": list(range(8))},
        }
    )


def _cells(plan, size = (64, 48)):
    from PIL import Image
    return [Image.new("RGB", size, (40 * c.ix, 40 * c.iy, 200)) for c in plan.cells]


def test_compose_grid_lays_out_headers_and_cells():
    plan = diffusion_xy.expand(
        _req(
            xy = {
                "x": {"field": "steps", "values": [4, 8, 12]},
                "y": {"field": "guidance", "values": [0, 1, 2]},
            }
        )
    )
    grid = diffusion_xy.compose_grid(_cells(plan), plan)
    left, top = 64 * 3 // 4, grid.height - 3 * 48
    assert grid.width == left + 3 * 64 and top > 0
    # Cell (ix=2, iy=1) sits at its slot; the header bands are white paper with dark text on it.
    assert grid.getpixel((left + 2 * 64 + 32, top + 48 + 24)) == (80, 40, 200)
    band = grid.crop((left, 0, grid.width, top))
    assert band.convert("L").getextrema()[0] < 128
    assert grid.getpixel((1, 1)) == (255, 255, 255)


def test_compose_grid_without_y_has_no_left_band():
    plan = diffusion_xy.expand(_req(xy = {"x": {"field": "steps", "values": [4, 8]}}))
    grid = diffusion_xy.compose_grid(_cells(plan), plan)
    assert grid.width == 2 * 64


def test_compose_grid_scales_down_past_the_pixel_budget(monkeypatch):
    monkeypatch.setattr(diffusion_xy, "MAX_GRID_PIXELS", 4 * 32 * 24)
    plan = diffusion_xy.expand(
        _req(
            xy = {"x": {"field": "steps", "values": [4, 8]}, "y": {"field": "seed", "values": [1, 2]}}
        )
    )
    grid = diffusion_xy.compose_grid(_cells(plan), plan)
    # 64x48 cells halve to 32x24, the row band to 3/4 of that.
    assert grid.width == 24 + 2 * 32


def test_route_renders_each_cell_and_saves_a_labelled_grid(client):
    import core.inference.diffusion as diffusion_module
    from PIL import Image

    client.post(
        "/api/inference/images/load",
        json = {"model_path": "x/z-image", "gguf_filename": "q.gguf"},
    )
    backend = diffusion_module.get_diffusion_backend()
    calls = []

    def _generate(**kwargs):
        calls.append(kwargs)
        return {
            "images": [Image.new("RGB", (64, 64), "gray")],
            "seed": kwargs["seed"],
            "repo_id": "x/z-image",
            "workflow": "txt2img",
        }

    backend.generate = _generate
    resp = client.post(
        "/api/inference/images/generate",
        json = {
            "prompt": "a sloth",
            "seed": 11,
            "xy": {
                "x": {"field": "steps", "values": [2, 4, 6]},
                "y": {"field": "guidance", "values": [0, 1, 2]},
            },
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert [(c["steps"], c["guidance"]) for c in calls] == [
        (2, 0.0),
        (4, 0.0),
        (6, 0.0),
        (2, 1.0),
        (4, 1.0),
        (6, 1.0),
        (2, 2.0),
        (4, 2.0),
        (6, 2.0),
    ]
    assert {c["seed"] for c in calls} == {11}
    cells = body["images"]
    assert [(c["steps"], c["guidance"]) for c in cells] == [
        (c["steps"], c["guidance"]) for c in calls
    ]
    grid = body["grid"]
    assert grid["xy_grid"]["cells"] == [c["id"] for c in cells]
    assert grid["xy_grid"]["x"]["labels"] == ["Steps: 2", "Steps: 4", "Steps: 6"]
    assert grid["width"] == 48 + 3 * 64 and grid["height"] > 3 * 64
    gallery = client.get("/api/inference/images/gallery").json()["images"]
    assert len(gallery) == 10


def test_route_refuses_a_bad_cell_before_rendering(client):
    import core.inference.diffusion as diffusion_module

    client.post(
        "/api/inference/images/load",
        json = {"model_path": "x/z-image", "gguf_filename": "q.gguf"},
    )
    backend = diffusion_module.get_diffusion_backend()
    backend.generate = lambda **kw: pytest.fail("no cell may render")
    resp = client.post(
        "/api/inference/images/generate",
        json = {"prompt": "a sloth", "xy": {"x": {"field": "steps", "values": [4, 500]}}},
    )
    assert resp.status_code == 400 and "Steps: 500" in resp.json()["detail"]
    over = client.post(
        "/api/inference/images/generate",
        json = {"prompt": "a sloth", "xy": {"x": {"field": "seed", "values": list(range(65))}}},
    )
    assert over.status_code == 422
