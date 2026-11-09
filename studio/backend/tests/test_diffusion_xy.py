# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""X/Y grid on /images/generate, renderer faked."""

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
    assert plan.order == [0, 1]


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
    assert grid.getpixel((left + 2 * 64 + 32, top + 48 + 24)) == (80, 40, 200)
    band = grid.crop((left, 0, grid.width, top))
    assert band.convert("L").getextrema()[0] < 128
    assert grid.getpixel((1, 1)) == (255, 255, 255)


def test_compose_grid_without_y_has_no_left_band():
    plan = diffusion_xy.expand(_req(xy = {"x": {"field": "steps", "values": [4, 8]}}))
    grid = diffusion_xy.compose_grid(_cells(plan), plan)
    assert grid.width == 2 * 64


def test_compose_grid_scales_down_past_the_pixel_budget(monkeypatch):
    monkeypatch.setattr(diffusion_xy, "MAX_GRID_PIXELS", 4 * 640 * 480)
    plan = diffusion_xy.expand(
        _req(
            xy = {"x": {"field": "steps", "values": [4, 8]}, "y": {"field": "seed", "values": [1, 2]}}
        )
    )
    grid = diffusion_xy.compose_grid(_cells(plan, size = (1280, 960)), plan)
    assert grid.width * grid.height <= 4 * 640 * 480 and grid.width < 2 * 640 * 1.75


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
    assert (grid["xy_grid"]["cell_width"], grid["xy_grid"]["cell_height"]) == (64, 64)
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


def test_stop_between_cells_ends_the_grid(client):
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
        assert client.post("/api/inference/images/generate/cancel").json() == {"cancelled": True}
        return {
            "images": [Image.new("RGB", (64, 64), "gray")],
            "seed": kwargs["seed"],
            "repo_id": "x/z-image",
            "workflow": "txt2img",
        }

    backend.generate = _generate
    backend.cancel_generate = lambda **kw: False
    resp = client.post(
        "/api/inference/images/generate",
        json = {"prompt": "a sloth", "xy": {"x": {"field": "steps", "values": [2, 4, 6]}}},
    )
    assert resp.status_code == 409, resp.text
    assert len(calls) == 1
    assert len(client.get("/api/inference/images/gallery").json()["images"]) == 1
    assert client.post("/api/inference/images/generate/cancel").json() == {"cancelled": False}


def test_grid_infotext_records_the_cell_size():
    from core.inference import image_gallery

    meta = {"prompt": "a sloth", "width": 2880, "height": 2441, "seed": 1, "steps": 8}
    assert "Size: 2880x2441," in image_gallery._params_text(meta)
    meta["xy_grid"] = {"cell_width": 768, "cell_height": 768}
    assert "Size: 768x768," in image_gallery._params_text(meta)


def test_lora_with_its_default_weight_expands():
    plan = diffusion_xy.expand(
        _req(loras = [{"id": "adapter"}], xy = {"x": {"field": "steps", "values": [2, 4]}})
    )
    assert [c.request.loras[0].weight for c in plan.cells] == [1.0, 1.0]


def test_variation_strength_sweep_shares_one_variation_seed():
    plan = diffusion_xy.expand(
        _req(xy = {"x": {"field": "variation_strength", "values": [0.1, 0.3, 0.5]}}),
        rng = random.Random(0),
    )
    seeds = {c.request.variation_seed for c in plan.cells}
    assert len(seeds) == 1 and None not in seeds
    plain = diffusion_xy.expand(_req(xy = {"x": {"field": "steps", "values": [2, 4]}}))
    assert {c.request.variation_seed for c in plain.cells} == {None}


def test_cells_with_any_adapter_render_before_adapter_free_ones():
    plan = diffusion_xy.expand(
        _req(
            loras = [{"id": "a", "weight": 1}, {"id": "b", "weight": 0}],
            xy = {"x": {"field": "lora_weight", "index": 0, "values": [0, 1]}},
        )
    )
    assert plan.order == [1, 0]


def test_stop_reaches_a_cell_still_waiting_for_the_engine(client):
    import core.inference.diffusion as diffusion_module
    import threading
    from core.inference.diffusion_families import DIFFUSION_CANCELLED_MSG
    from core.inference.generation_ticket import bind_generation_cancel

    client.post(
        "/api/inference/images/load",
        json = {"model_path": "x/z-image", "gguf_filename": "q.gguf"},
    )
    backend = diffusion_module.get_diffusion_backend()

    def _generate(**kwargs):
        # As the engine does: bind the cell's cancel, then (here) be stopped while queued for the slot.
        cancel = threading.Event()
        bind_generation_cancel(cancel)
        client.post("/api/inference/images/generate/cancel")
        assert cancel.is_set()
        raise RuntimeError(DIFFUSION_CANCELLED_MSG)

    backend.generate = _generate
    backend.cancel_generate = lambda **kw: False
    resp = client.post(
        "/api/inference/images/generate",
        json = {"prompt": "a sloth", "xy": {"x": {"field": "steps", "values": [2, 4]}}},
    )
    assert resp.status_code == 409, resp.text
    assert client.get("/api/inference/images/gallery").json()["images"] == []


def test_string_zero_lora_weight_drops_the_adapter_and_renders_last():
    plan = diffusion_xy.expand(
        _req(
            loras = [{"id": "a", "weight": 1}],
            xy = {"x": {"field": "lora_weight", "index": 0, "values": ["0", "1"]}},
        )
    )
    assert [c.request.loras for c in plan.cells][0] is None
    assert plan.order == [1, 0]


def test_long_labels_and_big_cells_stay_inside_the_pixel_budget():
    from PIL import Image

    long = " ".join(["word"] * 600)
    plan = diffusion_xy.expand(
        _req(
            prompt = "a red sloth",
            xy = {
                "x": {"field": "prompt_sr", "values": ["red", long] + [f"c{i}" for i in range(62)]}
            },
        )
    )
    cells = [Image.new("RGB", (720, 720), "gray") for _ in plan.cells]
    grid = diffusion_xy.compose_grid(cells, plan)
    assert grid.width * grid.height <= diffusion_xy.MAX_GRID_PIXELS


def test_fit_cell_keeps_each_render_to_its_share_of_the_budget():
    from PIL import Image

    plan = diffusion_xy.expand(_req(xy = {"x": {"field": "seed", "values": list(range(64))}}))
    small = diffusion_xy.fit_cell(Image.new("RGB", (2048, 2048)), plan)
    assert small.width * small.height <= diffusion_xy.MAX_GRID_PIXELS / 64
    same = Image.new("RGB", (512, 512))
    assert diffusion_xy.fit_cell(same, plan) is same


def test_stop_after_the_last_cell_ends_the_grid_unsaved(client):
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
        if len(calls) == 2:
            assert client.post("/api/inference/images/generate/cancel").json() == {
                "cancelled": True
            }
        return {
            "images": [Image.new("RGB", (64, 64), "gray")],
            "seed": kwargs["seed"],
            "repo_id": "x/z-image",
            "workflow": "txt2img",
        }

    backend.generate = _generate
    backend.cancel_generate = lambda **kw: False
    resp = client.post(
        "/api/inference/images/generate",
        json = {"prompt": "a sloth", "xy": {"x": {"field": "steps", "values": [2, 4]}}},
    )
    assert resp.status_code == 409, resp.text
    gallery = client.get("/api/inference/images/gallery").json()["images"]
    assert len(gallery) == 2 and not any(g.get("xy_grid") for g in gallery)


def test_layered_model_is_refused_before_any_cell_renders(client, monkeypatch):
    import core.inference.diffusion as diffusion_module
    import core.inference.diffusion_engine_router as router

    client.post(
        "/api/inference/images/load",
        json = {"model_path": "x/z-image", "gguf_filename": "q.gguf"},
    )
    backend = diffusion_module.get_diffusion_backend()
    backend.generate = lambda **kw: pytest.fail("no cell may render")
    engine = router.get_active_diffusion_engine()
    real_status = engine.status
    monkeypatch.setattr(engine, "status", lambda: {**real_status(), "family": "qwen-image-layered"})
    resp = client.post(
        "/api/inference/images/generate",
        json = {"prompt": "a sloth", "xy": {"x": {"field": "steps", "values": [2, 4]}}},
    )
    assert resp.status_code == 400 and "2 layers" in resp.json()["detail"], resp.text


def test_grid_stop_is_keyed_on_the_stable_account_id(monkeypatch):
    import routes.inference as routes_inference
    from core.inference.generation_ticket import GenerationTicket

    ticket = GenerationTicket()
    monkeypatch.setattr(routes_inference, "_xy_grid_stops", [("owner-id", ticket)])
    monkeypatch.setattr(routes_inference, "current_account_id", lambda: "owner-id")
    # Scope moves from None to the owner's id once the first managed account exists.
    monkeypatch.setattr(routes_inference.account_access, "account_scope", lambda: "something-else")
    assert routes_inference._stop_xy_grids() is True and ticket.cancelled()
