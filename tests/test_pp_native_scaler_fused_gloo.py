# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Real CPU/gloo AMP optimizers using PipelineParallel's scaler hooks, without a GPU import.
Run directly with Python, or collect with pytest. No model or network access is needed.
"""

import multiprocessing
import os

# Only spawned workers (and a direct run) hide the GPUs: pytest's own process keeps its mask for other tests.
if (
    __name__ == "__main__"
    or multiprocessing.parent_process() is not None
    or getattr(multiprocessing.current_process(), "_inheriting", False)
):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

import datetime
import importlib
import itertools
import pathlib
import pickle
import sys
import tempfile
import types
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def _pipeline_class():
    package = types.ModuleType("_pp_scaler_test")
    package.__path__ = [str(pathlib.Path(__file__).resolve().parents[1] / "unsloth/models")]
    sys.modules[package.__name__] = package
    return importlib.import_module(f"{package.__name__}.pp_native").PipelineParallel


def _worker(rank, rendezvous, output):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method = rendezvous,
        rank = rank,
        world_size = 2,
        timeout = datetime.timedelta(seconds = 30),
    )
    try:
        pipeline_class = _pipeline_class()
        results = {}
        cases = itertools.product(
            ("Adam", "AdamW"),
            (False, True),
            ("step_only", "scaler_unscale", "pp_unscale", "pp_unscale_unbound"),
            (-1, 0, 1, 2),
        )
        for name, fused, recipe, inf_rank in cases:
            parameter = torch.nn.Parameter(torch.tensor(1.0))
            reference = torch.nn.Parameter(torch.tensor(1.0))
            optimizer = getattr(torch.optim, name)([parameter], lr = 0.1, fused = fused)
            reference_optimizer = getattr(torch.optim, name)([reference], lr = 0.1, fused = False)
            scaler = torch.amp.GradScaler("cpu", init_scale = 1024.0, growth_interval = 2)
            pipeline = pipeline_class.__new__(pipeline_class)
            pipeline.module = torch.nn.ParameterList([parameter])
            pipeline.num_stages = 2
            pipeline.group = dist.group.WORLD
            pipeline._step_scaler = None
            history = []
            expected_scale, finite_steps = 1024.0, 0
            for step in range(3):
                if recipe != "pp_unscale_unbound":
                    pipeline._bind_scaler(scaler)
                    pipeline._bind_scaler(scaler)  # repeated binds must not stack hooks
                pipeline._step_scaler = None if recipe == "pp_unscale_unbound" else scaler
                scaler.scale(parameter * (step + 1)).backward()
                overflow = step == 1 and inf_rank != -1
                if overflow and (rank == inf_rank or inf_rank == 2):
                    parameter.grad.fill_(float("inf"))
                reductions = []
                original_reduce = dist.all_reduce

                def counted(*args, **kwargs):
                    reductions.append(1)
                    return original_reduce(*args, **kwargs)

                dist.all_reduce = counted
                try:
                    if recipe.startswith("pp_unscale"):
                        pipeline.unscale_(scaler, optimizer)
                    elif recipe == "scaler_unscale":
                        scaler.unscale_(optimizer)
                    scaler.step(optimizer)
                    scaler.update()
                finally:
                    dist.all_reduce = original_reduce
                if overflow:
                    expected_scale *= 0.5
                    finite_steps = 0
                else:
                    reference.grad = torch.tensor(float(step + 1))
                    reference_optimizer.step()
                    finite_steps += 1
                    if finite_steps == 2:
                        expected_scale *= 2
                        finite_steps = 0
                pickle.loads(pickle.dumps(scaler))
                history.append(
                    {
                        "weight": parameter.item(),
                        "reference": reference.item(),
                        "scale": scaler.get_scale(),
                        "expected_scale": expected_scale,
                        "reductions": len(reductions),
                        "cleaned": pipeline._step_scaler is None
                        and not any(
                            key in scaler.__dict__
                            for key in (
                                "unscale_",
                                "_check_inf_per_device",
                                "update",
                                "_unsloth_pp_agree",
                            )
                        ),
                    }
                )
                optimizer.zero_grad(set_to_none = True)
            results[name, fused, recipe, inf_rank] = history
        torch.save(results, pathlib.Path(output) / f"rank{rank}.pt")
    finally:
        dist.destroy_process_group()


class TestPipelineScalerFused(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not dist.is_available() or not dist.is_gloo_available():
            raise unittest.SkipTest("requires CPU gloo")
        with tempfile.TemporaryDirectory() as output:
            rendezvous = (pathlib.Path(output) / "store").resolve().as_uri()
            mp.spawn(_worker, args = (rendezvous, output), nprocs = 2, join = True)
            cls.ranks = [
                torch.load(pathlib.Path(output) / f"rank{rank}.pt", weights_only = False)
                for rank in range(2)
            ]

    def test_optimizer_skip_and_scale_agreement(self):
        for rank, results in enumerate(self.ranks):
            for case, history in results.items():
                with self.subTest(rank = rank, case = case):
                    for step, result in enumerate(history):
                        self.assertAlmostEqual(
                            result["weight"],
                            result["reference"],
                            places = 6,
                            msg = str((case, step, result)),
                        )
                        self.assertEqual(
                            result["scale"], result["expected_scale"], (case, step, result)
                        )

    def test_explicit_pp_unscale_agrees_once(self):
        for rank, results in enumerate(self.ranks):
            for case, history in results.items():
                if case[2].startswith("pp_unscale"):
                    with self.subTest(rank = rank, case = case):
                        self.assertEqual([r["reductions"] for r in history], [1, 1, 1])

    def test_update_removes_hooks_and_releases_pipeline(self):
        for rank, results in enumerate(self.ranks):
            for case, history in results.items():
                with self.subTest(rank = rank, case = case):
                    self.assertTrue(all(r["cleaned"] for r in history))


if __name__ == "__main__":
    unittest.main(verbosity = 2)
