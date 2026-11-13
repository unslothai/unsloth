# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Truthful first-step milestones shared by SFT, vision, audio, decision and embedding callbacks."""

STARTING = "Starting training loop..."
ACTIVE = "Training in progress..."
WAITING_FOR_LOSS = "First training step finished. Waiting for the first loss measurement..."


class TrainingStartupProgress:
    def __init__(self):
        self.started = False
        self.start_step = 0
        self.accumulation = 1
        self.batches = 0
        self.step_finished = False
        self.loss_reported = False

    def begin(self, args, state):
        self.started = True
        self.start_step = state.global_step
        self.accumulation = max(1, int(getattr(args, "gradient_accumulation_steps", 1) or 1))
        self.batches = 0
        self.step_finished = False
        self.loss_reported = False
        return STARTING

    def step_begin(self):
        if not self.started or self.step_finished or self.loss_reported:
            return None
        return self._batch_status()

    def substep_end(self):
        if not self.started or self.step_finished or self.loss_reported:
            return None
        self.batches += 1
        return self._batch_status()

    def _batch_status(self):
        batch = min(self.batches + 1, self.accumulation)
        return (
            f"Running first training step (batch {batch} of up to {self.accumulation})... "
            "Loss appears after the optimizer step."
        )

    def step_end(self, state):
        if not self.started or self.step_finished or state.global_step <= self.start_step:
            return None
        self.step_finished = True
        return None if self.loss_reported else WAITING_FOR_LOSS

    def log(self, logs):
        if not self.started or self.loss_reported or logs.get("loss") is None:
            return None
        self.loss_reported = True
        return ACTIVE
