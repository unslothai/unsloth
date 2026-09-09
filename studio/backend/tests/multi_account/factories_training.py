# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Factories for the training, diffusion dataset, scan folder and data recipe object routes."""

from .factory_base import Factory, seeder

SENTINEL = "Alice's training resource: café / 日本語"
EDITED = "training-matrix-edited"
START_REQUEST_ID = "training-matrix-start-request"
START_JOB_ID = "training-matrix-job"
DIFFUSION_JOB_ID = "a1b2c3d4e5f607182930a4b5c6d7e8f9"
DATASET_NAME = "training-matrix-dataset"
IMAGE_NAME = "training-matrix-image.png"
RECIPE_JOB_ID = "f9e8d7c6b5a40312092a8b7c6d5e4f30"
BLOCK_ID = "0123456789abcdef0123456789abcdef"
FILE_ID = "fedcba9876543210fedcba9876543210"
SCAN_FOLDER_DIRNAME = "training-matrix-scan-folder"

PNG_1X1 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmM"
    "IQAAAABJRU5ErkJggg=="
)

_SCAN_FOLDER_REASON = (
    "Removing a scan folder id is an idempotent delete in the caller's own studio.db, so a "
    "foreign account is told ok without seeing or touching alice's row."
)
_PUBLISH_REASON = (
    "Publishing uploads to the Hub, so the owning account is taken only as far as the 409 that "
    "proves it resolved its own preview job, while a foreign account resolves no job at all and "
    "is refused for a missing artifact path."
)
_BLOCK_REASON = (
    "Removing an upload block is an idempotent delete under the caller's own uploads root, so a "
    "foreign account is told ok with deleted false."
)
_EVENTS_REASON = (
    "The owning account's SSE stream ends only when the client disconnects, which an in-process "
    "TestClient request cannot do, so the case would hang rather than return."
)


def _claim_in_process_job(service) -> None:
    from utils.account_context import current_account
    with service._account_job_lock:
        service._result_account = service.job_account = current_account()


@seeder("training-start-request")
def seed_training_start_request(account) -> dict[str, str]:
    from core.training import get_training_backend
    from core.training.training import TrainingStartRequestRecord
    from utils.account_context import run_as

    backend = get_training_backend()

    def install() -> None:
        _claim_in_process_job(backend)
        record = TrainingStartRequestRecord(
            start_request_id = START_REQUEST_ID,
            job_id = START_JOB_ID,
            state = "rejected",
            message = SENTINEL,
            error = "Training start failed validation",
        )
        with backend._lock:
            backend._start_requests[START_REQUEST_ID] = record
            backend._start_cancel_tombstones.pop(START_REQUEST_ID, None)
            backend._start_cancel_tombstone_reservations.pop(START_REQUEST_ID, None)
            backend.current_start_request_id = None
            backend.current_job_id = None
            backend._pending_start_request_id = None
            backend._status_start_request_id = None

    run_as(account, install)
    return {"start_request_id": START_REQUEST_ID}


@seeder("training-diffusion-run")
def seed_diffusion_run(account) -> dict[str, str]:
    import json

    from utils.account_context import run_as
    from utils.paths.storage_roots import tensorboard_root

    def install() -> None:
        folder = tensorboard_root() / "diffusion"
        folder.mkdir(parents = True, exist_ok = True)
        (folder / f"{DIFFUSION_JOB_ID}.json").write_text(
            json.dumps(
                {
                    "job_id": DIFFUSION_JOB_ID,
                    "status": "completed",
                    "message": SENTINEL,
                    "step": 100,
                    "total_steps": 100,
                    "saved": True,
                    "config": {},
                }
            ),
            encoding = "utf-8",
        )

    run_as(account, install)
    return {"job_id": DIFFUSION_JOB_ID}


@seeder("training-diffusion-dataset")
def seed_diffusion_dataset(account) -> dict[str, str]:
    import base64

    from utils.account_context import run_as
    from utils.paths import datasets_root

    def install() -> None:
        folder = datasets_root() / DATASET_NAME
        folder.mkdir(parents = True, exist_ok = True)
        image = folder / IMAGE_NAME
        image.write_bytes(base64.b64decode(PNG_1X1))
        image.with_suffix(".txt").write_text(SENTINEL, encoding = "utf-8")

    run_as(account, install)
    return {"name": DATASET_NAME, "filename": IMAGE_NAME}


@seeder("training-scan-folder")
def seed_scan_folder(account) -> dict[str, str]:
    from storage.studio_db import add_scan_folder_with_status
    from utils.account_context import run_as
    from utils.paths import workspace_root

    def install() -> str:
        folder = workspace_root() / SCAN_FOLDER_DIRNAME
        folder.mkdir(parents = True, exist_ok = True)
        row, _ = add_scan_folder_with_status(str(folder))
        return str(row["id"])

    return {"folder_id": run_as(account, install)}


@seeder("training-recipe-job")
def seed_recipe_job(account) -> dict[str, str]:
    from core.data_recipe.jobs import get_job_manager
    from core.data_recipe.jobs.types import Job
    from utils.account_context import run_as

    manager = get_job_manager()

    def install() -> None:
        _claim_in_process_job(manager)
        job = Job(job_id = RECIPE_JOB_ID, status = "completed", started_at = 1000.0)
        job.finished_at = 2000.0
        job.execution_type = "preview"
        job.analysis = {"summary": SENTINEL}
        job.dataset = [{"text": SENTINEL}]
        job.rows = 1
        job.cols = 1
        with manager._lock:
            manager._job = job
            manager._proc = None
            manager._mp_q = None
            manager._events.clear()
            manager._subs.clear()
            manager._seq = 0

    run_as(account, install)
    return {"job_id": RECIPE_JOB_ID}


@seeder("training-unstructured-upload")
def seed_unstructured_upload(account) -> dict[str, str]:
    import json

    from utils.account_context import run_as
    from utils.paths import unstructured_uploads_root

    def install() -> None:
        block = unstructured_uploads_root() / BLOCK_ID
        block.mkdir(parents = True, exist_ok = True)
        (block / f"{FILE_ID}.txt").write_text(SENTINEL, encoding = "utf-8")
        (block / f"{FILE_ID}.extracted.txt").write_text(SENTINEL, encoding = "utf-8")
        (block / f"{FILE_ID}.meta.json").write_text(
            json.dumps({"original_filename": "training-matrix.txt", "size_bytes": 1}),
            encoding = "utf-8",
        )

    run_as(account, install)
    return {"block_id": BLOCK_ID, "file_id": FILE_ID}


FACTORIES = {
    "routes.training_history:DELETE:/runs/{run_id}": Factory("training", fragment = "deleted"),
    "routes.training:GET:/start-requests/{start_request_id}": Factory(
        "training-start-request", fragment = SENTINEL
    ),
    "routes.training:POST:/start-requests/{start_request_id}/acknowledge": Factory(
        "training-start-request", fragment = "ok"
    ),
    "routes.training:POST:/start-requests/{start_request_id}/cancel": Factory(
        "training-start-request", fragment = SENTINEL
    ),
    "routes.training:GET:/diffusion/runs/{job_id}": Factory(
        "training-diffusion-run", fragment = SENTINEL
    ),
    "routes.training:GET:/diffusion/dataset/{name}/images": Factory(
        "training-diffusion-dataset", fragment = SENTINEL
    ),
    "routes.training:GET:/diffusion/dataset/{name}/image/{filename}": Factory(
        "training-diffusion-dataset", fragment = "IHDR"
    ),
    "routes.training:DELETE:/diffusion/dataset/{name}/image/{filename}": Factory(
        "training-diffusion-dataset", fragment = IMAGE_NAME
    ),
    "routes.training:PUT:/diffusion/dataset/{name}/caption/{filename}": Factory(
        "training-diffusion-dataset", {"caption": EDITED}, fragment = EDITED
    ),
    "routes.models:DELETE:/scan-folders/{folder_id}": Factory(
        "training-scan-folder",
        owner = (200,),
        wrong = (200,),
        reason = _SCAN_FOLDER_REASON,
    ),
    "routes.data_recipe.jobs:GET:/jobs/{job_id}/status": Factory(
        "training-recipe-job", fragment = RECIPE_JOB_ID
    ),
    "routes.data_recipe.jobs:GET:/jobs/{job_id}/analysis": Factory(
        "training-recipe-job", fragment = SENTINEL
    ),
    "routes.data_recipe.jobs:GET:/jobs/{job_id}/dataset": Factory(
        "training-recipe-job", fragment = SENTINEL
    ),
    "routes.data_recipe.jobs:POST:/jobs/{job_id}/cancel": Factory(
        "training-recipe-job", fragment = RECIPE_JOB_ID
    ),
    "routes.data_recipe.jobs:POST:/jobs/{job_id}/publish": Factory(
        "training-recipe-job",
        {"repo_id": "training-matrix/dataset", "description": SENTINEL, "hf_token": "hf_matrix"},
        success = 409,
        fragment = "Only completed full runs can be published.",
        owner = (400,),
        wrong = (400,),
        reason = _PUBLISH_REASON,
    ),
    "routes.data_recipe.seed:DELETE:/seed/unstructured-block/{block_id}": Factory(
        "training-unstructured-upload",
        fragment = "true",
        absent = "true",
        owner = (200,),
        wrong = (200,),
        reason = _BLOCK_REASON,
    ),
    "routes.data_recipe.seed:DELETE:/seed/unstructured-file/{block_id}/{file_id}": Factory(
        "training-unstructured-upload", fragment = "ok"
    ),
}

SKIPPED = {
    "routes.data_recipe.jobs:GET:/jobs/{job_id}/events": _EVENTS_REASON,
    "routes.data_recipe.jobs:POST:/jobs/{job_id}/events": _EVENTS_REASON,
}
