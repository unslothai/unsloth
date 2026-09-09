from typing import Optional, Sequence

from hub.utils import download_registry
from hub.utils.hf_cache_state import TRANSPORT_HTTP, TRANSPORT_XET

LOAD_OWNER = "load"


def _job_key(repo_id: str) -> str:
    return download_registry.normalize_job_key(f"{download_registry.normalize_repo_key(repo_id)}::")


def is_load_owned(registry: download_registry.DownloadRegistry, key: str) -> bool:
    metadata = registry.get_job_metadata(key)
    if metadata is None or metadata.owner != LOAD_OWNER:
        return False
    return registry.get_job(key).state in ("running", "cancelling")


def claim_load_downloads(
    repo_ids: Sequence[str],
    *,
    xet_disabled: bool = False,
    hub_cache: Optional[str] = None,
) -> list[str]:
    registry = download_registry.get_models_registry()
    transport = TRANSPORT_HTTP if xet_disabled else TRANSPORT_XET
    claimed: list[str] = []
    for repo_id in dict.fromkeys(str(repo).strip() for repo in repo_ids if repo):
        if not repo_id:
            continue
        accepted, _state = registry.claim(
            _job_key(repo_id),
            transport,
            repo_type = "model",
            repo_id = repo_id,
            hub_cache = hub_cache,
            owner = LOAD_OWNER,
        )
        if accepted:
            claimed.append(_job_key(repo_id))
    return claimed


def release_load_downloads(keys: Sequence[str], state: download_registry.JobState) -> None:
    registry = download_registry.get_models_registry()
    for key in keys:
        if is_load_owned(registry, key):
            registry.set_job(key, state)
