"""Hugging Face Hub transport, kept separate from scheduling."""

from pathlib import Path
from typing import Callable, Optional

from .manifest import upload_files


class HuggingFaceCheckpointTransport:
    def __init__(self, token: str, repo_id: str) -> None:
        self.token = token
        self.repo_id = repo_id

    def validate_access(self) -> None:
        """Verify that an existing model repository is writable without mutating it."""
        from huggingface_hub import HfApi
        from huggingface_hub.errors import HfHubHTTPError, RepositoryNotFoundError

        api = HfApi(token = self.token)
        try:
            api.repo_info(repo_id = self.repo_id, repo_type = "model")
            user = api.whoami()
        except RepositoryNotFoundError as error:
            raise ValueError(
                "Repository not found. Create it on Hugging Face and try again."
            ) from error
        except HfHubHTTPError as error:
            status = getattr(getattr(error, "response", None), "status_code", None)
            if status == 401:
                raise PermissionError("Authentication required") from error
            if status == 403:
                raise PermissionError("No write permission") from error
            raise ConnectionError("Hugging Face unavailable") from error

        namespace = self.repo_id.split("/", 1)[0]
        username = user.get("name") if isinstance(user, dict) else None
        organizations = user.get("orgs", []) if isinstance(user, dict) else []
        writable_namespaces = {username}
        writable_namespaces.update(
            org.get("name") for org in organizations
            if isinstance(org, dict) and org.get("roleInOrg") in {"admin", "write"}
        )
        if namespace not in writable_namespaces:
            raise PermissionError("No write permission")

    def upload_checkpoint(
        self,
        run_id: str,
        checkpoint_path: Path,
        progress: Optional[Callable[[int, int, int, int], None]] = None,
    ) -> None:
        from huggingface_hub import HfApi

        files = upload_files(checkpoint_path)
        total = sum(path.stat().st_size for path in files)
        uploaded = 0
        api = HfApi(token = self.token)
        for index, path in enumerate(files, 1):
            api.upload_file(
                path_or_fileobj = str(path),
                path_in_repo = f"runs/{run_id}/checkpoints/{checkpoint_path.name}/{path.relative_to(checkpoint_path).as_posix()}",
                repo_id = self.repo_id,
            )
            uploaded += path.stat().st_size
            if progress:
                progress(index, len(files), uploaded, total)
