"""Hugging Face Hub transport, kept separate from scheduling."""

from pathlib import Path
from typing import Callable, Optional

from .manifest import upload_files


class HuggingFaceCheckpointTransport:
    def __init__(self, token: str, repo_id: str, private: bool = True) -> None:
        self.token = token
        self.repo_id = repo_id
        self.private = private

    def ensure_repository(self) -> None:
        from huggingface_hub import HfApi

        HfApi(token = self.token).create_repo(
            repo_id = self.repo_id, private = self.private, exist_ok = True
        )

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
