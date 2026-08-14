"""Validated, non-secret checkpoint backup policy."""

import re
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


_REPO_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,95}/[A-Za-z0-9][A-Za-z0-9._-]{0,95}")


class CheckpointBackupConfig(BaseModel):
    """Portable backup settings. Authentication is deliberately not represented."""

    model_config = ConfigDict(extra = "ignore")

    enabled: bool = False
    provider: Literal["huggingface"] = "huggingface"
    repo_id: Optional[str] = None
    interval_checkpoints: int = Field(1, gt = 0, le = 1000)
    strategy: Literal["latest"] = "latest"
    keep_remote: int = Field(1, ge = 1)
    upload_on_stop: bool = True
    upload_on_complete: bool = True

    @field_validator("repo_id")
    @classmethod
    def normalize_repo_id(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        value = value.strip().strip("/")
        if not value:
            return None
        if not _REPO_ID.fullmatch(value) or ".." in value or "--" in value:
            raise ValueError("repo_id must be a valid Hugging Face namespace/repository ID")
        return value

    @model_validator(mode = "after")
    def require_repo_when_enabled(self) -> "CheckpointBackupConfig":
        if self.enabled and not self.repo_id:
            raise ValueError("repo_id is required when checkpoint backup is enabled")
        return self

    def validate_for_save_steps(self, save_steps: int) -> "CheckpointBackupConfig":
        """Validate cadence only when periodic backup is enabled."""
        if not self.enabled:
            return self
        if save_steps <= 0:
            raise ValueError(
                "Interval-based backup requires local checkpoint save steps greater than zero."
            )
        return self

    def effective_backup_steps(self, save_steps: int) -> int:
        """Resolve the portable checkpoint multiplier for a training run."""
        self.validate_for_save_steps(save_steps)
        return save_steps * self.interval_checkpoints
