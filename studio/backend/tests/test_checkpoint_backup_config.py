from core.training.checkpoint_backup.config import CheckpointBackupConfig


def test_checkpoint_multiplier_resolves_against_save_steps():
    config = CheckpointBackupConfig(
        enabled = True,
        repo_id = "owner/backups",
        interval_checkpoints = 3,
    )

    assert config.effective_backup_steps(200) == 600
    assert config.effective_backup_steps(300) == 900


def test_legacy_visibility_is_accepted_but_not_serialized():
    for private in (True, False):
        config = CheckpointBackupConfig.model_validate({"private": private})
        assert "private" not in config.model_dump()
