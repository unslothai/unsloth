from core.training.checkpoint_backup.config import CheckpointBackupConfig


def test_checkpoint_multiplier_resolves_against_save_steps():
    config = CheckpointBackupConfig(
        enabled = True,
        repo_id = "owner/backups",
        interval_checkpoints = 3,
    )

    assert config.effective_backup_steps(200) == 600
    assert config.effective_backup_steps(300) == 900
