# SPDX-License-Identifier: Apache-2.0

import os

import pytest

from unsloth.launcher_world_size import world_size_from_env


@pytest.mark.parametrize(
    ("env", "expected"),
    [
        ({}, 1),
        ({"WORLD_SIZE": "auto"}, 1),
        ({"WORLD_SIZE": "4"}, 4),
        ({"OMPI_COMM_WORLD_SIZE": "8"}, 8),
        ({"PMI_SIZE": "2", "WORLD_SIZE": "1"}, 2),
    ],
)
def test_world_size_from_env(env, expected, monkeypatch):
    for key in list(os.environ):
        if key.startswith(
            (
                "WORLD_SIZE",
                "LOCAL_WORLD_SIZE",
                "MLX_",
                "OMPI_",
                "PMI",
                "MPI_",
                "MV2_",
            )
        ):
            monkeypatch.delenv(key, raising = False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    assert world_size_from_env() == expected
