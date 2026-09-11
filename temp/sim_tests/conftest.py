import os
from pathlib import Path
import socket
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
REPO = Path(os.environ.get('SIM_REPO', ROOT / 'work/unsloth')).resolve()
sys.path[:0] = [str(REPO / 'studio/backend'), str(REPO / 'studio/backend/tests')]
from test_diffusion_backend import fake_runtime


@pytest.fixture(autouse=True)
def isolate(monkeypatch, tmp_path):
    monkeypatch.setenv('UNSLOTH_STUDIO_HOME', str(tmp_path / 'studio'))
    monkeypatch.setenv('HF_HUB_OFFLINE', '1')
    real_connect = socket.socket.connect
    def connect(sock, address):
        if isinstance(address, tuple) and address[0] not in ('127.0.0.1', '::1', 'localhost'):
            raise OSError('Simulation forbids outbound network')
        return real_connect(sock, address)
    monkeypatch.setattr(socket.socket, 'connect', connect)
