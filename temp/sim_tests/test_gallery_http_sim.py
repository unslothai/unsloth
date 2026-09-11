import errno
from pathlib import Path
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image
from core.inference import image_gallery as gallery
from auth.authentication import get_current_subject
from routes.inference import studio_router

@pytest.mark.parametrize('stage', ['read', 'unlink'])
@pytest.mark.parametrize('code', [errno.EACCES, errno.EIO, errno.EBUSY])
def test_real_route_keeps_io_failures_as_500(monkeypatch, stage, code):
    row = gallery.save(Image.new('RGB', (16, 16)), dict(prompt='test', width=16, height=16, steps=9, guidance=0, seed=42, created_at=1))
    path = gallery.image_path(row['id'])
    app = FastAPI()
    app.include_router(studio_router, prefix='/api/inference')
    app.dependency_overrides[get_current_subject] = lambda: 'simulation'
    with TestClient(app, raise_server_exceptions=False) as client:
        with monkeypatch.context() as mp:
            def fail(*args, **kwargs): raise OSError(code, 'simulation failure')
            mp.setattr(Image if stage=='read' else Path, 'open' if stage=='read' else 'unlink', fail)
            response = client.delete('/api/inference/images/gallery/' + row['id'])
            assert response.status_code == 500, response.text
            assert path.exists()
        assert client.delete('/api/inference/images/gallery/' + row['id']).status_code == 200
        assert client.delete('/api/inference/images/gallery/' + row['id']).status_code == 404
