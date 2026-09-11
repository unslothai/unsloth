import errno
import os
from pathlib import Path
import threading

from hypothesis import given, settings, strategies as st
from PIL import Image
import pytest
from core.inference import image_gallery as gallery


def record():
    return gallery.save(Image.new('RGB', (16,16)), dict(prompt='simulation', width=16, height=16,
                        steps=9, guidance=0., seed=1, model='test', created_at=1.))


@pytest.mark.parametrize('code', [errno.EACCES, errno.EPERM, errno.EIO, errno.EROFS, errno.EBUSY, errno.ENOSPC])
@pytest.mark.parametrize('stage', ['read', 'unlink'])
def test_filesystem_errors_remain_errors(monkeypatch, stage, code):
    row = record()
    path = gallery.image_path(row['id'])
    failure = OSError(code, os.strerror(code))
    if stage == 'read':
        def fail(*args, **kwargs):
            raise failure
        monkeypatch.setattr(Image, 'open', fail)
    else:
        def fail(self, *args, **kwargs):
            raise failure
        monkeypatch.setattr(type(path), 'unlink', fail)
    with pytest.raises(OSError):
        gallery.delete(row['id'])
    assert path.exists()


def test_windows_sharing_violation(monkeypatch):
    row = record()
    path = gallery.image_path(row['id'])
    failure = PermissionError(errno.EACCES, 'The process cannot access the file because it is being used by another process')
    failure.winerror = 32
    def fail(self, *args, **kwargs):
        raise failure
    monkeypatch.setattr(type(path), 'unlink', fail)
    with pytest.raises(PermissionError):
        gallery.delete(row['id'])
    assert path.exists()


@pytest.mark.parametrize('phase', ['before_lookup', 'before_unlink'])
def test_external_deletion_races(monkeypatch, phase):
    row = record()
    path = gallery.image_path(row['id'])
    if phase == 'before_lookup':
        path.unlink()
    else:
        unlink = Path.unlink
        def race(self, *args, **kwargs):
            unlink(self)
            return unlink(self)
        monkeypatch.setattr(Path, 'unlink', race)
    assert gallery.delete(row['id']) is False
    assert not path.exists()


@pytest.mark.parametrize('workers', [2,4,16])
def test_concurrent_deletes(workers):
    row = record()
    results, errors = [], []
    barrier = threading.Barrier(workers)
    def delete():
        try:
            barrier.wait(5)
            results.append(gallery.delete(row['id']))
        except Exception as exc:
            errors.append(str(exc))
    threads = [threading.Thread(target=delete, daemon=True) for _ in range(workers)]
    for thread in threads: thread.start()
    for thread in threads: thread.join(5)
    assert not errors and all(not thread.is_alive() for thread in threads)
    assert sum(results) == 1
    assert gallery.list_images() == []


@settings(max_examples=400, derandomize=True)
@given(st.text(min_size=0, max_size=160))
def test_unknown_ids_cannot_delete_files(image_id):
    assert gallery.delete(image_id) is False


@pytest.mark.parametrize('payload', [b'', b'broken png', b'\x89PNG\r\n\x1a\n', b'{}'])
def test_unreadable_foreign_images_are_preserved(payload):
    path = gallery.gallery_dir() / 'foreign.png'
    path.write_bytes(payload)
    assert gallery.delete('foreign') is False
    assert path.read_bytes() == payload


def test_archive_pin_and_delete_order():
    first, second = record(), record()
    gallery.set_flags(first['id'], pinned=True, archived=True)
    assert [r['id'] for r in gallery.list_images()] == [second['id']]
    assert gallery.delete(first['id']) is True
    assert gallery.list_images(archived=True) == []
    assert gallery.delete(second['id']) is True
    assert gallery.list_images() == []
