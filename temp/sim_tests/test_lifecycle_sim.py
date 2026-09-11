import gc
import random
import threading
import time
import weakref

import pytest
from core.inference import diffusion as mod
from core.inference import diffusion_eager_patches as ep
from test_diffusion_backend import _FakePipeline, _FakeTransformer, _load_into

PHASES = ['transformer', 'pipeline', 'speed', 'quantize', 'placement', 'publication']


def hook_target(phase):
    return {
        'transformer': (_FakeTransformer, 'from_single_file'),
        'pipeline': (_FakePipeline, 'from_pretrained'),
        'speed': (mod, 'apply_speed_optims'),
        'quantize': (mod, 'quantize_text_encoders'),
        'placement': (mod, 'apply_memory_plan'),
        'publication': (mod, '_LoadState'),
    }[phase]


@pytest.mark.parametrize('phase', PHASES)
@pytest.mark.parametrize('ejectors', [1, 2, 4])
@pytest.mark.parametrize('seed', range(8))
def test_cancel_interleavings(fake_runtime, monkeypatch, tmp_path, phase, ejectors, seed):
    (tmp_path / 'model.gguf').write_bytes(b'weights')
    backend = mod.DiffusionBackend()
    entered, release = threading.Event(), threading.Event()
    results, errors = [], []
    target, method = hook_target(phase)
    original = getattr(target, method)
    def parked(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        return original(*args, **kwargs)
    def load():
        try:
            results.append(_load_into(backend, tmp_path, speed_mode='eager'))
        except Exception as exc:
            errors.append(str(exc))
    def unload():
        try:
            backend.unload()
        except BaseException as exc:
            errors.append('eject: ' + str(exc))
    with monkeypatch.context() as mp:
        mp.setattr(target, method, parked)
        loader = threading.Thread(target=load, daemon=True)
        workers = [threading.Thread(target=unload, daemon=True) for _ in range(ejectors)]
        loader.start()
        try:
            assert entered.wait(5)
            for worker in workers:
                worker.start()
            assert backend._cancel_event.wait(.3), 'cancellation blocked by construction'
            random.Random(seed).shuffle(workers)
            time.sleep(random.Random(seed).uniform(0, .002))
        finally:
            release.set()
            loader.join(5)
            for worker in workers:
                if worker.ident:
                    worker.join(5)
    assert not loader.is_alive()
    assert all(not worker.is_alive() for worker in workers)
    assert not results, 'cancelled load reported ready'
    assert len(errors) == 1 and 'cancelled' in errors[0], errors
    assert not backend.is_loaded and backend._teardown_waiters == 0
    assert not backend._transition_owns_slot and not ep.is_installed()
    _load_into(backend, tmp_path, speed_mode='off')
    state = backend._state
    for _ in range(2):
        assert len(backend.generate(prompt='recovery', steps=2)['images']) == 1
        assert backend._state is state
    backend.unload()


@pytest.mark.parametrize('phase', PHASES)
def test_constructor_failure_rolls_back(fake_runtime, monkeypatch, tmp_path, phase):
    (tmp_path / 'model.gguf').write_bytes(b'weights')
    backend = mod.DiffusionBackend()
    target, method = hook_target(phase)
    with monkeypatch.context() as mp:
        def fail(*args, **kwargs):
            raise RuntimeError('injected constructor failure')
        mp.setattr(target, method, fail)
        with pytest.raises(RuntimeError, match='injected constructor failure'):
            _load_into(backend, tmp_path, speed_mode='eager')
    assert backend._state is None and backend._teardown_waiters == 0
    assert not ep.is_installed()
    _load_into(backend, tmp_path, speed_mode='off')
    backend.unload()


def stage_background(monkeypatch, backend):
    monkeypatch.setattr(backend, '_estimate_download_bytes', lambda *a, **k: (0, []))
    monkeypatch.setattr(backend, '_te_prequant_plan_files', lambda *a, **k: {})
    monkeypatch.setattr(backend, 'declared_footprint_shortfall', lambda *a, **k: None)
    monkeypatch.setattr(mod, '_assert_base_repo_accessible', lambda *a, **k: None)


def test_superseded_worker_cannot_overwrite_current_load(fake_runtime, monkeypatch, tmp_path):
    old, new = tmp_path / 'old', tmp_path / 'new'
    for folder in (old, new):
        folder.mkdir()
        (folder / 'model.gguf').write_bytes(b'weights')
    backend = mod.DiffusionBackend()
    stage_background(monkeypatch, backend)
    entered, release, old_done, new_done = (threading.Event() for _ in range(4))
    run = backend._run_load
    def run_tracked(**kwargs):
        try:
            run(**kwargs)
        finally:
            (old_done if kwargs['repo_id'] == str(old) else new_done).set()
    def prefetch(repo_id, *args, **kwargs):
        if repo_id == str(old):
            entered.set()
            assert release.wait(5)
            raise RuntimeError('late old-worker failure')
        return None
    monkeypatch.setattr(backend, '_run_load', run_tracked)
    monkeypatch.setattr(backend, '_prefetch_files', prefetch)
    opts = dict(gguf_filename='model.gguf', family_override='z-image', base_repo='base/repo', speed_mode='off')
    backend.begin_load(str(old), **opts)
    try:
        assert entered.wait(5)
        old_cancel = backend._cancel_event
        backend.unload()
        backend.begin_load(str(new), **opts)
        assert new_done.wait(5)
        assert backend.status()['repo_id'] == str(new)
        assert not backend._cancel_event.is_set() and old_cancel.is_set()
    finally:
        release.set()
        assert old_done.wait(5)
    assert backend.status()['repo_id'] == str(new)
    assert backend.load_progress()['phase'] == 'ready'
    backend.unload()


@pytest.mark.parametrize('seed', range(20))
def test_repeated_load_generate_eject(fake_runtime, tmp_path, seed):
    (tmp_path / 'model.gguf').write_bytes(b'weights')
    backend = mod.DiffusionBackend()
    refs = []
    randomizer = random.Random(seed)
    for _ in range(10):
        _load_into(backend, tmp_path, speed_mode='off')
        refs.append(weakref.ref(backend._state.pipe))
        for _ in range(randomizer.randrange(4)):
            assert len(backend.generate(prompt='repeat', steps=2)['images']) == 1
        backend.unload()
        assert backend._state is None and backend._teardown_waiters == 0
        assert not backend._queued_generate_cancels
    gc.collect()
    assert all(ref() is None for ref in refs), 'pipeline retained after eject'
