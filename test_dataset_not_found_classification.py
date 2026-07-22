import sys
from pathlib import Path

import pytest

pytest.importorskip("fastapi")

_BACKEND = Path(__file__).resolve().parent / "studio" / "backend"
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from fastapi import HTTPException  # noqa: E402

from models.datasets import DatasetSplitsRequest  # noqa: E402
import routes.datasets as rd  # noqa: E402


GENERIC_MSG = "Dataset 'owner/x' doesn't exist on the Hub or cannot be accessed."


def _not_found(msg):
    from datasets.exceptions import DatasetNotFoundError

    return DatasetNotFoundError(msg)


def _attach_cause(err, status_code):
    class _R:
        pass

    _R.status_code = status_code
    cause = Exception("underlying")
    cause.response = _R()
    err.__cause__ = cause
    return err


def test_generic_message_without_http_status_returns_404():
    err = _not_found(GENERIC_MSG)
    assert rd._classify_dataset_not_found_status(err) == 404


def test_generic_message_with_http_404_returns_404():
    err = _attach_cause(_not_found(GENERIC_MSG), 404)
    assert rd._classify_dataset_not_found_status(err) == 404


def test_generic_message_with_http_401_returns_403():
    err = _attach_cause(_not_found(GENERIC_MSG), 401)
    assert rd._classify_dataset_not_found_status(err) == 403


def test_generic_message_with_http_403_returns_403():
    err = _attach_cause(_not_found(GENERIC_MSG), 403)
    assert rd._classify_dataset_not_found_status(err) == 403


def test_http_status_overrides_gated_keyword_match():
    err = _attach_cause(
        _not_found("this is a gated dataset but the real status is 404"),
        404,
    )
    assert rd._classify_dataset_not_found_status(err) == 404


def test_gated_keyword_without_status_returns_403():
    err = _not_found("This is a gated dataset on the Hub.")
    assert rd._classify_dataset_not_found_status(err) == 403


def test_is_private_keyword_without_status_returns_403():
    err = _not_found("Dataset X is private.")
    assert rd._classify_dataset_not_found_status(err) == 403


def test_ask_for_access_keyword_without_status_returns_403():
    err = _not_found("You must ask for access to this dataset.")
    assert rd._classify_dataset_not_found_status(err) == 403


def test_must_be_authenticated_keyword_returns_403():
    err = _not_found("You must be authenticated to view this.")
    assert rd._classify_dataset_not_found_status(err) == 403


def test_outer_handler_misspelled_name_returns_404(
    monkeypatch, fake_datasets, fake_structlog
):
    import datasets as ds

    def _raise(*a, **k):
        raise _not_found(
            "Dataset 'misspelled/name' doesn't exist on the Hub or cannot be accessed."
        )

    monkeypatch.setattr(ds, "get_dataset_config_names", _raise)

    req = DatasetSplitsRequest(dataset_name = "misspelled/name")
    with pytest.raises(HTTPException) as ei:
        rd.get_dataset_splits(req, current_subject = "t")
    assert ei.value.status_code == 404
    assert "was not found" in ei.value.detail.lower()


def test_cannot_be_accessed_phrase_is_no_longer_a_gated_keyword():
    err = _not_found("cannot be accessed due to some other reason")
    assert rd._classify_dataset_not_found_status(err) == 404


def test_cannot_access_phrase_is_no_longer_a_gated_keyword():
    err = _not_found("the request cannot access the upstream system")
    assert rd._classify_dataset_not_found_status(err) == 404
