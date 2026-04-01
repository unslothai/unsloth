"""Shared pytest configuration for tests/python/."""


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "server: heavyweight tests requiring studio venv"
    )
    config.addinivalue_line(
        "markers", "e2e: end-to-end tests requiring network and venv creation"
    )
