"""Shared pytest configuration for tests/python/."""


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "server: heavyweight tests requiring studio venv"
    )
