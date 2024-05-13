def pytest_make_parametrize_id(config, val, argname):
    return f"{argname}={val}"
