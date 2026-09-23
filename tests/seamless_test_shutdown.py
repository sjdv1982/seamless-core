"""Pytest plugin that closes Seamless before each isolated test process exits."""


def pytest_sessionfinish(session, exitstatus):
    try:
        import seamless
    except ImportError:
        return

    seamless.close()
