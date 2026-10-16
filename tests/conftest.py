"""Apply the narrowly scoped filters for known external warnings (f1podium.compat) to every test.

pytest wraps each test (setup, call, teardown) in its own warnings context with the command-line `-W error`. This
hook runs inside that context before any fixture is created (module-scoped fixtures included), so the filters cover
fixture setup too, take precedence over `-W error` only for the exact messages listed in compat, and are discarded
again when the test ends.
"""

import pytest

from f1podium.compat import silence_known_external_warnings


@pytest.hookimpl(wrapper=True)
def pytest_runtest_setup(item):
    silence_known_external_warnings()
    return (yield)
