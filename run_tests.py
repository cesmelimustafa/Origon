"""Pytest runner with CPython 3.10.0 bpo-46404 frozen-slot dataclass compatibility."""

import sys
import _pytest.nodeid
import pytest


def _safe_nodeid_str(self: _pytest.nodeid.NodeId) -> str:
    try:
        cached = object.__getattribute__(self, "_str_cache")
        if cached is not None:
            return cached
    except AttributeError:
        pass
    s = "::".join((self.path, *self.names))
    if self.params is not None:
        s += f"[{self.params}]"
    object.__setattr__(self, "_str_cache", s)
    return s


_pytest.nodeid.NodeId.__str__ = _safe_nodeid_str

if __name__ == "__main__":
    args = sys.argv[1:] or ["tests", "-v"]
    sys.exit(pytest.main(args))
