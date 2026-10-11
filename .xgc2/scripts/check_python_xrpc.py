#!/usr/bin/env python3
"""Require the released native SDK supplied by the build/runtime image."""
from importlib.metadata import version
from xgc2_xrpc.http import Context

assert version("xgc2-xrpc") == "0.2.0", "Python XRPC 0.2.0 is required; update the XGC2 image"
assert "query" in Context.__dataclass_fields__, "Python XRPC must expose discovery Context.query"
print("Python XRPC 0.2.0 native discovery API is available")
