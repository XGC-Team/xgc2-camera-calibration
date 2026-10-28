#!/usr/bin/env python3
"""Require the released Python SDK 0.2.1 on the interpreter path (build script wheel, runtime image)."""
from importlib.metadata import version
from xgc2_xrpc.http import Context

assert version("xgc2-xrpc") == "0.2.1", "Python XRPC 0.2.1 is required on the interpreter path"
assert "query" in Context.__dataclass_fields__, "Python XRPC must expose discovery Context.query"
print("Python XRPC 0.2.1 native discovery API is available")
