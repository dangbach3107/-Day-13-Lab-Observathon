"""Minimal wrapper for debugging — pure passthrough, no imports."""
from __future__ import annotations


def mitigate(call_next, question, config, context):
    result = call_next(question, config)
    return result
