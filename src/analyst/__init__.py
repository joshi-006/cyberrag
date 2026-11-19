"""Analyst-facing tools built on top of the verification core.

Everything here is deterministic or delegates to ``VerificationService``; nothing
in this package calls an LLM or changes the baseline corpus on disk.
"""
