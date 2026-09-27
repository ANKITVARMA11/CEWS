"""Optional AI layers: embeddings-based topic discovery and announcement extraction.

Every feature here is off by default (``ENABLE_AI_*`` in ``.env``) and has a deterministic
fallback that runs when it is off, when a dependency is missing, or when a call fails: keyword
rules for announcements, plain keyword matching for topics. Nothing in the scoring or forecasting
pipeline depends on anything in this package being installed or enabled.

Organization matching (``org_matching.py``) is not implemented yet.
"""
