"""Read-only views of the database for the dashboard, the API and the exports.

Nothing here computes anything: every number has already been produced by the collection,
scoring and forecasting passes and stored with its reasoning. These functions only fetch and
shape it, so the dashboard cannot quietly invent a figure that the pipeline never produced.
"""
