"""Running CEWS on a schedule: collect, analyse, export, repeat.

* ``job_state``  - the lock that stops two cycles overlapping, and the persisted run history.
* ``jobs``       - one refresh cycle: fetch every enabled source, then re-run the analysis chain.
* ``scheduler``  - the interval loop (APScheduler) that runs a cycle every ``FETCH_INTERVAL_MINUTES``.

Nothing here decides *what* to compute; it only decides *when*, and makes sure a cycle that is
already running is never started a second time.
"""
