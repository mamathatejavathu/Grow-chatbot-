"""Run identifiers shared by the online stages.

The same `run_id` ties the retrieval trace, the generation record, and the
combined `artifacts/runs/<run_id>.json` together, so one question can be
replayed end to end.
"""

from __future__ import annotations

import datetime as dt
import uuid


def new_run_id() -> str:
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{uuid.uuid4().hex[:4]}"
