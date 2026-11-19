"""Time-machine checkpoints for a verdict.

The verdict for a claim can only change on a date when evidence for it was
published, so those dates are the stops of the workbench slider. Each stop is
re-verified with ``verify_claim(..., as_of_date=stop)``.
"""
from __future__ import annotations

from typing import Any, List


def checkpoints(v: Any, limit: int = 14) -> List[str]:
    """Distinct publication dates of the evidence behind a verdict (the slider stops).

    If there are more than ``limit`` dates, every contradicting date is kept plus
    evenly spaced supporting dates and the last date, so the moments when the
    verdict can flip are never skipped.
    """
    sup = sorted({e.pub_date[:10] for e in v.supporting if e.pub_date})
    con = sorted({e.pub_date[:10] for e in v.contradicting if e.pub_date})
    dates = sorted(set(sup) | set(con))
    if len(dates) <= limit:
        return dates
    keep = set(con) | {dates[0], dates[-1]}
    rest = [d for d in dates if d not in keep]
    slots = max(0, limit - len(keep))
    if slots and rest:
        step = len(rest) / slots
        keep.update(rest[int(i * step)] for i in range(slots))
    return sorted(keep)[: max(limit, len(con) + 2)]
