"""Shared helpers for the internal-auth guard and per-handler gates.

The blanket guard in ``app/__init__.py`` protects ``/api/*`` with
``MIROSHARK_INTERNAL_KEY``. A small set of spectator endpoints are exempt
from the blanket check (browsers' ``EventSource`` cannot send custom
headers) and instead enforce their own ``is_public`` gates — but those
gates must still recognise an operator carrying the key, so keyed callers
keep full access to private sims.
"""

import hmac
import os

from flask import request


def request_has_internal_key() -> bool:
    """True iff the current request carries the valid internal key."""
    internal_key = os.environ.get('MIROSHARK_INTERNAL_KEY')
    if not internal_key:
        return False
    provided = request.headers.get('x-miroshark-internal-key')
    return bool(provided) and hmac.compare_digest(provided, internal_key)


def caller_may_view_private() -> bool:
    """May the current caller see private-sim data on spectator endpoints?

    - Key configured: only requests carrying it (operator traffic).
    - No key configured: mirror the blanket guard's posture — open in
      local dev (``Config.DEBUG``, where the whole API is open anyway,
      including the SPA's own run-status polling), closed in deployed /
      non-debug environments (spectators get the ``is_public`` surface
      instead of the blanket 503).
    """
    if os.environ.get('MIROSHARK_INTERNAL_KEY'):
        return request_has_internal_key()
    from ..config import Config
    return bool(Config.DEBUG)
