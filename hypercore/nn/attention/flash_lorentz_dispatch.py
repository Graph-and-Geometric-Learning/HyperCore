"""Version selector for flash Lorentz attention.

Routes to v1 (memory-optimal, ~20x mem win, slower) or v2 (hybrid:
saves P from forward in _bwd_kv, ~1.26x faster bwd, ~3-5x mem win).

Default: v2. Set FLASH_LORENTZ_VERSION='v1' to revert.
"""

import os

from .flash_lorentz_attention import flash_attention_core as _core_v1
from .flash_lorentz_attention_v2 import flash_attention_core_v2 as _core_v2

_DEFAULT = os.environ.get("FLASH_LORENTZ_VERSION", "v2").lower()


def flash_attention_core(q, k, v, c, scale, mask=None, version=None):
    v_ = (version or _DEFAULT).lower()
    if v_ == "v1":
        return _core_v1(q, k, v, c, scale, mask=mask)
    if v_ == "v2":
        return _core_v2(q, k, v, c, scale, mask=mask)
    raise ValueError(f"unknown flash lorentz version: {v_}")
