# -*- coding: utf-8 -*-
"""让 tests/ 能直接 import scripts/ 下的脚本（scripts 不是包）。"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
