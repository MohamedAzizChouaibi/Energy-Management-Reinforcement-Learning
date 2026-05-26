"""Shared Day 4 configuration: credentials, cache dir, and HTTP session.

Keys are read from the git-ignored ``.env`` at the project root (see
``.env.example``). Nothing here ever logs a key value.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

try:
    from dotenv import load_dotenv
except ImportError as exc:  # pragma: no cover - dependency is in requirements.txt
    raise SystemExit(
        "python-dotenv is required for Day 4. Install with:\n"
        "    pfa/bin/pip install python-dotenv"
    ) from exc

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = PROJECT_ROOT / "gps" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Make ``import gps...`` work when a module in this package is run as a script.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Load .env once, from the project root explicitly (find_dotenv() is unreliable
# under heredocs / non-file frames).
load_dotenv(PROJECT_ROOT / ".env")


def get_key(name: str) -> str:
    """Return an API key from the environment or fail with guidance."""
    value = os.environ.get(name, "").strip()
    if not value:
        raise SystemExit(
            f"Missing {name}. Copy .env.example to .env and set your key:\n"
            f"    cp .env.example .env  # then edit {name}=..."
        )
    return value


def session(total_retries: int = 3, backoff: float = 0.5) -> requests.Session:
    """A requests session with sane retries for flaky public APIs."""
    sess = requests.Session()
    retry = Retry(
        total=total_retries,
        backoff_factor=backoff,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    sess.mount("https://", adapter)
    sess.mount("http://", adapter)
    sess.headers.update({"User-Agent": "ths-ems-day4/1.0"})
    return sess
