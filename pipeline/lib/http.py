"""Polite single-URL HTTP for the source list (implementation.md section 2).

This module never discovers links. There is no crawl queue, no sitemap reader
and no "follow related page" logic - one URL in, one response out. The only
URLs that are ever fetched are the rows of `sources.csv`.
"""

from __future__ import annotations

import time
import urllib.robotparser as robotparser
from urllib.parse import urlparse

import httpx

REQUEST_TIMEOUT = 20.0
_robots_cache: dict[str, robotparser.RobotFileParser | None] = {}


class FetchError(RuntimeError):
    """A source could not be fetched after every retry."""


class RobotsDisallowed(RuntimeError):
    """robots.txt forbids fetching this source."""


def _robots(url: str) -> robotparser.RobotFileParser | None:
    parts = urlparse(url)
    origin = f"{parts.scheme}://{parts.netloc}"
    if origin in _robots_cache:
        return _robots_cache[origin]

    # Fetch through httpx so the request has the same timeout as a page fetch.
    # RobotFileParser.read() calls urlopen() with no timeout and can block the
    # stage forever on a stalled connection.
    parser = robotparser.RobotFileParser()
    parser.set_url(f"{origin}/robots.txt")
    try:
        response = httpx.get(f"{origin}/robots.txt",
                             timeout=REQUEST_TIMEOUT, follow_redirects=True)
        if response.status_code == 200:
            parser.parse(response.text.splitlines())
        else:
            parser = None
    except httpx.HTTPError:
        parser = None      # unreachable robots.txt -> allow, but report it
    _robots_cache[origin] = parser
    return parser


def robots_verdict(url: str, user_agent: str) -> tuple[bool, str]:
    """Return (allowed, note). A missing robots.txt is treated as allowed."""
    parser = _robots(url)
    if parser is None:
        return True, "robots.txt unreachable - allowed by default"
    if not parser.can_fetch(user_agent, url):
        return False, f"robots.txt disallows {url}"
    return True, ""


def fetch(url: str, user_agent: str, delay: float,
          retries: int = 3, timeout: float = REQUEST_TIMEOUT) -> httpx.Response:
    """GET one URL. Sleeps `delay` seconds after every successful request."""
    last = ""
    for attempt in range(1, retries + 1):
        try:
            response = httpx.get(
                url,
                headers={
                    "User-Agent": user_agent,
                    "Accept": "text/html,application/xhtml+xml",
                    "Accept-Language": "en-IN,en;q=0.9",
                },
                timeout=timeout,
                follow_redirects=True,
            )
            if response.status_code == 200:
                time.sleep(delay)
                return response
            if response.status_code in (401, 403, 404, 410):
                raise FetchError(f"HTTP {response.status_code}: {url}")
            last = f"HTTP {response.status_code}"
        except httpx.HTTPError as exc:
            last = f"{type(exc).__name__}: {exc}"
        if attempt < retries:
            time.sleep(delay * (2 ** (attempt - 1)))
    raise FetchError(f"fetch failed after {retries} attempts: {url} ({last})")
