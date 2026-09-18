"""Playwright environment: the fly 'sees' the page and 'clicks' one of the internal links."""
from __future__ import annotations

import os
import re
from urllib.parse import urljoin, urlparse

import numpy as np

START = "https://betking.com.ua/casino/"
SESSION_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "session.txt")

# The fly never touches these: registration, login, money, account settings, legal pages.
# With a live session cookie this list is what keeps the fly a spectator - extend it, do not shrink it.
FORBIDDEN = re.compile(
    r"/(register|login|logout|cashout|deposit|payment|pay|withdraw|wallet|balance|self-exclusion|invite-friends"
    r"|agreement|profile|cabinet|account|settings|verification|bet-slip|place-bet|play|launch|online-game)(?![a-z0-9])",
    re.I)


def start_screencast(context, page, on_frame, max_width: int = 1024, max_height: int = 640, quality: int = 50):
    """Live video of the page for the visualiser: Chromium's own screencast (a JPEG per repaint, so
    reels spin at the page's frame rate instead of one screenshot per decision). Frames are delivered
    while the main thread is inside Playwright calls (wait_for_timeout etc.)."""
    import base64
    cdp = context.new_cdp_session(page)

    def on(params):
        try:
            on_frame(base64.b64decode(params["data"]))
        finally:
            try:
                cdp.send("Page.screencastFrameAck", {"sessionId": params["sessionId"]})
            except Exception:
                pass

    cdp.on("Page.screencastFrame", on)
    cdp.send("Page.startScreencast", {"format": "jpeg", "quality": quality, "maxWidth": max_width,
                                      "maxHeight": max_height, "everyNthFrame": 1})
    return cdp


def load_session_cookie(path: str = SESSION_FILE) -> str | None:
    """PHPSESSID value from $FLY_PHPSESSID or data/session.txt (one line). Never logged, never committed."""
    val = os.environ.get("FLY_PHPSESSID", "").strip()
    if not val and os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            val = f.read().strip()
    return val or None


class WebEnv:
    def __init__(self, start: str = START, headless: bool = True, max_actions: int = 32,
                 max_steps: int = 10, viewport=(1280, 800), session_cookie: str | None = None):
        from playwright.sync_api import sync_playwright
        self.start = start
        self.host = urlparse(start).netloc
        self.max_actions = max_actions
        self.max_steps = max_steps
        self._pw = sync_playwright().start()
        self.browser = self._pw.chromium.launch(headless=headless)
        self.context = self.browser.new_context(viewport={"width": viewport[0], "height": viewport[1]},
                                                locale="uk-UA")
        self.session = False
        if session_cookie:
            self.context.add_cookies([
                {"name": "PHPSESSID", "value": session_cookie, "domain": self.host, "path": "/",
                 "httpOnly": True, "secure": True, "sameSite": "Lax"},
                {"name": "PHPSESSID", "value": session_cookie, "domain": "." + self.host, "path": "/",
                 "httpOnly": True, "secure": True, "sameSite": "Lax"},
            ])
            self.session = True
        self.page = self.context.new_page()
        self.visited: set[str] = set()
        self.links: list[tuple[str, str]] = []
        self.boxes: list[list[int]] = []     # viewport bbox [x, y, w, h] per link
        self.step_i = 0
        self._on_frame = None
        self._cast = None

    @property
    def on_frame(self):
        return self._on_frame

    @on_frame.setter
    def on_frame(self, cb):
        """Set by the agent when a visualiser is attached: starts the live screencast."""
        self._on_frame = cb
        if cb and self._cast is None:
            try:
                self._cast = start_screencast(self.context, self.page, lambda jpg: self._on_frame and self._on_frame(jpg))
            except Exception as e:
                print(f"browser: screencast unavailable ({str(e)[:80]})")

    def idle(self, seconds: float):
        """Sleep while still pumping browser events (live frames keep flowing)."""
        self.page.wait_for_timeout(seconds * 1000)

    def _norm(self, href: str) -> str:
        u = urlparse(urljoin(self.page.url, href))
        return u._replace(fragment="", query="").geturl().rstrip("/") + "/"

    def _collect_links(self):
        raw = self.page.eval_on_selector_all(
            "a[href]",
            "els => els.map(e => { const r = e.getBoundingClientRect(); "
            "return [e.getAttribute('href'), (e.innerText || '').trim().slice(0, 40), "
            "[Math.round(r.x), Math.round(r.y), Math.round(r.width), Math.round(r.height)]] })")
        seen, links = set(), []
        self.boxes = []
        for href, text, box in raw:
            if not href or href.startswith(("javascript:", "mailto:", "tel:", "#")):
                continue
            url = self._norm(href)
            if urlparse(url).netloc != self.host or FORBIDDEN.search(url) or url in seen:
                continue
            seen.add(url)
            links.append((url, text))
            self.boxes.append(box)
        self.links = links[: self.max_actions]
        self.boxes = self.boxes[: self.max_actions]

    def _goto(self, url: str) -> bool:
        try:
            self.page.goto(url, wait_until="domcontentloaded", timeout=30000)
            self.page.wait_for_timeout(1500)
            return True
        except Exception:
            return False

    def reset(self) -> dict:
        self.visited.clear()
        self.step_i = 0
        self._goto(self.start)
        return self._observe()

    def _observe(self) -> dict:
        self._collect_links()
        self.visited.add(self._norm(self.page.url))
        mask = np.zeros(self.max_actions, bool)
        mask[: len(self.links)] = True
        try:
            title = self.page.title()
        except Exception:
            title = ""
        return {"png": self.page.screenshot(type="png"), "mask": mask, "url": self.page.url, "links": self.links,
                "boxes": self.boxes, "title": title, "session": self.session}

    def step(self, action: int) -> tuple[dict, float, bool]:
        """Reward stub: +1 new page, -0.2 already visited, -1 navigation failure."""
        self.step_i += 1
        url, _ = self.links[action]
        new = url not in self.visited
        ok = self._goto(url)
        reward = -1.0 if not ok else (1.0 if new else -0.2)
        obs = self._observe()
        done = self.step_i >= self.max_steps or not obs["mask"].any()
        return obs, reward, done

    def close(self):
        for f in (self.context.close, self.browser.close, self._pw.stop):
            try:
                f()
            except Exception:      # driver may already be gone (Ctrl+C / supervisor restart)
                pass
