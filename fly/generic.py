"""Other providers: a game client we do not know (Pragmatic, NetGame, Playson ...).

The fly has to figure the game out by itself, so the generic driver gives it
  * a grid of click targets over the game area (plus Space, which spins most HTML5 slots) as actions,
  * whatever the client leaks about money on the wire as observation: XHR / fetch / WebSocket
    responses are sniffed for balance / win fields (`"balance":123.45`, `balance=12345`, `totalWin`,
    ...). The fly's own bank is charged with the balance drop it sees after an action (the stake) and
    credited with the win (or the balance rise).

Demo guard (do not shrink): the launch URL carries isMoney=false, real-money endpoints are aborted at
the network layer by CasinoEnv, and before every action at least one request of the game itself must
have shown a demo marker (demo / fun / free / practice / isMoney=false) while none showed a real-money
marker. A client that gives no such evidence is marked unplayable instead of being poked at.
"""
from __future__ import annotations

import json
import re
import threading
import time
from urllib.parse import urlparse

DEMO_MARK = re.compile(r"demo|fun|free|practice|trial|play_for_fun|ismoney=false|mode=demo|real=0", re.I)
REAL_MARK = re.compile(r"ismoney=true|mode=real|real=1|realmoney|/cashier|/deposit", re.I)
NUM = r'["\']?\s*[:=]\s*["\']?(-?\d+(?:[.,]\d+)?)'
BALANCE_RE = re.compile(r'(?<![a-z])(?:balance|credit|credits|bal|cash|money)(?:Cents|_cents)?' + NUM, re.I)
WIN_RE = re.compile(r'(?<![a-z])(?:totalWin|total_win|winAmount|win_amount|win|tw|roundWin|wonAmount)(?:Cents|_cents)?' + NUM, re.I)
CENTS_HINT = re.compile(r"cents", re.I)
GRID = (6, 4)          # click targets over the game area: columns x rows
SPLASH_WORDS = re.compile(r"грати|играть|play|start|continue|продовжити|ok|close|закрити|×", re.I)


def _num(s: str) -> float | None:
    try:
        return float(s.replace(",", "."))
    except ValueError:
        return None


class NetSniffer:
    """Reads money-like fields out of the game's network traffic (responses + websocket frames)."""

    def __init__(self, page, site_host: str = "", ignore_hosts: tuple[str, ...] = ("google", "gemius", "doubleclick",
                                                                                   "facebook", "tiktok", "gstatic")):
        self.page = page
        self.site = site_host                    # the casino site itself: its widgets talk about money too
        self.ignore = ignore_hosts
        self.lock = threading.Lock()
        self.balance: float | None = None
        self.win: float | None = None
        self.balance_at = 0.0
        self.win_at = 0.0
        self.demo_evidence = 0
        self.real_evidence = 0
        self.hosts: dict[str, int] = {}
        self.samples: list[str] = []
        page.on("response", self._on_response)
        page.on("websocket", self._on_ws)

    # ---- hooks --------------------------------------------------------------------------------
    def _relevant(self, url: str) -> bool:
        host = urlparse(url).netloc
        return url.startswith("http") and not any(h in host for h in self.ignore)

    def _on_response(self, resp):
        try:
            url = resp.url
            if not self._relevant(url):
                return
            self._marks(url)
            if self.site and urlparse(url).netloc.endswith(self.site):
                return                            # site pages / scripts / activity feeds: not the game
            ctype = (resp.headers.get("content-type") or "").lower()
            if "javascript" in ctype or url.endswith(".js"):
                return                            # code, not state
            if not any(t in ctype for t in ("json", "text", "xml", "x-www-form")):
                return
            if int(resp.headers.get("content-length") or 0) > 200_000:
                return
            body = resp.text()
            if body:
                self._scan(body[:200_000], url)
        except Exception:
            pass

    def _on_ws(self, ws):
        if not self._relevant(ws.url):
            return
        self._marks(ws.url)
        if self.site and urlparse(ws.url).netloc.endswith(self.site):
            return
        ws.on("framereceived", lambda p: self._scan(p if isinstance(p, str) else p.decode("utf-8", "replace"), ws.url))

    def _marks(self, url: str):
        with self.lock:
            self.hosts[urlparse(url).netloc] = self.hosts.get(urlparse(url).netloc, 0) + 1
            if REAL_MARK.search(url):
                self.real_evidence += 1
            elif DEMO_MARK.search(url):
                self.demo_evidence += 1

    def _scan(self, text: str, url: str):
        if len(text) > 200_000:
            return
        bal = BALANCE_RE.search(text)
        win = WIN_RE.search(text)
        if not bal and not win:
            return
        now = time.time()
        with self.lock:
            if bal:
                v = _num(bal.group(1))
                if v is not None:
                    if CENTS_HINT.search(bal.group(0)) or (v > 10_000_000 and "." not in bal.group(1)):
                        v /= 100.0
                    self.balance, self.balance_at = v, now
            if win:
                v = _num(win.group(1))
                if v is not None:
                    if CENTS_HINT.search(win.group(0)):
                        v /= 100.0
                    self.win, self.win_at = v, now
            if len(self.samples) < 40:
                u = urlparse(url)
                self.samples.append(f"{u.netloc}{u.path[:50]}: {bal and bal.group(0)} {win and win.group(0)}")

    # ---- api ----------------------------------------------------------------------------------
    def reset(self):
        with self.lock:
            self.balance = self.win = None
            self.balance_at = self.win_at = 0.0
            self.demo_evidence = self.real_evidence = 0
            self.hosts.clear()
            self.samples.clear()

    def snapshot(self) -> dict:
        with self.lock:
            return {"balance": self.balance, "win": self.win, "balance_at": self.balance_at, "win_at": self.win_at,
                    "demo": self.demo_evidence, "real": self.real_evidence}

    @property
    def is_demo(self) -> bool:
        with self.lock:
            return self.demo_evidence > 0 and self.real_evidence == 0


class GenericGame:
    """Actions over an unknown client: a grid of click points inside the game box + Space."""

    def __init__(self, page, box: dict, sniffer: NetSniffer):
        self.page, self.box, self.sniffer = page, box, sniffer
        cols, rows = GRID
        self.targets: list[tuple[str, list[int], str]] = []       # (label, [x, y, w, h], kind)
        w, h = box["width"] / cols, box["height"] / rows
        for r in range(rows):
            for c in range(cols):
                x, y = box["x"] + c * w, box["y"] + r * h
                self.targets.append((f"tap {c + 1}.{r + 1}", [int(x), int(y), int(w), int(h)], "click"))
        self.targets.append(("space", [int(box["x"] + box["width"] - 60), int(box["y"] + box["height"] - 30), 50, 20], "key"))

    def links(self) -> list[tuple[str, str]]:
        return [(f"game:{lab}", "Space key" if kind == "key" else lab) for lab, _, kind in self.targets]

    def prior(self) -> list[float]:
        """Log-prior: spin controls live bottom-right in nearly every slot; Space spins most of them."""
        cols, rows = GRID
        out = []
        for lab, _, kind in self.targets:
            if kind == "key":
                out.append(1.0)
                continue
            c, r = (int(x) for x in lab.split()[1].split("."))
            out.append(0.5 * (r == rows) + 0.3 * (c >= cols - 1) + 0.2 * (r == rows - 1))
        return out

    def boxes(self) -> list[list[int]]:
        return [b for _, b, _ in self.targets]

    def act(self, i: int):
        lab, b, kind = self.targets[min(i, len(self.targets) - 1)]
        if kind == "key":
            self.page.keyboard.press("Space")
        else:
            self.page.mouse.click(b[0] + b[2] / 2, b[1] + b[3] / 2)

    def dismiss_splash(self, frames) -> bool:
        """Splash screens ("PLAY", "continue", a close cross): click the first such button we can see."""
        for f in frames:
            try:
                r = f.evaluate("""() => { for (const el of document.querySelectorAll('button, [role=button], a, div, span')) {
                    const t = (el.innerText || el.getAttribute('aria-label') || '').trim(); if (!t || t.length > 20) continue;
                    if (!/^(грати|играть|play|start|continue|продовжити|ok|закрити|close|×)$/i.test(t)) continue;
                    const r = el.getBoundingClientRect(); if (r.width < 8 || r.height < 8) continue;
                    return [r.x + r.width / 2, r.y + r.height / 2]; } return null; }""")
            except Exception:
                r = None
            if r:
                try:
                    fe = f.frame_element().bounding_box() if f != f.page.main_frame else {"x": 0, "y": 0}
                except Exception:
                    fe = {"x": 0, "y": 0}
                self.page.mouse.click(r[0] + fe["x"], r[1] + fe["y"])
                return True
        return False


def money_delta(before: dict, after: dict) -> tuple[float | None, float | None]:
    """(stake, win) inferred from two sniffer snapshots: the balance drop = stake, a fresh win field or
    a balance rise = win. None = nothing observable happened."""
    b0, b1 = before.get("balance"), after.get("balance")
    stake = win = None
    if b0 is not None and b1 is not None and after.get("balance_at", 0) > before.get("balance_at", 0):
        if b1 < b0:
            stake = round(b0 - b1, 2)
        elif b1 > b0:
            win = round(b1 - b0, 2)
    if after.get("win") and after.get("win_at", 0) > before.get("win_at", 0):
        win = max(win or 0.0, float(after["win"]))
    return stake, win


def summarize(sniffer: NetSniffer) -> str:
    s = sniffer.snapshot()
    hosts = sorted(sniffer.hosts, key=lambda h: -sniffer.hosts[h])[:6]
    return (f"balance={s['balance']} win={s['win']} demo_marks={s['demo']} real_marks={s['real']} hosts={hosts}"
            + "".join("\n   " + x for x in sniffer.samples[:12]))


if __name__ == "__main__":     # python -m fly.generic <slug> : open a demo game, sniff, click PLAY, press Space once
    import sys
    from playwright.sync_api import sync_playwright
    from .browser import SESSION_FILE, load_session_cookie
    slug = sys.argv[1] if len(sys.argv) > 1 else "leopatra-bonus-combo"
    cookie = load_session_cookie(SESSION_FILE)
    with sync_playwright() as p:
        b = p.chromium.launch(headless=True, channel="chromium")
        ctx = b.new_context(viewport={"width": 1280, "height": 800}, locale="uk-UA")
        if cookie:
            ctx.add_cookies([{"name": "PHPSESSID", "value": cookie, "domain": "betking.com.ua", "path": "/",
                              "httpOnly": True, "secure": True, "sameSite": "Lax"}])
        pg = ctx.new_page()
        sn = NetSniffer(pg, "betking.com.ua")
        pg.goto(f"https://betking.com.ua/casino/?game={slug}&isMoney=false", wait_until="domcontentloaded", timeout=60000)
        pg.wait_for_timeout(12000)
        print("after load:", summarize(sn))
        g = GenericGame(pg, {"x": 110, "y": 125, "width": 1060, "height": 595}, sn)
        print("splash:", g.dismiss_splash(pg.frames))
        pg.mouse.click(640, 650)          # where NetGame draws its PLAY button
        pg.wait_for_timeout(8000)
        print("after splash:", summarize(sn))
        before = sn.snapshot()
        pg.keyboard.press("Space")
        pg.wait_for_timeout(9000)
        print("after space:", summarize(sn), "delta:", money_delta(before, sn.snapshot()))
        b.close()
