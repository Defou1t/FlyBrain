"""Casino environment: the fly plays a slot on the DEMO account (isMoney=false, play-money "FUN" credits).

Actions are the game's own bet buttons (in Amusnet/EGT games a click on a bet value places that bet
and spins) plus "spin again" (Space). Reward = balance change / bet, clipped to [-1, 3]:
losing the bet = -1 -> PPL101 (aversive dopamine); a win = + -> PAM cluster (appetitive dopamine).
Every step is appended to data/dopamine.csv.

Hard guards (do not remove):
  * the game URL must contain isMoney=false, any request with isMoney=true or to cashier/deposit/
    withdraw/payment endpoints is aborted at the network layer;
  * before the first action the game frame must show the play-money currency "FUN",
    otherwise the environment refuses to act.
"""
from __future__ import annotations

import csv
import json
import os
import re
import time
from collections import deque
from urllib.parse import urlparse

import numpy as np

from .browser import SESSION_FILE, load_session_cookie

DEMO_GAME = "https://betking.com.ua/casino/?game=olympus-glory-buy-bonus&isMoney=false"
DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
LOG = os.path.join(DATA, "dopamine.csv")
MONEY = re.compile(r"isMoney=true|/(cashier|deposit|withdraw|payment|pay|balance/(add|topup))\b", re.I)
NUM = r"(\d[\d\s ]*[.,]\d{2})"
BAL_RE = re.compile(r"ЗАЛИШОК\s*СУМИ:?\s*" + NUM, re.I)
WIN_RE = re.compile(r"ОСТАНН[ІI]Й\s*ВИГРАШ:?\s*" + NUM, re.I)
BET_BTN_JS = """
() => {
  const out = [];
  for (const el of document.querySelectorAll('*')) {
    if (el.children.length > 4) continue;
    const t = (el.innerText || '').replace(/\\s+/g, ' ').trim();
    const m = t.match(/^FUN\\s+(\\d+[.,]\\d+)\\s+СТАВКА$/i);
    if (!m) continue;
    const r = el.getBoundingClientRect();
    if (r.width < 10 || r.height < 10) continue;
    if (out.some(o => Math.abs(o.x - r.x) < 4 && Math.abs(o.y - r.y) < 4)) continue;   // nested duplicates
    out.push({ value: parseFloat(m[1].replace(',', '.')), x: r.x, y: r.y, w: r.width, h: r.height });
  }
  return out;
}
"""


def _num(s: str) -> float:
    return float(re.sub(r"[\s ]", "", s).replace(",", "."))


class CasinoEnv:
    def __init__(self, game: str = DEMO_GAME, headless: bool = True, max_actions: int = 32, max_steps: int = 20,
                 viewport=(1280, 800), session_cookie: str | None = None, frame_hint: str = "amusnet",
                 log_path: str = LOG, spin_timeout: float = 15.0):
        if "ismoney=false" not in game.lower():
            raise ValueError("casino env only runs demo games: the URL must contain isMoney=false")
        from playwright.sync_api import sync_playwright
        self.game = game
        self.host = urlparse(game).netloc
        self.max_actions = max_actions
        self.max_steps = max_steps
        self.frame_hint = frame_hint
        self.spin_timeout = spin_timeout
        self.log_path = log_path
        self._pw = sync_playwright().start()
        self.browser = self._pw.chromium.launch(headless=headless)
        self.context = self.browser.new_context(viewport={"width": viewport[0], "height": viewport[1]},
                                                locale="uk-UA")
        self.context.route("**/*", self._guard)
        self.session = False
        if session_cookie:
            for dom in (self.host, "." + self.host):
                self.context.add_cookies([{"name": "PHPSESSID", "value": session_cookie, "domain": dom, "path": "/",
                                           "httpOnly": True, "secure": True, "sameSite": "Lax"}])
            self.session = True
        self.page = self.context.new_page()
        self.page.on("websocket", self._on_ws)
        self.ws_log: deque = deque(maxlen=300)
        self.ws_balance: float | None = None
        self.ws_win: float | None = None
        self.balance: float | None = None
        self.win: float | None = None
        self.bet: float | None = None
        self.delta = 0.0
        self.step_i = 0
        self.episode = 0
        self.cum_reward = 0.0
        self.links: list[tuple[str, str]] = []
        self.boxes: list[list[int]] = []
        self._buttons: list[dict] = []
        os.makedirs(DATA, exist_ok=True)
        if not os.path.exists(log_path):
            with open(log_path, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(["ts", "episode", "step", "action", "bet", "balance_before", "balance_after",
                                        "delta", "reward", "dopamine_pam", "dopamine_ppl1", "cum_reward"])

    # ---- guards -------------------------------------------------------------------------------
    @staticmethod
    def _guard(route, request):
        if MONEY.search(request.url):
            route.abort()
        else:
            route.continue_()

    def _demo_check(self, frame):
        if "ismoney=true" in (frame.url + self.page.url).lower():
            raise RuntimeError("real-money mode detected - refusing to act")
        text = self._frame_text(frame)
        if "FUN" not in text:
            raise RuntimeError("play-money currency 'FUN' not visible in the game - refusing to act")

    # ---- websocket / dom readers --------------------------------------------------------------
    def _on_ws(self, ws):
        if self.frame_hint and self.frame_hint not in ws.url:
            return
        ws.on("framereceived", self._ws_frame)

    def _ws_frame(self, payload):
        s = payload if isinstance(payload, str) else payload.decode("utf-8", "replace")
        if s in ("ping", "pong"):
            return
        self.ws_log.append(s[:2000])
        try:
            obj = json.loads(s)
        except Exception:
            return
        found = {}

        def walk(o, depth=0):
            if depth > 6:
                return
            if isinstance(o, dict):
                for k, v in o.items():
                    kl = k.lower()
                    if isinstance(v, (int, float)) and not isinstance(v, bool):
                        if "balance" in kl or kl in ("credit", "credits"):
                            found["balance"] = float(v)
                        elif "win" in kl and "line" not in kl:
                            found["win"] = max(found.get("win", 0.0), float(v))
                    else:
                        walk(v, depth + 1)
            elif isinstance(o, list):
                for v in o:
                    walk(v, depth + 1)

        walk(obj)
        if "balance" in found:
            self.ws_balance = found["balance"]
        if "win" in found:
            self.ws_win = found["win"]

    def game_frame(self):
        cands = [f for f in self.page.frames if self.frame_hint in f.url]
        if cands:
            return cands[0]
        for f in self.page.frames:   # fallback: any cross-origin iframe
            if f.url.startswith("http") and urlparse(f.url).netloc != self.host:
                return f
        return None

    @staticmethod
    def _frame_text(frame) -> str:
        try:
            return re.sub(r"[ \t\r\n]+", " ", frame.evaluate("() => document.body.innerText") or "")
        except Exception:
            return ""

    def _read(self) -> tuple[float | None, float | None]:
        """(balance, last win) from the game DOM; falls back to values seen on the websocket."""
        frame = self.game_frame()
        bal = win = None
        if frame is not None:
            text = self._frame_text(frame)
            m = BAL_RE.search(text)
            if m:
                bal = _num(m.group(1))
            m = WIN_RE.search(text)
            if m:
                win = _num(m.group(1))
        if bal is None and self.ws_balance is not None:
            bal = self.ws_balance
        if win is None and self.ws_win is not None:
            win = self.ws_win
        return bal, win

    def _bet_buttons(self) -> list[dict]:
        frame = self.game_frame()
        if frame is None:
            return []
        try:
            btns = frame.evaluate(BET_BTN_JS)
            box = frame.frame_element().bounding_box() or {"x": 0, "y": 0}
        except Exception:
            return []
        for b in btns:
            b["x"] += box["x"]
            b["y"] += box["y"]
        return sorted(btns, key=lambda b: b["value"])

    def _spin_box(self) -> list[int]:
        frame = self.game_frame()
        try:
            b = frame.frame_element().bounding_box()
            return [int(b["x"] + b["width"] * 0.91), int(b["y"] + b["height"] * 0.40), int(b["width"] * 0.06),
                    int(b["height"] * 0.12)]
        except Exception:
            return [0, 0, 0, 0]

    # ---- gym-like api -------------------------------------------------------------------------
    def reset(self) -> dict:
        self.episode += 1
        self.step_i = 0
        self.page.goto(self.game, wait_until="domcontentloaded", timeout=60000)
        self.page.wait_for_timeout(9000)
        self.page.keyboard.press("Escape")            # registration / promo modals
        self.page.wait_for_timeout(1500)
        frame = self.game_frame()
        if frame is None:
            raise RuntimeError("game iframe not found")
        self._demo_check(frame)
        self.balance, self.win = self._read()
        self.delta = 0.0
        return self._observe()

    def _observe(self) -> dict:
        self._buttons = self._bet_buttons()[: self.max_actions - 1]
        self.links = [("spin", "спин (та же ставка)")] + [(f"bet:{b['value']:.2f}", f"ставка {b['value']:.2f}")
                                                          for b in self._buttons]
        self.boxes = [self._spin_box()] + [[int(b["x"]), int(b["y"]), int(b["w"]), int(b["h"])] for b in self._buttons]
        mask = np.zeros(self.max_actions, bool)
        mask[: len(self.links)] = True
        if self.bet is None and self._buttons:       # first spin must pick a bet
            mask[0] = False
        try:
            title = self.page.title()
        except Exception:
            title = ""
        return {"png": self.page.screenshot(type="png"), "mask": mask, "url": self.page.url, "links": self.links,
                "boxes": self.boxes, "title": title, "session": self.session,
                "casino": {"balance": self.balance, "bet": self.bet, "win": self.win, "delta": self.delta,
                           "cum": self.cum_reward}}

    def _wait_settle(self, before: float | None):
        """Wait until the balance moved away from `before` and then stayed put for 2.5 s."""
        t0 = time.time()
        last, last_change = before, time.time()
        moved = False
        while time.time() - t0 < self.spin_timeout:
            time.sleep(0.5)
            bal, _ = self._read()
            if bal is not None and bal != last:
                last, last_change, moved = bal, time.time(), True
            if moved and time.time() - last_change > 2.5:
                break

    def step(self, action: int) -> tuple[dict, float, bool]:
        self.step_i += 1
        frame = self.game_frame()
        self._demo_check(frame)
        before, _ = self._read()
        self.ws_win = None
        if action == 0:
            box = frame.frame_element().bounding_box()
            self.page.mouse.click(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
            self.page.keyboard.press("Space")
        else:
            b = self._buttons[action - 1]
            self.bet = b["value"]
            self.page.mouse.click(b["x"] + b["w"] / 2, b["y"] + b["h"] / 2)
        self._wait_settle(before)
        after, win = self._read()
        self.balance, self.win = after, win
        self.delta = (after - before) if (after is not None and before is not None) else 0.0
        reward = float(np.clip(self.delta / self.bet, -1.0, 3.0)) if self.bet else 0.0
        self.cum_reward += reward
        with open(self.log_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([f"{time.time():.1f}", self.episode, self.step_i, self.links[action][0], self.bet,
                                    before, after, f"{self.delta:.2f}", f"{reward:.3f}", f"{max(reward, 0):.3f}",
                                    f"{max(-reward, 0):.3f}", f"{self.cum_reward:.3f}"])
        obs = self._observe()
        broke = self.balance is not None and self.bet is not None and self.balance < self.bet
        done = self.step_i >= self.max_steps or broke
        return obs, reward, done

    def close(self):
        self.context.close()
        self.browser.close()
        self._pw.stop()

    # ---- diagnostics ---------------------------------------------------------------------------
    def probe(self, seconds: float = 20.0) -> dict:
        """Dump what the env can see (frames, balance text, bet buttons, websocket frames) without acting."""
        self.page.goto(self.game, wait_until="domcontentloaded", timeout=60000)
        self.page.wait_for_timeout(9000)
        self.page.keyboard.press("Escape")
        self.page.wait_for_timeout(seconds * 1000)
        frame = self.game_frame()
        return {"frames": [f.url[:160] for f in self.page.frames], "game_frame": frame.url[:200] if frame else None,
                "text": self._frame_text(frame)[:1500] if frame else "", "read": self._read(),
                "bet_buttons": self._bet_buttons(), "ws": list(self.ws_log)[:40]}


if __name__ == "__main__":   # python -m fly.casino [--headed]  -> data/casino_probe.json (no bets placed)
    import sys
    env = CasinoEnv(headless="--headed" not in sys.argv, session_cookie=load_session_cookie(SESSION_FILE))
    try:
        info = env.probe()
    finally:
        env.close()
    out = os.path.join(DATA, "casino_probe.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=1)
    print(f"balance/win: {info['read']}  bet buttons: {len(info['bet_buttons'])}  ws frames: {len(info['ws'])}")
    print("written", out)
