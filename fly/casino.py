"""Casino environment: the fly plays a slot on the DEMO account (isMoney=false, play-money "FUN" credits).

Actions = the game's own bet buttons (Amusnet/EGT: a click on a bet value places that bet and spins).
Reward per spin = (win - bet) / bet, clipped to [-1, 3]: losing the bet = -1 -> PPL101 (aversive
dopamine); a win -> + -> PAM cluster (appetitive dopamine). Every spin is appended to data/dopamine.csv.

What the game exposes in its DOM (Amusnet HTML5 client):
  #balance-amount-field   "49 999.90"     balance (a win is credited only when the NEXT spin starts)
  #win-amount-field       "0.10"          win of the last spin (counts up during the animation)
  #bet-slider button[id^=bet-]            bet buttons, id = bet in cents (bet-10 = 0.10 FUN)
  #bet-slider-left-arrow / -right-arrow   scroll the strip; only ~9 buttons are on screen at a time
  #info-line-text         "БУДЬ ЛАСКА, ЗРОБІТЬ СТАВКУ" before the first bet

Hard guards (do not remove):
  * the game URL must contain isMoney=false; any request with isMoney=true or to cashier/deposit/
    withdraw/payment endpoints is aborted at the network layer;
  * before every action the game frame must show the play-money currency "FUN".
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

READ_JS = """
() => {
  const t = s => { const e = document.querySelector(s); return e ? e.innerText : null; };
  return { balance: t('#balance-amount-field'), win: t('#win-amount-field'), currency: t('#balance-currency-field'),
           info: t('#info-line') };
}
"""
BET_BTN_JS = """
() => {
  // the strip is a carousel clipped by #outer-bet-slider (~447 px, ~5 buttons); an arrow click shifts it by ~5
  const clip = (document.querySelector('#outer-bet-slider') || document.body).getBoundingClientRect();
  return [...document.querySelectorAll('#bet-slider button[id^=bet-]')].filter(b => /^bet-\\d+$/.test(b.id)).map(b => {
    const r = b.getBoundingClientRect();
    return { id: b.id, value: parseInt(b.id.split('-')[1], 10) / 100, x: r.x, y: r.y, w: r.width, h: r.height,
             visible: r.x >= clip.x - 2 && r.x + r.width <= clip.x + clip.width + 2 };
  });
}
"""
ARROW_JS = "(sel) => { const e = document.querySelector(sel); if (!e) return null; const r = e.getBoundingClientRect(); return [r.x, r.y, r.width, r.height]; }"


def _num(s) -> float | None:
    if s is None:
        return None
    s = re.sub(r"[^\d.,-]", "", str(s)).replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return None


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
        self.balance: float | None = None     # balance field (pending win not yet credited)
        self.win: float | None = None         # last spin's win
        self.bet: float | None = None
        self.delta = 0.0                      # win - bet of the last spin
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
                                        "win", "delta", "reward", "dopamine_pam", "dopamine_ppl1", "cum_reward"])

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
        cur = (self._dom(frame).get("currency") or "").strip().upper()
        if cur != "FUN":
            raise RuntimeError(f"play-money currency 'FUN' not shown (got {cur!r}) - refusing to act")

    # ---- readers -------------------------------------------------------------------------------
    def _on_ws(self, ws):
        if self.frame_hint and self.frame_hint not in ws.url:
            return
        ws.on("framereceived", lambda p: self.ws_log.append((p if isinstance(p, str) else p.decode("utf-8", "replace"))[:2000]))

    def game_frame(self):
        cands = [f for f in self.page.frames if self.frame_hint in f.url]
        if cands:
            return cands[0]
        for f in self.page.frames:   # fallback: any cross-origin iframe
            if f.url.startswith("http") and urlparse(f.url).netloc != self.host:
                return f
        return None

    @staticmethod
    def _dom(frame) -> dict:
        try:
            return frame.evaluate(READ_JS) or {}
        except Exception:
            return {}

    def _read(self) -> tuple[float | None, float | None]:
        """(balance field, win field) as floats."""
        frame = self.game_frame()
        if frame is None:
            return None, None
        d = self._dom(frame)
        return _num(d.get("balance")), _num(d.get("win"))

    def _frame_box(self):
        frame = self.game_frame()
        try:
            return frame.frame_element().bounding_box()
        except Exception:
            return None

    def _bet_buttons(self) -> list[dict]:
        """All bet buttons in page coordinates; `visible` = fully inside the iframe right now."""
        frame = self.game_frame()
        box = self._frame_box()
        if frame is None or box is None:
            return []
        try:
            btns = frame.evaluate(BET_BTN_JS)
        except Exception as e:
            print(f"casino: bet buttons unreadable: {str(e)[:120]}")
            return []
        for b in btns:
            b["x"] += box["x"]
            b["y"] += box["y"]
        return sorted(btns, key=lambda b: b["value"])

    def _arrow_box(self, side: str) -> list[int] | None:
        frame, box = self.game_frame(), self._frame_box()
        try:
            r = frame.evaluate(ARROW_JS, f"#bet-slider-{side}-arrow")
        except Exception:
            r = None
        if not r or not box:
            return None
        return [int(r[0] + box["x"]), int(r[1] + box["y"]), int(r[2]), int(r[3])]

    def _close_modals(self):
        try:
            btn = self.page.locator(".modal-template__close-button, .modal-template [class*='close-button']")
            if btn.count() and btn.first.is_visible():
                btn.first.click(timeout=3000)
        except Exception:
            pass
        self.page.keyboard.press("Escape")
        self.page.wait_for_timeout(800)

    def _scroll_to(self, btn_id: str, tries: int = 14) -> dict | None:
        """Page the bet strip with its arrows until `btn_id` is fully visible; returns its button dict.
        The strip is paged, not circular: 8 pages x 5 stakes, arrows do nothing at either end."""
        for _ in range(tries):
            btns = self._bet_buttons()
            b = next((x for x in btns if x["id"] == btn_id), None)
            if b is None:
                return None
            if b["visible"]:
                # wait until the transition has finished (position stable), then return fresh coordinates
                for _ in range(8):
                    self.page.wait_for_timeout(150)
                    b2 = next((x for x in self._bet_buttons() if x["id"] == btn_id), None)
                    if b2 and abs(b2["x"] - b["x"]) < 1 and b2["visible"]:
                        return b2
                    b = b2 or b
                return b if b["visible"] else None
            vis = [i for i, x in enumerate(btns) if x["visible"]]
            if not vis:                                   # mid-transition: wait, re-read
                self.page.wait_for_timeout(300)
                continue
            side = "right" if btns.index(b) > vis[-1] else "left"
            arrow = self._arrow_box(side)
            if not arrow:
                return None
            self.page.mouse.click(arrow[0] + arrow[2] / 2, arrow[1] + arrow[3] / 2)   # raw click: no actionability wait
            self.page.wait_for_timeout(650)
        return None

    def _recover(self):
        """After a win the game may wait for collect/gamble: collect, dismiss overlays, give it a moment."""
        frame = self.game_frame()
        try:
            col = frame.locator("#collect-button")
            if col.count() and col.first.is_visible():
                col.first.click(timeout=2000)
        except Exception:
            pass
        self.page.keyboard.press("Escape")
        self.page.wait_for_timeout(1500)

    # ---- gym-like api -------------------------------------------------------------------------
    def reset(self) -> dict:
        self.episode += 1
        self.step_i = 0
        self.page.goto(self.game, wait_until="domcontentloaded", timeout=60000)
        self.page.wait_for_timeout(6000)
        self._close_modals()
        frame = None
        for i in range(50):                          # the game client can take a while to boot
            frame = self.game_frame()
            if frame is not None and (self._dom(frame).get("currency") or "").strip():
                break
            if i % 5 == 4:
                self._close_modals()
            self.page.wait_for_timeout(1000)
        if frame is None:
            raise RuntimeError("game iframe not found")
        self._demo_check(frame)
        self.balance, _ = self._read()
        self.win, self.bet, self.delta = 0.0, None, 0.0
        return self._observe()

    def _observe(self) -> dict:
        self._buttons = self._bet_buttons()[: self.max_actions]
        self.links = [(f"bet:{b['value']:.2f}", f"ставка {b['value']:.2f} FUN") for b in self._buttons]
        right = self._arrow_box("right") or [0, 0, 0, 0]
        # off-screen bets: the fly lands on the strip's arrow, the env scrolls the strip for it
        self.boxes = [[int(b["x"]), int(b["y"]), int(b["w"]), int(b["h"])] if b["visible"] else right
                      for b in self._buttons]
        mask = np.zeros(self.max_actions, bool)
        mask[: len(self.links)] = True
        try:
            title = self.page.title()
        except Exception:
            title = ""
        bal = (self.balance or 0.0) + (self.win or 0.0)   # pending win counts: it is credited at the next spin
        return {"png": self.page.screenshot(type="png"), "mask": mask, "url": self.page.url, "links": self.links,
                "boxes": self.boxes, "title": title, "session": self.session,
                "casino": {"balance": bal if self.balance is not None else None, "bet": self.bet, "win": self.win,
                           "delta": self.delta, "cum": self.cum_reward}}

    def _wait_settle(self, before: tuple) -> tuple[tuple, bool]:
        """Wait until the (balance, win) fields moved away from `before` and then stayed put for 2 s.
        Returns ((balance, win), won): `won` is True if the win field changed or a win line was shown -
        the win field keeps the last non-zero win, so a losing spin after a win still displays it."""
        t0 = time.time()
        last, last_change, moved, won = before, time.time(), False, False
        frame = self.game_frame()
        while time.time() - t0 < self.spin_timeout:
            time.sleep(0.5)
            cur = self._read()
            if "=" in (self._dom(frame).get("info") or ""):     # "Лінія 5 4x = 0.40 FUN"
                won = True
            if cur != last and cur[0] is not None:
                if cur[1] != last[1]:
                    won = True
                last, last_change, moved = cur, time.time(), True
            if moved and time.time() - t0 > 3.0 and time.time() - last_change > 2.0:
                break
        return last, won

    def step(self, action: int) -> tuple[dict, float, bool]:
        self.step_i += 1
        frame = self.game_frame()
        self._demo_check(frame)
        before = self._read()
        want = self._buttons[action]["id"] if action < len(self._buttons) else None
        b = None
        for attempt in range(3):
            b = self._scroll_to(want) if want else None
            if b is not None:
                break
            self._recover()
        if b is None:   # strip unreachable (game overlay?): take the nearest visible stake instead of crashing
            vis = [x for x in self._bet_buttons() if x["visible"]]
            if not vis:
                raise RuntimeError(f"bet button for action {action} not reachable and no stake visible")
            wanted = self._buttons[action]["value"] if action < len(self._buttons) else vis[0]["value"]
            b = min(vis, key=lambda x: x["value"])          # smallest visible stake: never bet more than intended
            print(f"casino: stake {wanted:.2f} unreachable, using visible {b['value']:.2f}")
        self.bet = b["value"]
        self.page.mouse.click(b["x"] + b["w"] / 2, b["y"] + b["h"] / 2)
        after, won = self._wait_settle(before)
        self.balance = after[0]
        self.win = (after[1] or 0.0) if won else 0.0
        win = self.win
        self.delta = win - self.bet
        reward = float(np.clip(self.delta / self.bet, -1.0, 3.0))
        self.cum_reward += reward
        with open(self.log_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([f"{time.time():.1f}", self.episode, self.step_i, self.links[action][0], self.bet,
                                    before[0], after[0], f"{win:.2f}", f"{self.delta:.2f}", f"{reward:.3f}",
                                    f"{max(reward, 0):.3f}", f"{max(-reward, 0):.3f}", f"{self.cum_reward:.3f}"])
        obs = self._observe()
        broke = self.balance is not None and self.balance + win < self.bet
        done = self.step_i >= self.max_steps or broke
        return obs, reward, done

    def close(self):
        self.context.close()
        self.browser.close()
        self._pw.stop()

    # ---- diagnostics ---------------------------------------------------------------------------
    def probe(self, seconds: float = 15.0) -> dict:
        """Dump what the env can see (frames, balance/win fields, bet buttons, websocket frames) without betting."""
        self.page.goto(self.game, wait_until="domcontentloaded", timeout=60000)
        self.page.wait_for_timeout(9000)
        self._close_modals()
        self.page.wait_for_timeout(seconds * 1000)
        frame = self.game_frame()
        return {"frames": [f.url[:160] for f in self.page.frames], "game_frame": frame.url[:200] if frame else None,
                "dom": self._dom(frame) if frame else {}, "read": self._read(),
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
    print(f"balance/win: {info['read']}  bet buttons: {len(info['bet_buttons'])} "
          f"(visible {sum(b['visible'] for b in info['bet_buttons'])})  ws frames: {len(info['ws'])}")
    print("written", out)
