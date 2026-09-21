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

from .browser import SESSION_FILE, launch_chromium, load_session_cookie, start_screencast
from .drive import Drive
from .generic import GenericGame, NetSniffer, money_delta, summarize

DEMO_GAME = "https://betking.com.ua/casino/?game=olympus-glory-buy-bonus&isMoney=false"
LOBBY = "https://betking.com.ua/casino/"                              # every provider; the fly picks any card
AMUSNET = "https://betking.com.ua/games/top-provider-games-amusnet/"
LOBBIES = [AMUSNET, LOBBY, AMUSNET, "https://betking.com.ua/games/all-slots/"]   # the clients we drive best, twice as often
EXPLORE_LIMIT = 12    # generic client: this many actions without any money signal -> unplayable
BONUS_CAP = 150.0     # a free-spins bonus keeps the fields moving; never wait longer than this for one spin
BOUGHT_CAP = 420.0    # a bought bonus (10 free spins + intro + count-up) takes ~100 s; hard cap
MAX_STAKES = 31       # stakes use action slots 0..30; "buy the bonus" takes the slot right after the stakes
BONUS_JS = """
() => {   // the buy-bonus toggle: text "BUY BONUS" when off, a picture + check mark when on; the modal is another route
  const b = document.querySelector('#buy-bonus-button'); if (!b) return null;
  const r = b.getBoundingClientRect();
  return { on: !/BUY/i.test(b.innerText || ''), x: r.x, y: r.y, w: r.width, h: r.height,
           modal: !!document.querySelector('#bonus-modal'), collect: !!(document.querySelector('#collect-button') && document.querySelector('#collect-button').getBoundingClientRect().width > 0) };
}
"""
CARDS_JS = """
() => {   // lobby slot cards: <div class="game-item" data-app-process-url="/online-game/<slug>/"><img alt="name">
  const out = [], seen = new Set();
  for (const el of document.querySelectorAll('.game-item[data-app-process-url]')) {
    const m = (el.getAttribute('data-app-process-url') || '').match(/online-game\\/([^/]+)/);
    const r = el.getBoundingClientRect();
    if (!m || seen.has(m[1]) || r.width < 40 || r.bottom < 0 || r.top > innerHeight) continue;
    seen.add(m[1]);
    const img = el.querySelector('img');
    out.push({ slug: m[1], name: (img && img.alt) || m[1], box: [Math.round(r.x), Math.round(r.y), Math.round(r.width), Math.round(r.height)] });
  }
  return out;
}
"""
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
                 log_path: str = LOG, spin_timeout: float = 12.0, lobby: str | None = LOBBY):
        if "ismoney=false" not in game.lower():
            raise ValueError("casino env only runs demo games: the URL must contain isMoney=false")
        from playwright.sync_api import sync_playwright
        self.game = game
        self.lobby = lobby                           # None -> open `game` directly
        self._on_frame = None                        # callback(jpeg bytes): live view for the visualiser
        self._cast = None
        self.drive = Drive()                         # appetite, stake pattern, lucky/unlucky slots, events
        self.cur_slug, self.cur_name = None, ""
        self.rest = False                            # set when the fly has lost its appetite: take a walk
        self.new_events: list[dict] = []
        self.last_png: bytes | None = None
        u = urlparse(game)
        self.game_base = f"{u.scheme}://{u.netloc}{u.path}"
        self.phase = "game"
        self._cards: list[dict] = []
        self.host = urlparse(game).netloc
        self.max_actions = max_actions
        self.max_steps = max_steps
        self.frame_hint = frame_hint
        self.spin_timeout = spin_timeout
        self.log_path = log_path
        self._pw = sync_playwright().start()
        self.browser = launch_chromium(self._pw, headless)
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
        self.sniffer = NetSniffer(self.page, self.host)   # other providers: money fields on the wire
        self.kind = "amusnet"                            # driver for the open game: amusnet | generic
        self.generic: GenericGame | None = None
        self.explored = 0                                # generic actions without any money signal
        self.on_idle = None                              # agent hook: run the brain while we wait (live retina)
        self.last_frame: bytes | None = None             # latest screencast JPEG
        self.lobby_i = 0
        self.balance: float | None = None     # balance field (pending win not yet credited)
        self.pending_win: float | None = None  # win shown by the game, credited at the next spin
        self.bonus: dict | None = None         # buy-bonus toggle state of the open client (None = no such feature)
        self.bonus_ratio: float | None = None  # bonus price / stake (Amusnet: 90x)
        self.last_bonus: dict | None = None    # what the last purchase cost and paid
        self.bonus_i: int | None = None        # action index of "buy the bonus" in the current observation
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
                                        "win", "delta", "reward", "dopamine_pam", "dopamine_ppl1", "cum_reward",
                                        "bank", "desire"])

    # ---- guards -------------------------------------------------------------------------------
    @staticmethod
    def _guard(route, request):
        if MONEY.search(request.url):
            route.abort()
        else:
            route.continue_()

    def _demo_check(self, frame):
        if "ismoney=true" in ((frame.url if frame else "") + self.page.url).lower():
            raise RuntimeError("real-money mode detected - refusing to act")
        if self.kind == "generic":
            if "ismoney=false" not in self.page.url.lower() or not self.sniffer.is_demo:
                raise RuntimeError("no demo evidence from the game client - refusing to act")
            return
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
                    self.page.wait_for_timeout(100)
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
            self.page.wait_for_timeout(350)
        return None

    # ---- buy bonus (Amusnet "BUY BONUS" toggle) -----------------------------------------------------
    def _bonus_state(self) -> dict | None:
        frame = self.game_frame()
        try:
            return frame.evaluate(BONUS_JS) if frame else None
        except Exception:
            return None

    def _close_bonus_modal(self):
        frame = self.game_frame()
        try:
            c = frame.locator("#close-modal")
            if c.count() and c.first.is_visible():
                c.first.click(timeout=2000)
                self.page.wait_for_timeout(600)
        except Exception:
            pass

    def _open_bonus_dialog(self) -> bool:
        """Click BUY BONUS until its price dialog (#bonus-modal) shows up: the client swallows the first
        click after a page load, the next ones open the dialog."""
        for _ in range(4):
            st = self._bonus_state()
            if not st:
                return False
            if st["modal"]:
                return True
            box = self._frame_box() or {"x": 0, "y": 0}
            self.page.mouse.click(box["x"] + st["x"] + st["w"] / 2, box["y"] + st["y"] + st["h"] / 2)
            for _ in range(8):
                self.page.wait_for_timeout(250)
                self._tick_brain()
                st2 = self._bonus_state()
                if st2 and st2["modal"]:
                    return True
        return False

    def _dialog_read(self) -> tuple[float | None, float | None]:
        """(stake, price) shown by the bonus dialog."""
        frame = self.game_frame()
        try:
            r = frame.evaluate("() => { const t = s => { const e = document.querySelector(s); return e ? e.innerText : null; };"
                               " return [t('#bonus-active-bet-amount'), t('#bonus-amount')]; }")
        except Exception:
            return None, None
        return _num(r[0]), _num(r[1])

    def _dialog_click(self, sel: str) -> bool:
        frame, box = self.game_frame(), self._frame_box()
        try:
            r = frame.evaluate(ARROW_JS, sel)
        except Exception:
            r = None
        if not r or not box:
            return False
        self.page.mouse.click(box["x"] + r[0] + r[2] / 2, box["y"] + r[1] + r[3] / 2)
        return True

    def _leave_bonus_mode(self):
        """After a bought feature the strip may show bonus prices instead of stakes: toggle back."""
        for _ in range(3):
            btns = self._bet_buttons()
            st = self._bonus_state()
            if st and st["modal"]:
                self._close_bonus_modal()
                continue
            if not btns or btns[0]["value"] < 5.0 and not (st and st["on"]):
                return
            box = self._frame_box() or {"x": 0, "y": 0}
            self.page.mouse.click(box["x"] + st["x"] + st["w"] / 2, box["y"] + st["y"] + st["h"] / 2)
            self.page.wait_for_timeout(1200)
            self._tick_brain()

    def _buy_bonus(self, target: float | None) -> tuple[float, float, bool, float | None]:
        """Buy the feature through its dialog at the stake whose price is nearest to `target` (or the
        cheapest affordable): returns (price, win, ok, balance right after the purchase). The bought
        bonus plays itself out; we wait for the collect button, take it, and put the strip back."""
        if not self._open_bonus_dialog():
            return 0.0, 0.0, False, None
        stake, price = self._dialog_read()
        if price is None:
            self._close_bonus_modal()
            return 0.0, 0.0, False, None
        cap = min(self.drive.bank, self.drive.bonus_cap())     # never more than a small share of the bank
        for _ in range(45):                                # walk the dialog's stake to the wanted price
            stake, price = self._dialog_read()
            if price is None:
                break
            step_dir = None
            if price > cap + 1e-9:
                step_dir = "left"
            elif target and price < target * 0.75 and price * 1.3 <= cap:
                step_dir = "right"
            elif target and price > target * 1.5:
                step_dir = "left"
            if not step_dir:
                break
            before_price = price
            self._dialog_click(f"#bonus-arrow-{step_dir}")
            self.page.wait_for_timeout(200)
            _, price2 = self._dialog_read()
            if price2 == before_price:                     # end of the range
                break
        stake, price = self._dialog_read()
        if price is None or price > cap + 1e-9:
            print(f"casino: cheapest feature costs {price} FUN, above the fly's limit {cap:.2f} - not buying")
            self._close_bonus_modal()
            return 0.0, 0.0, False, None
        before = self._read()
        self._dialog_click("#bonus-buy-button")
        print(f"casino: buying the bonus for {price:.2f} FUN (stake {stake})")
        frame = self.game_frame()
        t0 = time.time()
        started = False
        start_balance = None
        best_win = 0.0
        last_change = time.time()
        last = None
        while time.time() - t0 < BOUGHT_CAP:
            self.page.wait_for_timeout(250)
            self._tick_brain()
            d = self._dom(frame)
            cur = (_num(d.get("balance")), _num(d.get("win")))
            if not started:
                if cur[0] is not None and before[0] is not None and cur[0] < before[0] - 1e-6:
                    started, start_balance = True, cur[0]
                elif time.time() - t0 > 6.0:
                    print("casino: the purchase did not go through")
                    self._close_bonus_modal()
                    return 0.0, 0.0, False, None
                continue
            if cur[1]:
                best_win = max(best_win, cur[1])
            if cur != last:
                last, last_change = cur, time.time()
            st = self._bonus_state() or {}
            # the feature is over when the game offers to collect / gamble and nothing has moved for a while
            if st.get("collect") and time.time() - last_change > 4.0 and time.time() - t0 > 10.0:
                break
            if time.time() - last_change > 45.0:         # nothing at all for a long time: assume it ended quietly
                break
        self._recover()                                   # collect
        self._leave_bonus_mode()
        real_price = round(before[0] - start_balance, 2) if before[0] is not None and start_balance is not None else price
        return (real_price if real_price > 0 else price), best_win, True, start_balance

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

    # ---- lobby: the fly looks for a slot itself ---------------------------------------------------
    def _lobby_cards(self) -> list[dict]:
        try:
            cards = self.page.evaluate(CARDS_JS)
        except Exception:
            cards = []
        cards = [c for c in cards if not (self.drive.slots.get(c["slug"]) or {}).get("unplayable")]
        return cards[: self.max_actions]

    def _open_lobby(self) -> bool:
        lobby = self.lobby if self.lobby not in LOBBIES else LOBBIES[self.lobby_i % len(LOBBIES)]
        self.lobby_i += 1
        try:
            self.page.goto(lobby, wait_until="domcontentloaded", timeout=60000)
            self.page.wait_for_timeout(3000)
            self._close_modals()
        except Exception:
            return False
        self._cards = self._lobby_cards()
        if not self._cards:
            return False
        self.phase = "lobby"
        self.links = [(f"slot:{c['slug']}", c["name"]) for c in self._cards]
        self.boxes = [c["box"] for c in self._cards]
        return True

    def _open_game(self, url: str):
        self.sniffer.reset()
        self.page.goto(url, wait_until="domcontentloaded", timeout=60000)
        self.page.wait_for_timeout(4000)
        self._close_modals()
        frame = None
        self.kind, self.generic, self.explored = "amusnet", None, 0
        for i in range(30):                          # an Amusnet client shows FUN + a stake strip within ~30 s
            frame = self.game_frame()
            if frame is not None and (self._dom(frame).get("currency") or "").strip():
                break
            if i % 5 == 4:
                self._close_modals()
            self.page.wait_for_timeout(1000)
            self._tick_brain()
        if frame is not None and (self._dom(frame).get("currency") or "").strip() and self._bet_buttons():
            self._demo_check(frame)
            self.phase = "game"
            self.balance, _ = self._read()
            self.win, self.bet, self.delta = 0.0, None, 0.0
            self.pending_win = 0.0
            return
        self._open_generic()

    def _game_box(self) -> dict | None:
        """Where the game is drawn: the biggest cross-origin iframe, else the site's game container."""
        best = None
        for f in self.page.frames:
            if f == self.page.main_frame or not f.url.startswith("http"):
                continue
            try:
                b = f.frame_element().bounding_box()
            except Exception:
                b = None
            if b and b["width"] > 300 and b["height"] > 200 and (best is None or b["width"] * b["height"] > best["width"] * best["height"]):
                best = b
        if best:
            return best
        try:
            return self.page.evaluate("""() => { for (const sel of ['#casinoClient', '[class*=game-client]', '[class*=gameClient]', 'canvas', 'iframe']) {
                const e = document.querySelector(sel); if (!e) continue; const r = e.getBoundingClientRect();
                if (r.width > 300 && r.height > 200) return { x: r.x, y: r.y, width: r.width, height: r.height }; } return null; }""")
        except Exception:
            return None

    def _open_generic(self):
        """Not an Amusnet client: let the fly explore it - if the wire shows it is a demo."""
        self.kind = "generic"
        box = self._game_box()
        if not box:
            raise RuntimeError("no game area found")
        self.generic = GenericGame(self.page, box, self.sniffer)
        for _ in range(3):                           # splash screens / "PLAY" buttons
            if not self.generic.dismiss_splash(self.page.frames):
                break
            self.page.wait_for_timeout(2500)
            self._tick_brain()
        t0 = time.time()
        while time.time() - t0 < 20 and not self.sniffer.is_demo:
            self.page.wait_for_timeout(500)
            self._tick_brain()
        self._demo_check(None)
        print(f"casino: unknown client, exploring it  ({summarize(self.sniffer).splitlines()[0]})")
        self.phase = "game"
        snap = self.sniffer.snapshot()
        self.balance = snap["balance"]
        if snap["balance"] is None:
            self.explored += 4                       # not even a balance on the wire: little hope, fewer tries
        self.win, self.bet, self.delta = 0.0, None, 0.0

    # ---- gym-like api -------------------------------------------------------------------------
    @property
    def broke(self) -> bool:
        return self.drive.broke

    def reset(self) -> dict:
        self.episode += 1
        self.step_i = 0
        self.rest = False
        if self.phase == "game" and self.cur_slug and self.drive.wants_switch():
            why = self.drive.reason
            self.drive.close_slot(why)
            print(f"casino: leaving '{self.cur_name}' ({why}), looking for another slot")
        if self.lobby and self._open_lobby():        # phase 1: pick a slot from the Amusnet list
            return self._observe()
        self._open_game(self.game)
        self._register_slot(self.game)
        return self._observe()

    def _register_slot(self, url: str, name: str | None = None):
        m = re.search(r"game=([^&]+)", url)
        self.cur_slug = m.group(1) if m else url
        self.cur_name = name or self.cur_slug
        self.drive.open_slot(self.cur_slug, self.cur_name, self.balance)

    @property
    def on_frame(self):
        return self._on_frame

    @on_frame.setter
    def on_frame(self, cb):
        """Set by the agent when a visualiser is attached: starts Chromium's screencast (live reels)."""
        self._on_frame = cb
        if cb and self._cast is None:
            def got(jpg):
                self.last_frame = jpg
                if self._on_frame:
                    self._on_frame(jpg)
            try:
                self._cast = start_screencast(self.context, self.page, got)
            except Exception as e:
                print(f"casino: screencast unavailable ({str(e)[:80]})")

    def idle(self, seconds: float):
        """Wait while still pumping browser events (live frames keep flowing) and letting the brain run."""
        t0 = time.time()
        while time.time() - t0 < seconds:
            self.page.wait_for_timeout(100)
            self._tick_brain()

    def _tick_brain(self):
        if self.on_idle:
            try:
                self.on_idle(self.last_frame)
            except Exception as e:
                print(f"casino: idle brain hook failed ({str(e)[:60]})")
                self.on_idle = None

    def action_prior(self) -> np.ndarray | None:
        """Log-prior for the readout: wanted stake size in the game, lucky/unlucky slots in the lobby."""
        if self.phase == "lobby":
            return self.drive.lobby_prior([c["slug"] for c in self._cards])
        if self.kind == "generic":
            return np.array(self.generic.prior()[: self.max_actions], np.float32)
        prior = np.zeros(self.max_actions, np.float32)
        bp = self.drive.bet_prior([b["value"] for b in self._buttons])
        prior[: len(bp)] = bp
        if self.bonus_i is not None:
            prior[self.bonus_i] = self.drive.bonus_prior()
        return prior

    def _observe(self) -> dict:
        if self.phase == "lobby":
            mask = np.zeros(self.max_actions, bool)
            mask[: len(self.links)] = True
            self.last_png = self.page.screenshot(type="png")
            return {"png": self.last_png, "mask": mask, "url": self.page.url,
                    "links": self.links, "boxes": self.boxes, "title": "lobby", "session": self.session,
                    "casino": {"balance": self.drive.bank, "demo": self.balance, "bet": None, "win": None,
                               "delta": 0.0, "cum": self.cum_reward, "phase": "lobby", "drive": self.drive.snapshot(),
                               "events": []}}
        if self.kind == "generic":
            self._buttons = []
            self.links = self.generic.links()[: self.max_actions]
            self.boxes = self.generic.boxes()[: self.max_actions]
            mask = np.zeros(self.max_actions, bool)
            mask[: len(self.links)] = not self.drive.broke
            self.last_png = self.page.screenshot(type="png")
            try:
                title = self.page.title()
            except Exception:
                title = ""
            return {"png": self.last_png, "mask": mask, "url": self.page.url, "links": self.links,
                    "boxes": self.boxes, "title": title, "session": self.session, "ghost": True,
                    "casino": {"balance": self.drive.bank, "demo": self.balance, "bet": self.bet, "win": self.win,
                               "delta": self.delta, "cum": self.cum_reward, "phase": "game", "kind": "generic",
                               "drive": self.drive.snapshot(), "events": self.new_events}}
        self.bonus = self._bonus_state()
        if self.bonus and (self.bonus["on"] or self.bonus["modal"]):   # left in bonus mode: back to stakes
            self._leave_bonus_mode()
            self.bonus = self._bonus_state()
        self._buttons = self._bet_buttons()[: MAX_STAKES]
        self.bonus_i = None
        self.links = [(f"bet:{b['value']:.2f}", f"bet {b['value']:.2f} FUN") for b in self._buttons]
        right = self._arrow_box("right") or [0, 0, 0, 0]
        # off-screen bets: the fly lands on the strip's arrow, the env scrolls the strip for it
        self.boxes = [[int(b["x"]), int(b["y"]), int(b["w"]), int(b["h"])] if b["visible"] else right
                      for b in self._buttons]
        mask = np.zeros(self.max_actions, bool)
        for i, b in enumerate(self._buttons):            # only stakes the fly can still afford
            mask[i] = b["value"] <= self.drive.bank + 1e-9
        if self.bonus and self._buttons:                 # the feature can be bought: one more action
            if self.bonus_ratio is None:
                self.bonus_ratio = 90.0                  # Amusnet: price = 90 x stake (read exactly on first purchase)
            min_price = self._buttons[0]["value"] * self.bonus_ratio
            self.bonus_i = len(self._buttons)
            self.links.append((f"bonus:{min_price:.2f}", f"buy bonus (from {min_price:.2f} FUN)"))
            fb = self._frame_box() or {"x": 0, "y": 0}
            self.boxes.append([int(fb["x"] + self.bonus["x"]), int(fb["y"] + self.bonus["y"]), int(self.bonus["w"]), int(self.bonus["h"])])
            mask[self.bonus_i] = min_price <= self.drive.bonus_cap() and self.drive.bonus_allowed()
        try:
            title = self.page.title()
        except Exception:
            title = ""
        bal = (self.balance or 0.0) + (self.pending_win or 0.0)   # a pending win counts: credited at the next spin
        self.last_png = self.page.screenshot(type="png")
        return {"png": self.last_png, "mask": mask, "url": self.page.url, "links": self.links,
                "boxes": self.boxes, "title": title, "session": self.session,
                "casino": {"balance": self.drive.bank, "demo": bal if self.balance is not None else None,
                           "bet": self.bet, "win": self.win,
                           "delta": self.delta, "cum": self.cum_reward, "phase": "game", "kind": "amusnet",
                           "drive": self.drive.snapshot(), "events": self.new_events}}

    def _wait_settle(self, before: tuple) -> tuple[tuple, bool, float]:
        """Follow one spin to its end, streaming live frames meanwhile. Returns ((balance, win), won, started_balance).

        Timeline of the Amusnet client (measured): ~0.2 s after the click the balance drops by the stake
        and #info-line goes blank; the reels stop 2.3-5 s later. A dead spin brings the "place your bet"
        prompt back; a win shows "Line N 4x = 0.40 FUN" and #win-amount-field counts up from 0 to the
        total over 1-2 s (it keeps that value through later dead spins, so a change alone is not a win).
        A bonus (free spins, gamble) keeps the fields moving for much longer: as long as anything keeps
        changing the spin is not over (cap BONUS_CAP). The win is credited to the balance at the next spin."""
        t0 = time.time()
        frame = self.game_frame()
        started = saw_blank = False
        won = False
        last = before
        best_win = 0.0
        started_balance = None
        stable_since = None
        stop_info = None
        while time.time() - t0 < BONUS_CAP:
            self.page.wait_for_timeout(100)                      # pumps screencast frames meanwhile
            self._tick_brain()                                   # the fly watches the reels
            d = self._dom(frame)
            cur = (_num(d.get("balance")), _num(d.get("win")))
            info = (d.get("info") or "").strip()
            if not started:
                if cur[0] is not None and cur != before or (info == "" and time.time() - t0 > 0.1):
                    started = True                              # the stake was taken / reels are moving
                    started_balance = cur[0]
                elif time.time() - t0 > 3.0:
                    return before, False, None                  # the click did not start a spin
                continue
            if started_balance is None and cur[0] is not None:
                started_balance = cur[0]
            if info == "":                                      # reels still turning
                saw_blank = True
                continue
            if not saw_blank:                                   # still the previous spin's win line
                continue
            if stop_info is None:
                stop_info = info
            if "=" in info:
                won = True
            if cur != last:
                last, stable_since = cur, time.time()
            elif stable_since is None:
                stable_since = time.time()
            if won and cur[1]:
                best_win = max(best_win, cur[1])
            # reels stopped: a dead spin settles at once, a win once the odometer has stopped counting,
            # a bonus screen (the info text moved on) only after a longer quiet period
            bonus = info != stop_info and "=" not in info          # win lines ("Line 3 = 0.40") cycle; a bonus text does not
            need = 3.0 if bonus else 1.2 if won else 0.3
            if stable_since and time.time() - stable_since > need and time.time() - t0 > (12.0 if bonus else 0):
                break
        if won and best_win:
            last = (last[0], best_win)
        return last, won, started_balance

    def step(self, action: int) -> tuple[dict, float, bool]:
        self.step_i += 1
        if self.phase == "lobby":                    # open the chosen slot in demo mode (reward 0)
            card = self._cards[action] if action < len(self._cards) else self._cards[0]
            url = f"{self.game_base}?game={card['slug']}&isMoney=false"
            print(f"casino: opening slot '{card['name']}' in demo mode")
            try:
                self._open_game(url)
                self._register_slot(url, card["name"])
                if self.kind == "generic":
                    self._note_explored(self.explored)
            except Exception as e:                   # could not open it: remember; three failures = unplayable
                print(f"casino: {card['name']} did not open ({str(e)[:80]}), back to the lobby")
                self.drive.note_open_failure(card["slug"], card["name"])
                if not self._open_lobby():
                    self._open_game(self.game)
                    self._register_slot(self.game)
            return self._observe(), 0.0, False
        if self.kind == "generic":
            return self._step_generic(action)
        frame = self.game_frame()
        self._demo_check(frame)
        if self.bonus_i is not None and action == self.bonus_i:
            return self._step_bonus()
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
        after, won, at_start = self._wait_settle(before)
        self.balance = after[0]
        self.win = (after[1] or 0.0) if won else 0.0
        win = self.win
        # the demo balance is the ground truth: at spin start it must equal the previous balance + the
        # previous win - this stake. Anything above that is a win we did not see (bonus rounds, gamble,
        # a missed odometer) - credit it now so the fly's bank tracks the demo account exactly
        if at_start is not None and before[0] is not None and self.pending_win is not None:
            missed = round(at_start - (before[0] + self.pending_win - self.bet), 2)
            if missed > 0.005:
                print(f"casino: +{missed:.2f} FUN credited by the game that we had not seen (bonus?) - counted now")
                win += missed
                self.win = win
        self.pending_win = self.win
        self.delta = win - self.bet
        reward = float(np.clip(self.delta / self.bet, -1.0, 3.0))
        self.cum_reward += reward
        with open(self.log_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([f"{time.time():.1f}", self.episode, self.step_i, self.links[action][0], self.bet,
                                    before[0], after[0], f"{win:.2f}", f"{self.delta:.2f}", f"{reward:.3f}",
                                    f"{max(reward, 0):.3f}", f"{max(-reward, 0):.3f}", f"{self.cum_reward:.3f}",
                                    f"{self.drive.bank - self.bet + win:.2f}", f"{self.drive.desire:.3f}"])
        self.new_events = self.drive.spin(self.bet, win, reward, None)
        obs = self._observe()
        broke = self.drive.broke or not obs["mask"].any()
        if broke:
            print(f"casino: the fly is broke (bank {self.drive.bank:.2f} FUN) - it stops and waits for a refill")
        switch = self.drive.wants_switch() and not broke
        self.rest = self.drive.wants_rest() and not broke
        if self.rest:
            print(f"casino: appetite is gone (desire {self.drive.desire:.2f}) - the fly goes for a walk")
        done = self.step_i >= self.max_steps or broke or switch or self.rest
        return obs, reward, done

    def _note_explored(self, n: int):
        """Exploration is remembered per slot across sessions: a game that never shows money is dropped."""
        st = self.drive.slots.get(self.cur_slug)
        if st is not None:
            st["explored"] = st.get("explored", 0) + n

    def _explored_total(self) -> int:
        st = self.drive.slots.get(self.cur_slug) or {}
        return max(self.explored, int(st.get("explored", 0)))

    def _step_bonus(self) -> tuple[dict, float, bool]:
        """Buy the feature: the price is the stake, the whole bought round is the win."""
        target = (self.drive.target_bet or self._buttons[0]["value"]) * (self.bonus_ratio or 90.0)
        price, win, ok, start_balance = self._buy_bonus(min(target, self.drive.bonus_cap()))
        if not ok:
            obs = self._observe()
            return obs, 0.0, False
        if self._buttons and price > 0:
            stake_guess = min((b["value"] for b in self._buttons), key=lambda v: abs(v * (self.bonus_ratio or 90.0) - price))
            if stake_guess > 0:
                self.bonus_ratio = round(price / stake_guess, 2)
        self.bet, self.win = price, win
        self.delta = win - price
        self.balance, _ = self._read()
        # collected: some clients credit the win at once, others at the next spin - tell which by the balance
        self.pending_win = 0.0 if (self.balance is not None and start_balance is not None
                                   and abs(self.balance - start_balance) > 0.005) else win
        reward = float(np.clip(self.delta / price, -1.0, 3.0))
        self.cum_reward += reward
        with open(self.log_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([f"{time.time():.1f}", self.episode, self.step_i, f"bonus:{price:.2f}", price,
                                    "", self.balance, f"{win:.2f}", f"{self.delta:.2f}", f"{reward:.3f}",
                                    f"{max(reward, 0):.3f}", f"{max(-reward, 0):.3f}", f"{self.cum_reward:.3f}",
                                    f"{self.drive.bank - price + win:.2f}", f"{self.drive.desire:.3f}"])
        self.new_events = self.drive.bonus(price, win, reward)
        self.last_bonus = {"price": price, "win": win}
        print(f"casino: bonus bought for {price:.2f} FUN paid {win:.2f} FUN (reward {reward:+.2f})")
        obs = self._observe()
        broke = self.drive.broke or not obs["mask"].any()
        switch = self.drive.wants_switch() and not broke
        self.rest = self.drive.wants_rest() and not broke
        done = self.step_i >= self.max_steps or broke or switch or self.rest
        return obs, reward, done

    def _step_generic(self, action: int) -> tuple[dict, float, bool]:
        """Unknown client: do the action, watch the wire for a balance drop (stake) and a win."""
        try:
            self._demo_check(None)
        except RuntimeError as e:                   # the guard is a reason to leave the game, not to crash the fly
            print(f"casino: leaving '{self.cur_name}': {e}")
            self._note_explored(4)
            self.drive.reason = "unplayable"
            return self._observe(), 0.0, True
        before = self.sniffer.snapshot()
        self.generic.act(action)
        t0 = time.time()
        stake = win = None
        while time.time() - t0 < self.spin_timeout:
            self.page.wait_for_timeout(150)
            self._tick_brain()
            stake, win = money_delta(before, self.sniffer.snapshot())
            if stake is not None and time.time() - t0 > 3.0 and time.time() - max(self.sniffer.balance_at, self.sniffer.win_at) > 1.5:
                break
            if stake is None and win is None and time.time() - t0 > 6.0:
                break                                # nothing happened on the wire: next try
        snap = self.sniffer.snapshot()
        self.balance = snap["balance"]
        if stake is None and win is None:
            self.explored += 1
            self._note_explored(1)
            self.bet, self.win, self.delta = None, 0.0, 0.0
            reward = 0.0
        else:
            self.explored = 0
            self.bet = stake or 0.0
            self.win = win or 0.0
            self.delta = self.win - self.bet
            reward = float(np.clip(self.delta / self.bet, -1.0, 3.0)) if self.bet else float(np.clip(self.win / 10.0, 0.0, 3.0))
            self.cum_reward += reward
            with open(self.log_path, "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow([f"{time.time():.1f}", self.episode, self.step_i, self.links[action][0], self.bet,
                                        before.get("balance"), snap["balance"], f"{self.win:.2f}", f"{self.delta:.2f}",
                                        f"{reward:.3f}", f"{max(reward, 0):.3f}", f"{max(-reward, 0):.3f}",
                                        f"{self.cum_reward:.3f}", f"{self.drive.bank - self.bet + self.win:.2f}",
                                        f"{self.drive.desire:.3f}"])
            self.new_events = self.drive.spin(self.bet, self.win, reward, None)
        obs = self._observe()
        if self._explored_total() >= EXPLORE_LIMIT:
            print(f"casino: '{self.cur_name}' shows no money signal after {self._explored_total()} actions - giving up on it")
            self.drive.mark_unplayable(self.cur_slug, self.cur_name)
            self.drive.reason = "unplayable"
            return obs, 0.0, True
        broke = self.drive.broke
        switch = self.drive.wants_switch() and not broke
        self.rest = self.drive.wants_rest() and not broke
        done = self.step_i >= self.max_steps or broke or switch or self.rest
        return obs, reward, done

    def close(self):
        for f in (self.context.close, self.browser.close, self._pw.stop):
            try:
                f()
            except Exception:      # driver may already be gone (Ctrl+C / supervisor restart)
                pass

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
