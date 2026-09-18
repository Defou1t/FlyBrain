"""The fly's appetite for slots: desire to play, stake patterns shaped by results, slot switching,
lucky / unlucky slots and a record of goals (+15 % of the start balance) and busts.

State is kept in data/drive.json (slot statistics, desire) and every notable moment is appended to
data/events.jsonl with a timestamp, so runs can be compared.
"""
from __future__ import annotations

import json
import math
import os
import time
from datetime import datetime

import numpy as np

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
STATE = os.path.join(DATA, "drive.json")
EVENTS = os.path.join(DATA, "events.jsonl")

LUCKY_PCT = 0.10       # slot counts as lucky when the session balance is +10 % over its start
GOAL_PCT = 0.15        # goal: +15 % over the start balance -> recorded, start re-based
BUST_PCT = -0.30       # bust: -30 % from the start (the demo balance never really hits zero)
LOSS_STREAK_SWITCH = 6 # this many losses in a row -> change the slot
REST_DESIRE = 0.10     # below this the fly leaves the casino for a walk
SWITCH_DESIRE = 0.18   # below this it tries another slot
MAX_BET = 40.0


class Drive:
    def __init__(self, path: str = STATE, events_path: str = EVENTS):
        self.path, self.events_path = path, events_path
        self.desire = 0.6
        self.slots: dict[str, dict] = {}
        self.events: list[dict] = []
        self.slot: str | None = None
        self.name = ""
        self.start = None          # balance when the slot was opened (re-based after a goal)
        self.opened_at = 0.0
        self.session_spins = 0
        self.session_net = 0.0
        self.loss_streak = 0
        self.win_streak = 0
        self.target_bet: float | None = None
        self.reason = ""            # why the last switch / rest happened (for the page)
        self._load()
        self.desire = max(self.desire, 0.55)   # a fresh start (or a walk) restores some appetite

    # ---- persistence ------------------------------------------------------------------------
    def _load(self):
        try:
            with open(self.path, encoding="utf-8") as f:
                d = json.load(f)
            self.desire = float(d.get("desire", self.desire))
            self.slots = d.get("slots", {})
        except (OSError, ValueError):
            pass
        try:
            with open(self.events_path, encoding="utf-8") as f:
                self.events = [json.loads(line) for line in f if line.strip()][-200:]
        except (OSError, ValueError):
            self.events = []

    def _save(self):
        os.makedirs(DATA, exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump({"desire": self.desire, "slots": self.slots}, f, ensure_ascii=False, indent=1)

    def _event(self, kind: str, balance: float | None, **extra) -> dict:
        pct = (balance - self.start) / self.start if (balance is not None and self.start) else None
        ev = {"ts": time.time(), "time": datetime.now().strftime("%d.%m %H:%M:%S"), "type": kind,
              "slot": self.slot, "name": self.name, "balance": balance, "start": self.start,
              "pct": None if pct is None else round(pct * 100, 2), "spins": self.session_spins,
              "desire": round(self.desire, 3), **extra}
        self.events.append(ev)
        self.events = self.events[-200:]
        os.makedirs(DATA, exist_ok=True)
        with open(self.events_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(ev, ensure_ascii=False) + "\n")
        print(f"drive: {kind} {self.name} balance={balance} ({ev['pct']}%) after {self.session_spins} spins")
        return ev

    # ---- slot sessions ------------------------------------------------------------------------
    def open_slot(self, slug: str, name: str, balance: float | None):
        self.slot, self.name = slug, name
        self.start = balance
        self.opened_at = time.time()
        self.session_spins = 0
        self.session_net = 0.0
        self.loss_streak = self.win_streak = 0
        self.target_bet = None
        s = self.slots.setdefault(slug, {"name": name, "spins": 0, "wins": 0, "net": 0.0, "best_pct": 0.0,
                                         "lucky": False, "unlucky": False, "sessions": 0, "goals": 0, "busts": 0})
        s["sessions"] += 1
        self._save()

    def spin(self, bet: float, win: float, reward: float, balance: float | None) -> list[dict]:
        """Update after one spin. Returns the notable events it produced (goal / lucky / bust / switch)."""
        events = []
        s = self.slots.get(self.slot)
        self.session_spins += 1
        self.session_net += win - bet
        if s:
            s["spins"] += 1
            s["net"] = round(s["net"] + win - bet, 2)
            if win > 0:
                s["wins"] += 1
        # desire: wins feed it (dopamine), losses wear it down, it drifts back to neutral
        if reward > 0:
            self.win_streak += 1
            self.loss_streak = 0
            self.desire += 0.10 * min(reward, 3.0)
        else:
            self.loss_streak += 1
            self.win_streak = 0
            self.desire -= 0.04
        self.desire += (0.5 - self.desire) * 0.02
        self.desire = float(np.clip(self.desire, 0.0, 1.0))
        # stake pattern: after a win chase it with a bigger stake, after losses back off
        if reward > 0:
            self.target_bet = min(MAX_BET, bet * (2.0 if reward >= 1 else 1.4))
        else:
            self.target_bet = max(0.1, bet * (0.75 if self.loss_streak < 3 else 0.5))
        if self.desire > 0.75:
            self.target_bet = min(MAX_BET, self.target_bet * 1.3)
        # milestones against the start balance of this slot
        if balance is not None and self.start:
            pct = (balance - self.start) / self.start
            if s:
                s["best_pct"] = max(s.get("best_pct", 0.0), round(pct * 100, 2))
            if pct >= LUCKY_PCT and s and not s["lucky"]:
                s["lucky"] = True
                s["unlucky"] = False
                events.append(self._event("lucky", balance))
            if pct >= GOAL_PCT:
                if s:
                    s["goals"] += 1
                events.append(self._event("goal", balance, gain=round(balance - self.start, 2)))
                self.start = balance                     # re-base: the next +15 % counts from here
                self.desire = min(1.0, self.desire + 0.2)
            elif pct <= BUST_PCT or (balance < 0.1):
                if s:
                    s["busts"] += 1
                    s["unlucky"] = True
                events.append(self._event("bust", balance, loss=round(self.start - balance, 2)))
                self.desire = max(0.0, self.desire - 0.3)
                self.reason = "слив"
        self._save()
        return events

    # ---- decisions ----------------------------------------------------------------------------
    def wants_switch(self) -> bool:
        if self.loss_streak >= LOSS_STREAK_SWITCH:
            self.reason = f"{self.loss_streak} проигрышей подряд"
            return True
        if self.desire < SWITCH_DESIRE and self.session_spins >= 5:
            self.reason = "пропал азарт"
            return True
        return self.reason == "слив"

    def wants_rest(self) -> bool:
        return self.desire < REST_DESIRE

    def close_slot(self, why: str):
        s = self.slots.get(self.slot)
        if s and self.session_net < 0 and not s["lucky"] and self.session_spins >= 3:
            s["unlucky"] = True
        self._event("switch", None, why=why)
        self.reason = ""
        self.desire = max(self.desire, 0.45)       # a fresh slot restores some appetite
        self._save()

    def bet_prior(self, values: list[float], sharpness: float = 0.9) -> np.ndarray:
        """Log-prior over the available stakes: peaks at the stake the fly currently wants."""
        if not self.target_bet:
            return np.zeros(len(values), np.float32)
        return np.array([-sharpness * abs(math.log(max(v, 0.01) / self.target_bet)) for v in values], np.float32)

    def lobby_prior(self, slugs: list[str]) -> np.ndarray:
        """Prefer lucky slots, avoid unlucky ones, a little curiosity for unknown ones."""
        out = []
        for slug in slugs:
            s = self.slots.get(slug)
            out.append(1.2 if (s and s["lucky"]) else -2.0 if (s and s["unlucky"]) else 0.3 if s is None else 0.0)
        return np.array(out, np.float32)

    def snapshot(self) -> dict:
        s = self.slots.get(self.slot) or {}
        return {"desire": round(self.desire, 3), "slot": self.slot, "name": self.name, "start": self.start,
                "session_spins": self.session_spins, "session_net": round(self.session_net, 2),
                "loss_streak": self.loss_streak, "win_streak": self.win_streak,
                "target_bet": self.target_bet and round(self.target_bet, 2), "lucky": bool(s.get("lucky")),
                "unlucky": bool(s.get("unlucky")), "reason": self.reason,
                "events": self.events[-12:][::-1],
                "slots": sorted(({"slug": k, **v} for k, v in self.slots.items()), key=lambda x: -x["spins"])[:12]}
