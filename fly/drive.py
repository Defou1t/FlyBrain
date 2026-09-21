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
LOSS_STREAK_SWITCH = 6 # this many dead spins (win = 0) in a row -> change the slot
UNLUCKY_TTL = 45 * 60  # an unlucky slot is avoided for this long, then it gets another chance
REST_DESIRE = 0.10     # below this the fly leaves the casino for a walk
SWITCH_DESIRE = 0.18   # below this it tries another slot
MAX_BET = 40.0
BANK = 50_000.0      # the fly's own money: it believes it has this much and spends it across slots


class Drive:
    def __init__(self, path: str = STATE, events_path: str = EVENTS):
        self.path, self.events_path = path, events_path
        self.desire = 0.6
        self.bank = BANK            # virtual balance, persists across slots and restarts
        self.bank_min = self.bank_max = BANK      # extremes since the money was handed over (refill resets)
        self.bank_min_at = self.bank_max_at = ""
        self.since = datetime.now().strftime("%d.%m %H:%M")
        self.since_ts = time.time()
        self.spins_total = 0
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
            self.bank = float(d.get("bank", BANK))
            self.bank_min = float(d.get("bank_min", self.bank))
            self.bank_max = float(d.get("bank_max", max(self.bank, BANK)))
            for v in self.slots.values():      # opened many times, never spun: a client we cannot drive
                if v.get("spins", 0) == 0 and v.get("sessions", 0) >= 5:
                    v["unplayable"] = True
            self.bank_min_at = d.get("bank_min_at", "")
            self.bank_max_at = d.get("bank_max_at", "")
            self.since = d.get("since", self.since)
            self.since_ts = float(d.get("since_ts", 0.0))
            self.spins_total = int(d.get("spins_total", 0))
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
            json.dump({"desire": self.desire, "bank": round(self.bank, 2), "bank_min": round(self.bank_min, 2),
                       "bank_max": round(self.bank_max, 2), "bank_min_at": self.bank_min_at,
                       "bank_max_at": self.bank_max_at, "since": self.since, "since_ts": self.since_ts,
                       "spins_total": self.spins_total,
                       "slots": self.slots}, f, ensure_ascii=False, indent=1)

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
    def open_slot(self, slug: str, name: str, balance: float | None = None):
        self.slot, self.name = slug, name
        self.start = self.bank                      # milestones are measured on the fly's own bank
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
        self.bank = max(0.0, round(self.bank - bet + win, 2))
        balance = self.bank
        now = datetime.now().strftime("%d.%m %H:%M:%S")
        if self.bank < self.bank_min:
            self.bank_min, self.bank_min_at = self.bank, now
        if self.bank > self.bank_max:
            self.bank_max, self.bank_max_at = self.bank, now
        self.session_spins += 1
        self.spins_total += 1
        self.session_net += win - bet
        if s:
            s["spins"] += 1
            s["net"] = round(s["net"] + win - bet, 2)
            if win > 0:
                s["wins"] += 1
        # desire: wins feed it (dopamine), dead spins wear it down, it drifts back to neutral.
        # The streak counts dead spins (nothing won at all): a small win still keeps the fly hooked.
        if win > 0:
            self.win_streak += 1
            self.loss_streak = 0
            self.desire += 0.10 * min(reward, 3.0) if reward > 0 else 0.01
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
            elif pct <= BUST_PCT and not self.broke:
                if s:
                    s["busts"] += 1
                    s["lucky"] = False
                    self._mark_unlucky(s)
                events.append(self._event("bust", balance, loss=round(self.start - balance, 2)))
                self.desire = max(0.0, self.desire - 0.3)
                self.reason = "bust"
        if self.broke:
            events.append(self._event("broke", balance))
            self.reason = "broke"
        self._save()
        return events

    # ---- decisions ----------------------------------------------------------------------------
    def wants_switch(self) -> bool:
        if self.loss_streak >= LOSS_STREAK_SWITCH:
            self.reason = "streak"
            return True
        if self.desire < SWITCH_DESIRE and self.session_spins >= 5:
            self.reason = "bored"
            return True
        return self.reason == "bust"

    def wants_rest(self) -> bool:
        return self.desire < REST_DESIRE and not self.broke

    @property
    def broke(self) -> bool:
        return self.bank < 0.1

    def refill(self, amount: float = BANK):
        """Somebody gave the fly money again (gear menu on localhost)."""
        self.bank = float(amount)
        self.start = self.bank
        self.bank_min = self.bank_max = self.bank
        self.bank_min_at = self.bank_max_at = ""
        self.since = datetime.now().strftime("%d.%m %H:%M")
        self.since_ts = time.time()
        self.spins_total = 0
        self.desire = max(self.desire, 0.6)
        self.reason = ""
        self._event("refill", self.bank)
        self._save()

    @staticmethod
    def _mark_unlucky(s: dict):
        s["unlucky"] = True
        s["unlucky_until"] = time.time() + UNLUCKY_TTL

    @staticmethod
    def is_unlucky(s: dict | None) -> bool:
        """Unlucky wears off: after UNLUCKY_TTL the slot is neutral again (else every slot ends up avoided)."""
        return bool(s and s.get("unlucky") and s.get("unlucky_until", 0) > time.time())

    def mark_unplayable(self, slug: str, name: str):
        """The slot opened but has no stake strip we can drive: never pick it again."""
        s = self.slots.setdefault(slug, {"name": name, "spins": 0, "wins": 0, "net": 0.0, "best_pct": 0.0,
                                         "lucky": False, "unlucky": False, "sessions": 0, "goals": 0, "busts": 0})
        s["unplayable"] = True
        self._save()

    def reset(self):
        """Forget everything: bank back to the start, no slot memory, no events, fresh appetite."""
        self.slots = {}
        self.events = []
        self.slot, self.name, self.start = None, "", None
        self.session_spins, self.session_net = 0, 0.0
        self.loss_streak = self.win_streak = 0
        self.target_bet = None
        self.reason = ""
        self.desire = 0.6
        try:
            os.remove(self.events_path)
        except OSError:
            pass
        self.refill()

    def close_slot(self, why: str):
        s = self.slots.get(self.slot)
        if s and self.session_net < 0 and not s["lucky"] and self.session_spins >= 3:
            self._mark_unlucky(s)
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
            out.append(-9.0 if (s and s.get("unplayable")) else 1.2 if (s and s["lucky"])
                       else -2.0 if self.is_unlucky(s) else 0.3 if s is None else 0.0)
        return np.array(out, np.float32)

    def snapshot(self) -> dict:
        s = self.slots.get(self.slot) or {}
        return {"desire": round(self.desire, 3), "bank": round(self.bank, 2), "broke": self.broke,
                "bank_min": round(self.bank_min, 2), "bank_max": round(self.bank_max, 2),
                "bank_min_at": self.bank_min_at, "bank_max_at": self.bank_max_at, "since": self.since,
                "spins_total": self.spins_total,
                "slot": self.slot, "name": self.name, "start": self.start,
                "session_spins": self.session_spins, "session_net": round(self.session_net, 2),
                "loss_streak": self.loss_streak, "win_streak": self.win_streak,
                "target_bet": self.target_bet and round(self.target_bet, 2), "lucky": bool(s.get("lucky")),
                "unlucky": self.is_unlucky(s), "reason": self.reason,
                "events": self.events[-12:][::-1],
                "slots": sorted(({"slug": k, **v, "unlucky": self.is_unlucky(v)} for k, v in self.slots.items()
                                 if not v.get("unplayable")), key=lambda x: -x["spins"])[:12]}
