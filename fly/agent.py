"""Episode loop: page -> retina -> LIF -> descending neurons -> readout -> click."""
from __future__ import annotations

import queue
import time

import numpy as np

from .brain import LIF
from .connectome import Brain
from .motor import Readout
from .senses import dopamine, encode


class Restart(Exception):
    """The page changed the session cookie: exit cleanly so fly.serve starts a fresh fly."""


class SwitchMode(Exception):
    """Raised inside an episode when the page asks for another environment (browse <-> casino)."""

    def __init__(self, mode: str):
        super().__init__(mode)
        self.mode = mode


class Control:
    """Commands from the visualiser page (pause / resume / casino / browse)."""

    def __init__(self, viz, mode: str):
        self.viz = viz
        self.mode = mode
        self.paused = False
        self.broke = False
        self.readout = None
        if viz:
            viz.set_state(paused=False, mode=mode, broke=False)

    def poll(self, block: bool = False, timeout: float = 0.0) -> str | None:
        if not self.viz:
            return None
        try:
            cmd = self.viz.commands.get(block=block, timeout=timeout) if block else self.viz.commands.get_nowait()
        except queue.Empty:
            return None
        if cmd == "pause":
            self.paused = True
            print("  fly shooed away: no more actions until 'resume'")
        elif cmd == "resume":
            self.paused = False
            print("  fly is back to work")
        elif cmd in ("casino", "browse") and cmd != self.mode:
            self.paused = False
            raise SwitchMode(cmd)
        elif cmd == "restart":
            raise Restart()
        self.viz.set_state(paused=self.paused, mode=self.mode, broke=self.broke)
        return cmd

    def wait_for_refill(self, brain, sim, env, ticks: int = 16):
        """Broke: sit and smoke until the gear menu hands over money ('refill'); the brain keeps looking."""
        self.broke = True
        self.viz.set_state(paused=self.paused, mode=self.mode, broke=True)
        print("  broke - waiting for a refill (gear menu on localhost)")
        while True:
            png = getattr(env, "last_png", None)
            sim.run(ticks, encode(brain, png) if png else None, hook=self.viz.tick)
            cmd = self.poll(block=True, timeout=0.5)
            if cmd == "reset" and self.readout is not None:
                reset_all(env, self.readout, self.viz)
                cmd = "refill"
            if cmd == "refill":
                if env.drive.broke:
                    env.drive.refill()
                self.broke = False
                self.viz.set_state(paused=self.paused, mode=self.mode, broke=False)
                print("  refilled: the fly is back in the game")
                return


def reset_all(env, readout: Readout, viz):
    """The gear menu's 'reset': bank and slot memory (drive), learned readout, dopamine log, page charts."""
    import os
    if hasattr(env, "drive"):
        env.drive.reset()
        env.cum_reward = 0.0
    readout.reset()
    log_path = getattr(env, "log_path", None)
    if log_path and os.path.exists(log_path):
        try:
            with open(log_path, encoding="utf-8") as f:
                header = f.readline()
            with open(log_path, "w", encoding="utf-8") as f:
                f.write(header)
        except OSError:
            pass
    if viz:
        viz.clear_history()
    print("  reset: bank, slot memory, readout weights and the dopamine log start over")


def run_episode(brain: Brain, sim: LIF, readout: Readout, env, rng: np.random.Generator,
                ticks: int = 96, train: bool = True, verbose: bool = True, viz=None, episode: int = 0,
                pace: float = 2.5, control: Control | None = None) -> float:
    if viz and hasattr(env, "on_frame"):
        env.on_frame = viz.frame           # live frames while reels spin / the strip scrolls
    if viz and hasattr(env, "on_idle"):    # the brain keeps looking at the live picture while the env waits
        def watch(jpg, _n=[0]):
            _n[0] += 1
            sim.run(4, encode(brain, jpg) if jpg and _n[0] % 3 == 0 else None, hook=viz.tick)
        env.on_idle = watch
    if control:
        control.readout = readout
    obs = env.reset()
    trajectory, rewards, last_reward = [], [], 0.0
    done = False
    while not done:
        if not obs["mask"].any():          # nothing clickable (blank / failed page): end the episode
            print("  no actions available on this page, ending episode")
            break
        if viz:
            viz.page(obs, env.step_i, episode)
        ext = encode(brain, obs["png"]) + dopamine(brain, last_reward)
        counts = sim.run(ticks, ext, hook=viz.tick if viz else None)
        if control:
            cmd = control.poll()
            while control.paused:          # shooed away: keep the brain alive on the same picture, no clicks
                sim.run(max(8, ticks // 4), encode(brain, obs["png"]), hook=viz.tick if viz else None)
                cmd = control.poll(block=True, timeout=0.5) or cmd
            if cmd == "refill" and hasattr(env, "drive"):     # money handed over mid-game
                env.drive.refill()
            if cmd == "reset":                                # forget results and memory, start over
                reset_all(env, readout, viz)
                if hasattr(env, "drive"):
                    obs = env.reset()
                    continue
        mask = obs["mask"].copy()
        prior = env.action_prior() if hasattr(env, "action_prior") else None   # appetite / stake pattern
        a, p = readout.act(counts, mask, rng, prior)
        url, text = obs["links"][a]
        if viz:
            viz.decision(p, a, sim.firing_rate())
            if hasattr(env, "idle"):
                env.idle(pace)          # let the fly land on the link; live frames keep flowing
            else:
                time.sleep(pace)
        obs, r, done = env.step(a)
        trajectory.append((counts, mask, a, p))
        rewards.append(r)
        last_reward = r
        if viz:
            viz.reward(r, float(sum(rewards)), done, obs.get("casino"))
        if verbose:
            print(f"  step {env.step_i:2d} rate={sim.firing_rate():.3f} p={p[a]:.2f} r={r:+.1f} -> {url}  [{text}]")
    if viz:
        viz.page(obs, env.step_i, episode)
    if train and trajectory:
        readout.update(trajectory, rewards)
        readout.save()
    return float(sum(rewards))
