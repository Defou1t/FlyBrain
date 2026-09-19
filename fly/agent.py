"""Episode loop: page -> retina -> LIF -> descending neurons -> readout -> click."""
from __future__ import annotations

import queue
import time

import numpy as np

from .brain import LIF
from .connectome import Brain
from .motor import Readout
from .senses import dopamine, encode


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
            if cmd == "refill":
                env.drive.refill()
                self.broke = False
                self.viz.set_state(paused=self.paused, mode=self.mode, broke=False)
                print("  refilled: the fly is back in the game")
                return


def run_episode(brain: Brain, sim: LIF, readout: Readout, env, rng: np.random.Generator,
                ticks: int = 96, train: bool = True, verbose: bool = True, viz=None, episode: int = 0,
                pace: float = 2.5, control: Control | None = None) -> float:
    if viz and hasattr(env, "on_frame"):
        env.on_frame = viz.frame           # live frames while reels spin / the strip scrolls
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
