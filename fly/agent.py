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
        if viz:
            viz.set_state(paused=False, mode=mode)

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
        self.viz.set_state(paused=self.paused, mode=self.mode)
        return cmd


def run_episode(brain: Brain, sim: LIF, readout: Readout, env, rng: np.random.Generator,
                ticks: int = 96, train: bool = True, verbose: bool = True, viz=None, episode: int = 0,
                pace: float = 2.5, control: Control | None = None) -> float:
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
            control.poll()
            while control.paused:          # shooed away: keep the brain alive on the same picture, no clicks
                sim.run(max(8, ticks // 4), encode(brain, obs["png"]), hook=viz.tick if viz else None)
                control.poll(block=True, timeout=0.5)
        mask = obs["mask"].copy()
        a, p = readout.act(counts, mask, rng)
        url, text = obs["links"][a]
        if viz:
            viz.decision(p, a, sim.firing_rate())
            time.sleep(pace)            # let the fly land on the link before the page changes
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
