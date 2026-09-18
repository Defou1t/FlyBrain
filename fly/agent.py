"""Episode loop: page -> retina -> LIF -> descending neurons -> readout -> click."""
from __future__ import annotations

import time

import numpy as np

from .brain import LIF
from .browser import WebEnv
from .connectome import Brain
from .motor import Readout
from .senses import dopamine, encode


def run_episode(brain: Brain, sim: LIF, readout: Readout, env: WebEnv, rng: np.random.Generator,
                ticks: int = 96, train: bool = True, verbose: bool = True, viz=None, episode: int = 0,
                pace: float = 2.5) -> float:
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
