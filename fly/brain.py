"""Leaky integrate-and-fire simulation of the whole connectome (numpy / scipy.sparse)."""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from .connectome import Brain


class LIF:
    def __init__(self, brain: Brain, leak: float = 0.85, target_rate: float = 0.03,
                 homeostasis_gain: float = 12.0, noise: float = 0.05, seed: int = 0):
        self.brain = brain
        self.W = brain.W
        self.leak = leak
        self.noise = noise
        self.target_rate = target_rate
        self.gain = homeostasis_gain
        self.rng = np.random.default_rng(seed)
        n = brain.n
        self.v = np.zeros(n, np.float32)
        self.threshold = np.ones(n, np.float32)
        self.rate = np.full(n, target_rate, np.float32)   # running firing-rate estimate
        self.spikes = np.zeros(n, bool)
        self.tick = 0

    def step(self, external: np.ndarray | None = None) -> np.ndarray:
        """One tick. `external` = injected current per neuron (dense [n] or None)."""
        n = self.brain.n
        fired = np.flatnonzero(self.spikes)
        if fired.size:
            row = sp.csr_matrix((np.ones(fired.size, np.float32), (np.zeros(fired.size, np.int64), fired)),
                                shape=(1, n))
            syn = np.asarray((row @ self.W).todense()).ravel()
        else:
            syn = np.zeros(n, np.float32)
        self.v = self.leak * self.v + syn + self.noise * self.rng.standard_normal(n).astype(np.float32)
        if external is not None:
            self.v += external
        self.spikes = self.v >= self.threshold
        self.v[self.spikes] = 0.0
        # homeostatic threshold: keep each neuron's long-run rate near target_rate
        self.rate += 0.01 * (self.spikes.astype(np.float32) - self.rate)
        self.threshold += self.gain * 0.01 * (self.rate - self.target_rate)
        np.clip(self.threshold, 0.2, 10.0, out=self.threshold)
        self.tick += 1
        return self.spikes

    def run(self, ticks: int, external: np.ndarray | None = None, hook=None) -> np.ndarray:
        """Run `ticks` steps with constant input; return spike counts of the readout neurons.
        `hook(spikes, tick)` is called after every tick (used by the visualiser)."""
        counts = np.zeros(self.brain.n, np.int32)
        for _ in range(ticks):
            spikes = self.step(external)
            counts += spikes
            if hook is not None:
                hook(spikes, self.tick)
        return counts[self.brain.readout].astype(np.float32)

    def firing_rate(self) -> float:
        return float(self.rate.mean())
