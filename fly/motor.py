"""Motor readout: descending/motor spike counts -> softmax over available links. Trained with REINFORCE.

Only this linear layer learns; the connectome itself stays frozen (same recipe as FLYT3 / DOOMFLY).
"""
from __future__ import annotations

import os

import numpy as np


class Readout:
    def __init__(self, n_in: int, n_actions: int, lr: float = 1e-4, seed: int = 0, path: str | None = None):
        rng = np.random.default_rng(seed)
        self.W = rng.standard_normal((n_in, n_actions)).astype(np.float32) * 0.01
        self.b = np.zeros(n_actions, np.float32)
        self.value = np.zeros(n_in + 1, np.float32)   # linear value baseline
        self.lr = lr
        self.path = path
        if path and os.path.exists(path):
            z = np.load(path)
            if z["W"].shape == self.W.shape:
                self.W, self.b, self.value = z["W"], z["b"], z["value"]

    @staticmethod
    def _norm(x: np.ndarray) -> np.ndarray:
        return (x - x.mean()) / (x.std() + 1e-6)

    def policy(self, counts: np.ndarray, mask: np.ndarray) -> np.ndarray:
        x = self._norm(counts)
        logits = x @ self.W + self.b
        logits = np.where(mask, logits, -1e9)
        p = np.exp(logits - logits.max())
        return p / p.sum()

    def act(self, counts: np.ndarray, mask: np.ndarray, rng: np.random.Generator) -> tuple[int, np.ndarray]:
        p = self.policy(counts, mask)
        return int(rng.choice(len(p), p=p)), p

    def update(self, trajectory: list[tuple[np.ndarray, np.ndarray, int, np.ndarray]], rewards: list[float],
               gamma: float = 0.95, entropy: float = 0.01):
        """trajectory: [(counts, mask, action, probs)], rewards: per step. Policy gradient with value baseline."""
        G, returns = 0.0, []
        for r in reversed(rewards):
            G = r + gamma * G
            returns.append(G)
        returns.reverse()
        for (counts, mask, a, p), G in zip(trajectory, returns):
            x = self._norm(counts)
            xb = np.append(x, 1.0)
            adv = G - float(xb @ self.value)
            self.value += self.lr * adv * xb
            g = -p.copy()
            g[a] += 1.0
            g += entropy * (p * (np.log(p + 1e-9) + 1.0)) * mask   # entropy bonus keeps exploration alive
            self.W += self.lr * adv * np.outer(x, g)
            self.b += self.lr * adv * g

    def save(self):
        if self.path:
            np.savez(self.path, W=self.W, b=self.b, value=self.value)
