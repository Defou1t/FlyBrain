"""Sensory encoding: browser screenshot -> photoreceptor currents on the hex retina."""
from __future__ import annotations

import io

import numpy as np

from .connectome import Brain, HEX_H, HEX_W


def screenshot_to_retina(png: bytes) -> np.ndarray:
    """Grayscale image -> [2, HEX_H, HEX_W] luminance for (left eye, right eye) in [0, 1]."""
    from PIL import Image
    img = Image.open(io.BytesIO(png)).convert("L")
    w, h = img.size
    eyes = []
    for half in (img.crop((0, 0, w // 2, h)), img.crop((w // 2, 0, w, h))):
        eyes.append(np.asarray(half.resize((HEX_W, HEX_H), Image.BILINEAR), np.float32) / 255.0)
    return np.stack(eyes)


def encode(brain: Brain, png: bytes, gain: float = 1.5, contrast: bool = True) -> np.ndarray:
    """Dense external current [n]: photoreceptors get (contrast-normalised) luminance."""
    lum = screenshot_to_retina(png)
    if contrast:
        lum = (lum - lum.mean()) / (lum.std() + 1e-6)
        lum = np.clip(lum * 0.5 + 0.5, 0, 1)
    ext = np.zeros(brain.n, np.float32)
    for eye, retina in zip(lum, (brain.retina_L, brain.retina_R)):
        m = retina >= 0
        ext[retina[m]] = gain * eye[m]
    return ext


def dopamine(brain: Brain, reward: float, amplitude: float = 3.0) -> np.ndarray:
    """Aversive signal: negative reward -> current into PPL101 (as in DOOMFLY / FLYT3)."""
    ext = np.zeros(brain.n, np.float32)
    if reward < 0 and brain.dopamine.size:
        ext[brain.dopamine] = amplitude * min(1.0, -reward)
    return ext
