"""Load the MaleCNS v1.0 flat connectome into a sparse matrix + neuron index.

Source (public, CC-BY, HHMI Janelia): gs://flyem-male-cns/v1.0/connectome-data/flat-connectome/
Files expected in data/:
  annotations.feather       body-annotations-male-cns-v1.0-minconf-0.5.feather
  neurotransmitters.feather body-neurotransmitters-male-cns-v1.0.feather
  weights.feather           connectome-weights-male-cns-v1.0-minconf-0.5-significant-only.feather

`build()` converts them once into data/brain.npz (CSR, pre -> post) which `load()` reads.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
CACHE = os.path.join(DATA, "brain.npz")

BASE = "https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/"
FILES = {
    "annotations.feather": "body-annotations-male-cns-v1.0-minconf-0.5.feather",
    "neurotransmitters.feather": "body-neurotransmitters-male-cns-v1.0.feather",
    "weights.feather": "connectome-weights-male-cns-v1.0-minconf-0.5-significant-only.feather",
}

HEX_W, HEX_H = 36, 39  # optic-lobe column grid (assignedOlHex1 x assignedOlHex2)


@dataclass
class Brain:
    W: sp.csr_matrix            # [n, n]  W[pre, post] signed synaptic weight
    body_id: np.ndarray         # [n] MaleCNS bodyId
    superclass: np.ndarray      # [n] str
    cell_type: np.ndarray       # [n] str
    retina_L: np.ndarray        # [HEX_H, HEX_W] -> neuron index of an R1-R6 cell (or -1), left eye
    retina_R: np.ndarray        # same, right eye
    descending: np.ndarray      # neuron indices of top descending neurons (by out-degree)
    motor: np.ndarray           # neuron indices of top VNC motor neurons
    dopamine: np.ndarray        # PPL101 pair (aversive dopamine)
    soma: np.ndarray            # [n, 3] soma position (nm-ish voxel coords), NaN if unknown
    reward_dopamine: np.ndarray = None   # PAM cluster (appetitive dopamine)

    @property
    def n(self) -> int:
        return self.W.shape[0]

    @property
    def readout(self) -> np.ndarray:
        return np.concatenate([self.descending, self.motor])


def download():
    import urllib.request
    os.makedirs(DATA, exist_ok=True)
    for local, remote in FILES.items():
        path = os.path.join(DATA, local)
        if os.path.exists(path):
            continue
        print(f"downloading {remote} ...")
        urllib.request.urlretrieve(BASE + remote, path)


def build(n_readout: int = 512) -> Brain:
    import pandas as pd

    ann = pd.read_feather(os.path.join(DATA, "annotations.feather"))
    nt = pd.read_feather(os.path.join(DATA, "neurotransmitters.feather"))
    w = pd.read_feather(os.path.join(DATA, "weights.feather"), columns=["body_pre", "body_post", "weight"])

    bodies = np.unique(np.concatenate([w.body_pre.values, w.body_post.values, ann.bodyId.values]))
    idx = pd.Series(np.arange(len(bodies)), index=bodies)
    n = len(bodies)

    # neurotransmitter polarity: GABA / glutamate are inhibitory in the fly (consensus prediction)
    sign = pd.Series(1.0, index=bodies)
    inh = nt[nt.consensus_nt.isin(["gaba", "glutamate"])].body.unique()
    sign.loc[sign.index.intersection(inh)] = -1.0

    pre = idx.loc[w.body_pre.values].values
    post = idx.loc[w.body_post.values].values
    strength = np.log1p(w.weight.values.astype(np.float32))
    strength *= sign.loc[w.body_pre.values].values.astype(np.float32)
    W = sp.csr_matrix((strength, (pre, post)), shape=(n, n), dtype=np.float32)
    # normalise by fan-out so a hub neuron does not blow up its targets
    fanout = np.maximum(np.diff(W.indptr), 1).astype(np.float32)
    W = (sp.diags(1.0 / np.sqrt(fanout)) @ W).tocsr()

    superclass = np.full(n, "", dtype=object)
    cell_type = np.full(n, "", dtype=object)
    ai = idx.loc[ann.bodyId.values].values
    superclass[ai] = ann.superclass.fillna("").values
    cell_type[ai] = ann.type.fillna("").values
    soma = np.full((n, 3), np.nan, np.float32)
    has = ann.somaLocation.notna().values
    soma[ai[has]] = np.stack(ann.somaLocation[has].values).astype(np.float32)

    # Retina: R1-R6 photoreceptors carry no hex coords, but their lamina target L1 does.
    # Give each photoreceptor the hex column of its strongest L1 postsynaptic partner.
    l1 = ann[(ann.type == "L1") & ann.assignedOlHex1.notna()]
    l1_idx = idx.loc[l1.bodyId.values].values
    l1_hex = {i: (int(h1) - 1, int(h2) - 1, s) for i, h1, h2, s in
              zip(l1_idx, l1.assignedOlHex1.values, l1.assignedOlHex2.values, l1.somaSide.values)}
    photoreceptors = idx.loc[ann[ann.type == "R1-R6"].bodyId.values].values
    retina = {"L": -np.ones((HEX_H, HEX_W), dtype=np.int64), "R": -np.ones((HEX_H, HEX_W), dtype=np.int64)}
    Wabs = abs(W)
    for p in photoreceptors:
        lo, hi = Wabs.indptr[p], Wabs.indptr[p + 1]
        targets = [(Wabs.data[k], Wabs.indices[k]) for k in range(lo, hi) if Wabs.indices[k] in l1_hex]
        if not targets:
            continue
        _, t = max(targets)
        h1, h2, side = l1_hex[t]
        if side in retina and retina[side][h2, h1] < 0:
            retina[side][h2, h1] = p
    # columns whose photoreceptors are not traced: fall back to the column's L1 cell itself
    for i, (h1, h2, side) in l1_hex.items():
        if side in retina and retina[side][h2, h1] < 0:
            retina[side][h2, h1] = i

    outdeg = np.asarray(Wabs.sum(axis=1)).ravel()

    def top(mask, k):
        cand = np.flatnonzero(mask)
        return cand[np.argsort(-outdeg[cand])[:k]]

    descending = top(superclass == "descending_neuron", n_readout)
    motor = top(np.isin(superclass, ["vnc_motor", "cb_motor"]), n_readout)
    dopamine = np.flatnonzero(cell_type == "PPL101")
    # appetitive dopamine: PAM neurons (mushroom-body reward DANs)
    reward_dopamine = np.flatnonzero(np.char.startswith(cell_type.astype(str), "PAM"))

    brain = Brain(W, bodies, superclass, cell_type, retina["L"], retina["R"], descending, motor, dopamine, soma,
                  reward_dopamine)
    np.savez_compressed(
        CACHE, data=W.data, indices=W.indices, indptr=W.indptr, shape=np.array(W.shape),
        body_id=bodies, superclass=superclass.astype(str), cell_type=cell_type.astype(str),
        retina_L=retina["L"], retina_R=retina["R"], descending=descending, motor=motor, dopamine=dopamine,
        soma=soma, reward_dopamine=reward_dopamine,
    )
    return brain


def load() -> Brain:
    if not os.path.exists(CACHE):
        download()
        return build()
    z = np.load(CACHE, allow_pickle=False)
    if "soma" not in z or "reward_dopamine" not in z:   # cache from an older build
        return build()
    W = sp.csr_matrix((z["data"], z["indices"], z["indptr"]), shape=tuple(z["shape"]))
    return Brain(W, z["body_id"], z["superclass"], z["cell_type"], z["retina_L"], z["retina_R"],
                 z["descending"], z["motor"], z["dopamine"], z["soma"], z["reward_dopamine"])


def synthetic(n: int = 20_000, k: int = 40, seed: int = 0) -> Brain:
    """Random sparse network with the same interface, for running without the 500 MB download."""
    rng = np.random.default_rng(seed)
    pre = rng.integers(0, n, n * k)
    post = rng.integers(0, n, n * k)
    sign = np.where(rng.random(n) < 0.3, -1.0, 1.0)[pre]
    W = sp.csr_matrix((rng.random(n * k).astype(np.float32) * sign, (pre, post)), shape=(n, n))
    W = (sp.diags(1.0 / np.sqrt(np.maximum(np.diff(W.indptr), 1))) @ W).tocsr()
    perm = rng.permutation(n)
    n_ret = HEX_H * HEX_W
    retina_L = perm[:n_ret].reshape(HEX_H, HEX_W)
    retina_R = perm[n_ret:2 * n_ret].reshape(HEX_H, HEX_W)
    sc = np.full(n, "intrinsic", dtype=object)
    sc[retina_L.ravel()] = sc[retina_R.ravel()] = "ol_sensory"
    descending = perm[2 * n_ret:2 * n_ret + 512]
    motor = perm[2 * n_ret + 512:2 * n_ret + 1024]
    sc[descending] = "descending_neuron"
    sc[motor] = "vnc_motor"
    dopamine = perm[-2:]
    # fake anatomy: brain = flattened ellipsoid, VNC = elongated blob below it
    soma = rng.standard_normal((n, 3)).astype(np.float32) * np.array([1.0, 0.6, 0.45], np.float32)
    vnc = rng.random(n) < 0.15
    soma[vnc] = rng.standard_normal((vnc.sum(), 3)).astype(np.float32) * np.array([0.35, 0.35, 1.2], np.float32)         + np.array([0, 0, 2.6], np.float32)
    soma = soma * 15000 + 40000
    return Brain(W, np.arange(n), sc, np.full(n, "", dtype=object), retina_L, retina_R, descending, motor, dopamine,
                 soma, perm[-40:-2])
