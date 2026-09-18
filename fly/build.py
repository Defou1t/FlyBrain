"""Download MaleCNS (~560 MB) and build data/brain.npz.  python -m fly.build"""
from . import connectome

if __name__ == "__main__":
    connectome.download()
    b = connectome.build()
    print(f"built: {b.n:,} neurons, {b.W.nnz:,} synapses, retina {int((b.retina_L >= 0).sum())}+{int((b.retina_R >= 0).sum())}, "
          f"DN {b.descending.size}, motor {b.motor.size}, PPL101 {b.dopamine.size}")
