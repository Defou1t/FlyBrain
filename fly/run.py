"""CLI.  python -m fly.run --synthetic --episodes 3 --headed --viz   (--episodes 0 = run forever)"""
from __future__ import annotations

import argparse
import itertools
import os
import signal
import sys
import time

import numpy as np

from . import connectome
from .agent import Control, SwitchMode, run_episode
from .brain import LIF
from .browser import SESSION_FILE, START, WebEnv, load_session_cookie
from .motor import Readout


def _interrupt(*_):
    raise KeyboardInterrupt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true", help="random 20k-neuron brain instead of MaleCNS")
    ap.add_argument("--episodes", type=int, default=1, help="0 = forever (used by fly.serve)")
    ap.add_argument("--steps", type=int, default=8, help="clicks per episode")
    ap.add_argument("--ticks", type=int, default=48, help="LIF ticks per decision")
    ap.add_argument("--headed", action="store_true", help="show the browser window")
    ap.add_argument("--no-train", action="store_true")
    ap.add_argument("--start", default=START)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--viz", action="store_true", help="open the live 3-D visualiser (slows the sim to real time)")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-open", action="store_true", help="do not open the visualiser in a browser tab")
    ap.add_argument("--pace", type=float, default=1.1, help="seconds to wait after a decision (viz only)")
    ap.add_argument("--session-file", default=SESSION_FILE,
                    help="file with the PHPSESSID cookie value (or set $FLY_PHPSESSID); the fly browses logged in")
    ap.add_argument("--casino", action="store_true", help="play a demo slot (bets = actions) instead of browsing")
    ap.add_argument("--game", default=None, help="demo game URL (must contain isMoney=false)")
    ap.add_argument("--public", action="store_true",
                    help="let colleagues watch the visualiser at http://<your-ip>:<port>/ (toggle in the page)")
    ap.add_argument("--tunnel", nargs="?", const="auto", choices=["auto", "ngrok", "lhr", "cloudflare"], default=None,
                    help="also open a public https link for viewers outside the network: "
                         "auto = ngrok if installed else lhr; ngrok; lhr = localhost.run over ssh; cloudflare")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(signal, "SIGBREAK"):   # fly.serve stops us with Ctrl+Break -> same clean path as Ctrl+C
        signal.signal(signal.SIGBREAK, _interrupt)

    t0 = time.time()
    brain = connectome.synthetic(seed=args.seed) if args.synthetic else connectome.load()
    print(f"brain: {brain.n:,} neurons, {brain.W.nnz:,} synapses, retina L/R "
          f"{int((brain.retina_L >= 0).sum())}/{int((brain.retina_R >= 0).sum())} photoreceptors, "
          f"readout {brain.readout.size} DN+motor ({time.time() - t0:.1f}s)")

    sim = LIF(brain, seed=args.seed)
    print("warming up ...", end=" ", flush=True)
    t0 = time.time()
    sim.run(200)
    print(f"baseline rate {sim.firing_rate():.3f} ({(time.time() - t0) / 200 * 1000:.1f} ms/tick)")

    viz = None
    if args.viz:
        from .viz import VizServer
        viz = VizServer(brain, port=args.port, open_browser=not args.no_open, public=args.public,
                        tunnel_provider=args.tunnel or "auto")
        if args.tunnel:
            viz.set_tunnel(True, args.tunnel)
        viz.wait_for_client(timeout=3 if args.episodes <= 0 else 30)   # supervised: do not wait for viewers

    cookie = load_session_cookie(args.session_file)
    print("session: PHPSESSID loaded from file/env, browsing as the logged-in user" if cookie else "session: guest")

    def make_env(mode: str):
        if mode == "casino":
            from .casino import DEMO_GAME, LOBBY, CasinoEnv
            e = CasinoEnv(game=args.game or DEMO_GAME, headless=not args.headed, max_steps=args.steps,
                          session_cookie=cookie, lobby=None if args.game else LOBBY)
            print(f"casino: {'demo game ' + e.game if args.game else 'the fly picks a slot from ' + LOBBY}"
                  f"  (log -> {e.log_path})")
        else:
            e = WebEnv(start=args.start, headless=not args.headed, max_steps=args.steps, session_cookie=cookie)
        ck = os.path.join(connectome.DATA, "readout" + ("_synthetic" if args.synthetic else "")
                          + ("_casino" if mode == "casino" else "") + ".npz")
        return e, Readout(brain.readout.size, e.max_actions, seed=args.seed, path=ck)

    mode = "casino" if args.casino else "browse"
    env, readout = make_env(mode)
    control = Control(viz, mode)
    rng = np.random.default_rng(args.seed)
    interrupted = False
    return_to = None                                 # 'casino' while the fly takes a walk to recover its appetite
    episodes = itertools.count(1) if args.episodes <= 0 else range(1, args.episodes + 1)
    total_label = "∞" if args.episodes <= 0 else str(args.episodes)

    def switch(new_mode: str):
        nonlocal env, readout, mode
        print(f"switching to {new_mode} mode")
        env.close()
        mode = new_mode
        env, readout = make_env(mode)
        control.mode = mode
        if viz:
            viz.set_state(paused=False, mode=mode)

    try:
        for ep in episodes:
            print(f"episode {ep}/{total_label}")
            try:
                total = run_episode(brain, sim, readout, env, rng, ticks=args.ticks, train=not args.no_train,
                                    viz=viz, episode=ep, pace=args.pace, control=control)
            except SwitchMode as sw:                 # the page asked for the other activity
                return_to = None
                switch(sw.mode)
                continue
            extra = (f"balance {env.balance} FUN, cumulative dopamine {env.cum_reward:+.2f}" if mode == "casino"
                     else f"unique pages {len(env.visited)}")
            print(f"  return {total:+.1f}, {extra}")
            if mode == "casino" and getattr(env, "broke", False) and control.viz:
                try:
                    control.wait_for_refill(brain, sim, env, ticks=max(8, args.ticks // 3))
                except SwitchMode as sw:
                    control.broke = False
                    switch(sw.mode)
                    continue
                continue                             # refilled: a fresh episode picks a slot again
            if mode == "casino" and getattr(env, "rest", False):
                return_to = "casino"                 # lost its appetite: one walk around the site, then back
                switch("browse")
            elif return_to and mode == "browse":
                return_to = None
                switch("casino")
            elif total == 0 and env.step_i == 0:
                time.sleep(5)                        # page gave nothing to click: do not hammer the site
    except KeyboardInterrupt:
        interrupted = True
        print("interrupted - closing the browser, readout is saved after every episode")
    finally:
        env.close()
    if viz and not interrupted:
        print("episodes done; visualiser still up — Ctrl+C to exit")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
    if viz and viz.tunnel:
        viz.tunnel.stop()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:     # Ctrl+C / fly.serve restart during start-up: leave quietly
        pass
