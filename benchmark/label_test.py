"""
Label test (task transfer check): same edge model, same router logic, new prompt. No training, no tuning.

Rules are in data/labels/RUBRIC.md (commit it with labels.csv BEFORE running this):
  - confidence = MIN per-token probability over the answer tokens (not joint: joint shrinks with length)
  - threshold fixed at 0.90 in advance
  - scoring = uppercase, strip spaces - _ . , then exact match; "a|b" alternates
One run gives both views:
  - offline: every frame answered by the edge; low-confidence ones flagged and queued
  - after sync: queued frames re-read by Muse Spark (the online router result is the same thing)
The jersey pipeline (route.py, run_benchmark.py scoring, config/threshold.json) is not touched.

Usage (repo root, fresh-ish memory):
  python benchmark/label_test.py
  python benchmark/label_test.py --muse-all     # also read every image with Muse, for comparison
Edge model: router/edge_model.py (Meta Perception LM 1B). The Qwen2.5-VL-3B label results in the
README were produced by the code at git tag `qwen-comparison` (non-Meta model, not part of this project).
"""
import os, sys, re, csv, json, math, time, base64, inspect, argparse
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
from benchmark import run_benchmark as rb

LABEL_PROMPT = ("Read the main identifier on the label or tag in this image: a tracking number, "
                "barcode number, part number, location code, or postal code. Reply with only that identifier.")
THRESHOLD = 0.90          # fixed before any run (RUBRIC.md); do not change after seeing results
MAX_TOKENS = 40           # IDs run to ~13-17 tokens; jersey mode used 16


def norm(s):
    s = (s or "").strip().strip('"\'`').upper()
    if ":" in s:                       # "TRACKING NUMBER: EC..." -> "EC..."
        s = s.rsplit(":", 1)[1]
    return re.sub(r"[\s\-_.,]", "", s) or None


def score(raw, truth):
    pred = norm(raw)
    return pred is not None and pred in [norm(t) for t in truth.split("|")], pred


def edge_read(path, prompt=None):
    return rb.edge_chat(path, prompt or LABEL_PROMPT, max_tokens=MAX_TOKENS, top_logprobs=3)


def confidence(toks):
    """min = weakest answer token (what the router uses here); joint shown for comparison only."""
    ans = [t for t in toks if t.get("token", "").strip()]
    if not ans:
        return {"min": 0.0, "joint": 0.0, "n_tokens": 0, "weakest": None}
    lps = [t["logprob"] for t in ans]
    i = lps.index(min(lps))
    return {"min": round(math.exp(lps[i]), 4), "joint": round(math.exp(sum(lps)), 4),
            "n_tokens": len(ans), "weakest": ans[i]["token"]}


def muse_read(path):
    from router import meta_client as mc
    params = inspect.signature(mc.muse_read).parameters
    if "prompt" not in params:
        sys.exit("[MUSE] meta_client.muse_read has no `prompt` argument; it would ask for a jersey number. "
                 "Add `prompt=PROMPT` to it first (see the patch notes).")
    return mc.muse_read(path, prompt=LABEL_PROMPT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="data/labels/labels.csv")
    ap.add_argument("--images", default="data/labels/images")
    ap.add_argument("--muse-all", action="store_true", help="also read every image with Muse (comparison)")
    ap.add_argument("--prompt", help="override the edge/Muse prompt (EXPLORATORY: any non-default prompt was "
                    "chosen after seeing these images; never report its number as a clean held-out result)")
    ap.add_argument("--tiles", type=int, default=None,
                    help="edge image tiles (default = edge_model.MAX_TILES, the pre-registered setting). "
                         "Anything else is EXPLORATORY and is reported separately.")
    args = ap.parse_args()
    from router import edge_model as em
    default_tiles = em.MAX_TILES
    if args.tiles:
        em.MAX_TILES = args.tiles                  # read by edge_model.load()
    global LABEL_PROMPT
    if args.prompt:
        LABEL_PROMPT = args.prompt
    exploratory = em.MAX_TILES != default_tiles or bool(args.prompt)
    print(f"[CFG] edge {em.NAME}, {em.MAX_TILES} tiles, threshold {THRESHOLD}"
          + ("  ** EXPLORATORY (post-hoc), not the pre-registered run **" if exploratory else ""))
    if args.prompt:
        print(f"[CFG] custom prompt: {LABEL_PROMPT!r}")

    rows = list(csv.DictReader(open(args.labels, encoding="utf-8")))
    rb.mem_check()

    out = []
    proc, _ = rb.start_vlm_server()
    try:
        for r in rows:
            path = os.path.join(args.images, r["file"])
            raw, toks, t = edge_read(rb.prep_read(path))
            ok, pred = score(raw, r["truth"])
            c = confidence(toks)
            out.append({"image": r["file"], "truth": r["truth"], "difficulty": r["difficulty"],
                        "edge_raw": raw, "edge_pred": pred, "edge_correct": ok, "edge_s": t,
                        "conf": c, "kept": pred is not None and c["min"] >= THRESHOLD,
                        "bytes": os.path.getsize(path)})
            print(f"[EDGE] {r['file']}  {str(pred):<16} truth={r['truth']:<28} {'✓' if ok else '✗'}  "
                  f"min={c['min']:.3f} joint={c['joint']:.3f} weakest={c['weakest']!r}  "
                  f"{'KEEP' if out[-1]['kept'] else 'QUEUE'}")
    finally:
        rb.stop_vlm_server(proc)

    # Sync: Muse re-reads what the edge was unsure about (and everything, with --muse-all).
    for o in out:
        if o["kept"] and not args.muse_all:
            continue
        raw, t = muse_read(os.path.join(args.images, o["image"]))
        ok, pred = score(raw, o["truth"])
        o.update({"muse_raw": raw, "muse_pred": pred, "muse_correct": ok, "muse_s": t})
        print(f"[MUSE] {o['image']}  {str(pred):<16} {'✓' if ok else '✗'}  ({t}s)")

    for o in out:
        o["final_pred"] = o["edge_pred"] if o["kept"] else o.get("muse_pred")
        o["final_correct"] = o["edge_correct"] if o["kept"] else o.get("muse_correct", False)

    n = len(out)
    kept = [o for o in out if o["kept"]]
    queued = [o for o in out if not o["kept"]]
    print(f"\n{'image':<12}{'diff':<8}{'edge':<17}{'min':>6}  {'tier':<6}{'final':<17}ok")
    for o in out:
        print(f"{o['image']:<12}{o['difficulty']:<8}{str(o['edge_pred']):<17}{o['conf']['min']:>6.3f}  "
              f"{'EDGE' if o['kept'] else 'MUSE':<6}{str(o['final_pred']):<17}{'✓' if o['final_correct'] else '✗'}")
    print(f"\nOffline (edge only):  {sum(o['edge_correct'] for o in out)}/{n} correct; "
          f"{len(kept)} kept ({sum(o['edge_correct'] for o in kept)} correct), {len(queued)} queued")
    print(f"After sync (routed):  {sum(o['final_correct'] for o in out)}/{n} correct; "
          f"{len(queued)} Muse calls, {sum(o['bytes'] for o in queued) / 1e6:.2f} MB uploaded")
    if args.muse_all:
        print(f"Muse alone:           {sum(o.get('muse_correct', False) for o in out)}/{n} correct")
    print(f"Confident errors (kept but wrong): {[o['image'] for o in kept if not o['edge_correct']]}")
    print(f"n = {n}, qualitative task-transfer check; threshold fixed in advance, not tuned."
          + (" EXPLORATORY: settings chosen after seeing the pre-registered run." if exploratory else ""))
    print(f"Mean edge read: {sum(o['edge_s'] for o in out) / n:.2f}s")

    tag = f"_explore_{em.MAX_TILES}tiles" if exploratory else ""
    path = os.path.join(rb.RESULTS_DIR, f"labels{tag}_{datetime.now():%Y%m%d_%H%M%S}.json")
    json.dump({"prompt": LABEL_PROMPT, "threshold": THRESHOLD, "signal": "min token prob",
               "edge_model": f"{em.MODEL_ID}@{em.REVISION[:7]}", "edge_tiles": em.MAX_TILES,
               "exploratory": exploratory,
               "muse_all": args.muse_all, "rows": out}, open(path, "w"), indent=2)
    print(f"[LOG] Wrote {path}")


if __name__ == "__main__":
    main()
