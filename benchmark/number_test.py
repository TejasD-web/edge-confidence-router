"""
Transfer test: British locomotive running numbers (asset identification), with a FORMAT HINT. PRE-REGISTERED
(rules in data/numbers/RUBRIC.md, committed with labels.csv before any run).

Question: if the robot is told what a valid ID looks like, does the same training-free router read asset
numbers on-device, and does a simple format check catch bad reads offline?

Reused unchanged from the jersey router:
  - confidence: joint digit probability (benchmark/logprob_probe.confidence -> "joint")
  - threshold: config/threshold.json (chosen on the jersey DEV set, frozen)
New, fixed before any image was collected:
  - prompt with the format (five digits)
  - format rule: normalized answer must match ^[0-9]{5}$
  - routing rule: keep on-device only if format passes AND joint >= threshold
Also reported: what confidence alone would have kept.

One run gives both views: offline (edge answers every frame; failing frames queue) and after sync (Muse re-reads
the queued frames with the same prompt).

Usage (repo root):
  python benchmark/number_test.py
  python benchmark/number_test.py --muse-all     # also read every image with Muse, for comparison
"""
import os, sys, re, csv, json, argparse
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
from benchmark import run_benchmark as rb
from benchmark.logprob_probe import confidence as digit_confidence   # joint digit probability
from benchmark.label_test import norm, score                         # alternates + normalized scoring

PROMPT = ("This is a British locomotive. Its number has exactly five digits, usually painted as two digits, "
          "a space, then three digits. Read the locomotive's number from the image. "
          "Reply with only the five digits, no spaces and nothing else.")
FORMAT = re.compile(r"^[0-9]{5}$")
MAX_TOKENS = 16


def load_threshold():
    return float(json.load(open("config/threshold.json"))["threshold"])


def format_ok(pred):
    return bool(pred) and bool(FORMAT.match(pred))


def muse_read(path):
    from router import meta_client as mc
    return mc.muse_read(path, prompt=PROMPT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="data/numbers/labels.csv")
    ap.add_argument("--images", default="data/numbers/images")
    ap.add_argument("--muse-all", action="store_true", help="also read every image with Muse (comparison)")
    args = ap.parse_args()

    threshold = load_threshold()
    rows = list(csv.DictReader(open(args.labels, encoding="utf-8")))
    if not rows:
        sys.exit("[DATA] data/numbers/labels.csv is empty")
    rb.mem_check()
    print(f"[CFG] edge {rb.em.NAME}; joint digit prob, frozen threshold {threshold}; format ^[0-9]{{5}}$; "
          f"PRE-REGISTERED (data/numbers/RUBRIC.md)")

    out = []
    proc, _ = rb.start_vlm_server()
    try:
        for r in rows:
            path = os.path.join(args.images, r["file"])
            raw, toks, t = rb.edge_chat(rb.prep_read(path), PROMPT, max_tokens=MAX_TOKENS, top_logprobs=3)
            ok, pred = score(raw, r["truth"])
            c = digit_confidence(toks)
            fmt = format_ok(pred)
            conf_ok = pred is not None and c["joint"] >= threshold
            o = {"image": r["file"], "truth": r["truth"], "difficulty": r["difficulty"],
                 "edge_raw": raw, "edge_pred": pred, "edge_correct": ok, "edge_s": t, "conf": c,
                 "format_ok": fmt, "kept_conf_only": conf_ok, "kept": conf_ok and fmt,
                 "bytes": os.path.getsize(path)}
            out.append(o)
            print(f"[EDGE] {r['file']:<12} {str(pred):<10} truth={r['truth']:<13} {'✓' if ok else '✗'}  "
                  f"joint={c['joint']:.3f}  format={'ok' if fmt else 'FAIL'}  {'KEEP' if o['kept'] else 'QUEUE'}")
    finally:
        rb.stop_vlm_server(proc)

    for o in out:
        if o["kept"] and not args.muse_all:
            continue
        raw, t = muse_read(os.path.join(args.images, o["image"]))
        ok, pred = score(raw, o["truth"])
        o.update({"muse_raw": raw, "muse_pred": pred, "muse_correct": ok, "muse_s": t})
        print(f"[MUSE] {o['image']:<12} {str(pred):<10} {'✓' if ok else '✗'}  ({t}s)")

    for o in out:
        o["final_pred"] = o["edge_pred"] if o["kept"] else o.get("muse_pred")
        o["final_correct"] = o["edge_correct"] if o["kept"] else o.get("muse_correct", False)

    n = len(out)
    kept = [o for o in out if o["kept"]]
    kept_c = [o for o in out if o["kept_conf_only"]]
    queued = [o for o in out if not o["kept"]]
    print(f"\n{'image':<12}{'diff':<8}{'edge':<10}{'joint':>7}  {'fmt':<5}{'tier':<6}{'final':<10}ok")
    for o in out:
        print(f"{o['image']:<12}{o['difficulty']:<8}{str(o['edge_pred']):<10}{o['conf']['joint']:>7.3f}  "
              f"{'ok' if o['format_ok'] else 'FAIL':<5}{'EDGE' if o['kept'] else 'MUSE':<6}"
              f"{str(o['final_pred']):<10}{'✓' if o['final_correct'] else '✗'}")
    print(f"\nEdge alone:                         {sum(o['edge_correct'] for o in out)}/{n} correct; "
          f"format passed {sum(o['format_ok'] for o in out)}/{n}")
    print(f"On-device, format + confidence:     {len(kept)}/{n} kept ({sum(o['edge_correct'] for o in kept)} correct); "
          f"confident errors {[o['image'] for o in kept if not o['edge_correct']] or 'none'}")
    print(f"On-device, confidence only (compare): {len(kept_c)}/{n} kept ({sum(o['edge_correct'] for o in kept_c)} correct); "
          f"confident errors {[o['image'] for o in kept_c if not o['edge_correct']] or 'none'}")
    print(f"After sync (routed):                {sum(o['final_correct'] for o in out)}/{n} correct; "
          f"{len(queued)} Muse calls, {sum(o['bytes'] for o in queued) / 1e6:.2f} MB uploaded")
    if args.muse_all:
        print(f"Muse alone:                         {sum(o.get('muse_correct', False) for o in out)}/{n} correct")
    print(f"Mean edge read: {sum(o['edge_s'] for o in out) / n:.2f}s   n = {n}, pre-registered, one run.")

    path = os.path.join(rb.RESULTS_DIR, f"numbers_{datetime.now():%Y%m%d_%H%M%S}.json")
    json.dump({"prompt": PROMPT, "format": FORMAT.pattern, "signal": "joint digit prob (jersey)",
               "threshold": threshold, "threshold_source": "config/threshold.json (jersey dev, frozen)",
               "routing_rule": "format AND joint >= threshold", "pre_registered": True,
               "edge_model": f"{rb.em.MODEL_ID}@{rb.em.REVISION[:7]}, {rb.em.MAX_TILES} tiles",
               "muse_all": args.muse_all, "rows": out}, open(path, "w"), indent=2)
    print(f"[LOG] Wrote {path}")


if __name__ == "__main__":
    main()
