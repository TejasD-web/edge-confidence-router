"""
EXPLORATORY: does SAM 2.1 crop-then-read help the edge on labels? (on the already-seen 9 label images)

Pipeline per image (all on-device):
  1. Qwen grounds the label  -> bounding box          (reuses run_benchmark.ground)
  2. SAM 2.1 segments inside that box -> tight crop    (reuses run_benchmark.run_sam_batch; CPU)
  3. Edge reads the crop at the 256-token budget       (reuses label_test.edge_read + confidence)
  4. Route on min-token confidence, threshold 0.90     (same as the label test)
Compared against the full-frame label run (results/labels_20261102_004720.json).

This is EXPLORATORY: these images were already seen, SAM adds a grounding+segment cascade (slow, not the
~1s edge path), and nothing here is a pre-registered result. Truths are unchanged.

Usage (repo root, fresh-ish memory):
  python benchmark/sam_label_test.py
  python benchmark/sam_label_test.py --prompt "<read prompt>"   # combine SAM + a different prompt
  python benchmark/sam_label_test.py --point-only               # skip Qwen grounding; SAM center-point crop
"""
import os, sys, csv, json, argparse
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
from benchmark import run_benchmark as rb
from benchmark import label_test as lt

GROUND_LABEL = ('Locate the main printed identifier on the label or tag (a tracking number, barcode number, '
                'part number, location code or postal code). '
                'Output its bounding box as JSON: {"bbox_2d": [x1, y1, x2, y2]}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="data/labels/labels.csv")
    ap.add_argument("--images", default="data/labels/images")
    ap.add_argument("--prompt", help="edge read-prompt override (default: the label-test prompt)")
    ap.add_argument("--point-only", action="store_true", help="skip Qwen grounding; SAM center-point crop only")
    args = ap.parse_args()
    if args.prompt:
        lt.LABEL_PROMPT = args.prompt
    rb.GROUND_PROMPT = GROUND_LABEL          # ground() reads this at call time

    rows = list(csv.DictReader(open(args.labels, encoding="utf-8")))
    mem = next(int(l.split()[1]) for l in open("/proc/meminfo") if l.startswith("MemAvailable"))
    if mem < 4_500_000:
        sys.exit(f"[MEM] Only {mem // 1024} MB available. Drop caches: sync && sudo sysctl -w vm.drop_caches=3")
    print(f"[CFG] ** EXPLORATORY ** SAM 2.1 crop-then-read on labels. threshold {lt.THRESHOLD}, "
          f"read prompt {lt.LABEL_PROMPT!r}")
    print(f"[CFG] grounding: {'OFF (center point)' if args.point_only else 'Qwen box -> SAM'}")

    out = []
    proc, _ = rb.start_vlm_server()
    try:
        boxes = {}
        if not args.point_only:
            for r in rows:
                box, graw, gt = rb.ground(os.path.join(args.images, r["file"]))
                boxes[r["file"]] = box
                print(f"[GRND] {r['file']:<12} {box if box else 'NO BOX ' + repr(graw[:50])}  ({gt}s)")
        jobs = [{"path": os.path.join(args.images, r["file"]), "box": boxes.get(r["file"])} for r in rows]
        crops, _ = rb.run_sam_batch(jobs)                 # SAM on CPU; crop files in results/crops/
        for r in rows:
            ev = crops.get(r["file"], {})
            crop_path = ev.get("box_crop_path") or ev.get("crop_path")
            used = "sam-box" if ev.get("box_crop_path") else ("sam-point" if ev.get("crop_path") else "FULL(fallback)")
            if not crop_path:
                crop_path = os.path.join(args.images, r["file"])
            raw, toks, t = lt.edge_read(rb.prep_read(crop_path))
            ok, pred = lt.score(raw, r["truth"])
            c = lt.confidence(toks)
            o = {"image": r["file"], "truth": r["truth"], "difficulty": r["difficulty"], "crop": used,
                 "crop_size": ev.get("box_crop_size") or ev.get("crop_size"), "edge_raw": raw,
                 "edge_pred": pred, "edge_correct": ok, "edge_s": t, "conf": c,
                 "kept": pred is not None and c["min"] >= lt.THRESHOLD,
                 "bytes": os.path.getsize(os.path.join(args.images, r["file"]))}
            out.append(o)
            print(f"[EDGE] {r['file']:<12} {str(pred):<16} truth={r['truth']:<28} {'✓' if ok else '✗'}  "
                  f"min={c['min']:.3f}  {used:<12} {o['crop_size']}  {'KEEP' if o['kept'] else 'QUEUE'}")
    finally:
        rb.stop_vlm_server(proc)

    for o in out:
        if o["kept"]:
            continue
        raw, t = lt.muse_read(os.path.join(args.images, o["image"]))   # Muse reads the FULL image
        ok, pred = lt.score(raw, o["truth"])
        o.update({"muse_pred": pred, "muse_correct": ok, "muse_s": t})
        print(f"[MUSE] {o['image']:<12} {str(pred):<16} {'✓' if ok else '✗'}  ({t}s)")
    for o in out:
        o["final_correct"] = o["edge_correct"] if o["kept"] else o.get("muse_correct", False)

    n = len(out)
    kept = [o for o in out if o["kept"]]
    print(f"\nEdge (SAM crop) alone:  {sum(o['edge_correct'] for o in out)}/{n} correct")
    print(f"On-device (kept):       {len(kept)}/{n}  ({sum(o['edge_correct'] for o in kept)} correct)")
    print(f"After sync:             {sum(o['final_correct'] for o in out)}/{n}")
    print(f"CONFIDENT ERRORS:       {[o['image'] for o in kept if not o['edge_correct']] or 'none'}")
    print(f"Grounding found a box:  {sum(o['crop'] == 'sam-box' for o in out)}/{n}")
    print(f"Mean edge read: {sum(o['edge_s'] for o in out) / n:.2f}s   (+ grounding + SAM, not counted)")
    print("Baseline full-frame label run: edge 2/9, 0 kept, 8/9 after sync. EXPLORATORY, seen images.")

    path = os.path.join(rb.RESULTS_DIR, f"labels_sam_explore_{datetime.now():%Y%m%d_%H%M%S}.json")
    json.dump({"read_prompt": lt.LABEL_PROMPT, "ground_prompt": GROUND_LABEL, "threshold": lt.THRESHOLD,
               "point_only": args.point_only, "exploratory": True, "rows": out}, open(path, "w"), indent=2)
    print(f"[LOG] Wrote {path}")


if __name__ == "__main__":
    main()
