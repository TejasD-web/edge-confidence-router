"""
Cloud-tier comparison: SAM 3.1 (Meta API) pipelines on the LICENSED image sets.

Replaces the early-prototype SAM 3.1 numbers (which were run on images later excluded for
data-rights reasons and are not reported). All conditions here are fixed pipelines with no
tunable choices, evaluated once:

  muse_full   : Muse Spark, full image              (cloud baseline)
  sam_edge    : SAM 3.1 box -> padded crop -> EDGE Qwen read   (cloud locate, edge read)
  sam_muse    : SAM 3.1 box -> padded crop -> Muse Spark read  (fully cloud)

Sets:
  --set jerseys  data/own/heldout.json (n=25). Supplementary comparison on the held-out set;
                 the shipped router result (23/25) is unaffected and nothing is tuned here.
  --set labels   data/labels/labels.csv (n=9). EXPLORATORY (images already seen). Purpose:
                 localization control -- SAM 3.1 locates the label well, so if the edge still
                 fails on good crops, the bottleneck is reading capability, not localization.

Everything cloud-side is labeled cloud. Nothing here estimates local SAM 3 accuracy (weights
pending); at most it bounds what a future on-device segmentation tier could approach.

Usage (repo root, fresh-ish memory):
  python benchmark/cloud_sam_test.py --set jerseys
  python benchmark/cloud_sam_test.py --set labels
"""
import os, sys, csv, json, argparse
from datetime import datetime
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
from benchmark import run_benchmark as rb
from benchmark import label_test as lt
from router import meta_client as mc

SETS = {
    "jerseys": {"manifest": "data/own/heldout.json", "images": "data/own/images",
                "sam_prompt": "jersey number", "read_prompt": rb.PROMPT, "scorer": rb.score,
                "note": "held-out set; supplementary fixed-pipeline comparison (router result unaffected)"},
    "labels":  {"manifest": "data/labels/labels.csv", "images": "data/labels/images",
                "sam_prompt": "printed label or tag", "read_prompt": lt.LABEL_PROMPT, "scorer": lt.score,
                "note": "EXPLORATORY (seen images); localization control for the edge reading wall"},
}


def load_manifest(path):
    if path.endswith(".csv"):
        return [{"image": r["file"], "truth": r["truth"], "difficulty": r["difficulty"]}
                for r in csv.DictReader(open(path, encoding="utf-8"))]
    data = json.load(open(path))
    return [{"image": m["image"], "truth": str(m["truth"]), "difficulty": m.get("difficulty")} for m in data]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=SETS, required=True)
    args = ap.parse_args()
    cfg = SETS[args.set]
    rows = load_manifest(cfg["manifest"])
    scorer, read_prompt = cfg["scorer"], cfg["read_prompt"]

    mem = next(int(l.split()[1]) for l in open("/proc/meminfo") if l.startswith("MemAvailable"))
    if mem < 4_500_000:
        sys.exit(f"[MEM] Only {mem // 1024} MB available. Drop caches: sync && sudo sysctl -w vm.drop_caches=3")
    print(f"[CFG] set={args.set} n={len(rows)}  SAM prompt={cfg['sam_prompt']!r}\n[CFG] {cfg['note']}")

    out = []
    proc, _ = rb.start_vlm_server()          # edge stays warm for the sam_edge reads
    try:
        for m in rows:
            path = os.path.join(cfg["images"], m["image"])
            o = {"image": m["image"], "truth": m["truth"], "difficulty": m["difficulty"]}
            try:                              # 1. Muse, full image
                raw, t = mc.muse_read(path, prompt=read_prompt)
                ok, pred = scorer(raw, m["truth"])
                o["muse_full"] = {"raw": raw, "pred": pred, "correct": ok, "s": t}
            except Exception as e:
                o["muse_full"] = {"error": str(e)[:150]}
            try:                              # 2. SAM 3.1 locate
                boxes, raw_sam, t_sam = mc.sam_segment(path, prompt=cfg["sam_prompt"])
                o["sam"] = {"n_boxes": len(boxes), "s": t_sam}
            except Exception as e:
                boxes = []
                o["sam"] = {"error": str(e)[:150]}
            if boxes:
                box = max(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
                img = Image.open(path).convert("RGB")
                crop = mc.pad_crop(img, box)
                crop_path = os.path.join(rb.CROP_DIR, f"cst_{args.set}_{os.path.splitext(m['image'])[0]}.jpg")
                crop.save(crop_path, quality=95)
                o["sam"]["crop_size"] = f"{crop.size[0]}x{crop.size[1]}"
                try:                          # 3. edge reads the SAM 3.1 crop
                    raw, t = rb.query_vlm(rb.prep_read(crop_path), prompt=read_prompt)
                    ok, pred = scorer(raw, m["truth"])
                    o["sam_edge"] = {"raw": raw, "pred": pred, "correct": ok, "s": t}
                except Exception as e:
                    o["sam_edge"] = {"error": str(e)[:150]}
                try:                          # 4. Muse reads the same crop
                    raw, t = mc.muse_read(crop, prompt=read_prompt)
                    ok, pred = scorer(raw, m["truth"])
                    o["sam_muse"] = {"raw": raw, "pred": pred, "correct": ok, "s": t}
                except Exception as e:
                    o["sam_muse"] = {"error": str(e)[:150]}
            out.append(o)
            def _p(k):
                c = o.get(k) or {}
                return f"{str(c.get('pred')):>14}{'✓' if c.get('correct') else '✗'}" if "pred" in c else f"{'--':>14} "
            print(f"{m['image']:<12} truth={m['truth']:<28} muse={_p('muse_full')}  "
                  f"sam_edge={_p('sam_edge')}  sam_muse={_p('sam_muse')}  boxes={o['sam'].get('n_boxes', '-')}")
    finally:
        rb.stop_vlm_server(proc)

    n = len(out)
    print(f"\n=== {args.set} (n={n}) ===")
    for k, label in (("muse_full", "Muse Spark, full image (cloud)"),
                     ("sam_edge", "SAM 3.1 crop -> edge Qwen read"),
                     ("sam_muse", "SAM 3.1 crop -> Muse read (cloud)")):
        answered = [o for o in out if "pred" in (o.get(k) or {})]
        print(f"{label:<38} {sum(o[k]['correct'] for o in answered):>2}/{n}  (answered {len(answered)})")
    print(f"SAM 3.1 found a box: {sum(bool((o.get('sam') or {}).get('n_boxes')) for o in out)}/{n}")
    print(f"[NOTE] {cfg['note']}")
    print("[NOTE] Cloud results, labeled cloud. Not an estimate of local SAM 3 accuracy (weights pending).")

    path = os.path.join(rb.RESULTS_DIR, f"cloud_sam_{args.set}_{datetime.now():%Y%m%d_%H%M%S}.json")
    json.dump({"set": args.set, **{k: v for k, v in cfg.items() if k != "scorer"}, "rows": out},
              open(path, "w"), indent=2)
    print(f"[LOG] Wrote {path}")


if __name__ == "__main__":
    main()
