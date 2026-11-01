"""
Build dev / held-out manifests and the attribution table from data/own/labels.csv.

Pre-registered (commit this script + labels.csv + RUBRIC.md BEFORE any model run):
  - Split is stratified by difficulty with a fixed seed (SEED below); DEV_FRACTION of each
    difficulty goes to dev (rounded), the rest to held-out. No hand-picking.
  - `truth` may hold pre-registered alternates as "a|b" (decided by the size rule in RUBRIC.md).
    `alt_candidates` is informational only and is NOT used for scoring.

Usage (repo root):
  python scripts/build_manifests.py            # writes data/own/dev.json, heldout.json, ATTRIBUTION.md
  python scripts/build_manifests.py --check    # validate only (images present, fields filled)
"""
import os, sys, csv, json, random, argparse
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OWN = os.path.join(ROOT, "data", "own")
LABELS = os.path.join(OWN, "labels.csv")
IMAGES = os.path.join(OWN, "images")
SEED = 20261101
DEV_FRACTION = 15 / 40
DIFFS = ("easy", "medium", "hard")
REQUIRED = ("file", "commons_url", "author", "license", "truth", "difficulty")


def load():
    rows = list(csv.DictReader(open(LABELS, newline="", encoding="utf-8")))
    errors = []
    for r in rows:
        for k in REQUIRED:
            if not (r.get(k) or "").strip():
                errors.append(f"{r.get('file')}: missing {k}")
        if r.get("difficulty") not in DIFFS:
            errors.append(f"{r['file']}: difficulty must be one of {DIFFS}")
        if not all(t.strip().isdigit() for t in r["truth"].split("|")):
            errors.append(f"{r['file']}: truth must be digits or digits|digits, got {r['truth']!r}")
        if not os.path.exists(os.path.join(IMAGES, r["file"])):
            errors.append(f"{r['file']}: not found in {IMAGES}")
        if "CONFIRM" in (r.get("notes") or ""):
            errors.append(f"{r['file']}: notes still say CONFIRM")
    files = [r["file"] for r in rows]
    if len(files) != len(set(files)):
        errors.append("duplicate file names")
    return rows, errors


def split(rows):
    rng = random.Random(SEED)
    by = defaultdict(list)
    for r in sorted(rows, key=lambda r: r["file"]):
        by[r["difficulty"]].append(r)
    dev, held = [], []
    for d in DIFFS:
        group = by[d][:]
        rng.shuffle(group)
        k = round(len(group) * DEV_FRACTION)
        dev += group[:k]
        held += group[k:]
    return sorted(dev, key=lambda r: r["file"]), sorted(held, key=lambda r: r["file"])


def entry(r):
    return {"image": r["file"], "truth": r["truth"], "difficulty": r["difficulty"],
            "source_url": r["commons_url"], "author": r["author"], "license": r["license"]}


def attribution(rows, dev_files):
    out = ["# Image attribution", "",
           "All evaluation images are from Wikimedia Commons under the licenses below. "
           "Images were downloaded at reduced resolution and converted to JPEG where needed; "
           "no other edits were made. Inclusion does not imply endorsement by the authors or the people pictured.", "",
           "| File | Split | Author | License | Source |", "| --- | --- | --- | --- | --- |"]
    for r in sorted(rows, key=lambda r: r["file"]):
        split_ = "dev" if r["file"] in dev_files else "held-out"
        out.append(f"| {r['file']} | {split_} | {r['author']} | {r['license']} | [Commons]({r['commons_url']}) |")
    return "\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    rows, errors = load()
    for e in errors:
        print("[ERR]", e)
    n_alt = sum("|" in r["truth"] for r in rows)
    counts = {d: sum(r["difficulty"] == d for r in rows) for d in DIFFS}
    print(f"[OK] {len(rows)} images  {counts}  alternates: {n_alt}")
    if args.check or errors:
        sys.exit(1 if errors else 0)
    dev, held = split(rows)
    json.dump([entry(r) for r in dev], open(os.path.join(OWN, "dev.json"), "w"), indent=2)
    json.dump([entry(r) for r in held], open(os.path.join(OWN, "heldout.json"), "w"), indent=2)
    open(os.path.join(OWN, "ATTRIBUTION.md"), "w").write(attribution(rows, {r["file"] for r in dev}))
    for name, part in (("dev", dev), ("held-out", held)):
        c = {d: sum(r["difficulty"] == d for r in part) for d in DIFFS}
        print(f"[SPLIT] {name:<8} {len(part):>2}  {c}  alternates: {sum('|' in r['truth'] for r in part)}")
    print("[WROTE] data/own/dev.json, data/own/heldout.json, data/own/ATTRIBUTION.md")


if __name__ == "__main__":
    main()
