"""
Label/tag mode for the live router: same routing logic as the jersey router, different task.

Kept separate from router/route.py on purpose, so the evaluated jersey path is untouched.
Settings match the pre-registered label test (data/labels/RUBRIC.md, benchmark/label_test.py):
  - prompt: label_test.LABEL_PROMPT
  - confidence: MIN per-token probability over the answer (joint shrinks with length)
  - threshold: 0.90, fixed in advance
Label frames use their own offline queue (results/queue_label/), so sync re-asks Muse the label
question, not the jersey one.
"""
import os, sys, json, shutil
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from router import route as rt               # new_record(), cloud_available(); chdirs to ROOT
from benchmark import label_test as lt       # LABEL_PROMPT, edge_read(), confidence(), norm(), score()
rb, mc = rt.rb, rt.mc

THRESHOLD = lt.THRESHOLD                      # 0.90
QUEUE_DIR = os.path.join(rb.RESULTS_DIR, "queue_label")
QUEUE_FILE = os.path.join(QUEUE_DIR, "queue.jsonl")


def route_label(image_path, max_tier=3, threshold=THRESHOLD):
    rec, step, done = rt.new_record(image_path)
    rec["task"] = "label"
    rec["threshold"] = threshold
    raw, toks, t = lt.edge_read(rb.prep_read(image_path))
    pred, c = lt.norm(raw), lt.confidence(toks)
    step("edge_full", pred, t, raw=raw, conf=c["min"])
    rec["edge_conf"] = c
    rec["conf_value"] = c["min"]                 # what the dashboard bar shows for this task
    rec["conf_signal"] = "min token"
    if pred is not None and c["min"] >= threshold:
        return done(1, pred, f"edge confidence (min token) {c['min']:.3f} >= {threshold}")
    if max_tier < 3:
        rec["needs_queue"] = True
        return done(1, pred, f"LOW CONFIDENCE: min token {c['min']:.3f} < {threshold}, offline; queued for verification")
    if not rt.cloud_available():
        rec["needs_queue"] = True
        return done(1, pred, f"LOW CONFIDENCE: min token {c['min']:.3f} < {threshold}, no connection; queued for verification")
    try:
        raw_m, t_m = mc.muse_read(image_path, prompt=lt.LABEL_PROMPT)
        rec["muse_calls"] += 1
        p_muse = lt.norm(raw_m)
        step("muse_full", p_muse, t_m, raw=raw_m)
        return done(3, p_muse, "escalated to Muse Spark")
    except Exception as e:
        step("muse_full", None, None, error=str(e)[:200])
        rec["needs_queue"] = True
        return done(3, pred, "LOW CONFIDENCE: Muse call failed; edge answer kept, queued for verification")


def enqueue(image_path, rec, truth=None):
    os.makedirs(os.path.join(QUEUE_DIR, "images"), exist_ok=True)
    dst = os.path.join(QUEUE_DIR, "images", rec["image"])
    shutil.copy2(image_path, dst)
    entry = {"image": rec["image"], "path": dst, "task": "label", "edge_answer": rec["answer"],
             "edge_conf": rec.get("conf_value"), "truth": truth, "bytes": os.path.getsize(dst),
             "status": "unverified", "queued_at": datetime.now().isoformat(timespec="seconds")}
    with open(QUEUE_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")
    rec["queued"] = True


def entries():
    if not os.path.exists(QUEUE_FILE):
        return []
    return [json.loads(l) for l in open(QUEUE_FILE) if l.strip()]


def sync():
    """Verify queued label frames with Muse Spark (label prompt). Same output shape as rt.sync()."""
    all_e = entries()
    pending = [e for e in all_e if e["status"] == "unverified"]
    if not pending:
        return
    if not rt.cloud_available():
        sys.exit(f"[SYNC] Meta Model API unreachable; {len(pending)} label frames stay queued.")
    print(f"[SYNC] Verifying {len(pending)} queued label frames with Muse Spark...")
    for e in pending:
        try:
            raw, t = mc.muse_read(e["path"], prompt=lt.LABEL_PROMPT)
        except Exception as ex:
            print(f"[SYNC] {e['image']}: ERROR {str(ex)[:120]} (stays queued)")
            continue
        ans = lt.norm(raw)
        e.update({"status": "verified", "cloud_answer": ans, "cloud_latency_s": t,
                  "corrected": ans != e["edge_answer"], "verified_at": datetime.now().isoformat(timespec="seconds")})
        if e.get("truth"):
            e["edge_correct"] = lt.score(e["edge_answer"] or "", e["truth"])[0]
            e["correct"] = lt.score(ans or "", e["truth"])[0]
        verb = "changed" if e["corrected"] else "confirmed"
        print(f"[SYNC] {e['image']}: edge {e['edge_answer']} (min {e['edge_conf']}) -> Muse {ans}  {verb}  {t}s")
    with open(QUEUE_FILE, "w") as f:
        for e in all_e:
            f.write(json.dumps(e) + "\n")
