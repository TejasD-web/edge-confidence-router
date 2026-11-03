"""
Task registry + routing for every task except the jersey benchmark (which stays in router/route.py,
untouched, because it is the evaluated path).

Built-in tasks
  jersey  numbers, the benchmark: joint digit probability, threshold from config/threshold.json
          (selected on the dev set). Routed by router/route.py; listed here for the dashboard only.
  label   labels and tags, pre-registered (data/labels/RUBRIC.md): label prompt, weakest-token
          confidence, threshold 0.90 fixed in advance.
  loco    locomotive numbers, pre-registered (data/numbers/RUBRIC.md): the prompt states the format
          (five digits); kept on-device only if the answer is exactly five digits AND the joint digit
          probability >= the jersey dev threshold. Same rule as benchmark/number_test.py.

Custom prompts
  Anyone can ask their own question: "Custom..." on the dashboard, POST /tasks {"prompt": ...}, or
  feed.py --prompt. Kept in memory only, so they disappear when the server stops. Same router,
  weakest-token confidence, threshold 0.90, UNCALIBRATED: no dev set picked the threshold, so
  on-device answers carry no tested accuracy. Token confidence is weak evidence for free-text
  answers; short answers (a number, a code, a word) route best.

Queued frames (offline) for label and custom tasks go to results/queue_tasks/ with their prompt,
so sync asks Muse the same question the edge was asked.
"""
import os, sys, re, json, shutil, threading
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from router import route as rt               # new_record(), cloud_available(), CONF_THRESHOLD; chdirs to ROOT
from benchmark import label_test as lt       # pre-registered label prompt, threshold, confidence, scoring
from benchmark import number_test as nt      # pre-registered locomotive prompt + format rule
rb, mc, lp = rt.rb, rt.mc, rt.lp

QUEUE_DIR = os.path.join(rb.RESULTS_DIR, "queue_tasks")
QUEUE_FILE = os.path.join(QUEUE_DIR, "queue.jsonl")
DEFAULT_THRESHOLD = 0.90
MAX_TOKENS = 64
MAX_PROMPT = 600

BUILTIN = {
    "jersey": {"name": "jersey", "title": "Numbers (jersey benchmark)", "prompt": rb.PROMPT,
               "threshold": rt.CONF_THRESHOLD, "signal": "joint digit probability",
               "status": "calibrated on the dev set", "builtin": True},
    "label": {"name": "label", "title": "Labels and tags", "prompt": lt.LABEL_PROMPT,
              "threshold": lt.THRESHOLD, "signal": "weakest token",
              "status": "fixed in advance, tested on 9 images", "builtin": True},
    "loco": {"name": "loco", "title": "Locomotive numbers (format check)", "prompt": nt.PROMPT,
             "threshold": rt.CONF_THRESHOLD, "signal": "joint digit probability + five-digit format check",
             "status": "fixed in advance, tested on 25 images", "builtin": True, "format": nt.FORMAT.pattern},
}


# ── registry ──

_session = []                    # custom prompts for this server run only (never written to disk)
_lock = threading.Lock()


def all_tasks():
    with _lock:
        return list(BUILTIN.values()) + list(_session)


def get_task(name):
    return next((t for t in all_tasks() if t["name"] == name), None)


def add_custom(prompt):
    """Register a custom prompt for this server run; returns the task. Reuses an identical prompt."""
    prompt = re.sub(r"\s+", " ", (prompt or "").strip())
    if not prompt or len(prompt) > MAX_PROMPT:
        raise ValueError(f"prompt: 1-{MAX_PROMPT} characters")
    with _lock:
        same = next((t for t in _session if t["prompt"] == prompt), None)
        if same:
            return same
        n = len(_session) + 1
        short = prompt if len(prompt) <= 40 else prompt[:39] + "…"
        t = {"name": f"custom-{n}", "title": f"Custom: {short}", "prompt": prompt,
             "threshold": DEFAULT_THRESHOLD, "signal": "weakest token",
             "status": "uncalibrated · this session only", "builtin": False}
        _session.append(t)
        return t


# ── reading ──

def clean(task_name, raw):
    """Label task: the pre-registered normalizer (uppercase, no separators). Custom: tidy free text."""
    if task_name in ("label", "loco"):
        return lt.norm(raw)
    s = re.sub(r"\s+", " ", (raw or "").strip().strip('"\'`')).strip()
    return s[:200] or None


def score(task_name, answer, truth):
    """Normalized exact match (uppercase, spaces and - _ . , removed); 'a|b' lists alternates."""
    return lt.score(answer or "", truth)


def edge_read(image_path, prompt):
    """Edge read with per-token logprobs (Meta Perception LM 1B, router/edge_model.py)."""
    return rb.edge_chat(image_path, prompt, max_tokens=MAX_TOKENS, top_logprobs=1)


def route_task(image_path, task, max_tier=3):
    name, prompt, th = task["name"], task["prompt"], task["threshold"]
    note = "" if task["builtin"] else " · uncalibrated threshold"
    rec, step, done = rt.new_record(image_path)
    rec.update({"task": name, "threshold": th, "conf_signal": task["signal"]})
    raw, toks, t = edge_read(rb.prep_read(image_path), prompt)
    pred = clean(name, raw)
    if task.get("format"):
        # Locomotive rule: joint digit probability (as for jerseys) AND the answer must match the format.
        c = lp.confidence(toks)
        conf, label = c["joint"], "joint digit probability"
        fmt_ok = bool(pred) and bool(re.match(task["format"], pred))
        rec["format_ok"] = fmt_ok
        why_low = (f"format check failed ({pred!r} is not five digits)" if not fmt_ok
                   else f"{label} {conf:.3f} < {th}")
        ok = fmt_ok and conf >= th
        why_ok = f"{label} {conf:.3f} >= {th}, format ok (five digits)"
    else:
        c = lt.confidence(toks)
        conf, label = c["min"], "weakest token"
        ok = pred is not None and conf >= th
        why_low, why_ok = f"{label} {conf:.3f} < {th}", f"edge confidence ({label}) {conf:.3f} >= {th}"
    step("edge_full", pred, t, raw=raw, conf=conf)
    rec["edge_conf"] = c
    rec["conf_value"] = conf
    if ok:
        return done(1, pred, f"{why_ok}{note}")
    if max_tier < 3:
        rec["needs_queue"] = True
        return done(1, pred, f"LOW CONFIDENCE: {why_low}, offline; queued for verification{note}")
    if not rt.cloud_available():
        rec["needs_queue"] = True
        return done(1, pred, f"LOW CONFIDENCE: {why_low}, no connection; queued for verification{note}")
    try:
        raw_m, t_m = mc.muse_read(image_path, prompt=prompt)
        rec["muse_calls"] += 1
        p_muse = clean(name, raw_m)
        step("muse_full", p_muse, t_m, raw=raw_m)
        return done(3, p_muse, f"escalated to Muse Spark{note}")
    except Exception as e:
        step("muse_full", None, None, error=str(e)[:200])
        rec["needs_queue"] = True
        return done(3, pred, "LOW CONFIDENCE: Muse call failed; edge answer kept, queued for verification")


# ── offline queue ──

def enqueue(image_path, rec, task, truth=None):
    os.makedirs(os.path.join(QUEUE_DIR, "images"), exist_ok=True)
    dst = os.path.join(QUEUE_DIR, "images", rec["image"])
    shutil.copy2(image_path, dst)
    entry = {"image": rec["image"], "path": dst, "task": task["name"], "prompt": task["prompt"],
             "edge_answer": rec["answer"], "edge_conf": rec.get("conf_value"), "truth": truth,
             "bytes": os.path.getsize(dst), "status": "unverified",
             "queued_at": datetime.now().isoformat(timespec="seconds")}
    with open(QUEUE_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")
    rec["queued"] = True


def entries():
    if not os.path.exists(QUEUE_FILE):
        return []
    return [json.loads(l) for l in open(QUEUE_FILE) if l.strip()]


def sync():
    """Verify queued frames with Muse Spark, asking each frame's own task prompt."""
    all_e = entries()
    pending = [e for e in all_e if e["status"] == "unverified"]
    if not pending:
        return
    if not rt.cloud_available():
        sys.exit(f"[SYNC] Meta Model API unreachable; {len(pending)} task frames stay queued.")
    print(f"[SYNC] Verifying {len(pending)} queued task frames with Muse Spark...")
    for e in pending:
        try:
            raw, t = mc.muse_read(e["path"], prompt=e["prompt"])
        except Exception as ex:
            print(f"[SYNC] {e['image']}: ERROR {str(ex)[:120]} (stays queued)")
            continue
        ans = clean(e["task"], raw)
        e.update({"status": "verified", "cloud_answer": ans, "cloud_latency_s": t,
                  "corrected": ans != e["edge_answer"], "verified_at": datetime.now().isoformat(timespec="seconds")})
        if e.get("truth"):
            e["edge_correct"] = score(e["task"], e["edge_answer"], e["truth"])[0]
            e["correct"] = score(e["task"], ans, e["truth"])[0]
        print(f"[SYNC] {e['image']} [{e['task']}]: edge {e['edge_answer']} -> Muse {ans}  "
              f"{'changed' if e['corrected'] else 'confirmed'}  {t}s")
    with open(QUEUE_FILE, "w") as f:
        for e in all_e:
            f.write(json.dumps(e) + "\n")
