"""
Mode 3 (EXPLORATORY): structured read + caption. Same 9 label images, same id truths, same router rule.

The model returns JSON: {"id": ..., "id_type": ..., "caption": ...}
  - id       : routed. Confidence = MIN token probability over the id value's tokens only; threshold 0.90.
  - caption  : NOT routed, never used for decisions. Shown as unverified. Scored by hand against the facts
               in data/labels/facts.csv, written BEFORE this run: facts hit (x/3) + any false statement.
Reported separately from the pre-registered label test (these images were already seen).

Usage (repo root, fresh-ish memory):
  python benchmark/describe_test.py
Then mark captions in the printed table (or in results/describe_*.json).
"""
import os, sys, re, csv, json, math, time, base64, argparse
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
from benchmark import run_benchmark as rb
from benchmark.label_test import norm, score, THRESHOLD

PROMPT = ('Look at this image from a robot camera. Reply with JSON only, no other text: '
          '{"id": "<the main identifier on a label or tag: tracking number, barcode number, part number, '
          'location code or postal code; empty string if none is readable>", '
          '"id_type": "<tracking|barcode|part|location|postal|none>", '
          '"caption": "<one sentence, at most 15 words, describing what is in the image>"}')
MAX_TOKENS = 120


def parse(text):
    """JSON if possible, else regex per field. Missing fields -> ''."""
    m = re.search(r"\{.*\}", text or "", re.S)
    if m:
        try:
            d = json.loads(m.group(0))
            return {k: str(d.get(k, "") or "") for k in ("id", "id_type", "caption")}
        except json.JSONDecodeError:
            pass
    out = {}
    for k in ("id", "id_type", "caption"):
        mm = re.search(rf'"{k}"\s*:\s*"([^"]*)"', text or "")
        out[k] = mm.group(1) if mm else ""
    return out


def id_confidence(toks):
    """Min probability over the tokens that make up the id VALUE (not the JSON scaffolding or caption)."""
    text, spans = "", []
    for t in toks:
        s = t.get("token", "")
        spans.append((len(text), len(text) + len(s), t))
        text += s
    m = re.search(r'"id"\s*:\s*"([^"]*)"', text)
    if not m or not m.group(1).strip():
        return {"min": 0.0, "joint": 0.0, "n_tokens": 0, "weakest": None}
    a, b = m.span(1)
    ans = [t for (s, e, t) in spans if s < b and e > a and t.get("token", "").strip().strip('"')]
    if not ans:
        return {"min": 0.0, "joint": 0.0, "n_tokens": 0, "weakest": None}
    lps = [t["logprob"] for t in ans]
    i = lps.index(min(lps))
    return {"min": round(math.exp(lps[i]), 4), "joint": round(math.exp(sum(lps)), 4),
            "n_tokens": len(ans), "weakest": ans[i]["token"]}


def edge_read(path):
    import requests
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    t0 = time.time()
    r = requests.post(rb.VLM_URL, json={
        "model": "qwen2.5-vl",
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
            {"type": "text", "text": PROMPT}]}],
        "max_tokens": MAX_TOKENS, "temperature": 0, "logprobs": True, "top_logprobs": 1,
    }, timeout=180)
    t = round(time.time() - t0, 3)
    ch = r.json()["choices"][0]
    return ch["message"]["content"].strip(), (ch.get("logprobs") or {}).get("content") or [], t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="data/labels/labels.csv")
    ap.add_argument("--facts", default="data/labels/facts.csv")
    ap.add_argument("--images", default="data/labels/images")
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.labels, encoding="utf-8")))
    facts = {r["file"]: r["facts"] for r in csv.DictReader(open(args.facts, encoding="utf-8"))}
    mem = next(int(l.split()[1]) for l in open("/proc/meminfo") if l.startswith("MemAvailable"))
    if mem < 4_500_000:
        sys.exit(f"[MEM] Only {mem // 1024} MB available. Drop caches: sync && sudo sysctl -w vm.drop_caches=3")
    print(f"[CFG] mode 3 (EXPLORATORY), image budget {rb.READ_MAX_TOKENS} tokens, id threshold {THRESHOLD}")

    out = []
    proc, _ = rb.start_vlm_server()
    try:
        for r in rows:
            raw, toks, t = edge_read(rb.prep_read(os.path.join(args.images, r["file"])))
            p = parse(raw)
            ok, pred = score(p["id"], r["truth"])
            c = id_confidence(toks)
            o = {"image": r["file"], "truth": r["truth"], "facts": facts.get(r["file"], ""),
                 "edge_raw": raw, "edge": p, "edge_pred": pred, "edge_correct": ok, "edge_s": t, "conf": c,
                 "kept": pred is not None and c["min"] >= THRESHOLD}
            out.append(o)
            print(f"[EDGE] {r['file']}  id={str(pred):<16} {'✓' if ok else '✗'} min={c['min']:.3f} "
                  f"{'KEEP' if o['kept'] else 'QUEUE'}  type={p['id_type']!r}\n       caption: {p['caption']!r}")
    finally:
        rb.stop_vlm_server(proc)

    from router import meta_client as mc
    for o in out:                      # Muse reads all 9: needed to compare captions edge vs cloud
        raw, t = mc.muse_read(os.path.join(args.images, o["image"]), prompt=PROMPT)
        p = parse(raw)
        ok, pred = score(p["id"], o["truth"])
        o.update({"muse_raw": raw, "muse": p, "muse_pred": pred, "muse_correct": ok, "muse_s": t,
                  "final_pred": o["edge_pred"] if o["kept"] else pred,
                  "final_correct": o["edge_correct"] if o["kept"] else ok})
        print(f"[MUSE] {o['image']}  id={str(pred):<16} {'✓' if ok else '✗'} type={p['id_type']!r}\n"
              f"       caption: {p['caption']!r}")

    n = len(out)
    print("\nCAPTIONS (mark each: facts hit x/3, any false statement?)")
    for o in out:
        print(f"\n{o['image']}  facts: {o['facts']}\n  edge: {o['edge']['caption']}\n  muse: {o['muse']['caption']}")
    print(f"\nid, offline (edge):  {sum(o['edge_correct'] for o in out)}/{n}; kept {sum(o['kept'] for o in out)}")
    print(f"id, after sync:      {sum(o['final_correct'] for o in out)}/{n}")
    print(f"id, Muse alone:      {sum(o['muse_correct'] for o in out)}/{n}")
    print(f"Edge answers with an id or caption parsed: {sum(bool(o['edge']['id'] or o['edge']['caption']) for o in out)}/{n}")
    print(f"Mean edge read: {sum(o['edge_s'] for o in out) / n:.2f}s   EXPLORATORY, n = {n}")

    path = os.path.join(rb.RESULTS_DIR, f"describe_explore_{datetime.now():%Y%m%d_%H%M%S}.json")
    json.dump({"prompt": PROMPT, "threshold": THRESHOLD, "image_tokens": rb.READ_MAX_TOKENS,
               "exploratory": True, "rows": out}, open(path, "w"), indent=2)
    print(f"[LOG] Wrote {path}")


if __name__ == "__main__":
    main()
