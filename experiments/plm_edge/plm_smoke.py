"""
Smoke test: can Meta's Perception LM run on the Orin Nano as the edge reader, with per-token confidence?

Run in sam3-env (has CUDA torch). Reads one or more images with the jersey prompt and prints the
answer, latency, per-token probabilities, joint probability and weakest token, plus peak GPU memory.

  python experiments/plm_edge/plm_smoke.py data/own/images/wm_013.jpg data/own/images/wm_007.jpg
  python experiments/plm_edge/plm_smoke.py --model facebook/Perception-LM-1B --tiles 4 IMG...
"""
import argparse, time, math, sys, os

import torch
from transformers import AutoProcessor, AutoModelForImageTextToText

JERSEY_PROMPT_FALLBACK = ("What is the jersey number of the player in this image? "
                          "Answer with only the number.")


def jersey_prompt():
    """Read PROMPT from benchmark/run_benchmark.py without importing it (sam3-env lacks its deps)."""
    import ast
    src = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                       "benchmark", "run_benchmark.py")
    try:
        for node in ast.parse(open(src).read()).body:
            if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "PROMPT" for t in node.targets):
                return ast.literal_eval(node.value)
    except Exception as e:
        print(f"[warn] couldn't read PROMPT from {src} ({e})")
    print("[warn] using fallback prompt (not the benchmark prompt)")
    return JERSEY_PROMPT_FALLBACK


ap = argparse.ArgumentParser()
ap.add_argument("images", nargs="+")
ap.add_argument("--model", default="facebook/Perception-LM-1B")
ap.add_argument("--tiles", type=int, default=4, help="max image tiles (default processor uses many more)")
ap.add_argument("--prompt", default=None)
ap.add_argument("--max-new", type=int, default=16)
a = ap.parse_args()

prompt = a.prompt or jersey_prompt()
print(f"model {a.model} | tiles {a.tiles} | torch {torch.__version__} | cuda {torch.cuda.is_available()}")
print(f"prompt: {prompt!r}\n")

t0 = time.time()
proc = AutoProcessor.from_pretrained(a.model)
if hasattr(proc, "image_processor") and hasattr(proc.image_processor, "max_num_tiles"):
    proc.image_processor.max_num_tiles = a.tiles
# Load straight onto the GPU. Loading to CPU then .to("cuda") holds two copies at once, which
# doesn't fit in the Orin Nano's shared 8 GB (CPU and GPU use the same memory).
model = AutoModelForImageTextToText.from_pretrained(a.model, dtype=torch.bfloat16, device_map="cuda").eval()
import gc; gc.collect()


def drop_file_cache(repo):
    """Evict the weights file from the Linux page cache. On Jetson the ~3 GB of cached safetensors
    pages can block NvMap GPU allocations (error 12) even though `free` counts them as available."""
    try:
        from huggingface_hub import try_to_load_from_cache
        p = try_to_load_from_cache(repo, "model.safetensors")
        if isinstance(p, str):
            fd = os.open(p, os.O_RDONLY)
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            os.close(fd)
            print(f"dropped page cache for {p}")
    except Exception as e:
        print(f"[warn] couldn't drop page cache ({e})")


drop_file_cache(a.model)
print(f"load {time.time() - t0:.1f}s | params {sum(p.numel() for p in model.parameters())/1e9:.2f}B")

for img in a.images:
    conv = [{"role": "user", "content": [{"type": "image", "url": os.path.abspath(img)},
                                         {"type": "text", "text": prompt}]}]
    inputs = proc.apply_chat_template([conv], add_generation_prompt=True, tokenize=True,
                                      return_dict=True, return_tensors="pt").to(model.device)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize(); t = time.time()
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=a.max_new, do_sample=False,
                             output_scores=True, return_dict_in_generate=True)
    torch.cuda.synchronize(); dt = time.time() - t
    n_in = inputs["input_ids"].shape[1]
    gen = out.sequences[0, n_in:]
    lp = model.compute_transition_scores(out.sequences, out.scores, normalize_logits=True)[0]
    toks = [(proc.tokenizer.decode([int(i)]), math.exp(float(l))) for i, l in zip(gen, lp)
            if int(i) not in proc.tokenizer.all_special_ids]
    text = proc.tokenizer.decode(gen, skip_special_tokens=True).strip()
    digits = [p for s, p in toks if s.strip().isdigit()]
    joint = math.prod(digits) if digits else None
    weakest = min((p for _, p in toks), default=None)
    print(f"\n{os.path.basename(img)}: {text!r}  {dt:.2f}s  input tokens {n_in}  "
          f"peak GPU {torch.cuda.max_memory_allocated()/2**30:.2f} GiB")
    print("  tokens: " + "  ".join(f"{s!r}:{p:.3f}" for s, p in toks))
    print(f"  joint digit prob {joint if joint is None else round(joint, 4)}  "
          f"weakest token {weakest if weakest is None else round(weakest, 4)}")
