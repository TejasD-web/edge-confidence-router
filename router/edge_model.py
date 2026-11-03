"""
On-device reader: Meta Perception LM 1B (PLM-1B), run in-process with Hugging Face transformers.

Every edge read in the project goes through chat(): the router (route.py, task_route.py), the
threshold selection (select_threshold.py), the label test and the experiments. It returns the answer,
per-token log-probabilities in the same shape the code already used (OpenAI-style
{"token", "logprob", "top_logprobs"}), and the wall-clock latency, so the confidence code
(logprob_probe.confidence, label_test.confidence) works unchanged.

Model: facebook/Perception-LM-1B (FAIR Noncommercial Research License), bf16, ~3.1 GB peak on the
Orin Nano. Pinned to a commit so results are reproducible. Images are tiled by the PLM processor
(MAX_TILES tiles of 448 px); the full frame is passed in, no resizing on our side.

Jetson notes:
  - Load straight onto the GPU (device_map="cuda"). Loading on CPU and then .to("cuda") holds two
    copies in the shared 8 GB and fails with NvMap error 12.
  - After loading, evict the weights file from the Linux page cache. ~3 GB of cached pages can block
    NvMap GPU allocations even though `free` reports them as available.
"""
import os, time

MODEL_ID = os.environ.get("EDGE_MODEL", "facebook/Perception-LM-1B")
REVISION = os.environ.get("EDGE_REVISION", "2b1a854663b80d6c8b9a10e4b229be97c7f6be1f")
MAX_TILES = int(os.environ.get("EDGE_TILES", "4"))   # fixed before the dev run; 4 tiles = 1347 input tokens
NAME = "Perception LM 1B (Meta)"

_proc = _model = None
_load_s = None


def _drop_page_cache(path):
    try:
        fd = os.open(path, os.O_RDONLY)
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        os.close(fd)
    except (OSError, AttributeError):
        pass


def load():
    """Load PLM-1B onto the GPU once per process. Returns load time in seconds."""
    global _proc, _model, _load_s
    if _model is not None:
        return _load_s
    import gc, torch
    from transformers import AutoProcessor, AutoModelForImageTextToText
    t0 = time.time()
    rev = REVISION if MODEL_ID == "facebook/Perception-LM-1B" else None
    _proc = AutoProcessor.from_pretrained(MODEL_ID, revision=rev)
    if hasattr(_proc, "image_processor") and hasattr(_proc.image_processor, "max_num_tiles"):
        _proc.image_processor.max_num_tiles = MAX_TILES
    _model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, revision=rev, dtype=torch.bfloat16, device_map="cuda").eval()
    gc.collect()
    try:
        from huggingface_hub import try_to_load_from_cache
        p = try_to_load_from_cache(MODEL_ID, "model.safetensors", revision=rev)
        if isinstance(p, str):
            _drop_page_cache(p)
    except Exception:
        pass
    _load_s = round(time.time() - t0, 3)
    print(f"[EDGE] {NAME} loaded in {_load_s}s ({MODEL_ID}@{(rev or 'main')[:7]}, {MAX_TILES} tiles)")
    return _load_s


def unload():
    global _proc, _model
    if _model is None:
        return
    import gc, torch
    _model = _proc = None
    gc.collect()
    torch.cuda.empty_cache()
    print("[EDGE] unloaded")


def loaded():
    return _model is not None


def chat(image_path, prompt, max_tokens=16, top_logprobs=0):
    """One greedy read. Returns (text, tokens, seconds); tokens = [{"token", "logprob", "top_logprobs"}]
    for the generated answer tokens (special tokens dropped)."""
    import torch
    if _model is None:
        load()
    conv = [{"role": "user", "content": [{"type": "image", "url": os.path.abspath(image_path)},
                                         {"type": "text", "text": prompt}]}]
    inputs = _proc.apply_chat_template([conv], add_generation_prompt=True, tokenize=True,
                                       return_dict=True, return_tensors="pt").to(_model.device)
    torch.cuda.synchronize()
    t0 = time.time()
    with torch.inference_mode():
        out = _model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False,
                              output_logits=True, return_dict_in_generate=True)
    torch.cuda.synchronize()
    t = round(time.time() - t0, 3)

    tok = _proc.tokenizer
    special = set(tok.all_special_ids)
    gen = out.sequences[0, inputs["input_ids"].shape[1]:]
    toks = []
    for tid, logits in zip(gen.tolist(), out.logits):
        if tid in special:
            continue
        lsm = torch.log_softmax(logits[0].float(), dim=-1)
        entry = {"token": tok.decode([tid]), "logprob": float(lsm[tid])}
        if top_logprobs:
            v, i = torch.topk(lsm, top_logprobs)
            entry["top_logprobs"] = [{"token": tok.decode([int(j)]), "logprob": float(x)}
                                     for x, j in zip(v.tolist(), i.tolist())]
        toks.append(entry)
    text = tok.decode(gen, skip_special_tokens=True).strip()
    return text, toks, t
