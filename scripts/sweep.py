#!/usr/bin/env python3
"""Resumable visual-token budget sweep for DeepSeek-OCR-2 on TexOCR-eval pages.

Everything lives here instead of in notebook cells, so a Colab session is:

    !git -C /content/ocr-layout-token-budget pull
    !python /content/ocr-layout-token-budget/scripts/sweep.py --work /content/drive/MyDrive/ocr-layout-budget

State is `runs.csv` on Drive: one row per (mode, page), appended and fsynced after
every generation, so a timeout or disconnect loses at most one page. Re-running
skips completed pairs; `--dry-run` prints the plan and ETA without loading a model.

Modes (DeepEncoder V2 only supports 144 and 256 visual tokens per view):

    Compact  768/768,  crop=False  -> 144
    Base    1024/1024, crop=False  -> 256
    Gundam  1024/768,  crop=True   -> 256 + n*144  (n = 2..6, aspect-driven)

Examples
--------
    # what would run, how long, no GPU needed
    python scripts/sweep.py --dry-run

    # one page across all modes
    python scripts/sweep.py --smoke

    # real run, 2 papers per bucket, 120 generations per session
    python scripts/sweep.py --papers-per-bucket 2 --limit 120

    # cap runaway generations (document it: truncation biases NED upward)
    python scripts/sweep.py --max-new-tokens 4096
"""

from __future__ import annotations

import argparse
import collections
import csv
import gc
import json
import os
import sys
import time
import traceback
from typing import Dict, List, Optional, Sequence, Tuple

FIELDS = ["mode", "page_id", "paper_id", "page", "bucket", "base_size", "image_size",
          "crop_mode", "num_crops", "hook_crops", "measured_visual_tokens", "token_source",
          "pred_path", "gold_paper", "gen_seconds", "pred_chars"]

CONFIGS: Dict[str, dict] = {
    "Compact": dict(base_size=768,  image_size=768,  crop_mode=False),
    "Base":    dict(base_size=1024, image_size=1024, crop_mode=False),
    "Gundam":  dict(base_size=1024, image_size=768,  crop_mode=True),
}
PATCH = 64                     # 64 px patch -> one visual token
N_CROP_MIN, N_CROP_MAX = 2, 6
SUPPORTED_VIEW_TOKENS = (144, 256)
PROMPT = "<image>\n<|grounding|>Convert the document to latex."
DEFAULT_MODEL = "deepseek-ai/DeepSeek-OCR-2"


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


# --------------------------------------------------------------------------- #
# Plan: pages, trimming, ordering, resume
# --------------------------------------------------------------------------- #

def load_pages(data_dir: str) -> List[dict]:
    path = os.path.join(data_dir, "pages.jsonl")
    if not os.path.exists(path):
        raise SystemExit("no %s -- run scripts/build_pages.py and upload data/" % path)
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def trim_per_bucket(rows: List[dict], per_bucket: int) -> List[dict]:
    """Keep all pages of the first `per_bucket` papers in each bucket (paper-complete)."""
    if per_bucket <= 0:
        return rows
    keep: Dict[str, set] = {}
    for r in sorted(rows, key=lambda x: (x["bucket"], x["paper_id"], x["page"])):
        s = keep.setdefault(r["bucket"], set())
        if r["paper_id"] in s or len(s) < per_bucket:
            s.add(r["paper_id"])
    kept = {p for s in keep.values() for p in s}
    return [r for r in rows if r["paper_id"] in kept]


def load_done(runs_path: str) -> set:
    done = set()
    if not os.path.exists(runs_path):
        return done
    with open(runs_path, newline="", encoding="utf-8", errors="replace") as fh:
        for row in csv.DictReader(fh):
            mode, page_id = (row.get("mode") or "").strip(), (row.get("page_id") or "").strip()
            if not mode or not page_id:
                continue
            try:
                chars = int(float(row.get("pred_chars") or 0))
            except ValueError:
                chars = 0
            if chars > 0:            # an empty row is not done; retry it
                done.add((mode, page_id))
    return done


def sweep_order(rows: List[dict], modes: Sequence[str]) -> List[Tuple[str, dict]]:
    """Paper-major: for each paper, all pages of one mode, then the next mode.

    A timeout therefore leaves *complete* papers, which is what paper-level NED needs.
    """
    by_paper: Dict[str, List[dict]] = {}
    for r in rows:
        by_paper.setdefault(r["paper_id"], []).append(r)
    order: List[Tuple[str, dict]] = []
    for pid in sorted(by_paper):
        for m in modes:
            for r in sorted(by_paper[pid], key=lambda x: x["page"]):
                order.append((m, r))
    return order


def smoke_target(rows: List[dict], done: set, modes: Sequence[str]) -> List[Tuple[str, dict]]:
    """First page still missing a mode, so the smoke always measures something new."""
    for r in rows:
        missing = [m for m in modes if (m, r["page_id"]) not in done]
        if missing:
            return [(m, r) for m in missing]
    return []


def view_tokens(cfg: dict, blocks: int) -> int:
    base = (cfg["base_size"] // PATCH) ** 2
    if not cfg["crop_mode"]:
        return base
    return base + blocks * (cfg["image_size"] // PATCH) ** 2


def encoder_supports(cfg: dict) -> bool:
    size = cfg["image_size"] if cfg["crop_mode"] else cfg["base_size"]
    return (size // PATCH) ** 2 in SUPPORTED_VIEW_TOKENS


def describe_plan(rows: List[dict], target: List[Tuple[str, dict]], runs_path: str,
                  modes: Sequence[str], per_bucket: int) -> None:
    per_bucket_counts = collections.Counter(r["bucket"] for r in rows)
    papers = collections.Counter()
    for r in rows:
        papers[(r["bucket"], r["paper_id"])] = 1
    log("[plan] papers per bucket: %s" % dict(collections.Counter(
        {b: sum(1 for (bb, _) in papers if bb == b) for b in per_bucket_counts})))
    log("[plan] pages per bucket:  %s" % dict(per_bucket_counts))
    log("[plan] modes: %s" % ", ".join(
        "%s(%s..%s tok)" % (m, view_tokens(CONFIGS[m], 0 if not CONFIGS[m]["crop_mode"] else N_CROP_MIN),
                            view_tokens(CONFIGS[m], 0 if not CONFIGS[m]["crop_mode"] else N_CROP_MAX))
        for m in modes))
    log("[plan] queued %d generations (of %d total)" % (len(target), len(modes) * len(rows)))

    per_mode: Dict[str, List[float]] = {}
    if os.path.exists(runs_path):
        with open(runs_path, newline="", encoding="utf-8", errors="replace") as fh:
            for row in csv.DictReader(fh):
                try:
                    per_mode.setdefault(row.get("mode", "?"), []).append(float(row["gen_seconds"]))
                except (KeyError, TypeError, ValueError):
                    continue
    if per_mode:
        allsecs = sorted(s for v in per_mode.values() for s in v)
        med = allsecs[len(allsecs) // 2]
        log("[plan] observed: " + ", ".join("%s n=%d median %.0fs"
                                           % (m, len(v), sorted(v)[len(v) // 2])
                                           for m, v in sorted(per_mode.items())))
        log("[plan] rough ETA at global median %.0fs: %.1f h  (~%.0f CU on L4 at 1.8 CU/h)"
            % (med, med * len(target) / 3600, med * len(target) / 3600 * 1.8))
    else:
        log("[plan] no timings yet -- run --smoke first")


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #

def build_model(model_id: str, attn: str, dtype: str, max_new_tokens: Optional[int]):
    import torch
    from transformers import AutoModel, AutoTokenizer

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    if not torch.cuda.is_available():
        raise SystemExit("no CUDA device -- Runtime -> Change runtime type -> L4 GPU")
    gc.collect(); torch.cuda.empty_cache()
    log("[model] free VRAM before load: %.1f GB" % (torch.cuda.mem_get_info()[0] / 1e9))

    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    # torch_dtype converts during load: `.cuda().to(bfloat16)` would hold fp32 on the
    # GPU while allocating the bf16 copy and OOM a 22 GB L4.
    model = AutoModel.from_pretrained(
        model_id, trust_remote_code=True, use_safetensors=True,
        _attn_implementation=attn,
        torch_dtype=getattr(torch, dtype),
    ).eval().cuda()

    if max_new_tokens:
        _gen = model.generate
        model.generate = lambda *a, **kw: _gen(*a, **{**kw, "max_new_tokens": int(max_new_tokens)})
        log("[model] generation capped at %d new tokens (truncation biases NED upward)"
            % max_new_tokens)

    p0 = next(model.parameters())
    log("[model] %s | dtype %s | allocated %.1f GB | free %.1f GB"
        % (p0.device, p0.dtype, torch.cuda.memory_allocated() / 1e9,
           torch.cuda.mem_get_info()[0] / 1e9))

    mod = sys.modules[model.__class__.__module__]
    return torch, tok, model, mod


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", default="/content/drive/MyDrive/ocr-layout-budget",
                    help="Drive directory holding data/, outputs/, runs.csv")
    ap.add_argument("--data", default=None, help="override the data dir (default <work>/data)")
    ap.add_argument("--runs", default=None, help="override runs.csv path")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--modes", default="Compact,Base,Gundam")
    ap.add_argument("--papers-per-bucket", type=int, default=10,
                    help="0 = use every sampled paper (236 pages / 708 generations)")
    ap.add_argument("--limit", type=int, default=120, help="generations this session")
    ap.add_argument("--smoke", action="store_true", help="one page across all modes")
    ap.add_argument("--dry-run", action="store_true", help="print the plan, no model, no GPU")
    ap.add_argument("--attn", default="eager", help="eager | flash_attention_2")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--max-new-tokens", type=int, default=None)
    ap.add_argument("--hook", action="store_true",
                    help="also cross-check crop counts with a vision-tower hook")
    args = ap.parse_args(argv)

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    for m in modes:
        if m not in CONFIGS:
            raise SystemExit("unknown mode %r (have: %s)" % (m, ", ".join(CONFIGS)))
        if not encoder_supports(CONFIGS[m]):
            raise SystemExit("mode %r is not supported by DeepEncoder V2 "
                             "(only 144 and 256 tokens per view)" % m)

    data_dir = args.data or os.path.join(args.work, "data")
    runs_path = args.runs or os.path.join(args.work, "runs.csv")
    out_root = os.path.join(args.work, "outputs")

    pages = trim_per_bucket(load_pages(data_dir), args.papers_per_bucket)
    done = load_done(runs_path)
    target = (smoke_target(pages, done, modes) if args.smoke
              else sweep_order(pages, modes))
    target = [t for t in target if (t[0], t[1]["page_id"]) not in done]
    if args.smoke:
        target = target[:len(modes)]

    describe_plan(pages, target, runs_path, modes, args.papers_per_bucket)
    if args.dry_run:
        return 0
    if not target:
        log("[done] nothing queued")
        return 0

    torch, tok, model, mod = build_model(args.model, args.attn, args.dtype, args.max_new_tokens)
    dynamic_preprocess = getattr(mod, "dynamic_preprocess", None)
    log("[model] dynamic_preprocess: %s" % ("imported" if dynamic_preprocess else "NOT FOUND"))

    hook_state = {"n_crops": None}
    if args.hook:
        def _hook(module, inputs, kwargs, output):
            for obj in list(inputs) + list(kwargs.values()):
                if torch.is_tensor(obj) and obj.dim() == 5:
                    hook_state["n_crops"] = int(obj.shape[1]); return
                if isinstance(obj, (list, tuple)) and obj and torch.is_tensor(obj[0]) and obj[0].dim() == 4:
                    hook_state["n_crops"] = len(obj); return
        for attr in ("vision_model", "vision_tower", "vision_encoder", "deepencoder", "encoder"):
            m = getattr(model, attr, None)
            if m is not None and hasattr(m, "register_forward_hook"):
                m.register_forward_hook(_hook)
                log("[model] hook on %s" % attr)

    from PIL import Image

    def crop_count(image_path: str, cfg: dict) -> Tuple[int, str]:
        if not cfg["crop_mode"]:
            return 0, "no_crop"
        if dynamic_preprocess is None:
            return N_CROP_MAX, "assumed_max"
        im = Image.open(image_path).convert("RGB")
        blocks = dynamic_preprocess(im, min_num=N_CROP_MIN, max_num=N_CROP_MAX,
                                    image_size=cfg["image_size"])
        return len(blocks), "dynamic_preprocess"

    if not os.path.exists(runs_path):
        os.makedirs(os.path.dirname(runs_path) or ".", exist_ok=True)
        with open(runs_path, "w", newline="", encoding="utf-8") as fh:
            csv.DictWriter(fh, fieldnames=FIELDS).writeheader()
            fh.flush(); os.fsync(fh.fileno())

    failures = 0
    ran = 0
    t_start = time.time()
    for mode, page in target:
        if ran >= args.limit:
            log("[stop] session limit %d reached; re-run to continue" % args.limit)
            break
        cfg = CONFIGS[mode]
        img = os.path.join(data_dir, page["image"])
        dest = os.path.join(out_root, mode, page["page_id"] + ".tex")
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        hook_state["n_crops"] = None
        t0 = time.time()
        try:
            text = model.infer(
                tok, prompt=PROMPT, image_file=img,
                output_path="/tmp/ds_out",     # infer() makedirs() this unconditionally
                base_size=cfg["base_size"],    # explicit: model default is 640
                image_size=cfg["image_size"],
                crop_mode=cfg["crop_mode"],
                test_compress=False,
                save_results=False,
                eval_mode=True,                # otherwise it streams and returns None
            )
            if isinstance(text, (list, tuple)):
                text = text[0] if text else ""
            text = text or ""
        except Exception:
            failures += 1
            log("FAIL %s %s" % (mode, page["page_id"]))
            traceback.print_exc(limit=3)
            continue

        if not text.strip():
            failures += 1
            log("EMPTY %s %s (check eval_mode) -- not logging" % (mode, page["page_id"]))
            continue

        with open(dest, "w", encoding="utf-8") as fh:
            fh.write(text)
        blocks, source = crop_count(img, cfg)
        hook_crops = hook_state["n_crops"]
        if hook_crops is not None and cfg["crop_mode"] and hook_crops != blocks:
            log("  ! crop mismatch: dynamic_preprocess=%d hook=%d (%s)"
                % (blocks, hook_crops, page["page_id"]))
        tokens = view_tokens(cfg, blocks)

        with open(runs_path, "a", newline="", encoding="utf-8") as fh:
            csv.DictWriter(fh, fieldnames=FIELDS).writerow({
                "mode": mode, "page_id": page["page_id"], "paper_id": page["paper_id"],
                "page": page["page"], "bucket": page["bucket"],
                "base_size": cfg["base_size"], "image_size": cfg["image_size"],
                "crop_mode": cfg["crop_mode"], "num_crops": blocks, "hook_crops": hook_crops,
                "measured_visual_tokens": tokens, "token_source": source,
                "pred_path": dest, "gold_paper": os.path.join(data_dir, page["gold_paper"]),
                "gen_seconds": round(time.time() - t0, 2), "pred_chars": len(text),
            })
            fh.flush(); os.fsync(fh.fileno())
        ran += 1
        elapsed = time.time() - t_start
        eta = (elapsed / ran) * (len(target) - ran) / 3600
        print("%3d %-8s %s %5dtok n=%d %6.1fs %6dch  eta %.1fh"
              % (ran, mode, page["page_id"], tokens, blocks, time.time() - t0, len(text), eta),
              flush=True)

    log("[done] ran %d, failed %d, logged total %d"
        % (ran, failures, len(load_done(runs_path))))
    return 1 if failures and not ran else 0


if __name__ == "__main__":
    raise SystemExit(main())
