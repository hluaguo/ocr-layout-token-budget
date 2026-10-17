#!/usr/bin/env python3
"""Sample COMPLETE papers from TexOCR-eval and pull only the page images needed.

Why whole papers: TexOCR's own harness concatenates a paper's predicted pages and
compares that against `paper_gold/<id>.tex` (all top-level `section_set` entries).
Sampling a single page out of ten would inject ~90% edit distance from the nine
un-sampled pages. So every page `<arxiv_id>v1_<1..k>.png` of a selected paper is
included, and `paper_gold` is written next to it.

Why range requests: the 14 figure archives are 12.5 GB, but each is a plain zip
whose central directory we can read with a ~1 MB tail request. Page images are
then pulled member-by-member (~100-300 KB each), so a 240-page manifest costs
tens of MB of transfer, not gigabytes.

Buckets are paper-level tags from the annotation JSON -- there is no per-page gold
LaTeX in this dataset, so page-level tags like "contains \\begin{tabular}" are not
derivable. Priority is exclusive: table_heavy > math_heavy > mixed_layout.

Usage
-----
    python3 scripts/build_pages.py --dry-run                 # index + buckets only
    python3 scripts/build_pages.py                           # full manifest + PNGs
    python3 scripts/build_pages.py --per-bucket 10 --seed 42
    python3 scripts/build_pages.py --eval-root /path/to/TexOCR-eval   # offline

Outputs (under --out, default data/):
    pages.jsonl          one row per page: page_id, paper_id, page, bucket, image,
                         paper_gold, zip, member
    paper_gold/<id>.tex  gold LaTeX for the whole paper (top-level sections)
    images/<page_id>.png
    page_index.csv       every page render in the dataset (cached)
    buckets.csv          bucket -> candidate papers / sampled papers / pages
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import struct
import sys
import threading
import time
import urllib.error
import urllib.request
import zlib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

HF_REPO = "chengyewang/TexOCR-eval"
HF_API_TREE = "https://huggingface.co/api/datasets/{repo}/tree/main/{path}?recursive=1&expand=false"
HF_RESOLVE = "https://huggingface.co/datasets/{repo}/resolve/main/{path}"
UA = {"User-Agent": "ocr-layout-token-budget/build_pages"}

DISPLAY_MATH_RE = re.compile(r"\\begin\{(equation|align|gather|eqnarray|multline)\*?\}|\\\[")
PAGE_RE = re.compile(r"^(?P<paper>\d{4}\.\d{4,5}v\d+)_(?P<page>\d{4})\.png$")
VERSION_RE = re.compile(r"v\d+$")


def base_id(paper_id: str) -> str:
    """`2501.02066v1` -> `2501.02066`: annotation files are not versioned."""
    return VERSION_RE.sub("", paper_id)

_print_lock = threading.Lock()


def log(msg: str) -> None:
    with _print_lock:
        print(msg, file=sys.stderr, flush=True)


# --------------------------------------------------------------------------- #
# HTTP helpers
# --------------------------------------------------------------------------- #

def http_get(url: str, retries: int = 4, timeout: int = 180) -> bytes:
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as exc:
            last = exc
            if isinstance(exc, urllib.error.HTTPError) and exc.code in (401, 403, 404):
                raise
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError("GET failed: %s (%s)" % (url, last))


def http_range(url: str, start: int, end: int, retries: int = 4, timeout: int = 180) -> bytes:
    """Inclusive byte range. Falls back to a full read if the server ignores Range."""
    if end < start:
        return b""
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={**UA, "Range": "bytes=%d-%d" % (start, end)})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                data = r.read()
                if r.status == 206 or len(data) == (end - start + 1):
                    return data
                # server ignored Range: slice if we happened to get the whole file
                if len(data) > end:
                    return data[start:end + 1]
                return data
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as exc:
            last = exc
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError("RANGE failed: %s [%d-%d] (%s)" % (url, start, end, last))


# --------------------------------------------------------------------------- #
# Remote zip reading (central directory + member extraction by range)
# --------------------------------------------------------------------------- #

@dataclass
class ZipEntry:
    name: str
    offset: int          # local header offset
    comp_size: int
    uncomp_size: int
    method: int
    flags: int


class RemoteZip:
    def __init__(self, url: str, cache_path: Optional[str] = None):
        self.url = url
        self.cache_path = cache_path
        self._entries: Optional[List[ZipEntry]] = None

    def entries(self) -> List[ZipEntry]:
        if self._entries is not None:
            return self._entries
        if self.cache_path and os.path.exists(self.cache_path):
            with open(self.cache_path, encoding="utf-8") as fh:
                self._entries = [ZipEntry(**row) for row in json.load(fh)]
            return self._entries
        # EOCD lives near the end; 1 MiB of tail covers the central directory of
        # every archive in this dataset (largest seen: ~1,100 entries).
        window = 1 << 20
        tail = http_range_suffix(self.url, window)
        eocd = tail.rfind(b"PK\x05\x06")
        if eocd < 0:
            raise RuntimeError("no EOCD in tail of %s" % self.url)
        n_ent, cd_size, _cd_off = struct.unpack("<HII", tail[eocd + 10:eocd + 20])
        if cd_size > eocd:
            raise RuntimeError("central directory (%d B) larger than tail window for %s"
                               % (cd_size, self.url))
        cd = tail[eocd - cd_size:eocd]
        out: List[ZipEntry] = []
        i = 0
        while i + 46 <= len(cd) and cd[i:i + 4] == b"PK\x01\x02":
            method, = struct.unpack("<H", cd[i + 10:i + 12])
            _crc, = struct.unpack("<I", cd[i + 16:i + 20])
            comp, uncomp = struct.unpack("<II", cd[i + 20:i + 28])
            nlen, elen, clen = struct.unpack("<HHH", cd[i + 28:i + 34])
            _disk, _iattr, _eattr, off = struct.unpack("<HHII", cd[i + 34:i + 46])
            name = cd[i + 46:i + 46 + nlen].decode("utf-8", "replace")
            out.append(ZipEntry(name=name, offset=off, comp_size=comp,
                                uncomp_size=uncomp, method=method, flags=0))
            i += 46 + nlen + elen + clen
        if len(out) != n_ent:
            log("  ! %s: parsed %d of %d entries" % (self.url, len(out), n_ent))
        self._entries = out
        if self.cache_path:
            os.makedirs(os.path.dirname(self.cache_path) or ".", exist_ok=True)
            with open(self.cache_path, "w", encoding="utf-8") as fh:
                json.dump([e.__dict__ for e in out], fh)
        return out

    def read_member(self, entry: ZipEntry) -> bytes:
        head = http_range(self.url, entry.offset, entry.offset + 29)
        if head[:4] != b"PK\x03\x04":
            raise RuntimeError("bad local header for %s" % entry.name)
        nlen, elen = struct.unpack("<HH", head[26:30])
        data_start = entry.offset + 30 + nlen + elen
        raw = http_range(self.url, data_start, data_start + entry.comp_size - 1)
        if entry.method == 0:
            return raw
        if entry.method == 8:
            return zlib.decompress(raw, -15)
        raise RuntimeError("unsupported compression method %d for %s" % (entry.method, entry.name))


def http_range_suffix(url: str, length: int, retries: int = 4, timeout: int = 180) -> bytes:
    """Last `length` bytes of url (HTTP suffix range)."""
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={**UA, "Range": "bytes=-%d" % length})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as exc:
            last = exc
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError("SUFFIX RANGE failed: %s (%s)" % (url, last))


# --------------------------------------------------------------------------- #
# Dataset metadata
# --------------------------------------------------------------------------- #

def list_files(repo: str, path: str) -> List[dict]:
    url = HF_API_TREE.format(repo=repo, path=path)
    data = json.loads(http_get(url).decode("utf-8"))
    if isinstance(data, dict) and "error" in data:
        raise RuntimeError("HF API error for %s: %s" % (repo, data["error"]))
    return [x for x in data if x.get("type") == "file"]


def paper_ids_from_index(index_rows: Iterable[dict]) -> List[str]:
    return sorted({r["paper_id"] for r in index_rows})


def load_paper_json(repo: str, path: str, cache_dir: str) -> dict:
    local = os.path.join(cache_dir, os.path.basename(path))
    if os.path.exists(local):
        with open(local, encoding="utf-8") as fh:
            return json.load(fh)
    raw = http_get(HF_RESOLVE.format(repo=repo, path=path))
    os.makedirs(cache_dir, exist_ok=True)
    with open(local, "wb") as fh:
        fh.write(raw)
    return json.loads(raw.decode("utf-8"))


def paper_gold_tex(paper: dict) -> str:
    """Exactly what TexOCR's harness compares against: top-level sections, in order."""
    parts = []
    for key, val in (paper.get("section_set") or {}).items():
        if "." not in key:
            parts.append(val.get("content") or "")
    return "".join(parts)


def paper_tags(paper: dict, min_math: int, math_max_tables: int = 1) -> Dict[str, object]:
    sections = paper.get("section_set") or {}
    body = "".join((v.get("content") or "") for v in sections.values())
    math_hits = len(DISPLAY_MATH_RE.findall(body))
    n_tables = len(paper.get("table_set") or {})
    n_images = len(paper.get("image_set") or {})
    n_twocol = len(re.findall(r"\\twocolumn|\\begin\{multicols\}", body))
    buckets: List[str] = []
    if n_tables >= 3:
        buckets.append("table_heavy")
    if n_tables <= math_max_tables and math_hits >= min_math:
        # `n_tables == 0` was the intended strict rule, but no paper in the
        # 6-10 page eligible pool has zero extracted tables (min is 1), so the
        # math bucket is "math-dominant": <= math_max_tables tables plus a high
        # display-math count. The strict count is logged and written to
        # buckets.csv so the relaxation stays visible in the write-up.
        buckets.append("math_heavy")
    if n_tables >= 1 and n_images >= 1 and math_hits >= 1:
        buckets.append("mixed_layout")
    return {"bucket": buckets[0] if buckets else None, "buckets": buckets,
            "n_tables": n_tables, "n_images": n_images,
            "math_hits": math_hits, "n_twocol": n_twocol, "n_sections": len(sections)}


# --------------------------------------------------------------------------- #
# Build
# --------------------------------------------------------------------------- #

def build_index(repo: str, cache_dir: str, workers: int) -> List[dict]:
    """Every page render in the dataset: paper_id, page, zip, member."""
    figure_files = list_files(repo, "arxiv/figures")
    zips = [f for f in figure_files if f["path"].endswith(".zip")]
    log("[index] %d figure archives" % len(zips))
    rows: List[dict] = []

    def handle(f: dict) -> List[dict]:
        path = f["path"]
        url = HF_RESOLVE.format(repo=repo, path=path)
        cache = os.path.join(cache_dir, "cd", os.path.basename(path) + ".json")
        rz = RemoteZip(url, cache_path=cache)
        try:
            entries = rz.entries()
        except Exception as exc:  # network hiccup on one archive should not kill the run
            log("  ! %s: %s" % (path, exc))
            return []
        out = []
        for e in entries:
            m = PAGE_RE.match(os.path.basename(e.name))
            if not m:
                continue
            out.append({"paper_id": m.group("paper"), "page": int(m.group("page")),
                        "zip": path, "member": e.name, "offset": e.offset,
                        "comp_size": e.comp_size, "method": e.method})
        log("  %s: %d page renders" % (os.path.basename(path), len(out)))
        return out

    with ThreadPoolExecutor(max_workers=workers) as ex:
        for part in ex.map(handle, zips):
            rows.extend(part)
    rows.sort(key=lambda r: (r["paper_id"], r["page"]))
    log("[index] %d pages across %d papers" % (len(rows), len({r["paper_id"] for r in rows})))
    return rows


def select_papers(repo: str, index: List[dict], cache_dir: str, *, per_bucket: int,
                  seed: int, pages_min: int, pages_max: int, min_math: int,
                  math_max_tables: int = 1, workers: int = 8,
                  max_papers: Optional[int] = None) -> Tuple[List[dict], Dict[str, dict]]:
    by_paper: Dict[str, List[dict]] = {}
    for r in index:
        by_paper.setdefault(r["paper_id"], []).append(r)
    complete = {p: sorted(rs, key=lambda x: x["page"]) for p, rs in by_paper.items()}
    # complete papers only: pages 1..k with no gaps
    eligible = {}
    for p, rs in complete.items():
        pages = [r["page"] for r in rs]
        if pages == list(range(1, len(pages) + 1)) and pages_min <= len(pages) <= pages_max:
            eligible[p] = rs
    log("[select] %d papers with %d-%d gapless pages (of %d)"
        % (len(eligible), pages_min, pages_max, len(complete)))

    paper_json_files = list_files(repo, "arxiv/papers")
    by_id = {}
    for f in paper_json_files:
        pid = os.path.basename(f["path"]).replace(".json", "")
        by_id[pid] = f["path"]
    ids = [p for p in sorted(eligible) if base_id(p) in by_id]
    if max_papers:
        ids = ids[:max_papers]
    log("[select] %d papers have both pages and annotations" % len(ids))

    def load(pid: str) -> Tuple[str, Optional[dict]]:
        try:
            paper = load_paper_json(repo, by_id[base_id(pid)], cache_dir)
        except Exception as exc:
            log("  ! %s: %s" % (pid, exc))
            return pid, None
        return pid, paper

    tags: Dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for i, (pid, paper) in enumerate(ex.map(load, ids), 1):
            if paper is None:
                continue
            tags[pid] = {**paper_tags(paper, min_math, math_max_tables),
                         "gold": paper_gold_tex(paper)}
            if i % 100 == 0:
                log("  tagged %d/%d" % (i, len(ids)))

    strict_math = sum(1 for t in tags.values() if t["n_tables"] == 0 and t["math_hits"] >= min_math)
    log("[select] papers with n_tables == 0 and math >= %d: %d (strict math rule)"
        % (min_math, strict_math))

    # Pools are computed per rule (a paper can qualify for several), then sampled
    # disjointly in priority order, so the buckets do not share papers while each
    # pool keeps its full size.
    order = ("table_heavy", "math_heavy", "mixed_layout")
    rnd = random.Random(seed)
    taken: set = set()
    chosen: List[dict] = []
    stats = {}
    for name in order:
        pool = sorted(p for p in tags if name in tags[p]["buckets"] and p not in taken)
        pick = rnd.sample(pool, min(per_bucket, len(pool))) if pool else []
        pick.sort()
        taken.update(pick)
        stats[name] = {"candidates": len(pool), "papers": len(pick),
                       "pages": sum(len(eligible[p]) for p in pick)}
        for pid in pick:
            chosen.append({"paper_id": pid, "bucket": name, "n_pages": len(eligible[pid]),
                           **{k: tags[pid][k] for k in ("n_tables", "n_images", "math_hits")},
                           "gold": tags[pid]["gold"], "pages": eligible[pid]})
    log("[select] " + ", ".join("%s: %d papers / %d pages (of %d candidates)"
                                % (k, v["papers"], v["pages"], v["candidates"])
                                for k, v in stats.items()))
    return chosen, stats


def fetch_member(repo: str, row: dict, dest: str, workers_cache: Dict[str, RemoteZip]) -> str:
    if os.path.exists(dest) and os.path.getsize(dest) > 1000:
        return dest
    url = HF_RESOLVE.format(repo=repo, path=row["zip"])
    rz = workers_cache.get(row["zip"])
    if rz is None:
        rz = RemoteZip(url)
        workers_cache[row["zip"]] = rz
    entry = ZipEntry(name=row["member"], offset=row["offset"], comp_size=row["comp_size"],
                     uncomp_size=0, method=row["method"], flags=0)
    data = rz.read_member(entry)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp = dest + ".part"
    with open(tmp, "wb") as fh:
        fh.write(data)
    os.replace(tmp, dest)
    return dest


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--repo", default=HF_REPO)
    ap.add_argument("--out", default="data")
    ap.add_argument("--per-bucket", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--pages-min", type=int, default=6)
    ap.add_argument("--pages-max", type=int, default=10)
    ap.add_argument("--min-math", type=int, default=3, help="display-math count for math_heavy")
    ap.add_argument("--math-max-tables", type=int, default=1,
                    help="math_heavy allows at most this many tables (data has no 0-table paper)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-papers", type=int, default=None, help="cap annotation downloads (smoke test)")
    ap.add_argument("--dry-run", action="store_true", help="index + selection only, no images")
    args = ap.parse_args(argv)

    out = args.out
    cache = os.path.join(out, ".cache")
    os.makedirs(out, exist_ok=True)

    index_path = os.path.join(out, "page_index.csv")
    if os.path.exists(index_path):
        with open(index_path, encoding="utf-8") as fh:
            index = list(csv.DictReader(fh))
            for r in index:
                r["page"] = int(r["page"]); r["offset"] = int(r["offset"])
                r["comp_size"] = int(r["comp_size"]); r["method"] = int(r["method"])
        log("[index] cached: %d pages" % len(index))
    else:
        index = build_index(args.repo, cache, args.workers)
        with open(index_path, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(index[0].keys()))
            w.writeheader(); w.writerows(index)

    chosen, stats = select_papers(
        args.repo, index, cache, per_bucket=args.per_bucket, seed=args.seed,
        pages_min=args.pages_min, pages_max=args.pages_max, min_math=args.min_math,
        math_max_tables=args.math_max_tables,
        workers=args.workers, max_papers=args.max_papers)

    with open(os.path.join(out, "buckets.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["bucket", "candidate_papers", "sampled_papers", "sampled_pages"])
        for name, s in stats.items():
            w.writerow([name, s["candidates"], s["papers"], s["pages"]])

    gold_dir = os.path.join(out, "paper_gold")
    os.makedirs(gold_dir, exist_ok=True)
    for c in chosen:
        with open(os.path.join(gold_dir, base_id(c["paper_id"]) + ".tex"), "w", encoding="utf-8") as fh:
            fh.write(c["gold"])

    rows = []
    for c in chosen:
        for r in c["pages"]:
            page_id = "%s_p%03d" % (c["paper_id"].split("v")[0], r["page"])
            rows.append({
                "page_id": page_id,
                "paper_id": c["paper_id"],
                "page": r["page"],
                "n_pages": c["n_pages"],
                "bucket": c["bucket"],
                "image": os.path.join("images", page_id + ".png"),
                "paper_gold": os.path.join("paper_gold", base_id(c["paper_id"]) + ".tex"),
                "zip": r["zip"],
                "member": r["member"],
                "_offset": r["offset"], "_comp_size": r["comp_size"], "_method": r["method"],
            })

    with open(os.path.join(out, "pages.jsonl"), "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps({k: v for k, v in row.items() if not k.startswith("_")},
                                ensure_ascii=False) + "\n")
    log("[write] %d pages, %d papers -> %s/pages.jsonl"
        % (len(rows), len(chosen), out))

    if args.dry_run:
        log("[dry-run] skipping image downloads")
        return 0

    failures = 0
    cache_by_zip: Dict[str, RemoteZip] = {}
    lock = threading.Lock()
    done = [0]

    def job(row: dict) -> bool:
        dest = os.path.join(out, row["image"])
        try:
            fetch_member(args.repo, {**row, "offset": row["_offset"], "comp_size": row["_comp_size"],
                                     "method": row["_method"]}, dest, cache_by_zip)
            return True
        except Exception as exc:
            log("  ! %s: %s" % (row["page_id"], exc))
            return False
        finally:
            with lock:
                done[0] += 1
                if done[0] % 25 == 0:
                    log("  images %d/%d" % (done[0], len(rows)))

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for ok in ex.map(job, rows):
            if not ok:
                failures += 1
    log("[done] %d images, %d failed" % (len(rows) - failures, failures))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
