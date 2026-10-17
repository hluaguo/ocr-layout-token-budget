#!/usr/bin/env python3
"""Three-metric evaluator: NED (text), pdflatex pass rate (structure),
tabular well-formedness (layout proxy).

Deliberately stdlib-only so it runs anywhere: the Mac (no torch, no tex),
a Colab CPU runtime, or next to the GPU box.

Design notes that matter for interpreting the numbers
-----------------------------------------------------
1. ``pdflatex`` pass rate is only meaningful if the wrapper preamble is fixed
   and reported. A LaTeX fragment from OCR is not a document; we wrap it in
   TEX_TEMPLATE. If you change the template, pass rates are no longer
   comparable across runs -- hence ``--template-sha`` lands in the output.
2. NED is computed on a *text projection* of the LaTeX (control sequences
   dropped, braces dropped, math delimiters dropped, escaped chars unwrapped).
   On the equation bucket this projection is mostly symbols, so NED there is
   expected to be flat-ish by construction -- do not read it as "text is fine".
3. The tabular proxy is not TEDS. It checks balanced \\begin/\\end, and that
   every row in a table has the same ampersand count as the first row, with
   `\\multicolumn{k}` rows exempted (they legitimately span k columns).
   Escaped ``\\&`` is ignored. It catches merged/duplicated columns, which is
   the failure mode we care about; it does not catch content in the wrong cell.

Usage
-----
    python3 scripts/score.py --self-test
    python3 scripts/score.py --runs runs.csv --out scores.csv --jobs 4
    python3 scripts/score.py --runs runs.csv --out scores.csv --no-compile

``runs.csv`` (written by the sweep notebook) needs a header and one row per
(mode, page, bucket):

    mode,page_id,bucket,vision_tokens,pred_path,gold_path

``gold_path`` may be "" if unavailable (NED then skipped for that row).
"""

from __future__ import annotations

import argparse
import csv
import difflib
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Tuple

# --------------------------------------------------------------------------- #
# LaTeX wrapper for the compile test (part of the metric -- keep it fixed)
# --------------------------------------------------------------------------- #

TEX_TEMPLATE = r"""\documentclass[10pt]{article}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{amsmath,amssymb,amsfonts,mathtools}
\usepackage{geometry}
\usepackage{booktabs,longtable,array,multirow,tabularx,makecell}
\usepackage{graphicx,xcolor,hyperref,caption,float}
\geometry{margin=2cm}
\setlength{\parindent}{0pt}
\begin{document}
%BODY%
\end{document}
"""

TEMPLATE_SHA = hashlib.sha256(TEX_TEMPLATE.encode()).hexdigest()[:12]

# Fallback text projection when rapidfuzz/Levenshtein is unavailable: the
# opcode cost of difflib is a valid (occasionally loose) edit script length.
try:  # pragma: no cover - depends on env
    from rapidfuzz.distance import Levenshtein as _RFLevenshtein  # type: ignore

    _HAVE_RF = True
except Exception:  # pragma: no cover
    try:
        import Levenshtein as _L  # type: ignore

        _RFLevenshtein = None
        _HAVE_RF = True
    except Exception:
        _RFLevenshtein = None
        _HAVE_RF = False


# --------------------------------------------------------------------------- #
# 1. Text projection + NED
# --------------------------------------------------------------------------- #

_COMMENT_RE = re.compile(r"(?<!\\)%.*$", re.MULTILINE)
_ESCAPED = {
    r"\&": "&", r"\%": "%", r"\_": "_", r"\#": "#", r"\$": "$",
    r"\{": "{", r"\}": "}", r"\textbackslash": "\\", r"\,": " ",
    r"\;": " ", r"\:": " ", r"\!": "", r"\\": " ",
}
_CTRL_RE = re.compile(r"\\[a-zA-Z@]+\*?|\\[^a-zA-Z@\s]")
_BRACES_RE = re.compile(r"[{}]")
_MATH_DELIM_RE = re.compile(r"\$+|\\\[|\\\]|\\\(|\\\)")
_WS_RE = re.compile(r"\s+")


def strip_latex(tex: str) -> str:
    """Project LaTeX to comparable plain text (deterministic, no deps)."""
    if not tex:
        return ""
    s = _COMMENT_RE.sub("", tex)
    for k, v in _ESCAPED.items():
        s = s.replace(k, v)
    # environments like \begin{tabular} become their name; useful signal that
    # structure existed, cheap to keep, symmetric between gold and pred.
    s = re.sub(r"\\(?:begin|end)\s*\{([^}]*)\}", r" \1 ", s)
    s = _CTRL_RE.sub(" ", s)
    s = _BRACES_RE.sub(" ", s)
    s = _MATH_DELIM_RE.sub(" ", s)
    s = s.replace("&", " & ")
    return _WS_RE.sub(" ", s).strip()


def _opcode_cost(a: str, b: str) -> int:
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    cost = 0
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        cost += max(i2 - i1, j2 - j1)
    return cost


def edit_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    if _HAVE_RF:
        if _RFLevenshtein is not None:  # rapidfuzz
            return int(_RFLevenshtein.distance(a, b))
        return int(_L.distance(a, b))  # python-Levenshtein
    return _opcode_cost(a, b)


def ned(ref: str, hyp: str) -> float:
    """Normalised edit distance in [0, 1]; 0 = identical."""
    if not ref and not hyp:
        return 0.0
    d = edit_distance(ref, hyp)
    return d / max(len(ref), len(hyp))


# --------------------------------------------------------------------------- #
# 2. Tabular well-formedness (layout proxy, not TEDS)
# --------------------------------------------------------------------------- #

_TAB_ENVS = ("tabular", "tabular*", "tabularx", "longtable", "array")


@dataclass
class TableReport:
    total: int = 0
    ok: int = 0
    problems: List[str] = field(default_factory=list)

    @property
    def parity(self) -> Optional[float]:
        return (self.ok / self.total) if self.total else None


def _find_env_blocks(tex: str, envs: Iterable[str]) -> List[str]:
    """Return inner text of every top-level matched env block (brace/stack aware)."""
    out: List[str] = []
    i = 0
    pattern = re.compile(r"\\(begin|end)\s*\{(%s)\*?\}" % "|".join(map(re.escape, envs)))
    depth = 0
    start = 0
    while True:
        m = pattern.search(tex, i)
        if not m:
            break
        if m.group(1) == "begin":
            if depth == 0:
                start = m.end()
            depth += 1
        else:
            depth -= 1
            if depth == 0 and start:
                out.append(tex[start:m.start()])
        i = m.end()
    return out


def _split_rows(body: str) -> List[str]:
    """Split on \\\\ at brace depth 0."""
    rows, buf, depth = [], [], 0
    i = 0
    while i < len(body):
        ch = body[i]
        if ch == "\\" and i + 1 < len(body):
            nxt = body[i + 1]
            if nxt == "\\" and depth == 0:
                rows.append("".join(buf))
                buf = []
                i += 2
                continue
            buf.append(body[i:i + 2])
            i += 2
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth = max(0, depth - 1)
        buf.append(ch)
        i += 1
    if "".join(buf).strip():
        rows.append("".join(buf))
    return rows


def _amp_count(row: str) -> int:
    """Unescaped & at brace depth 0, ignoring \\multicolumn spans."""
    depth, count, i = 0, 0, 0
    while i < len(row):
        ch = row[i]
        if ch == "\\" and i + 1 < len(row):
            i += 2
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth = max(0, depth - 1)
        elif ch == "&" and depth == 0:
            count += 1
        i += 1
    return count


_ROW_NOISE_RE = re.compile(r"\\(hline|toprule|midrule|bottomrule|cline\s*\{[^}]*\}|addlinespace|endhead|endfirsthead)")


def check_tabulars(tex: str) -> TableReport:
    rep = TableReport()
    for body in _find_env_blocks(tex, _TAB_ENVS):
        rep.total += 1
        rows = []
        for r in _split_rows(body):
            r = _ROW_NOISE_RE.sub(" ", r).strip()
            if r:
                rows.append(r)
        if not rows:
            rep.ok += 1  # empty table is not a layout failure
            continue
        # `\\multicolumn{k}` rows legitimately span k columns, and header rows
        # often use them, so the baseline column count comes from the widest
        # plain row, not from row 0.
        plain = [_amp_count(r) for r in rows if "\\multicolumn" not in r]
        if not plain:
            rep.ok += 1  # every row spans -> nothing to compare against
            continue
        expected = max(plain)
        bad = []
        for n, r in enumerate(rows, start=1):
            if "\\multicolumn" in r:
                continue  # legal span, exempt
            got = _amp_count(r)
            if got != expected:
                bad.append("row %d: %d & vs %d" % (n, got, expected))
        # unterminated \begin without matching \end shows up as no blocks at
        # all, so check counts globally too
        if bad:
            rep.problems.extend(bad[:3])
        else:
            rep.ok += 1
    n_begin = len(re.findall(r"\\begin\s*\{(?:%s)\*?\}" % "|".join(map(re.escape, _TAB_ENVS)), tex))
    n_end = len(re.findall(r"\\end\s*\{(?:%s)\*?\}" % "|".join(map(re.escape, _TAB_ENVS)), tex))
    if n_begin != n_end:
        rep.problems.append("unbalanced begin/end: %d/%d" % (n_begin, n_end))
        if rep.total:
            rep.ok = max(0, rep.ok - 1)
    return rep


# --------------------------------------------------------------------------- #
# 3. Compile test
# --------------------------------------------------------------------------- #

def have_pdflatex() -> bool:
    return shutil.which("pdflatex") is not None


def compile_tex(tex_body: str, workdir: Optional[str] = None, timeout: int = 60) -> dict:
    """Return {ok, exit_code, log_tail, pdf_bytes}. ok=False if pdflatex missing."""
    if not have_pdflatex():
        return {"ok": None, "exit_code": None, "log_tail": "pdflatex not found", "pdf_bytes": 0}
    tmp = workdir or tempfile.mkdtemp(prefix="texchk_")
    own = workdir is None
    tex_path = os.path.join(tmp, "doc.tex")
    with open(tex_path, "w", encoding="utf-8", errors="replace") as fh:
        fh.write(TEX_TEMPLATE.replace("%BODY%", tex_body))
    try:
        proc = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-halt-on-error",
             "-file-line-error", "-output-directory", tmp, tex_path],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            timeout=timeout, cwd=tmp,
        )
        out = proc.stdout.decode("utf-8", "replace")
        pdf = os.path.join(tmp, "doc.pdf")
        size = os.path.getsize(pdf) if os.path.exists(pdf) else 0
        ok = proc.returncode == 0 and size > 1000
        return {"ok": ok, "exit_code": proc.returncode,
                "log_tail": "\n".join(out.splitlines()[-12:]), "pdf_bytes": size}
    except subprocess.TimeoutExpired:
        return {"ok": False, "exit_code": None, "log_tail": "timeout", "pdf_bytes": 0}
    finally:
        if own:
            shutil.rmtree(tmp, ignore_errors=True)


# --------------------------------------------------------------------------- #
# 4. Scoring a run manifest
# --------------------------------------------------------------------------- #

@dataclass
class Row:
    mode: str
    page_id: str
    bucket: str
    vision_tokens: int
    pred_path: str
    gold_path: str


def _read_text(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            return fh.read()
    except OSError:
        return ""


def score_row(row: Row, do_compile: bool = True, keep_tex: bool = False) -> dict:
    pred = _read_text(row.pred_path)
    gold = _read_text(row.gold_path) if row.gold_path else ""
    rec = {
        "mode": row.mode, "page_id": row.page_id, "bucket": row.bucket,
        "vision_tokens": row.vision_tokens,
        "pred_chars": len(pred),
        "ned": None, "compile_ok": None, "tab_total": 0, "tab_ok": 0,
        "tab_parity": None,
    }
    if gold:
        rec["ned"] = round(ned(strip_latex(gold), strip_latex(pred)), 6)
    tabs = check_tabulars(pred)
    rec["tab_total"] = tabs.total
    rec["tab_ok"] = tabs.ok
    rec["tab_parity"] = None if tabs.parity is None else round(tabs.parity, 6)
    if tabs.problems:
        rec["tab_problems"] = "; ".join(tabs.problems[:3])
    if do_compile:
        cres = compile_tex(pred, keep_tex if keep_tex else None)
        rec["compile_ok"] = cres["ok"]
        if cres["ok"] is False:
            rec["compile_log"] = cres["log_tail"].replace("\n", " | ")[-400:]
    return rec


def load_runs(path: str) -> List[Row]:
    rows: List[Row] = []
    with open(path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            rows.append(Row(
                mode=(r.get("mode") or "").strip(),
                page_id=(r.get("page_id") or "").strip(),
                bucket=(r.get("bucket") or "").strip(),
                vision_tokens=int(float(r.get("vision_tokens") or 0)),
                pred_path=(r.get("pred_path") or "").strip(),
                gold_path=(r.get("gold_path") or "").strip(),
            ))
    return rows


def summarise(records: List[dict]) -> str:
    by = {}
    for r in records:
        by.setdefault((r["bucket"], r["mode"]), []).append(r)
    lines = ["bucket,mode,n,tokens_mean,ned_mean,compile_rate,tab_parity_mean"]
    for (bucket, mode), rs in sorted(by.items()):
        n = len(rs)
        toks = sum(x["vision_tokens"] for x in rs) / n if n else 0
        neds = [x["ned"] for x in rs if x["ned"] is not None]
        comps = [x["compile_ok"] for x in rs if x["compile_ok"] is not None]
        pars = [x["tab_parity"] for x in rs if x["tab_parity"] is not None]
        lines.append("%s,%s,%d,%.0f,%s,%s,%s" % (
            bucket, mode, n, toks,
            "%.4f" % (sum(neds) / len(neds)) if neds else "",
            "%.4f" % (sum(1 for c in comps if c) / len(comps)) if comps else "",
            "%.4f" % (sum(pars) / len(pars)) if pars else "",
        ))
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# 5. Self-test (no data, no GPU, no tex needed) -- run this first
# --------------------------------------------------------------------------- #

def self_test() -> int:
    fails = []

    def check(name, cond, extra=""):
        if not cond:
            fails.append("%s %s" % (name, extra))

    # text projection
    t = strip_latex(r"We have $x^2$ and 50\% of \textbf{bold} text. % comment")
    check("strip", "50%" in t and "bold" in t and "comment" not in t, repr(t))
    check("ned_same", ned("abc", "abc") == 0.0)
    check("ned_diff", 0.0 < ned("abcdef", "abXdef") <= 0.2, ned("abcdef", "abXdef"))

    # tabular parity
    good = r"""\begin{tabular}{ll}
\toprule
a & b \\
c & d \\
\bottomrule
\end{tabular}"""
    rep = check_tabulars(good)
    check("tab_good", rep.total == 1 and rep.ok == 1 and rep.parity == 1.0,
          "%s %s" % (rep.total, rep.errors if hasattr(rep, "errors") else rep.problems))

    merged = r"""\begin{tabular}{lll}
a & b & c \\
d & e \\
\end{tabular}"""
    rep = check_tabulars(merged)
    check("tab_merged", rep.total == 1 and rep.ok == 0, str(rep.problems))

    escaped = r"""\begin{tabular}{ll}
a \& b & c \\
d & e \\
\end{tabular}"""
    check("tab_escaped_amp", check_tabulars(escaped).ok == 1)

    multi = r"""\begin{tabular}{lll}
\multicolumn{2}{c}{hdr} & x \\
a & b & c \\
\end{tabular}"""
    check("tab_multicolumn", check_tabulars(multi).ok == 1)

    unterminated = r"""\begin{tabular}{ll}
a & b \\
"""
    rep = check_tabulars(unterminated)
    check("tab_unterminated", any("unbalanced" in p for p in rep.problems), str(rep.problems))

    # compile test availability is environment-dependent; only assert the shape
    res = compile_tex(r"hello")
    check("compile_shape", set(res) >= {"ok", "exit_code", "log_tail", "pdf_bytes"}, str(res))
    if have_pdflatex():
        check("compile_ok_true", compile_tex(r"$a^2+b^2=c^2$")["ok"] is True)
        check("compile_ok_false", compile_tex(r"\begin{tabular}{ll} a & b \\")["ok"] is False)

    print("self-test: %d checks, %d failed" % (13, len(fails)))
    for f in fails:
        print("  FAIL", f)
    print("  pdflatex:", "found" if have_pdflatex() else "NOT FOUND (compile metric will be null)")
    print("  edit distance:", "rapidfuzz/Levenshtein" if _HAVE_RF else "difflib fallback (upper bound)")
    print("  template sha:", TEMPLATE_SHA)
    return 1 if fails else 0


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--runs", help="manifest csv: mode,page_id,bucket,vision_tokens,pred_path,gold_path")
    ap.add_argument("--out", default="scores.csv")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    ap.add_argument("--no-compile", action="store_true")
    ap.add_argument("--keep-tex", action="store_true", help="keep wrapped .tex next to pdflatex logs")
    args = ap.parse_args(argv)

    if args.self_test:
        return self_test()
    if not args.runs:
        ap.print_help()
        return 2

    rows = load_runs(args.runs)
    print("scoring %d outputs (compile=%s, jobs=%d, template=%s)"
          % (len(rows), not args.no_compile, args.jobs, TEMPLATE_SHA), file=sys.stderr)
    records: List[dict] = []
    with ThreadPoolExecutor(max_workers=args.jobs) as ex:
        futs = [ex.submit(score_row, r, not args.no_compile, args.keep_tex) for r in rows]
        for k, f in enumerate(futs, 1):
            records.append(f.result())
            if k % 25 == 0 or k == len(futs):
                print("  %d/%d" % (k, len(futs)), file=sys.stderr)

    cols = ["mode", "page_id", "bucket", "vision_tokens", "pred_chars", "ned",
            "compile_ok", "tab_total", "tab_ok", "tab_parity", "tab_problems", "compile_log"]
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for rec in records:
            w.writerow(rec)
    summary_csv = os.path.splitext(args.out)[0] + "_summary.csv"
    summary = summarise(records)
    with open(summary_csv, "w", encoding="utf-8") as fh:
        fh.write(summary + "\n")
    print("\n" + summary, file=sys.stderr)
    print("\nwrote %s and %s" % (args.out, summary_csv), file=sys.stderr)

    meta = {"template_sha": TEMPLATE_SHA, "pdflatex": have_pdflatex(),
            "edit_distance_backend": "rapidfuzz" if _HAVE_RF else "difflib",
            "n_rows": len(records), "compile_metric": not args.no_compile}
    meta_path = os.path.splitext(args.out)[0] + "_meta.json"
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    print("wrote %s" % meta_path, file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
