#!/usr/bin/env python3
"""Plot the sweep: measured visual tokens on x, text and layout metrics on y.

Reads the CSVs written by scripts/score.py:

    scores_paper.csv   paper-level NED and compile pass rate (n = papers)
    scores_summary.csv page-level compile + tabular parity (n = pages)

and writes:
    figs/budget_vs_layout.png   two panels, one point per mode
    figs/budget_vs_layout.csv   the numbers behind it, so the README can quote them

Usage
-----
    python scripts/plot.py --work /content/drive/MyDrive/ocr-layout-budget
    python scripts/plot.py --scores scores.csv            # explicit prefix
"""

from __future__ import annotations

import argparse
import os
import sys

ORDER = ["Compact", "Base", "Gundam"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--work", default="/content/drive/MyDrive/ocr-layout-budget")
    ap.add_argument("--scores", default=None, help="prefix of scores*.csv (default <work>/scores)")
    ap.add_argument("--out", default=None, help="output png (default <work>/figs/budget_vs_layout.png)")
    args = ap.parse_args(argv)

    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    prefix = args.scores or os.path.join(args.work, "scores")
    paper_csv = prefix + "_paper.csv"
    page_csv = prefix + "_summary.csv"
    out_png = args.out or os.path.join(args.work, "figs", "budget_vs_layout.png")
    os.makedirs(os.path.dirname(out_png), exist_ok=True)

    if not os.path.exists(paper_csv):
        raise SystemExit("missing %s -- run scripts/score.py --group-by paper first" % paper_csv)
    paper = pd.read_csv(paper_csv)

    def mode_order(df):
        present = [m for m in ORDER if m in set(df["mode"])]
        return present + [m for m in sorted(set(df["mode"])) if m not in present]

    g = (paper.groupby("mode")
         .agg(tokens=("vision_tokens_mean", "mean"),
              ned=("ned", "mean"),
              n_papers=("ned", "count"),
              compile=("compile_ok", "mean"))
         .reindex(mode_order(paper)))

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(g.tokens, 1 - g.ned, "o-", label="1 - NED (text)")
    ax[0].set_ylabel("text fidelity")
    ax[0].set_title("Text survives compression")
    ax[0].set_xlabel("measured visual tokens / page")

    if g.compile.notna().any():
        ax[1].plot(g.tokens, g.compile, "s-", label="pdflatex pass rate (paper level)")
        ax[1].set_title("Structure does not")
    else:
        ax[1].text(0.5, 0.5, "compile pass not run yet\n(pdflatex on a CPU runtime)",
                   ha="center", va="center", transform=ax[1].transAxes)
    ax[1].set_ylabel("compile pass rate")
    ax[1].set_xlabel("measured visual tokens / page")

    if os.path.exists(page_csv):
        page = pd.read_csv(page_csv)
        if "tab_parity_mean" in page and page.tab_parity_mean.notna().any():
            h = (page.groupby("mode")
                 .agg(tokens=("tokens_mean", "mean"), parity=("tab_parity_mean", "mean"))
                 .reindex([m for m in mode_order(page) if m in set(g.index)]))
            for a in ax:
                a.plot(h.tokens, h.parity, "^--", alpha=.55, label="tabular parity (page level)")
    for a in ax:
        a.grid(alpha=.3)
        a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)

    csv_out = os.path.splitext(out_png)[0] + ".csv"
    g.round(4).to_csv(csv_out)
    print("wrote", out_png)
    print("wrote", csv_out)
    print(g.round(4).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
