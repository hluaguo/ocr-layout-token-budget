# Layout doesn't survive compression

DeepSeek-OCR-2 renders a page to LaTeX at a fixed visual-token budget. Lower the
budget and text still comes out — but tables, multi-column flow and environments
stop surviving, and the LaTeX stops compiling.

This repo measures exactly that, on annotated pages from [TexOCR](https://arxiv.org/abs/2604.22880)
(CC BY 4.0) with [DeepSeek-OCR-2](https://huggingface.co/deepseek-ai/DeepSeek-OCR-2) (Apache-2.0):

- **x-axis** = measured vision tokens per page, logged from the run — not the
  nominal mode label (Gundam's crop count varies with page aspect ratio).
- **y-axis** = three checks: `pdflatex` pass rate (structure), normalised edit
  distance on projected text (content), tabular well-formedness (merged or
  duplicated columns).
- **n** = 80 pages per bucket × 3 buckets (two-column, dense nested tables,
  multi-line equations) × 3 native modes (Base 256, Large 400, Gundam dynamic).

<!-- plots land here -->
<!-- ![budget vs layout](figs/budget_vs_layout.png) -->
<!-- ![failure case](figs/qualitative_fail.png) -->

**Status:** harness landed (`scripts/score.py`, 13 self-checks); page sampling and
the Colab sweep notebook are next. Preliminary 3-metric sweep — the full 9-metric
TexOCR evaluation and TAB-TEDS are separate, larger work.

```
python3 scripts/score.py --self-test     # no data, no GPU, no tex required
python3 scripts/score.py --runs runs.csv --out scores.csv --jobs 4
```

Compile results are only comparable at a fixed preamble: the wrapper's SHA is
printed by the self-test and recorded in `scores_meta.json`.
