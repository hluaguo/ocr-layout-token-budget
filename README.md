# Layout doesn't survive compression

DeepSeek-OCR-2 renders a page to LaTeX at a fixed visual-token budget. Lower the
budget and text still comes out — but tables, multi-column flow and environments
stop surviving, and the LaTeX stops compiling.

This repo measures exactly that, on annotated pages from
[TexOCR](https://arxiv.org/abs/2604.22880) (CC BY 4.0), with
[DeepSeek-OCR-2](https://huggingface.co/deepseek-ai/DeepSeek-OCR-2) (Apache-2.0):

- **x-axis** = measured vision tokens per page, logged from the run — not the
  nominal mode label (Gundam's token count varies with page aspect ratio).
- **y-axis** = two things a downstream user actually needs: `pdflatex` compile
  pass rate (structure) and normalised text edit distance (content).
- **n** = 90+ pages per budget across three page families: two-column academic,
  dense nested tables, multi-line equations.

<!-- plots land here -->
<!-- ![budget vs layout](figs/budget_vs_layout.png) -->
<!-- ![failure case](figs/qualitative_fail.png) -->

**Status:** scaffolding (v0.1) — sweep notebook and scoring scripts being added.
Metric set is deliberately small (compile rate, text edit distance, tabular
well-formedness); the full TexOCR 9-metric harness is separate work.

Notebook: `notebooks/sweep.ipynb` · scorer: `scripts/score.py`
