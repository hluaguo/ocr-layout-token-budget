# Layout doesn't survive compression

DeepSeek-OCR-2 renders a page to LaTeX at a fixed visual-token budget. Lower the
budget and text still comes out — but tables, multi-column flow and environments
stop surviving, and the LaTeX stops compiling.

Measured on [TexOCR](https://arxiv.org/abs/2604.22880) eval pages (CC BY 4.0) with
[DeepSeek-OCR-2](https://huggingface.co/deepseek-ai/DeepSeek-OCR-2) (Apache-2.0):

- **x-axis** = measured vision tokens per page, logged from the run — not the
  nominal mode label (Gundam's crop count follows the page aspect ratio).
- **y-axis** = text **NED** against whole-paper gold, plus `pdflatex` pass rate
  and tabular well-formedness (merged/duplicated columns).
- **n** = 30 complete papers / 236 pages, 3 buckets (table-heavy, math-heavy,
  mixed-layout) x 3 supported modes (Compact 144, Base 256, Gundam 256 + n x 144).

<!-- plots land here -->
<!-- ![budget vs layout](figs/budget_vs_layout.png) -->

**Status:** data + harness + sweep notebook landed; sweep not yet run.

## Why whole papers, and why paper-level gold

TexOCR-eval ships page images and **per-document** annotations (`section_set`,
`table_set`, `image_set`) — there is no per-page gold LaTeX. Their own harness
sorts a paper's predicted pages, concatenates them, and compares that against the
concatenation of the paper's top-level sections. So:

- page level (n=236): compile pass rate + tabular parity, no NED;
- paper level (n=30): NED and compile on the concatenated paper, which is the
  number comparable to their `edit_distance` column.

Sampling a single page out of ten would inject ~90% edit distance from the nine
un-sampled pages, which is why whole papers are sampled (6-10 gapless pages each).

## Run the sweep on Colab

Colab opens a single notebook, not a repo — the other files arrive from the setup cells.

1. Open <https://colab.research.google.com/github/hluaguo/ocr-layout-token-budget/blob/main/notebooks/sweep.ipynb>
   (or *File → Upload notebook* to use a local copy).
2. **Runtime → Change runtime type → L4 GPU.**
3. Put `data.tar.gz` (built with `tar -czf data.tar.gz data/`) in
   `MyDrive/ocr-layout-budget/` — the notebook untars it and clones this repo into
   `/content/ocr-layout-token-budget`.
4. Run cells 1–6 in order. Cell 5 is the sweep and ships with `RUN_LIMIT = 3` for the
   smoke test; cell 6 asserts the three rows are non-empty before you raise it.

## Reproduce

```bash
# 1. sample: reads the 14 figure archives' central directories over HTTP range,
#    pulls only the needed PNGs (~330 MB), writes paper_gold/<id>.tex
python3 scripts/build_pages.py --per-bucket 10 --seed 42

# 2. sweep on a Colab L4: notebooks/sweep.ipynb (resume-safe, writes on Drive)

# 3. score: page-level + paper-level, no GPU needed
python3 scripts/score.py --runs runs.csv --pages data/pages.jsonl \
    --group-by paper --out scores.csv --jobs 8

# no data, no GPU, no tex required:
python3 scripts/score.py --self-test
```

Compile results are only comparable at a fixed preamble: the wrapper's SHA is
printed by the self-test and recorded in `scores_meta.json`.

## Known caveats

- **DeepEncoder V2 only supports 144 and 256 visual tokens** (`deepencoderv2.py` defines query
  banks for exactly those two sizes). The 400-token `Large` mode from DeepSeek-OCR v1 raises
  `UnboundLocalError: param_img` on OCR-2, so the sweep uses Compact (768 px, 144 tokens),
  Base (1024 px, 256 tokens) and Gundam (256 + n x 144).
- `infer()` needs `eval_mode=True` (otherwise it streams and returns `None`) and a non-empty
  `output_path` (it calls `os.makedirs` unconditionally) — both are pinned in the notebook.
- `eval_mode` decodes with `no_repeat_ngram_size=35`; the repo's interactive path uses `20`.
  Numbers are therefore close to, not bit-identical with, the released demo.
- `max_new_tokens=8192`: a very dense Gundam page can hit the cap, so `pred_chars` is worth
  eyeballing for truncation before trusting a low NED.

## Dataset facts worth knowing

- 14 figure archives, 13,367 page renders across 1,000 papers; 978 annotation JSONs.
- 361 papers have 6-10 gapless pages, which is the sampling pool.
- **`table_set == 0` never happens** in that pool (minimum is 1 table per paper),
  so `math_heavy` is "math-dominant": at most 1 table and >= 3 display-math
  environments. The strict count is logged and written to `buckets.csv` rather than
  hidden — see the comment in `scripts/build_pages.py`.
