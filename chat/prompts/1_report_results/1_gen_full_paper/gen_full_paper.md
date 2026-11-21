# gen_full_paper — report_results

> Phase: `gen_paper_repo` · `gen_full_paper`
> Run: `run_SGYOniI23HcA-msgsum` — Zipf-Mandelbrot Law for Word Frequencies
>
> Full, verbatim record of every prompt the AI Inventor pipeline gave this agent — system-user, human-user and skill-input — in the order they landed. Nothing truncated.

## Task: `gen_full_paper` (sdk_openhands_agent)

### [1] SYSTEM-USER prompt · 2026-11-20 18:45:14 UTC

```
<task>
Typeset <paper_draft> as LaTeX with BibTeX, insert <available_figures>, and compile it to PDF.
</task>

<tool_use>
Maximize parallel tool calls. Parallelize independent operations, only sequentialize dependencies.
- Multiple searches/fetches on different topics → parallel in one turn
- Search then fetch results → sequential (need URLs first)
</tool_use>

<publishable_paper_rules>
This is the PUBLISHABLE PAPER. The run's internal report ships beside this paper as its own PDF
and already ho... [truncated, 84038 chars total]
```

### [2] SYSTEM-USER prompt · 2026-11-20 18:55:50 UTC

```
<task>
Typeset <paper_draft> as LaTeX with BibTeX, insert <available_figures>, and compile it to PDF.
</task>

<tool_use>
Maximize parallel tool calls. Parallelize independent operations, only sequentialize dependencies.
- Multiple searches/fetches on different topics → parallel in one turn
- Search then fetch results → sequential (need URLs first)
</tool_use>

<publishable_paper_rules>
This is the PUBLISHABLE PAPER. The run's internal report ships beside this paper as its own PDF
and already ho... [truncated, 84038 chars total]
```

### [3] HUMAN-USER prompt · 2026-11-20 18:55:50 UTC

```
Please continue from where it stopped.
```

### [4] SKILL-INPUT — aii-semscholar-bib · 2026-11-20 18:57:13 UTC

The agent loaded the **aii-semscholar-bib** skill; its `SKILL.md` (the instructions injected into the agent's context) follows verbatim.

````
---
name: aii-semscholar-bib
description: "Fetches real BibTeX entries in one batch from Semantic Scholar by DOI, ArXiv ID or title via aii_semscholar_bib__fetch, normalises citation keys to AuthorYYYY, injects DOIs, and merges the result into references.bib while recording each entry in references.json beside it; a paper it cannot fetch is not cited. ALWAYS use whenever a bibliography, reference list or .bib file is being built or extended, and whenever a citation needs a verified entry instead of an invented one — never hand-write BibTeX. Triggers: bibliography, references.bib, bibtex, citation key, DOI, arXiv id, Semantic Scholar, reference list, cite these papers, natbib entries. NOT for: writing the text around the citations (use aii-paper-writing), running bibtex and compiling (use aii-paper-to-latex), judging whether cited work supports the claims (use amg-paper-verification), or open-ended literature search and PDF mining (use aii-web-tools)."
---

## Tool: `aii_semscholar_bib__fetch`

Batch-fetch BibTeX entries from Semantic Scholar (OpenAlex, then Crossref, when S2 is rate-limited or down). Pass all references in a single call — the tool handles batching internally.

### How it works

1. **DOI/ArXiv refs** → batched into POST /paper/batch calls (up to 500 per API call, auto-chunked)
2. **Title-only refs** → individual GET /paper/search/match (1s delay between)
3. **Fallback when S2 is down** → if S2 still answers 429 (its shared anonymous pool saturates for every caller) or 5xx after its bounded retries, or cannot be reached, S2 is skipped for the rest of the call, and for the next 10-15 min in every call (then one probe decides whether it is back); every ref it did not answer resolves through **OpenAlex**, then **Crossref** (both keyless; set `AII_POLITE_CONTACT` for their higher-limit polite pool). DOI/arXiv hits must agree with the ref's title or first author, so a mislinked record is dropped rather than cited; title hits need a near-exact title. The BibTeX has the same layout, keys and fields as S2's, so `references.bib` cannot tell them apart; each entry's `source` (`semantic_scholar`, `openalex` or `crossref`) says which API answered.
4. **Post-process** → entry type, venue and year from the DOI's Crossref record (a preprint DOI Crossref links to its published version is cited as that version; S2's publication type decides when Crossref has no record), family names braced (`{De Domenico}`, `{DerSimonian}`); normalise fields so the entry renders cleanly (more than 10 authors keep 5 plus "and others", printed "et al."; S2's mangled accents like `Ram'e` restored; straight quotes as LaTeX quotes; arXiv records as `journal = {arXiv preprint arXiv:<id>}`, never `volume = {abs/<id>}`); fix citation key (AuthorYYYY, accents folded); inject DOI

The ability server runs a single worker (`max_threads: 1`). Multiple concurrent tool calls are queued — each runs independently (no cross-request aggregation). Batching happens within each request.

### Input format

```json
{
  "references": [
    {"doi": "10.48550/arXiv.1706.03762", "author": "Vaswani", "year": 2017},
    {"arxiv": "2201.11903", "author": "Wei", "year": 2022},
    {"title": "Tree of Thoughts", "author": "Yao", "year": 2023}
  ]
}
```

Each reference object can have:
- `doi` — DOI string (ArXiv DOIs like `10.48550/arXiv.XXXX.XXXXX` auto-convert to ArXiv IDs)
- `arxiv` — ArXiv ID (e.g. `"2305.14325"`)
- `title` — Paper title (used for search/match when no DOI/ArXiv)
- `author` — First author last name (for cleaner citation key)
- `year` — Publication year (int, for citation key)

At least one of `doi`, `arxiv`, or `title` is required per reference.

### Output format

```json
{
  "success": true,
  "bib_text": "@inproceedings{Vaswani2017, ...}\n\n@article{Wei2022, ...}",
  "total": 3,
  "found": 3,
  "failed_count": 0,
  "entries": [{"citation_key": "Vaswani2017", "bibtex": "...", "title": "...", "doi": "...", "arxiv": "", "source": "semantic_scholar"}],
  "failed": []
}
```

Called as a tool (or through `--json`), it returns these entries and writes no file. A bibliography
is built only through the CLI's `--out`, which also writes the record.

### Workflow

1. Collect DOIs, ArXiv IDs, or titles for all papers you need to cite
2. Run the CLI below with the full list in **one call** and `--out ./references.bib`
3. The script merges the fetched entries into `references.bib` (created if absent) and writes a
   record of each one to `references.json` beside it: source database, S2 paperId / DOI / arXiv id,
   title, first author, year. Later calls with `--out` append to both files and keep them in sync;
   a second paper under a key already taken gets a letter suffix (`Smith2020a`), which the output
   lists — cite the key it prints.
4. Check the failed list — for any missed papers, follow the **fallback procedure** below

`references.bib` and `references.json` are written ONLY by this script for papers. Never write or
paste a paper's BibTeX entry by hand, and never edit `references.json`: the paper step checks every
`\cite` key against `references.bib` and every entry's title and first author against its record,
and a paper entry the script did not write blocks the paper from being published.

The one exception is a source that is NOT a paper and has no record to fetch: a model card, a
software repository, a dataset. Append it to `references.bib` by hand as `@misc`, with its URL and
the date you accessed it:

```bibtex
@misc{Qwen2024,
  author = {{Qwen}},
  title = {Model card of {Qwen}'s {32B} instruction-tuned code model},
  howpublished = {\url{https://huggingface.co/Qwen/Qwen2.5-Coder-32B-Instruct}},
  note = {Hugging Face model Qwen2.5-Coder-32B-Instruct, accessed 27 October 2026},
  year = {2024}
}
```

`url = {...}` and `urldate = {...}` fields are accepted too. Anything with an arXiv, DOI or other
paper URL is a paper and must be fetched.

### Check every fetched entry against its published record (MANDATORY)

The script's entries are raw material, not proof. Reviewers rejected papers whose journal articles
printed as "In Science, volume ...", whose years were a preprint's, and whose particles dropped.

- **Cite the published version.** When a paper has a journal or proceedings version, fetch it by
  that DOI, never by its arXiv or preprint DOI. Look it up on Crossref by title
  (`curl -s "https://api.crossref.org/works?query.bibliographic=<title>&rows=3"`).
- **Compare each entry with its Crossref record** (`curl -s https://api.crossref.org/works/<DOI>`)
  and correct it in `references.bib` to match: entry type (a journal article is `@article` with
  `journal=` and no `booktitle=`), year, authors (every name, spelled and ordered as Crossref has
  them; the script's "and others" past five may stay), venue name and volume.
- **Brace-protect** particles and capitals BibTeX would mangle: `{De Domenico}` (else it prints
  "Domenico"), `{DerSimonian}`, `From {L}ouvain to {L}eiden`.
- Those are the only edits allowed. The title's words stay as fetched, bar a stray character such
  as the footnote digit in "good ideas1", and so does the first author.
  An entry whose title or first author is wrong is re-fetched by its correct DOI, and every
  `\cite` of it moves to the key the script prints.
- After compiling, check the PRINTED reference list in the PDF (see aii-paper-to-latex), not only
  the `.bib`.

### Fallback for failed references (MANDATORY)

NEVER fabricate BibTeX. For each failed reference:
1. **WebSearch** for `"Title" author year` (try `site:arxiv.org` too)
2. **WebFetch** the paper page → extract its DOI or ArXiv ID and exact title
3. Retry the script with that DOI / ArXiv ID / exact title (same `--out`)
4. Still not found → the paper is not cited. Remove the citation (and any claim that rests only on
   it); there is no hand-written fallback for a paper (only for a non-paper source, above).

---

### CLI (how to build a bibliography)

```bash
SKILL_DIR="$(realpath "$(git rev-parse --show-toplevel 2>/dev/null || echo /ai-inventor)/.claude/skills/aii-semscholar-bib")" && \
$SKILL_DIR/../.ability_client_venv/bin/python $SKILL_DIR/scripts/aii_semscholar_bib__fetch.py --out ./references.bib --refs '[
  {"doi": "10.48550/arXiv.1706.03762", "author": "Vaswani", "year": 2017},
  {"arxiv": "2201.11903", "author": "Wei", "year": 2022},
  {"title": "Tree of Thoughts", "author": "Yao", "year": 2023}
]'
```

`--out, -o PATH` — merge the entries into PATH and record them in `references.json` beside it (always use it for a bibliography)
`--json, -j` — output raw JSON instead of .bib text

**If the script fails** with a connection error (ability server not running): create a local `.venv`, install server deps from `server_requirements.txt` into it, then run the script with that `.venv`'s python (it falls back to the local core when the server is unreachable), `--out` included — bypassing the server:
```bash
uv venv .venv --python=3.12 && uv pip install --python=.venv/bin/python -r "$SKILL_DIR/scripts/server_requirements.txt"
```

**If it fails with `No module named 'requests'`** (the shared venv is missing or Python resolved
`..` through a symlink; `realpath` above prevents the latter), skip the venv and run the same
command through uv, `--out` and `--refs` unchanged:
```bash
uv run --no-project --python 3.12 --with-requirements "$SKILL_DIR/scripts/server_requirements.txt" python "$SKILL_DIR/scripts/aii_semscholar_bib__fetch.py" --out ./references.bib --refs '[{"arxiv": "1706.03762", "author": "Vaswani", "year": 2017}]'
```
````

### [5] SYSTEM-USER prompt · 2026-11-20 22:36:57 UTC

```
<task>
Typeset <paper_draft> as LaTeX with BibTeX, insert <available_figures>, and compile it to PDF.
</task>

<tool_use>
Maximize parallel tool calls. Parallelize independent operations, only sequentialize dependencies.
- Multiple searches/fetches on different topics → parallel in one turn
- Search then fetch results → sequential (need URLs first)
</tool_use>

<publishable_paper_rules>
This is the PUBLISHABLE PAPER. The run's internal report ships beside this paper as its own PDF
and already ho... [truncated, 84038 chars total]
```

### [6] SYSTEM-USER prompt · 2026-11-21 13:15:15 UTC

```
<task>
Typeset <paper_draft> as LaTeX with BibTeX, insert <available_figures>, and compile it to PDF.
</task>

<tool_use>
Maximize parallel tool calls. Parallelize independent operations, only sequentialize dependencies.
- Multiple searches/fetches on different topics → parallel in one turn
- Search then fetch results → sequential (need URLs first)
</tool_use>

<publishable_paper_rules>
This is the PUBLISHABLE PAPER. The run's internal report ships beside this paper as its own PDF
and already ho... [truncated, 84038 chars total]
```

### [7] SYSTEM-USER prompt · 2026-11-21 13:15:15 UTC

```
continue where you left off — reuse any partial work already written to disk. Do NOT start over.
```

### [8] SKILL-INPUT — aii-paper-to-latex · 2026-11-21 13:17:17 UTC

The agent loaded the **aii-paper-to-latex** skill; its `SKILL.md` (the instructions injected into the agent's context) follows verbatim.

````
---
name: aii-paper-to-latex
description: "Assembles and compiles a LaTeX paper into paper.pdf: preamble (generic; bundled sn-jnl for Springer, BMC, Nature; else the journal's own template fetched at run time), figure floats that includegraphics pre-generated .pdf and .jpg, float-placement and width rules, and the required pdflatex, bibtex, pdflatex, pdflatex sequence. Use whenever pre-written text and figures must become a compiled PDF, and when a build misbehaves — citations printing as ??, figures drifting to the end or above the title, shrunken axis labels, undefined references. Triggers: latex, pdflatex, bibtex, natbib, sn-jnl, journal template, includegraphics, figure float, htbp, compile the paper, paper.tex, paper.pdf. NOT for: writing the text (aii-paper-writing), figure images (aii-data-fig-gen, aii-concept-fig-gen), bibliography entries (aii-semscholar-bib); NOT for reshaping a PDF — merging, splitting, forms, tables (amg-meta-computer-use); NOT for compiling a user's own .tex in Downloads or Documents; that is amg-meta-computer-use."
---

## LaTeX Paper Assembly

Assembles a research paper from paper text, pre-generated figures (vector `.pdf` for data figures, `.jpg` for concept figures) and a bibliography into a compiled PDF.

### Document Setup

Which preamble depends on the research request:

- **No target journal named**: the generic preamble below.
- **A Springer, SpringerOpen, BMC or Nature Portfolio journal named**: that journal's
  official template, see "Target journal: Springer Nature template".
- **Any other publisher's journal named** (Elsevier, IEEE, ACM, Wiley, MDPI, ...): its
  template is not bundled, so download the official one at run time:
  1. Take the class from CTAN when it is there (elsarticle, IEEEtran, acmart, llncs,
     revtex, ...): `https://mirrors.ctan.org/macros/latex/contrib/<name>.zip`, which holds
     the `.cls` (or the `.ins`/`.dtx` to build it with `latex <name>.ins`) and its `.bst`.
     Otherwise use the template download on the journal's author-guidelines page.
  2. Put the `.cls` and `.bst` at the root of the paper directory, next to `paper.tex`, and
     any `.sty` the class loads that `kpsewhich <file>.sty` does not find. Subfolders other
     than `figures/` are dropped when the paper ships.
  3. Use the class options and reference style the journal's guidelines give (e.g.
     `\documentclass[preprint,12pt]{elsarticle}` with `elsarticle-num`), plus the generic
     preamble's packages the class does not already load, and its sections.
  4. Compile with the usual sequence below.

  Only when no official template can be found, or it will not compile, use the generic
  preamble and say in the paper-writing summary that the paper is not in the journal's own
  template.

Generic preamble:

```latex
\documentclass[11pt,letterpaper]{article}
\usepackage{graphicx, geometry, amsmath, hyperref, url, natbib, booktabs, longtable, xcolor, listings}
\geometry{margin=1in}
\hypersetup{colorlinks=true, linkcolor=black, citecolor=black, urlcolor=black}
\renewcommand{\topfraction}{0.9}
\renewcommand{\bottomfraction}{0.8}
\renewcommand{\textfraction}{0.07}
\renewcommand{\floatpagefraction}{0.75}
```

- **One typeface.** Add no font package to this preamble: no mathptmx, times, helvet, courier,
  newtx, newpx, lmodern, palatino or fontspec. Text, maths and captions stay Computer Modern.
  The figures are one family too, the paper's figure style, chosen from its field's top papers:
  CMU Serif like the text, or DejaVu Sans where that field draws sans-serif figures. A paper
  that loaded Times, Helvetica and Courier beside its Computer Modern figures showed three
  typefaces on one page. A journal's own class sets its own fonts.
- **Float fractions.** `\floatpagefraction` is the line that matters for figures: a page of
  floats alone is made only when they fill 75% of it, where LaTeX's default (0.5) left three
  float pages of one paper 35-50% blank. The `!` in every figure's `[!htbp]` already lifts the
  top, bottom and text limits, so the other three lines act only on floats placed without `!`.

### Target journal: Springer Nature template

This skill's own directory holds Springer Nature's official template (December 2024
package) in `templates/springer-nature/`: the class `sn-jnl.cls` and one BibTeX style per
reference style in `bst/`. Its `README.md` records the source and licence.

1. Pick the reference-style option from the journal's submission guidelines ("Preparing your
   manuscript", section References), which name the style and show example references:

   | the guidelines say                        | option           |
   |-------------------------------------------|------------------|
   | Basic Springer (Nature) reference style   | sn-basic         |
   | Math and Physical Sciences, numbered      | sn-mathphys-num  |
   | Math and Physical Sciences, author-year   | sn-mathphys-ay   |
   | American Physical Society (APS)           | sn-aps           |
   | Vancouver / BMC, numbered [1]             | sn-vancouver-num |
   | Vancouver, author-year                    | sn-vancouver-ay  |
   | Nature Portfolio journal                  | sn-nature        |
   | APA or Chicago style                      | not bundled      |

   `sn-basic` cites author-year, as Springer's example references do ("Smith J, Jones M
   (1999) Title. J Abbrev 965:325-329"); add the option `Numbered` only when the journal's
   published articles cite by number. When the guidelines name no style, use `sn-basic`.
   An APA or Chicago journal takes the generic preamble.
   Known journals:

   | journal                 | option   | checked                  |
   |-------------------------|----------|--------------------------|
   | Applied Network Science | sn-basic | guidelines, 2025 article |

   Applied Network Science's guidelines give "Examples of the Basic Springer reference
   style", and its articles cite "(Newman 2010)", so no `Numbered`.
2. Copy `sn-jnl.cls` and the option's `.bst` into the paper directory, next to `paper.tex`.
   The file is `bst/<option>.bst`, except `sn-aps` uses `bst/sn-APS.bst`. Both files are
   part of the paper's sources and ship with it, so the published repo rebuilds.
3. Preamble, with the journal's option:

   ```latex
   \documentclass[pdflatex,sn-basic]{sn-jnl}
   \usepackage{graphicx, amsmath, hyperref, url, booktabs, longtable, xcolor, listings}
   ```

   The class sets the page (A4, its own text block) and loads natbib and hyperref itself,
   and runs `\bibliographystyle` for the option. So no `geometry`, `\geometry`,
   `\hypersetup` or `natbib` options, and NO `\bibliographystyle` line: a second one is a
   BibTeX error ("Illegal, another \bibstyle command", exit status 2) that BibTeX then
   ignores, so it only ever adds an error to the build.
   `sn-nature` and `sn-aps` store no author names for natbib, so `\citet` prints
   "(author?) [2]" there: write the name in the sentence and cite with `\citep`.
4. Front matter, in this order after `\begin{document}`: `\title[<short running
   title>]{<full title>}`; `\author` and `\affil` only for authors the draft names (never
   placeholders); `\abstract{...}` (a command taking the abstract, not the `abstract`
   environment); `\keywords{...}`; then `\maketitle`, which prints all of them.
5. Back matter: `\backmatter`, then the journal's Declarations with each item under
   `\bmhead{<heading>}`, then `\bibliography{references}`.

Everything else in this skill (figures, code links, tables, length, compile sequence) is the
same for both setups.

### Figure Inclusion

CRITICAL: Include ALL figures. Every figure MUST appear in the paper.

```latex
\begin{figure}[!htbp]
  \centering
  \includegraphics[width=\linewidth,height=0.45\textheight,keepaspectratio]{figures/filename.pdf}
  \caption{Descriptive caption.}
  \label{fig:label}
\end{figure}
```

Rules:
- ALWAYS `[!htbp]` — all four options, so a float can never be deferred to the end of the
  document, which `[t]` or `[h]` alone risks. Do not ask for a page TOP: `[!t]` and
  `[!tbp]` both floated a figure ABOVE the paper's own title on page 1, where `[!htbp]`
  on the same document did not. Where a figure lands is decided by where it is declared
  in the text
- Use `figure`, never `figure*`. Both document setups are ONE column, so `figure*` is exactly
  as wide as `figure` (469.76pt generic, 372pt sn-jnl) and gains nothing, while restricting
  the float to a page top
- ALWAYS constrain with `width`, `height=0.45\textheight` and `keepaspectratio`.
  The height cap keeps a single plot under half a page, so it shares its page
  with text instead of standing alone on a half-empty float page. Figures are
  drawn at the 6.5 in text width, so a 16:9 or wider figure (3.7 in tall) is
  limited by its width and prints at 100%, and a 4:3 plot at 83%. A cap that
  binds harder shrinks the TEXT with the image: at `0.4\textheight` a square
  figure printed at 50.9%, putting 11 pt axis labels on the page at 5.6 pt,
  under the 7 pt minimum. So a 1:1 or taller plot is re-plotted at 16:9 or 4:3,
  never left for LaTeX to shrink. The figure generator measures legibility at
  the figure's OWN size, so it cannot see this happen
- Every figure needs `\caption`, `\label`, and a `\ref` in the text
- Do NOT convert figures to tables or describe them without inserting the image
- Do NOT skip any figures
- Never crop or trim with LaTeX options (`trim`, `clip`, `viewport`). Blank margins are cropped in
  code, in the file itself (PIL bounding box of non-white pixels), by the step that makes the figure,
  so the PDF, the repo and the website all show the same file

### Code links and footnotes

- No code link after each claim. The code is linked ONCE in the body, by a code-availability
  statement (`... are in the paper's code repository~\href{<repo URL>}{[code]}, and
  Appendix~\ref{app:code} links each result to the folder that produced it.`), and result by
  result in the last appendix section, `\section{Code for each result}\label{app:code}`: a
  `longtable` of Section | Result | Code folder, the Section cell `\S\ref{<label>}`, the Result
  cell naming the result and its figure or table, the Code folder cell
  `\href{<URL>}{\texttt{round-2/experiment-11}}`. Every Results subsection has a `\label` and a
  row. The finish check holds the table to the draft's code links and the run's code folders
- Never a footnote carrying a code URL: one paper's 33 code footnotes, each a 150-character URL
  broken mid-word, filled the foot of 12 pages
- Every URL is copied verbatim into its `\href`
- Any other URL (a model card, a dataset, another group's repository) goes in a footnote,
  `\footnote{\url{...}}`, or in its bibliography entry, never inline in a sentence, where it runs
  into the margin. A code link, or any URL over 70 characters, in a footnote sends the paper back
- A footnote mark goes after the sentence's closing punctuation, never directly after a digit or
  another mark: "IRR 1.19)123" reads as one number

### Tables

- At most 7 columns. Split a wider table in two, or transpose it so the many items are rows; an
  11-column table of narrow wrapping cells with 3-line headers is unreadable
- A table whose cells are sentences goes in the appendix
- Never shrink a table with `\resizebox`

### Length

The body, everything before the References heading, is at most 12 pages and about 7,000 words,
figures and tables included. Per-round detail, secondary analyses, full per-condition results and
text tables go in an appendix: `\clearpage` and `\appendix` after `\bibliography{references}`,
so the appendix starts on its own page, then `\section` headings, which is not counted. Measure the compiled PDF, not the source: `pdfinfo paper.pdf` for
the pages and `pdftotext paper.pdf - | wc -w` for the words. That count includes the text inside
vector figures (axis labels, ticks, legends) and tables, so the prose alone runs somewhat under
7,000.

### Compilation Process

Run each command separately (do NOT chain with `&&` — pdflatex often exits non-zero on warnings, which would skip bibtex and leave citations as `??`):

```bash
pdflatex -interaction=nonstopmode paper.tex
bibtex paper
pdflatex -interaction=nonstopmode paper.tex
pdflatex -interaction=nonstopmode paper.tex
```

On a Mac whose shell did not run `path_helper`, `pdflatex: command not found` means MacTeX is
off PATH, not missing: `export PATH="$PATH:/Library/TeX/texbin"` first.

All four commands are required. Skipping bibtex causes `??` in all citations.
Fix any errors between runs. Verify `./paper.pdf` was created.

After ANY change to `references.bib`, delete `paper.bbl` and `paper.aux` first and rerun all four:
a stale `.bbl` keeps printing the old entries. Then read the PRINTED reference list in the PDF, not
the `.bib`: no journal article printed as "In <Journal>, volume ...", no `??`, every name, year and
venue as its Crossref record has it.

### Output Files

- `./paper.tex` — LaTeX source
- `./references.bib` — bibliography file
- `./sn-jnl.cls` and `./<option>.bst` — only with a Springer Nature target journal; they ship
  with the paper's sources
- `./paper.pdf` — compiled PDF
- `./figures/` — all figure images (pre-generated, copied into workspace). Data
  figures are `.pdf` (vector — LaTeX renders their text at page resolution, which
  is what keeps axis labels sharp in print); concept figures are `.jpg`. Use each
  file's OWN extension in `\includegraphics`; there is no conversion step.
````

### [9] SYSTEM-USER prompt · 2026-11-21 13:28:43 UTC

```
STOP. You have reached the maximum number of turns.

Do NOT use any more tools. Finish what you are doing and provide your final output NOW.

Use whatever information you have gathered so far to produce the best response possible.
```

### [10] SYSTEM-USER prompt · 2026-11-21 13:28:55 UTC

````
<validation-feedback>
Attempt 1 failed validation.

The output file `/Volumes/aii-s20001/work/run_SGYOniI23HcA/4_gen_paper_repo/_4_assemble_paper/paper/workspace/.sdk_openhands_agent_struct_out.json` does not exist yet.



---

Output the result as JSON to: `/Volumes/aii-s20001/work/run_SGYOniI23HcA/4_gen_paper_repo/_4_assemble_paper/paper/workspace/.sdk_openhands_agent_struct_out.json`

JSON Schema:
```json
{
  "$defs": {
    "FullPaperExpectedFiles": {
      "description": "All expected output files from full paper generation.",
      "properties": {
        "paper_tex_path": {
          "description": "Path to LaTeX source file. Example: 'paper.tex'",
          "title": "Paper Tex Path",
          "type": "string"
        },
        "paper_pdf_path": {
          "description": "Path to compiled PDF. Example: 'paper.pdf'",
          "title": "Paper Pdf Path",
          "type": "string"
        },
        "references_bib_path": {
          "description": "Path to BibTeX bibliography file. Example: 'references.bib'",
          "title": "References Bib Path",
          "type": "string"
        },
        "figure_paths": {
          "description": "Paths to all figure image files. Example: ['figures/fig1_v0.jpg', 'figures/fig2_v0.jpg']",
          "items": {
            "type": "string"
          },
          "title": "Figure Paths",
          "type": "array"
        }
      },
      "required": [
        "paper_tex_path",
        "paper_pdf_path",
        "references_bib_path",
        "figure_paths"
      ],
      "title": "FullPaperExpectedFiles",
      "type": "object"
    }
  },
  "description": "Full paper \u2014 structured output from paper generation.",
  "properties": {
    "title": {
      "description": "Paper title in plain, everyday language \u2014 short and jargon-free so a non-expert grasps it at a glance. Aim for about 4-8 words (~40 characters).",
      "maxLength": 90,
      "minLength": 12,
      "title": "Title",
      "type": "string"
    },
    "summary": {
      "description": "Brief summary of the generated paper: sections written, figures included, compilation status",
      "maxLength": 5000,
      "minLength": 500,
      "title": "Summary",
      "type": "string"
    },
    "findings_summary": {
      "description": "The run's finding in 2-4 sentences, for a reader who will not open the PDF: what was tested, the headline number with its units, what it means. Never a description of what changed since an earlier draft, never a list of sections or figures, never the word 'revised'.",
      "maxLength": 1200,
      "minLength": 120,
      "title": "Findings Summary",
      "type": "string"
    },
    "out_expected_files": {
      "$ref": "#/$defs/FullPaperExpectedFiles",
      "description": "All output files you created. Must include paper.tex, paper.pdf, references.bib, and paths to all figure files."
    }
  },
  "required": [
    "title",
    "summary",
    "findings_summary",
    "out_expected_files"
  ],
  "title": "FullPaper",
  "type": "object"
}
```

IMPORTANT: this task is NOT complete until `/Volumes/aii-s20001/work/run_SGYOniI23HcA/4_gen_paper_repo/_4_assemble_paper/paper/workspace/.sdk_openhands_agent_struct_out.json` exists and contains JSON matching the schema above.

Produce `/Volumes/aii-s20001/work/run_SGYOniI23HcA/4_gen_paper_repo/_4_assemble_paper/paper/workspace/.sdk_openhands_agent_struct_out.json` again so it contains corrected JSON that matches the schema. Do not invent new fields.
</validation-feedback>
````

### [11] SYSTEM-USER prompt · 2026-11-21 13:29:21 UTC

```
<validation-feedback>
Attempt 2 failed validation.

Schema validation found 1 problem — fix ALL of them at once:
  - at `title`: "Zipf's Law Falls Short: The Zipf-Mandelbrot Model Accurately Captures Word Frequencies in Pride and Prejudice" is too long (at most 90 characters, got 109)
Every required field must be present and every field type must match the schema.

Produce `/Volumes/aii-s20001/work/run_SGYOniI23HcA/4_gen_paper_repo/_4_assemble_paper/paper/workspace/.sdk_openhands_agent_struct_out.json` again so it contains corrected JSON that matches the schema. Do not invent new fields.
</validation-feedback>
```

### [12] SYSTEM-USER prompt · 2026-11-21 13:30:16 UTC

```
CODE TABLE CHECK FAILED: 3 problem(s) with the code-availability statement or the appendix that maps each result to its code:
- https://github.com/ai-inventor-papers/ai-invention-624cdf-local-zipfian-curvature-as-narrative/tree/main/round-1/dataset-1: a code folder of the draft that no row of the code table links
- no code-availability statement in the body links the code repository
- https://github.com/ai-inventor-papers/ai-invention-624cdf-local-zipfian-curvature-as-narrative/tree/main/round-1/evaluation-1: not a code folder this run publishes

These links are the paper's only tie from a result to the code that produced it. You MUST, as <artifact_links> describes:
1. Keep or add the appendix section \section{Code for each result}\label{app:code}, a longtable with the columns Section | Result | Code folder
2. Give every URL listed above a row, verbatim inside \href{<URL>}{\texttt{<folder>}}, for the section where <paper_draft> cites it
3. Give every results part listed above a \label right after its heading and a row whose Section column is \S\ref{<that label>}
4. Link only code folders of this run; a URL reported as not one of them is replaced by the folder the draft's footnote for that result names
5. Keep one code-availability statement in the body that links the repository and points to Appendix~\ref{app:code}
6. A code folder linked outside the table: delete that link from the sentence and keep its row; an unfilled placeholder (TODO, <URL>): fill it from <paper_draft>
7. An address reported as containing braces or spaces (in a \newcommand or a link): fix it where it is written so it is one unbroken URL, the same characters as the draft's footnote URLs; every link built from it is broken until then
8. Do NOT change anything else: prose, numbers, claims, citations and figures stay as they are
9. Recompile the PDF with pdflatex/latexmk


LENGTH AND LOOK CHECK FAILED: 1 problem(s), measured on paper.tex and paper.pdf:
- tab:top10 has 8 columns; the limit is 7

Fix each kind as <length_and_look> says:
1. Over the body budget (12 pages, 7,000 words before the References): move per-round detail, secondary analyses, full per-condition results and text tables to an appendix after the bibliography (\clearpage, \appendix, then \section headings), and cut repetition, the headline restated in every section first. The finding, its numbers and every figure stay
2. A caption over 80 words: cut it to about 60, the takeaway in one sentence and only the encodings the axes do not show; the rest goes into the text
3. A footnote carrying a code link: delete it from the sentence and make sure the URL, verbatim, has its row in the \section{Code for each result} appendix table. A \url in a sentence: move it into a footnote, \footnote{\url{<URL>}}
4. A font package in the preamble: delete it, so the paper is all Computer Modern
5. A table over 7 columns: split it in two, or transpose it so the many items are rows; a table of sentences moves to the appendix
6. The appendix starting mid-page: put \clearpage right before \appendix

Then recompile (pdflatex, bibtex, pdflatex, pdflatex) and measure again with `pdfinfo paper.pdf` and `pdftotext paper.pdf - | wc -w`.
```

### [13] SYSTEM-USER prompt · 2026-11-21 13:32:07 UTC

```
CODE TABLE CHECK FAILED: 1 problem(s) with the code-availability statement or the appendix that maps each result to its code:
- https://github.com/ai-inventor-papers/ai-invention-624cdf-local-zipfian-curvature-as-narrative/tree/main/round-1/evaluation-1: not a code folder this run publishes

These links are the paper's only tie from a result to the code that produced it. You MUST, as <artifact_links> describes:
1. Keep or add the appendix section \section{Code for each result}\label{app:code}, a longtable with the columns Section | Result | Code folder
2. Give every URL listed above a row, verbatim inside \href{<URL>}{\texttt{<folder>}}, for the section where <paper_draft> cites it
3. Give every results part listed above a \label right after its heading and a row whose Section column is \S\ref{<that label>}
4. Link only code folders of this run; a URL reported as not one of them is replaced by the folder the draft's footnote for that result names
5. Keep one code-availability statement in the body that links the repository and points to Appendix~\ref{app:code}
6. A code folder linked outside the table: delete that link from the sentence and keep its row; an unfilled placeholder (TODO, <URL>): fill it from <paper_draft>
7. An address reported as containing braces or spaces (in a \newcommand or a link): fix it where it is written so it is one unbroken URL, the same characters as the draft's footnote URLs; every link built from it is broken until then
8. Do NOT change anything else: prose, numbers, claims, citations and figures stay as they are
9. Recompile the PDF with pdflatex/latexmk
```

### [14] SYSTEM-USER prompt · 2026-11-21 13:33:27 UTC

```
CODE TABLE CHECK FAILED: 2 problem(s) with the code-availability statement or the appendix that maps each result to its code:
- results part "Global Model Fits" (sec:global) has no code-table row
- results part "Residual Analysis" (sec:residual_results) has no code-table row

These links are the paper's only tie from a result to the code that produced it. You MUST, as <artifact_links> describes:
1. Keep or add the appendix section \section{Code for each result}\label{app:code}, a longtable with the columns Section | Result | Code folder
2. Give every URL listed above a row, verbatim inside \href{<URL>}{\texttt{<folder>}}, for the section where <paper_draft> cites it
3. Give every results part listed above a \label right after its heading and a row whose Section column is \S\ref{<that label>}
4. Link only code folders of this run; a URL reported as not one of them is replaced by the folder the draft's footnote for that result names
5. Keep one code-availability statement in the body that links the repository and points to Appendix~\ref{app:code}
6. A code folder linked outside the table: delete that link from the sentence and keep its row; an unfilled placeholder (TODO, <URL>): fill it from <paper_draft>
7. An address reported as containing braces or spaces (in a \newcommand or a link): fix it where it is written so it is one unbroken URL, the same characters as the draft's footnote URLs; every link built from it is broken until then
8. Do NOT change anything else: prose, numbers, claims, citations and figures stay as they are
9. Recompile the PDF with pdflatex/latexmk
```

### [15] SYSTEM-USER prompt · 2026-11-21 13:36:06 UTC

```
REFERENCE CHECK FAILED: 2 citation key(s) in paper.tex are not backed by a fetched reference:
- Zipf1935: in references.bib but never fetched (no references.json record)
- Mandelbrot1953: in references.bib but never fetched (no references.json record)

For each key: fetch the paper with the aii-semscholar-bib script (`--out ./references.bib`, which also records it in ./references.json) and cite it by the key the script prints, or remove the citation. Never write a paper's BibTeX entry or edit references.json by hand, and do not use \nocite{*}. A key that is not a paper (a model card, a software repository, a dataset) may instead be a hand-written @misc entry with howpublished = {\url{<URL>}} (or url =) and note = {Accessed <date>} (or urldate =), its URL no arXiv or DOI address. Then delete paper.bbl and paper.aux and re-run pdflatex, bibtex, pdflatex, pdflatex.
```
