# gen_paper_draft — report_results

> Phase: `gen_paper_repo` · `gen_paper_draft`
> Run: `run_fODCbnzS2ZPQ` — Zipf's Law Falls Short: The Zipf-Mandelbrot Model Accurately Captures Word Frequencies in Pride and Prejudice
>
> Full, verbatim record of every prompt the AI Inventor pipeline gave this agent — system-user, human-user and skill-input — in the order they landed. Nothing truncated.

## Task: `gen_paper_draft` (sdk_openhands_agent)

### [1] SYSTEM-USER prompt · 2026-11-20 04:54:36 UTC

```
<task>
Write this run's publishable paper out of its artifacts below, and emit it as structured JSON:
title, abstract, paper_text, figures, summary, headline_candidates. Prose and figure specs only: no LaTeX, no compiled
PDF, no image files. A later step typesets what you write.
</task>

<tool_use>
Maximize parallel tool calls. Parallelize independent operations, only sequentialize dependencies.
- Multiple searches/fetches on different topics → parallel in one turn
- Search then fetch results → ... [truncated, 86143 chars total]
```

### [2] SKILL-INPUT — aii-web-tools · 2026-11-20 04:55:17 UTC

The agent loaded the **aii-web-tools** skill; its `SKILL.md` (the instructions injected into the agent's context) follows verbatim.

````
---
name: aii-web-tools
description: "Runs web search, page fetch as markdown, and regex grep over full HTML or PDF text via this skill's own scripts (aii_fast_web_search.py, aii_fast_web_fetch.py) — a free-first keyless search stack with Serper fallback that works even where built-in WebSearch and WebFetch are absent. Use when a query, page, or paper must be searched, read, or mined for an exact quote, number, table value, or methodology sentence, and whenever a lossy summary would lose the detail. Triggers: web search, scholarly search, OpenAlex, Crossref, Serper, fetch a URL as markdown, read a PDF, arXiv, regex grep a page, exact quote, table value, citation check. NOT for: planning a broad multi-source literature review or mass verification campaign — use aii-web-research-tools; NOT for a PDF file already on disk — extraction, form filling, merging and PDF creation are amg-meta-computer-use (PDF); NOT for driving a browser or testing a UI."
---

## Web tools

You have three web capabilities: **search**, **fetch**, and **grep** (exact
regex extraction over a full page or PDF).

**Pick where they come from, in this order:**

1. **If you have built-in `WebSearch` / `WebFetch` tools, PREFER those over the
   scripts below.** They may be **deferred tools** (listed by name but with
   schemas not yet loaded) — if so, call `ToolSearch("select:WebSearch,WebFetch")`
   ONCE to load them, then use them normally. Do not skip them just because they
   need that one extra load step; they are the preferred path. Pair them with the
   `aii_web_tools__fetch_grep` script below when you need exact text / numbers /
   methodology that a summary would miss, or when reading a PDF.
2. **Only if you have NO built-in `WebSearch` / `WebFetch`** (e.g. the OpenHands
   backend), use the scripts in this skill (below). They are our own
   implementations — free-first web search (keyless general/scholarly engines,
   Serper fallback), html2text + PyMuPDF for fetch, and regex grep over the full
   document text. They work without any built-in web tools.

Workflow either way: **search** (discover) → **fetch** (read for the gist) →
**grep** (pull exact details / read PDFs).

---

## Running the scripts

Run every script with the skill's pre-provisioned interpreter (it already has
`requests`, `html2text`, `pymupdf`, `python-dotenv`). Set `PY` once:

```bash
export SKILL_DIR="$(realpath "$(git rev-parse --show-toplevel 2>/dev/null || echo /ai-inventor)/.claude/skills/aii-web-tools")"
export PY="$SKILL_DIR/../.ability_client_venv/bin/python"
```

`realpath` is required: in amg-os the skill folder is a symlink, and Python resolves the
`..` in its own path textually, so an unresolved `$SKILL_DIR/..` misses the venv and
silently runs a bare interpreter (`No module named 'requests'` or `'fitz'`).

If `$PY` is missing or an import still fails (a fresh worktree's venv holds only the
fig-gen deps), run any script below without a venv, same arguments:

```bash
uv run --no-project --python 3.12 --with-requirements "$SKILL_DIR/scripts/server_requirements.txt" python "$SKILL_DIR/scripts/aii_fast_web_fetch.py" fetch --url "https://arxiv.org/abs/2303.11366"
```

### 1. Search the web (free-first: general or scholarly)

```bash
# general web (default): keyless engines (ddgs, marginalia); Serper only if they miss
$PY "$SKILL_DIR/scripts/aii_fast_web_search.py" --query "neuro-symbolic FOL translation LLM" --max-results 10
# scholarly mode: OpenAlex + Crossref (DOIs, citation counts)
$PY "$SKILL_DIR/scripts/aii_fast_web_search.py" --query "neuro-symbolic FOL translation" --mode scholarly
```

Returns ranked title / URL / snippet lines. `--mode general` (default) uses
keyless general engines; `--mode scholarly` uses academic APIs. Both fall back
to Serper (paid) only when the free engines miss. Use search first to scan the
landscape; snippets are for discovery only — fetch a page before judging it.

### 2. Fetch a page as markdown (HTML or PDF)

```bash
$PY "$SKILL_DIR/scripts/aii_fast_web_fetch.py" fetch --url "https://arxiv.org/abs/2303.11366" --max-chars 10000
```

Built-in `WebFetch` cannot read PDFs, so for a paper's full text use `aii_fast_web_fetch.py fetch` on the arXiv `/pdf` URL (`https://arxiv.org/pdf/<id>`).

`--max-chars` caps output (default 10000); `--char-offset N` pages further in.
Handles PDFs transparently via PyMuPDF.

### 3. Grep a page or PDF (exact regex extraction)

```bash
$PY "$SKILL_DIR/scripts/aii_fast_web_fetch.py" grep --url "https://arxiv.org/pdf/2303.11366" --pattern "verbal reinforcement" --max-matches 20 --context-chars 200
```

Returns only the matching sections with surrounding context — the right tool
for exact numbers, table values, methodology, or long PDFs where a summary
would lose the detail. `-i` for case-insensitive.

**Parallelize** independent searches/fetches in one turn; only sequence a
fetch after the search that produced its URL.

---

## Notes

- The scripts call our ability server. If a script prints
  `Ability service not available`, the server is down — say so rather than
  silently improvising a different search method.
- Do **not** hand-roll your own `requests`/scraping for search when these
  tools are available: Serper returns clean Google results and the fetch/grep
  scripts already handle HTML, PDFs, and encoding.
````

### [3] SKILL-INPUT — aii-semscholar-bib · 2026-11-20 04:55:59 UTC

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

### [4] SKILL-INPUT — aii-concept-fig-gen · 2026-11-20 04:58:37 UTC

The agent loaded the **aii-concept-fig-gen** skill; its `SKILL.md` (the instructions injected into the agent's context) follows verbatim.

````
---
name: aii-concept-fig-gen
description: "Generates and edits CONCEPT FIGURES — architecture and pipeline diagrams, flow charts, cover and hero artwork — with OpenAI GPT Image and Gemini Nano Banana image models through OpenRouter, at a chosen aspect ratio and resolution, in parallel batches. Use whenever a figure must be DRAWN because no dataset sits behind it, or an existing image needs editing from a text instruction. Triggers: concept figure, figure_type='concept', architecture diagram, pipeline diagram, flow chart, cover image, conceptual artwork, image generation, image editing, nano banana, gemini image. NOT for: anything with numbers behind it — bars, curves, heatmaps, confusion matrices, scaling laws — which an image model only approximates, so use aii-data-fig-gen; multi-round variant batches are amg-iter-image-gen-human; calling a TEXT model over OpenRouter is aii-openrouter-llms; displaying a file is amg-meta-computer-use (Mac apps)."
---

# Image Generation & Editing (nano_banana)

> **Not for data figures.** An image model approximates numbers: bars come
> back close to but not equal to their labels, and axis ticks do not divide
> evenly. Nothing downstream detects it. If the figure has numbers behind
> it, use `aii-data-fig-gen`, which renders them deterministically.

Generate images via OpenRouter's dedicated images API (`/api/v1/images`) through the ability server, on four paid tiers tried in order with automatic fallback: `sunburst` (`openai/gpt-image-2.5-sunburst`, Sunburst 2.5, default) → `gpt` (`openai/gpt-5.4-image-2`) → `pro` (`google/gemini-3-pro-image-preview`, Nano Banana Pro) → `flash` (`google/gemini-3.1-flash-image-preview`, Nano Banana 2, the safety net every tier ends on). The `OPENROUTER_API_KEY` lives on the ability server — this skill routes requests through `call_server()`. Inside an AI Inventor run the call still goes to the run's `OPENROUTER_BASE_URL` on the run's own key, so every image counts against the run's OpenRouter budget; past it, generation fails with a message starting "AI Inventor per-run OpenRouter budget".

## Setup

```bash
SKILL_DIR="$(git rev-parse --show-toplevel 2>/dev/null || echo /ai-inventor)/.claude/skills/aii-concept-fig-gen"
G="$SKILL_DIR/scripts/concept_fig_gen.py"
PY="$SKILL_DIR/../.ability_client_venv/bin/python"
```

## Generate Image

```bash
$PY $G --prompt "prompt describing the image" --output output.jpg --aspect-ratio 16:9
```

## Cost — check before you generate

Every image is paid through OpenRouter, on free-backend runs too: there is no
$0 image path. The default is `openai/gpt-image-2.5-sunburst` (`--model
sunburst`) at $0.2469/image, max quality, and every tier can edit.

**None of the four tiers is one price, and `sunburst` is not tiered at all —
it always renders at the maximum.** `sunburst` ignores `--image-size`
entirely and always requests OpenRouter's highest quality setting for this
model (`quality: "max"` — its own catalog gives it an enum
`["auto","low","medium","high","xhigh","max"]` that no other tier has) at the
largest size OpenRouter accepts for the requested `--aspect-ratio` (a
3840px-longest-edge "size", OpenRouter's own ceiling for this model — see
below): $0.2469/image measured live (21:9 → 3840x1648). This is a standing
policy, not a per-call choice: Sunburst is meant to be the best a figure gets,
so it is never asked to render a cheaper draft. `gpt` is ~$0.04/image at 1K
but ~$0.19 at 2K and ~$0.42 at 4K (`size` and `quality: "high"` scale with
resolution — see below, and `gpt` DOES still honour `--image-size`, unlike
`sunburst`). `flash` is ~$0.067/image at 1K but ~$0.101 at 2K, measured live
at $0.1017 for a 2K edit. It matters because the figure step deliberately
uses both 1K and 2K for the non-`sunburst` tiers: it explores at 1K and then
makes exactly TWO 2K passes per figure. `pro` is flat at ~$0.134 across 1K
and 2K.

The paid path has four quality tiers, `sunburst` by default, selected with
`--model`, tried in order with automatic
fallback down the chain:

| `--model` | ~Price | Falls back to |
|---|---|---|
| `sunburst` | $0.2469/image, always max/4K | `gpt`, `pro`, `flash` |
| `gpt` | $0.04 / $0.19 / $0.42 (1K / 2K / 4K) | `pro`, then `flash` |
| `pro` | $0.134 / $0.134 / $0.24 (1K / 2K / 4K) | `flash` |
| `flash` | $0.067 / $0.101 / $0.15 | none (it IS the fallback) |

`sunburst`'s 4K/max figure is measured live; its 1K/2K entries in the price
table exist only as area-scaled estimates because current code never
requests them for `sunburst` — a call that falls back off `sunburst` onto
`gpt` after exhausted retries drops the max override and resumes `gpt`'s own
tiered behaviour (including whatever `--image-size` the caller asked for).
Every call reports its real charge back, so the table only ever stands in for
a response that carried no `usage` block.

**You do not normally choose this**: the pipeline sets it from the run's
`gen_paper_repo.gen_viz.image_model` config, which defaults to `sunburst` in
every preset: every drawn paper figure is Sunburst. A call that exhausts its retries on one
tier moves to the next one in
the chain, and every charge the provider reports is recorded — including one
on a response that came back priced and carrying no image, which is a refusal
(quota, moderation) rather than a blank a retry fills in. Such a response is
not asked for again at the same price, and the figure's failure still names
what the body said.

**The two OpenAI tiers ignore `resolution` outright** (verified live: they
return the same pixel dimensions whatever `resolution` tier is sent). Instead
this skill translates the requested size tier into an explicit `size` (pixel
width×height, derived from `--aspect-ratio`, both dimensions rounded to the
nearest multiple of 16 — OpenRouter rejects anything else live with HTTP 400
"Width and height must both be divisible by 16") plus a `quality` field —
`--aspect-ratio` itself is still honoured directly by all four models.
`gpt` uses whatever tier `--image-size` asked for, at `quality: "high"`.
**`sunburst` ignores `--image-size`, always resolving to the "4K" tier at
`quality: "max"`** — the largest size and highest quality OpenRouter accepts
for it, on any aspect ratio, regardless of what `--image-size` said (even a
missing or invalid value); a call that exhausts sunburst's retries and falls
back to `gpt` resumes `gpt`'s own tiered behaviour, not sunburst's max. Every
tier's `size` longest edge is capped at 3840px (OpenRouter's own ceiling: a
4096px request answers HTTP 400), so the "4K" tier renders at a 3840px
longest edge, not 4096. Edit mode (`--edit`) works identically across all
four tiers — the source image rides along the same way regardless of
`--model`.

Read the words in the image back yourself. `success: true` means a valid JPEG of the
right size arrived — nothing reads the words in it.

## Edit Image

```bash
$PY $G --edit input.jpg --prompt "Make the background blue" --output edited.jpg
```

**Parameters:**
- `--prompt` / `-p` (required) — image description or edit instruction
- `--output` / `-o` (default: `./generated_image.jpg`) — output file path (always saved as `.jpg`; suffix is forced)
- `--edit` — path to source image for editing (omit for generation)
- `--aspect-ratio` (default: `16:9`) — valid: `1:1`, `2:3`, `3:2`, `3:4`, `4:3`, `4:5`, `5:4`, `9:16`, `16:9`, `21:9`
- `--image-size` (default: `1K`) — resolution: `1K`, `2K`, `4K` (on `gpt` this becomes an explicit pixel `size`, not the `resolution` field — see above; `sunburst` ignores this flag entirely and always renders at its own max size)
- `--model` (default: `sunburst`) — paid tier: `sunburst` (`openai/gpt-image-2.5-sunburst`, Sunburst 2.5, always $0.2469/img at max quality/size regardless of `--image-size`, falls back to `gpt` then `pro` then `flash`), `gpt` (`openai/gpt-5.4-image-2`, ~$0.04-0.42/img depending on resolution, falls back to `pro` then `flash`), `pro` (Nano Banana Pro, ~$0.134/img @1K-2K, falls back to `flash`), or `flash` (Nano Banana 2, ~$0.067/img, cheapest, no further fallback). Normally set by the pipeline from `gen_paper_repo.gen_viz.image_model` (`sunburst` in every preset).
- `--style concept` — appends the paper's CONCEPT style: a top-venue Figure 1 for a
  single-column page 6.5 in wide, lettered panels with bold titles read left to right,
  the field's own objects instead of generic icons, thin grey panel frames, saturated
  colour that means one thing per colour, large labels, real text and numbers as given,
  no display formulas, the paper's typeface and the data figures' palette, the canvas
  filled edge to edge. Use it for every drawn paper figure. Without `--figure-style` the
  typeface is the house serif (Computer Modern style) and blue marks the method
- `--figure-style figure_style.json` — with `--style concept`, draws in the paper's
  figure style (see Figure style below): its typeface, serif or sans-serif, its role
  colours with the field's leading accent for the method, and its panel conventions
- `--style neurips` — appends the CHART style (axes, ticks, gridlines) for a plot
  drawn outside the paper pipeline; a paper's charts come from `aii-data-fig-gen`
- `--negative-prompt` — things to exclude from the image
- `--system` — system-level style instruction
- `--timeout` (default: `180`) — the WHOLE call's deadline, and therefore the
  retry budget. Each attempt gets the lesser of 180 s and whatever is left, and
  the loop will not start one it cannot finish: with 180 s and fast failures
  (a connection error, a 5xx) every attempt in the chain runs — up to twelve
  for the default `sunburst` tier (3 attempts on each of sunburst, gpt, pro,
  flash), nine for `gpt`, six for
  `pro`, three for `flash` — while on slow responses it stops and says how
  much budget was left rather than being cut off mid-request. Raise it if you
  want the full budget under slow responses — nine attempts of 180 s would
  need up to 1638 s.

## Print check (paper figures)

A paper prints a concept figure 6.5 inches wide; the image model drew it
thousands of pixels wide, where every label looks fine. After cropping, check
it the way the page shows it:

```bash
$PY "$SKILL_DIR/scripts/concept_print_check.py" fig1_v0.jpg --aspect 16:9
```

It fails a figure wider than 2.6:1 after the crop, or more than 15% wider than
the `--aspect` it was asked for (the drawing did not fill the canvas height),
then downscales it to 650 px wide and has a cheap vision model
(`anthropic/claude-sonnet-5.5`, about $0.003) list every label it cannot read
there, and every label printed under about 7 pt (capitals under about 7 px
tall at that width), readable or not. It writes `fig1_v0_print650.png` (read it back: it is what a reader
sees) and `fig1_v0.print_check.json`, the verdict the figure step reads, tied
to the image by its sha256. Exit 0 pass, 1 fail, 2 unchecked (no vision model
reachable, or a $0 run). `--no-vision` runs the aspect checks alone.

A legible figure can still be a sparse cartoon, so a paper figure is also
judged for MEANING AND STYLE against the field's own Figure 1s (see Domain
exemplars below):

```bash
$PY "$SKILL_DIR/scripts/concept_print_check.py" fig1_v0.jpg --aspect 16:9 \
  --style-note fig1_style_note.md \
  --reference exemplars/2412.07961/fig1_page.png --reference exemplars/2506.01939/fig1_page.png
```

The same model is shown up to three reference pages, the style note (your
notes on them, the claim the figure must show and its draft caption) and the
figure at 650 px, and fails it only for a blocking problem it names: a slogan
panel (a title over a generic icon or a lone word), no worked example, no real
number, far sparser than the references, a panel that does not show its own
claim, colours with no consistent meaning, flow drawn against the data (a
layer stack upside down, an arrow leaving or entering where the data does
not, input and output arrows in one colour side by side), or an arrow
crossing a label or brace. With `--figure-style figure_style.json` it also
fails labels set in the other typeface family than the paper's figure style
(serif where the style is sans-serif, or the reverse). Its answer sits under `style` in
the same verdict file and fails the figure like an unreadable label; the
figure step treats a verdict with no `style` entry as a check not run. Both
calls are booked to the run (about $0.02 together).

## Figure style (once per paper)

A paper's figures are drawn the way the top recent papers of its field draw
theirs, not in one house style: interpretability papers draw sans-serif
figures, a quantum-algorithms paper sets its figures in the serif of its
text. The paper draft step chooses the style ONCE, from the field's
best-cited recent papers:

```bash
$PY "$SKILL_DIR/scripts/figure_style.py" --field "mechanistic interpretability" \
  2403.19647 2406.04093 2408.05147 2211.00593 --out exemplars --write figure_style.json
```

It fetches each Figure 1 page as `fig1_exemplars.py` does (same `exemplars/`
folder), shows them to the vision model, which reports per page the
typeface family, accent colours, words and panel layout, and decides in
code: the majority family (a tie keeps serif), the house colour nearest the
field's leading accent for the method (grey stays the baseline, every colour
stays in the data figures' colourblind set), the word range from the median
words per figure, and one sentence of panel conventions. `figure_style.json`
is then the paper's: the pipeline hands it to every figure, a data figure
sets `"font_family": "DejaVu Sans"` when it is sans-serif, and a concept
figure passes `--figure-style figure_style.json` to `concept_fig_gen.py` and
to the print check. Without a vision model or a fetched page it writes the
house style with `"source": "default"` and exits 2. About $0.02.

## Domain exemplars (a paper's Figure 1)

Before drawing a paper's overview, look at how the top recent papers of its
field draw theirs: they draw the field's own objects on one real example,
print real numbers and give each colour one meaning, which no rule conveys as
well as two or three of them do.

```bash
$PY "$SKILL_DIR/scripts/fig1_exemplars.py" 2412.07961 2506.01939 2504.05228 --out exemplars
```

For each arXiv id it downloads the PDF, finds the page whose text carries the
"Figure 1" caption (pages 1 to 4) and renders it to
`exemplars/<id>/fig1_page.png`. Read each page back and keep the two or three
whose Figure 1 is an overview (a pipeline, a worked example, a concept beside
a result); a pure results chart is no model, so fetch the next candidate. The
ids come from the paper's own bibliography (the draft names them after "Style
references:" in the Figure 1 spec) or one web search such as
`<field keywords> arXiv 2025 ICLR OR NeurIPS OR ICML`. The ids run in
parallel, each download and render has a timeout, the call returns at
`--deadline` seconds (150 by default) with what it has, and a page fetched
before is reused. Nothing in it is paid. It needs poppler's `pdftotext` and
`pdftoppm`. Write 3 to 5 lines of notes per kept page (layout, roughly how
many words, the real objects and numbers it shows, what its colours mean) into
the style note, then adapt the closest one to the paper's own objects and
numbers: never copy an exemplar's content.

## Parallel Batch Generation

Use GNU `parallel` for multiple images:

```bash
export SKILL_DIR="$(git rev-parse --show-toplevel 2>/dev/null || echo /ai-inventor)/.claude/skills/aii-concept-fig-gen"
export PY="$SKILL_DIR/../.ability_client_venv/bin/python"
export G="$SKILL_DIR/scripts/concept_fig_gen.py"
parallel -j 5 -k --group --will-cite 'eval {}' ::: \
  "\$PY \$G -p \"prompt 1\" -o output_1.jpg --aspect-ratio 21:9" \
  "\$PY \$G -p \"prompt 2\" -o output_2.jpg --aspect-ratio 16:9" \
  "\$PY \$G -p \"prompt 3\" -o output_3.jpg --aspect-ratio 1:1"
```

## Preview

Do **NOT** open generated images in a GUI viewer (`loupe`, `xdg-open`, `eog`,
etc.). This skill is for automated / headless generation (e.g. pipeline figure
steps), and popping image windows clutters the user's desktop. Inspect images
programmatically if needed (read the file, check the returned JSON), not by
opening a viewer.

For interactive, human-curated review of multiple figure variants — where the
user wants to arrow-navigate batches in `loupe` — use the
`amg-iter-image-gen-human` skill instead; loupe-driven review is its job, not
this one's.

## Features

- **Model**: default `openai/gpt-5.4-image-2` (`--model gpt`); `--model pro` selects `gemini-3-pro-image-preview` (Nano Banana Pro); `--model flash` selects `gemini-3.1-flash-image-preview` (Nano Banana 2). Each tier that exhausts its retries falls back to the next one down the chain (gpt → pro → flash)
- **Auth**: API key on ability server (routed via `call_server()`)
- **Retries**: 3 attempts with exponential backoff per tier, then the next tier in the chain — as far as `--timeout` allows, since it is the deadline for the whole call
- **Edit mode**: Edit existing images with text instructions
- **Parallel**: GNU `parallel` with `-j 5` for batch generation
- **Headless**: never auto-opens a viewer (use `amg-iter-image-gen-human` for human review)

## Prompting Tips

- Name every element and where it sits — boxes, arrows, groupings, labels.
  The model places what you describe and invents what you leave out
- **Put the labels in their own closing sentence**, not inline in the sentence
  that describes the layout. "…three boxes joined by arrows. The boxes read
  Tokenizer, Transformer, Classifier." rendered all three words correctly;
  "…three labelled boxes left to right, Encoder, Router, Decoder, joined by
  arrows…" rendered `Enc:der`. Four out of four runs that stated the labels
  as a separate final sentence spelled every one of them right, including the
  same words the inline phrasing had corrupted. Word length was not the
  driver — `Transformer` and `Classifier` both came out clean
- Specify colors, fonts, layout, and what to exclude
- Use `--style concept` for a paper's drawn figure, with `--figure-style
  figure_style.json` when the paper has one. It sets the paper's typeface (the
  house serif, Computer Modern style, unless the figure style says sans-serif),
  pins the figure to the same colours every DATA figure in the paper uses — seaborn's
  `colorblind` — and tells the model not to let red-versus-green be the only
  difference between two elements, which is the one pairing that carries no
  meaning for about 8% of male readers
- Be detailed about LAYOUT and exact about TEXT: where each panel, object and
  arrow sits, its shape and colour, and the exact text, tokens and numbers it
  shows. A paper's concept figure prints 6.5 in wide and holds 2 to 4 lettered
  panels read left to right, about 80 to 120 words, mostly the paper's own text
  (a real prompt, the decisive tokens, a one-line code snippet), and labels of
  at most about eight words. Never ask for small text or generous white space:
  ask for large labels and a filled canvas
- Draw the field's own objects — token strips, code lines, layer stacks,
  probability bars, sampling trees, a small result thumbnail — never a generic
  icon (a gear, a brain, a thermometer, a question mark) and never a slogan
  panel, a title over an icon or a lone word
- **Draw the flow the way the data runs.** A model's layer stack has its
  input tokens at the BOTTOM, block 1 lowest and the last block highest, and
  its output leaving from the TOP (or the whole figure runs left to right);
  never a stack labelled "Layer 1" at the top while its input enters below.
  Drop per-layer labels rather than flip them: one side label ("Qwen3-8B, 36
  blocks") and a ⋮ for the hidden blocks. Every arrow starts where its data
  comes from and ends where it goes: an intervention applied at every block
  enters the whole column from the side, spanning all blocks; the next token
  leaves from the top of the position that predicts it. Input and output
  arrows get different colours, and never start side by side. Say each
  arrow's start, end and colour in the prompt, then check them on the image
- **Arrows never cross labels or braces.** Say in the prompt which side of
  each label, brace or box an arrow passes, or put the label where no arrow
  runs (a fork arrow through a brace label failed an independent review). Keep
  every label clear of the panel frame
- **Nothing under about 7 pt at print width.** The page prints the figure 6.5
  in wide: 7 pt is a capital letter about 7 px tall in the 650 px preview, or
  about 40 px in a 3840 px render. Small text comes from too many labels, so
  cut them before shrinking any: no per-layer or per-row labels, axis ticks
  only at the ends, one short side label
- Real numbers belong in an overview — the headline result, a k, a
  temperature, a count — each stated explicitly in the prompt and read back
  off the image to check it survived. If the figure is mostly numbers, it is a
  data figure: stop and use `aii-data-fig-gen`, which renders them instead of
  approximating them
- 1K resolution is default and most reliable

## Figure type templates

An image model draws what you name and invents what you leave out, so the
prompt for each kind of concept figure has a different set of things it
cannot omit. Each entry below names the kind, its usual aspect ratio and those
things; start from the one that matches and add the specifics.

- **Paper overview, Figure 1** (`16:9`) — the closest exemplar's template
  (see Domain exemplars): a left-to-right strip on one real example (input,
  then the method's steps, then the outcomes), a problem | solution or
  baseline | ours pair, or lettered panels going from a real artefact to the
  method to a small real result; the worked example's exact text and numbers;
  the headline number; what each colour means.
- **Architecture / pipeline diagram** (`16:9`, never wider than 2:1 in a
  paper) — every stage in order, left to right, drawn as the object it handles
  (a code file, a token strip, a layer stack); what flows along each arrow and which way it points; which stages are
  yours vs. baseline or off-the-shelf; where the boundary of the system sits.
- **Flow chart** (`16:9`) — each decision point and both of its
  outcomes; where a branch rejoins; the start and the terminal states; that
  arrows are labelled, not bare.
- **Side-by-side comparison** (`16:9`) — what the two panels are, in which
  order; that both use the SAME visual vocabulary so only the difference
  differs; a caption strip or heading per panel.
- **Conceptual artwork / cover** (`1:1`, `16:9`) — the single idea in one
  sentence; the metaphor and what maps to what; that no text appears unless you
  asked for it, since invented labels are the usual failure.

A well-scoped paper overview, as a model of the scale: the Figure 1 of
Forking Paths in Neural Text Generation (ICLR 2025), described the way a
prompt would ask for it: *"One wide strip read left to right on white. A grey
rounded box 'Initial Prompt' holds the real question 'Who is the current
British head of state? Let's think step by step.' A brown 'Base Path' line
carries the generated answer 'First, we know that the current year is 2024.
We also know that Queen Elizabeth was ...', with its forking tokens as small
red chips ('2024' against '2021', 'that' against 'who', 'was' against 'is')
and green 'Alternate Paths' curving off at each. On the right each path ends
in its 'Outcome': 'Queen Elizabeth II' and 'Keir Starmer' in pink, 'King
Charles III' in cyan. Brown means the base path, green an alternate, red the
forking token."* Real model text and real tokens, about 50 words, every
colour one meaning, no icons. A paper's own overview adds its headline number.
Never a row of slogan titles over a thermometer, a gear and a question mark.

Two things every entry shares: pass `--style concept` (it carries the paper's
typeface, palette and sizing), and read the image back to check that nothing was
invented — a stage that is not in your pipeline, an arrow that runs the wrong
way, a label you never wrote.

Reading it back is not optional, and re-running is a real fix. The same prompt
sent twice gave a correct three-box chain once and, the other time, four boxes
with `Encoder` in two of them and an arrow pointing at nothing — identical
text, different diagram. So a structure you cannot check by looking is a
structure you do not have; when it comes back wrong, generate it again rather
than editing the prompt, because the prompt was not what failed.

## Aspect Ratios

Pick by shape, not by venue. `--help` lists all ten; these are the ones a
paper figure normally wants.

| Ratio | Use Case |
|-------|----------|
| `16:9` | Wide: every paper concept figure (2 to 4 panels) |
| `21:9` | Ultra-wide: never a paper figure (labels too small) |
| `4:3`, `3:2`, `5:4` | Standard — one diagram with room around it |
| `1:1` | Square — a symmetric diagram, a cover image |
| `9:16`, `3:4`, `2:3`, `4:5` | Vertical — a stacked flow or poster |

**If the ability server is not running**, nothing needs doing: the CLI already
falls back to running the same function in-process, so `concept_fig_gen.py`
works standalone. Verified — with no server reachable it still reports its
own errors ("OPENROUTER_API_KEY not set")
rather than a connection failure.

What it needs is the deps. If the import fails, install them INTO THE VENV
`$PY` names above — creating a `.venv` in whatever directory you happen to be
standing in leaves `$PY` pointing at the same broken interpreter:
```bash
CLIENT_VENV="$SKILL_DIR/../.ability_client_venv"
uv venv "$CLIENT_VENV" --python=3.12          # only if it is not there yet
uv pip install --python="$CLIENT_VENV/bin/python" \
  -r "$SKILL_DIR/scripts/server_requirements.txt"
```
````

### [5] SKILL-INPUT — aii-data-fig-gen · 2026-11-20 04:59:20 UTC

The agent loaded the **aii-data-fig-gen** skill; its `SKILL.md` (the instructions injected into the agent's context) follows verbatim.

````
---
name: aii-data-fig-gen
description: "Renders publication-quality DATA FIGURES deterministically from a JSON spec via matplotlib — bar, line, scatter, heatmap, confusion matrix, box, violin, histogram, ECDF, ROC/PR, calibration, scaling law, Pareto frontier, forest/CI, volcano, dendrogram, clustermap, network graph, lettered multi-panel composites — as vector PDF plus PNG. Use whenever a figure plots numbers that already exist, so the drawing cannot disagree with them, and for hand-written matplotlib that must match the paper's house style. Triggers: chart, plot, graph, data figure, figure_type='data', confusion matrix, ablation grid, training curve, ROC, precision-recall, colourblind palette, Type 42 fonts, chart spec JSON. NOT for: figures with no dataset — architecture and flow diagrams, conceptual artwork, cover images — which go to aii-concept-fig-gen; charts that must live inside an Excel workbook are amg-meta-computer-use (Excel); displaying a rendered file is amg-meta-computer-use (Mac apps)."
---

# Data figures — charts rendered from their numbers

Deterministic figures from a JSON spec: the numbers go in, matplotlib draws
them, and the picture cannot disagree with the data. Nothing is generated by
a model, so a bar is the height of its value and every axis is computed.
Re-running a spec gives a byte-identical PNG; the PDF differs only in its
embedded creation timestamp.

## Data figure or concept figure?

| The figure is… | Use |
|---|---|
| A chart of numbers you have | **this skill** |
| A confusion matrix, ablation grid, correlation | **this skill** |
| A scaling law, training curve, Pareto trade-off | **this skill** |
| Artwork, a metaphor, a cover image | `aii-concept-fig-gen` |
| An architecture or flow diagram | `aii-concept-fig-gen` |

In that table **this skill** means a data figure and `aii-concept-fig-gen` a
concept figure. For an architecture or flow diagram, read *Limits* first.

The test is whether the figure has underlying numbers. If it does, an image
model will approximate them — bars that do not match their labels, axis
ticks that do not divide evenly, invented data points. That failure is
invisible to a reviewer of the prompt and obvious to a reviewer of the
paper.

## Use a generator when one fits — hand-write only when none does

The generators are a menu, not a fence. Every type below is a shortcut that
already has the house style, the data-integrity guards and the layout fixes
baked in, so reaching for one is almost always less work than plotting by
hand and the result is consistent with every other figure in the paper.

**Check `--list-types` first.** If a type matches what you need, use it.
Don't know the name? `--search "<the question your figure answers>"` ranks
the catalogue by intent rather than by name — `--search "before and after
per method"` puts `slope` first and `dumbbell` second.
Two-thirds of research figures are a bar, a line, a scatter or a heatmap,
and those are solved.

`--search` spans **two corpora** and labels every hit with which one it
came from:

| label | what it is | what to do |
|---|---|---|
| `ours: <type>` | one of our 61 types | `--example`, edit, render |
| `chartmimic: <task>/<id>` | a published figure | read its `.py` |

A `chartmimic:` hit is a **reference, not a spec.** It is a human-curated
figure from a STEM paper with the matplotlib that draws it — from
ChartMimic ([arXiv:2406.09961](https://arxiv.org/abs/2406.09961)), 4,800 of
them over 22 categories. Adapting one is a *hand-written* figure: no house
style, no data-integrity guards, no layout passes unless you call them, so
everything above about hand-written figures still applies. The search
prints the path to its code under every such hit. Generators outrank
exemplars on a tie, because a generator is the runnable answer.

Reach for an exemplar in exactly two cases: **nothing in the catalogue
fits** (see the gap table below), or you want to see how a published figure
did something — a twin axis, a labelled contour — in working code.
`--corpus ours|chartmimic|all` narrows the search; the default is `all`.

**If nothing fits, write matplotlib yourself** — that is expected and
supported, not a failure. Novel or one-off figures exist. When you do:

```python
import sys; sys.path.insert(0, "<skill>/scripts")
import matplotlib.pyplot as plt
from chart_geometry import assert_text_is_legible, fit_point_labels
from chart_style import (
    apply_house_style, apply_color_map, PALETTE, literal, place_legend, place_point_label,
    fit_legends, clear_legends_of_data, fit_tick_labels, fit_titles,
    rasterize_dense_clouds, fit_canvas_to_fixed_aspect, assert_legends_clear_of_data,
    assert_series_are_distinguishable, assert_axis_names_are_unique,
)

apply_house_style()                 # fonts, palette, grid, Type-42 PDF fonts
fig, ax = plt.subplots(figsize=(6.5, 3.66), layout="constrained")
...
place_legend(ax, loc="best")        # a legend fit_legends can reflow
place_point_label(ax, literal("Ours"), (1, 2))   # a name, nudged off the data
apply_color_map(fig, {"Ours": PALETTE[0]})  # the paper's colour map, by series name
fit_legends(fig)                    # reflow a legend wider than its axes
clear_legends_of_data(fig)          # move it below the axes if it sits on data
fit_tick_labels(fig)                # wrap/tilt tick labels that would collide
fit_titles(fig)                     # wrap any title wider than its axes
clear_legends_of_data(fig)          # AGAIN — the two above reshaped the axes
fit_point_labels(fig)               # move point names off markers and curves
rasterize_dense_clouds(fig)         # >25k points as a bitmap, text stays vector
fit_canvas_to_fixed_aspect(fig)     # square-cell imshow grid: height to the grid
assert_text_is_legible(fig)         # raises if any text collides or is cut off
assert_legends_clear_of_data(fig)   # raises if a legend still hides its data
assert_series_are_distinguishable(fig)  # raises on two identical legend keys
assert_axis_names_are_unique(fig)   # raises if one name labels two positions
fig.savefig("figX_v0.pdf")          # vector, so LaTeX renders text at page res
```

Call the fitters in that order — the legend decides how much room the axes
has, whether it then has to move out of the data is only knowable once it is
placed, tick labels change the axes height, the title is measured against the
axes it ends up on, and a point's name can only be placed once nothing above
it will move the point again. `clear_legends_of_data` appears TWICE on
purpose: it decides by measuring, and the two passes between its calls shrink
the axes under a legend that is already placed and a fixed size. A wrapped
title took a lone chart from 179 px of axes height to 141, and a legend that
covered nothing before covered half a curve after — with the mover's turn
already past, so the figure was refused rather than fixed. The first call
still has to happen first, because the room the legend needs is an input to
the passes below it. Two further gates are warning-based and so are
not in the snippet: `assert_layout_applied` and `assert_all_glyphs_rendered`
read what matplotlib warned about during the draw, so they need the figure
built inside `warnings.catch_warnings(record=True)` — worth doing, since a
missing glyph is only ever a warning and ships as a hollow box.
`place_legend` and `place_point_label` are how
the fitters find what to fix: a legend built with a bare `ax.legend` cannot
be reflowed, and a name written with a bare `ax.annotate` will not be moved
off the marker it landed on.

That keeps a hand-written figure looking like the rest of the paper and
still gets you colourblind-safe colours, submission-compliant fonts, no
clipped labels and no overprinted ones. What you lose is the data-integrity
checking — so verify the numbers yourself.

**If you hand-write the same figure type twice, add a renderer instead.**
`chart_renderers*.py` — one function, `(ax, spec) -> None`, registered in
its family's dict. That is how this catalogue got here.

## Use it

```bash
SKILL_DIR="$(git rev-parse --show-toplevel 2>/dev/null || echo /ai-inventor)/.claude/skills/aii-data-fig-gen"
G="$SKILL_DIR/scripts/chart_gen.py"

python "$G" --list-types            # the catalogue
python "$G" --search "compare distributions across groups"   # find it by intent
python "$G" --search "pie wedges" --corpus chartmimic         # exemplars only
python "$G" --audit                 # what ChartMimic has that we do not
python "$G" --example bar           # a complete spec to copy and edit
python "$G" --spec fig1.json --out figures/fig1
```

`python` here is the pipeline image's interpreter, which has matplotlib and
scipy installed system-wide. Outside the image use the project venv —
`.venv/bin/python` — since a bare `python3` will not have them.

Writes `figures/fig1.pdf` **and** `figures/fig1.png`. The PDF is the
deliverable — LaTeX renders vector text at page resolution, so it stays
sharp and selectable at any zoom. The PNG exists so you can read the figure
back and look at it.

`--format pdf`, `--format png`, `--format pdf,png,svg` narrows the output.
SVG keeps its labels as TEXT rather than paths, so it stays editable and
searchable. EPS is refused: the PostScript backend cannot draw transparency
and flattens it silently, which the house style uses on nine of every ten
figures — the file would not match the PNG you checked.
`--spec -` reads the spec from stdin.

Runs on `matplotlib` + `numpy`, both already `aii_pipeline` dependencies —
nothing to install.

## The catalogue

`--example <type>` prints a complete spec for any of these. The "choose it
over" half of each entry is the useful one: most figures have two plausible
types and the choice between them is what decides whether a reviewer reads
the point.

### Comparing categories

- `bar` — draws: Vertical bars, grouped or stacked, optional error bars.
  Choose it over: The default. `barh` if names are long.
- `barh` — draws: Horizontal bars — labels on the y-axis with room to run.
  Choose it over: `bar`, whenever names exceed ~40 chars, or for a ranking.
- `lollipop` — draws: A stem and a dot per category. Choose it over: `barh`,
  past ~20 categories, where bars become a picket fence.
- `dumbbell` — draws: Two markers per row joined by a line. Choose it over:
  Paired bars, when the GAP between them is the story.
- `slope` — draws: One line per item from a before value to an after value.
  Choose it over: Paired bars, when which items changed RANK is the story.
- `bump` — draws: Rank against time, one line per item; the crossings are
  the finding. Choose it over: `slope`, which shows a reordering for exactly
  TWO time points and cannot show the path between more.
- `volcano` — draws: Effect size against significance, with both thresholds
  drawn. Choose it over: A `bar` of effects, which cannot show what survived
  correction, or a table of p-values, which cannot show what was big enough
  to matter.
- `diverging` — draws: Signed bars either side of zero, sorted. Choose it
  over: `bar`, for deltas — direction reads instantly.
- `waterfall` — draws: Steps from a starting total to a final total. Choose
  it over: `bar`, for an ablation — it shows contributions compounding.
- `bar_sig` — draws: Grouped bars with significance brackets and stars.
  Choose it over: `bar`, when the comparison being claimed is pairwise.
- `forest` — draws: Point estimates with confidence intervals and a null
  line. Choose it over: `bar`, when whether an interval crosses zero is the
  question.
- `radar` — draws: A closed polygon per method over 3+ metrics. Choose it
  over: Several bar charts, for a multi-metric profile at a glance.
- `parallel` — draws: One polyline per configuration across independently
  scaled axes. Choose it over: A table, for a hyperparameter sweep — trends
  across axes show up.
- `funnel` — draws: Stage attrition with retention vs. previous and vs.
  intake. Choose it over: `barh`, when the stages are sequential and losses
  compound.
- `stacked_pct` — draws: Composition as percentages; every bar full height.
  Choose it over: Stacked `bar`, when categories have very different totals.
- `treemap` — draws: Nested rectangles with AREA proportional to value.
  Choose it over: `bar`, only when there are too many parts for one axis —
  length beats area for precise reading.
- `upset` — draws: Set intersections as sorted bars over a membership
  matrix. Choose it over: A Venn diagram, past 3 sets — circles cannot stay
  area-true and stop reading as sets.

### Trends and relationships

- `line` — draws: Multi-series lines with optional uncertainty bands. Choose
  it over: The default for anything against time or steps.
- `fan` — draws: A median with nested quantile bands around it. Choose it
  over: `line` with a band, when the spread is skewed or bounded — a
  symmetric ± band on an accuracy near its ceiling implies scores above
  100%.
- `step` — draws: A piecewise-constant series — value holds, then jumps.
  Choose it over: `line`, for schedules — a slope implies values that never
  occurred.
- `scatter` — draws: Points with an optional least-squares fit and R².
  Choose it over: `line`, when x is not ordered and the relationship is the
  point.
- `joint` — draws: Scatter with the marginal distribution of each variable
  beside it. Choose it over: `scatter`, when "and how is each one
  distributed?" is the obvious next question — which for a headline
  correlation it always is.
- `splom` — draws: Every pair of variables as its own scatter, distributions
  down the diagonal. Choose it over: `corr`, when the SHAPE of each
  relationship is the claim — one number cannot tell a straight line from
  two clusters or an outlier.
- `bubble` — draws: Scatter with a third variable as marker AREA, plus a
  size key, never inside the axes; the colour key takes the strip under them
  and `legend_loc` is refused. Choose it over:
  `scatter`, when a third quantity matters but not enough for its own axis.
  `"size_key": false` drops the key when every point's label already prints
  the sized value. Fixing a rendered figure means editing its spec and
  re-rendering, never re-drawing it in a hand-written script that moves a
  key into the data.
- `scaling` — draws: Log-log points with a fitted power law and its
  exponent. Choose it over: `line`, for scaling laws — the exponent is
  computed and annotated.
- `speedup` — draws: Measured speedup against worker count, with the ideal
  line. Choose it over: `line`, for parallel results — the ideal reference
  is what the claim is measured against.
- `pareto` — draws: Scatter with the non-dominated frontier drawn through
  it. Choose it over: `scatter`, for trade-offs where the frontier is the
  finding.
- `area` — draws: Stacked areas — a total and how it divides. Choose it
  over: `line`, when the total matters as much as the parts.
- `residual` — draws: Residuals against fitted values, with the zero line.
  Choose it over: Predicted-vs-actual, where heteroscedasticity hides on the
  diagonal.
- `bland_altman` — draws: Difference between two methods against their mean,
  with limits of agreement. Choose it over: A scatter of A against B, where
  the diagonal reads as agreement and r = 0.99 hides a 10% offset.
- `acf` — draws: Autocorrelation per lag as stems, with the significance
  band. Choose it over: `line`, which shows the level and hides whether each
  point predicts the next.
- `sankey` — draws: Flows between stages at proportional widths. Choose it
  over: `area`, when what matters is what became what.
- `timeline` — draws: Gantt-style spans, one row per task. Choose it over: A
  table of timestamps, when overlap and duration are the point.

### Model evaluation

Give these raw `labels` and `scores` rather than a precomputed curve wherever
you can: the renderer sweeps the threshold itself, so the AUC or AP in the
legend is integrated from the points actually drawn and cannot drift from
the curve beside it.

When only the curve survives — it came from a paper, or from a logged
artefact — pass it directly instead: `fpr`/`tpr` for `roc`, `recall`/
`precision` for `pr`, `probabilities`/`labels` for `calibration`. The
summary statistic is still integrated from the plotted points, so a PR curve
that stops short reports `AP = 0.375 up to recall 0.60` rather than quietly
extrapolating the rest. One evaluation set per figure: `pr`'s baseline and
`calibration`'s bins both move with class balance, so curves from different
test sets cannot share axes honestly.

- `roc` — draws: ROC curves with AUC in the legend, plus the chance
  diagonal. Choose it over: `pr`, when the classes are roughly balanced.
- `pr` — draws: Precision-recall curves with average precision and the
  prevalence baseline. Choose it over: `roc`, when positives are rare — ROC
  flatters a rare-class model.
- `calibration` — draws: Reliability diagram with the ideal diagonal, ECE,
  and per-bin counts. Choose it over: `roc`/`pr`, when whether to TRUST a
  probability is the question.
- `learning_curve` — draws: Score against training-set size, train and
  validation with ±std bands. Choose it over: `line`, to show whether more
  data or a better model is the bottleneck.
- `qq` — draws: Sample quantiles against theoretical normal quantiles, with
  a reference line. Choose it over: `hist`, for judging normality — the eye
  reads a straight line far better than a bell.
- `cd_diagram` — draws: Mean ranks over many datasets, joining methods a
  test cannot separate. Choose it over: `bar_sig`, which compares pairwise
  on ONE dataset — this is the many-datasets headline figure.

### Distributions

- `box` — draws: Median, quartiles, whiskers, outliers per group. Choose it
  over: The compact default for a few groups.
- `violin` — draws: Full mirrored density per group. Choose it over: `box`,
  when a distribution may be multi-modal — a box hides that.
- `strip` — draws: Every raw observation, jittered, with the mean marked.
  Choose it over: `box`, when n is small enough that each point should be
  visible.
- `beeswarm` — draws: Every observation, packed sideways so none hides
  another. Choose it over: `strip`, whose random jitter still overlaps at
  any real n — the eye reads the clumps as density and they are partly
  collision.
- `ridgeline` — draws: Stacked density curves, one row per group. Choose it
  over: `violin`, past ~6 groups, where a violin grid gets too wide.
- `raincloud` — draws: Half violin, box and jittered points together, with
  n. Choose it over: `violin`, when the reader must see the observations —
  twelve seeds look as smooth as twelve thousand.
- `hist` — draws: Binned counts or density. Choose it over: `ecdf`, only
  when the shape of ONE distribution is the point.
- `ecdf` — draws: Empirical cumulative distribution, stepped. Choose it
  over: `hist`, for comparing distributions — no bin width to argue about.
- `survival` — draws: Kaplan-Meier curves with censoring ticks and
  confidence bands. Choose it over: `ecdf`, when some subjects have not
  finished — an ECDF must drop or invent those.
- `hexbin` — draws: Hexagonal density bins with a colourbar. Choose it over:
  `scatter`, past ~2000 points where it becomes a solid blob.
- `hist2d` — draws: A joint distribution as a rectangular binned grid.
  Choose it over: `hexbin`, when the axes are naturally rectangular.

### Matrices and fields

- `heatmap` — draws: Annotated matrix with a colourbar. Choose it over: A
  table, when the pattern matters more than the digits.
- `seqheat` — draws: A per-token quantity drawn on the tokens themselves.
  Choose it over: `heatmap`, for anything measured per token — it puts
  indices on an axis and leaves the reader rebuilding the sentence from a
  legend.
- `corr` — draws: Correlation matrix, diverging map centred at zero. Choose
  it over: `heatmap`, for correlations — sign reads from colour direction.
- `contour` — draws: Filled contours of a 2-D field, levels labelled. Choose
  it over: `heatmap`, for a smooth field like a loss surface.
- `clustermap` — draws: Heatmap with rows and columns reordered into their
  clusters, trees drawn beside. Choose it over: `heatmap`, whenever the row
  order is arbitrary — block structure that is obvious once reordered is
  invisible in the order the log happened to emit.
- `catmap` — draws: A grid whose cells hold a CATEGORY, with a discrete
  legend and no scale. Choose it over: `heatmap`, for any nominal cell —
  expert IDs, pass/fail/timeout, which variant won. A ramp asserts that
  expert 4 is more than expert 1 and that 2 lies between them, and a reader
  takes the ordering as real.
- `quiver` — draws: A field of arrows: where each sample is, and where it
  went. Choose it over: A `scatter` of the before and after positions, which
  carries the same numbers and leaves the reader pairing points up by eye.

### Structure

- `dendrogram` — draws: Hierarchical clustering as a tree, branch heights
  the real merge distances. Choose it over: `corr`, which shows every
  pairwise relationship and no grouping.
- `tree` — draws: A rooted tree from a parent/child structure you already
  have. Choose it over: `dendrogram`, which computes its own linkage from a
  matrix and cannot be given a tree — and `network`, whose force layout
  loses depth.
- `network` — draws: A graph as nodes and links, node area and edge width
  from the data. Choose it over: A concept figure, for anything with REAL
  edges — an image model draws a plausible graph, not yours. Use `sankey`
  for flows between ordered stages and `heatmap` for a dense graph.

### Composites

- `panel` — draws: Any of the above in a lettered grid, `(a)`–`(p)`. Choose
  it over: Several separate figures, when they are read together.

## What ChartMimic has that we do not

Measured, not guessed: `chart_gen.py --audit` maps all 22 ChartMimic
categories onto our 61 types over the 4,800 indexed exemplars. Seventeen
categories are covered. These five are not — every one of them is a figure
shape real papers publish and no generator can produce:

| ChartMimic | n | what to do instead |
|---|---|---|
| Combination | 240 | Hand-write. Bars + a line, usually `twinx`. |
| Hard-to-Recognize | 200 | Hand-write; it is their catch-all. |
| 3D | 160 | `heatmap` or `contour`; 3-D scatter: see below. |
| Plot-in-Plot | 160 | Hand-write `ax.inset_axes`; not `panel`. |
| Pie | 160 | `barh`, `stacked_pct` or `treemap`. |

**Combination is the one real gap.** Bars with a line on a second y-axis is
a standard results figure and we have no type for it; the other four are
either deliberate refusals (pie, 3-D bars and surfaces — both read worse
than what we do have) or not a chart type at all (Hard-to-Recognize). Adding a
`bar_line`/dual-axis renderer would close the largest measured hole in the
catalogue.

The 17 covered categories are where an exemplar is a REFERENCE rather than
a gap-filler: our generator is still the answer, and the exemplar shows how
a published figure handled the same shape.

### The exemplar store, and rebuilding the index

The raw corpus — code, PNG and PDF per exemplar, ~450 MB — lives in
gitignored `aii_data/chartmimic/`. What is committed is
`scripts/chartmimic_index.json` (1.02 MB, 4,800 entries: id, category,
derived subtype, a one-line intent, up to six matplotlib feature keywords).
Images and code are never tracked — `rule-big-blob-public-only` caps a new
tracked file at 2 MB and the public export excludes data outright.

```bash
python "$SKILL_DIR/scripts/chartmimic_index_build.py" --fetch  # populate the store
python "$SKILL_DIR/scripts/chartmimic_index_build.py"          # rewrite the index
```

`--fetch` pins the dataset revision, which the index records alongside its
count and build date, so a rebuild is checkable. Without the store the
search still works — it just returns our own types only.

## Spec shape

```json
{
  "type": "bar",
  "title": "Accuracy by benchmark",
  "xlabel": "Benchmark",
  "ylabel": "Accuracy (%)",
  "aspect": "16:9",
  "categories": ["ARC", "GSM8K", "HumanEval"],
  "series": [
    {"label": "Baseline", "values": [41.2, 55.8, 33.1], "errors": [1.8, 2.4, 2.9]},
    {"label": "Ours",     "values": [48.9, 67.3, 45.6], "errors": [1.5, 2.0, 2.6]}
  ]
}
```

Keys every type takes: `title`, `show_title`, `aspect` (`"W:H"`), `width_in`
(default 6.5, the paper's `\linewidth`, so the figure prints at 100%),
`font_pt`, `font_family`, `color_map`.

- **No in-figure title by default.** `title` names the figure in the spec,
  and it is drawn only with `"show_title": true`. In a paper the caption is
  the title: drawn in the figure as well, it repeats the caption and takes a
  line of height from the plot. A `panel`'s sub-spec titles are always drawn,
  because there they are the cells' headings.
- **One colour map per paper: `color_map`.** `{"series name": colour}`, each
  colour one of the palette's hex values, applied by name to the drawn series
  and their legend entries. Renderers colour series by position, so without
  it the first series of every figure is blue whatever it is. Give every
  figure of a paper the same map; names a figure does not draw are skipped.
  A colour outside the palette is refused.

Keys that depend on what the type actually draws. Passing one to a type that
never reads it is REFUSED by name — *"nothing read this key"* — rather than
dropped quietly, so a figure never comes back missing what the spec asked
for. "Applies to" below is therefore the set that is accepted, not a hint:

- `xlabel`, `ylabel` — applies to: every type with axes, which is all of
  them but `panel` — a panel has none of its own, so put the labels on the
  sub-specs and a label at panel level is refused. `radar`, `treemap`,
  `sankey`, `parallel` and `upset` do read the key, but draw their own
  geometry with the axis turned off, so the label is accepted and never
  painted.
- `xlim`, `ylim` — applies to: every type — the shared layer applies them
  whatever the geometry, so these two are never refused as unread. Limits
  that would crop data are refused rather than applied.
- `legend_loc` — applies to: only the types that actually draw a legend,
  i.e. two or more named series. A one-series chart gets none, because a
  one-entry legend restates the y-label — and asking to place a legend that
  is not drawn is refused. Takes matplotlib's in-axes placements (`best`,
  `upper right`, `lower left`, …) and NOT `outside …`: that is what the
  layout pass itself uses when it moves a legend off the data, and
  matplotlib accepts it only on a figure legend. You do not need to ask for
  it — the move happens on its own. `bubble` refuses it: its colour key
  always sits under the axes.
- `cmap` — applies to: only the eight types that encode a value as colour —
  `heatmap`, `clustermap`, `corr`, `hist2d`, `hexbin`, `contour`, `quiver`,
  `seqheat`. Anywhere else it is refused: a bar chart given a colour map is
  a spec expecting colour to carry a meaning that chart never encodes. The
  default is already perceptually uniform (`cividis`, or `RdBu_r` where the
  scale has a meaningful zero), so reach for this only with a reason.
  Rainbow and cyclic maps are refused: `jet` puts a bright band in the
  middle of a run that is monotonic in the data, and a reader takes the band
  for a boundary in the result.

`font_family` goes in FRONT of the default CMU Serif and DejaVu Serif, and
matplotlib draws each glyph from the first of the three that has it: the
font you name draws everything it covers, the Latin labels and digits
included. Two uses: the paper's figure style (`figure_style.json`, chosen
from the top papers of the paper's field) is sans-serif, which every data
figure of that paper sets as `"font_family": "DejaVu Sans"` (its maths then
switches to DejaVu Sans too); or a script the default cannot draw: CJK,
Devanagari, Thai. See *Legibility*.

Per-type keys are documented by `--example <type>`; start from the example
rather than the schema.

**Series marks are on by default; do not hand-code them.** Every type whose
series are lines or points (`line`, `ecdf`, `step`, `roc`, `pr`,
`calibration`, `learning_curve`, `bump`, `radar`, `parallel`, `survival`,
`scaling`, `scatter`, `residual`, `bland_altman`, `pareto`, `qq`, `dumbbell`,
`speedup`, and any `panel` built from them) draws series *i* with its own marker shape and
dash pattern as well as its colour: circle solid, square dashed, triangle
dotted, diamond dash-dot, then down-triangle, plus, cross, left-triangle.
The order is tied to the series index like the colour, so series 1 looks
the same in every figure of a paper; a dense line carries about eight markers,
staggered so two overlapping curves both show. Override per series with
`"marker"` (`o s ^ D v P X < > d p h * 8`, or `"none"` on a line) and
`"linestyle"` (`-`, `--`, `:`, `-.`), or set `"series_marks": false` on the
spec for colour only. `survival` keeps its censoring ticks as markers and
`scaling` and `speedup` keep their dashes for the fit and the efficiency
line, so those three vary only the other channel.

### Multi-panel

```json
{"type": "panel", "title": "Overview", "ncols": 2, "panels": [
  {"type": "bar", "categories": ["A", "B"], "series": [{"values": [3, 5]}]},
  {"type": "line", "series": [{"values": [1, 2, 4, 8]}]}
]}
```

Any chart type nests inside `panels`. Sub-panels are lettered `(a)`, `(b)`…
automatically — do not put the letter in the panel's own `title`, which is
how panel labels end up collided with their titles.

`ncols` and `aspect` both default from the panel count: the grid is squared
(capped at three columns, which is the most that fits at the 6.5-inch text
width) and the canvas is sized so each cell is about 4:3. Pinning `ncols: 4`
is allowed but leaves each cell 1.6 inches wide, which is narrower than a
labelled chart needs — it will be refused rather than drawn on top of
itself.

## How long text may be

Hard caps, checked before anything is drawn, so an over-long string is a
message rather than a figure with its labels cut off. Each was set by
growing that slot until the figure broke, then backing off. Each entry is the
key, its cap, then what happened past it:

- `title`, max **120** — never refused, never collided; it just ate the
  canvas. At 600 characters the chart was 38% of its own figure.
- `xlabel`, `ylabel`, `cbar_label`, max **80** — silently CLIPPED. An x-label
  ran off both edges from ~90 characters, a y-label from ~50, cut mid-word, at
  exit 0.
- `series[].label`, max **60** — legend entries collided at 80 and collapsed
  the layout at 100.
- `categories[]` and any other text, max **80** — under a *vertical* bar the
  limit is 40, with a pointer to `barh`; see *Legibility*.

A title is a heading; an axis label is a quantity and its unit. Detail
belongs in the caption, which has the full column width and as many lines as
it needs.

These are coarse budgets that cannot know the figure's real width — a
3.5-inch column fits about half as much — so the drawn result is measured
too, and anything that still does not fit is refused with the same kind of
message.

## It refuses rather than lying

The generator exits non-zero, writing nothing, when the figure would not
match its data or a reader would not be able to read it. These were live
defects, each of which exited 0 and produced a confident, plausible, wrong
picture:

- **Length mismatches.** Five categories against three values used to render
  three bars and silently drop two categories. Ragged series were zero-filled,
  inventing measurements nobody made.
- **NaN / Infinity / null / strings in values.** matplotlib draws NaN as
  *nothing*, so the gap reads as a measured zero.
- **Right-to-left text.** matplotlib does no bidi reordering and no Arabic
  joining, so Hebrew and Arabic draw left to right in isolated forms —
  reversed and unjoined. Every glyph exists, so the missing-glyph gate above
  sees nothing; the reader who can read the script is the first to know.
- **Glyphs the font cannot draw.** A missing glyph renders as a hollow box
  and matplotlib only warns. It is machine-dependent too: CJK looks right on
  a laptop with a CJK font and ships as boxes from the pipeline image.
- **Labels printed over each other.** Measured on the drawn figure, on the
  ORIENTED box of each label so a tilted tick is judged on its ink rather
  than on the much larger box around it. A 7x7 correlation matrix forced to
  `21:9` rendered its cells as `0.290.360.581.00`.
- **Labels running off the canvas.** A 300-character x-label was drawn with
  30% of itself visible, cut mid-word at both ends, with no warning.
- **A legend sitting on the data it explains.** The legend is opaque by
  design, so whatever is under it is gone rather than faint. A lone chart's
  legend is measured after layout and moved below the axes; a panel cell has
  nowhere to move it and is refused. A `timeline` in a two-column grid drew
  its legend over eight of its nine bars, and the `bar` cell beside it had
  its bar TOPS masked — GSM8K reading as ~40 where the spec said 55.8.
- **Keys nothing reads.** `x_label`/`y_label` instead of `xlabel`/`ylabel` is
  a natural guess; it used to be accepted in silence and the figure came back
  with no axis labels at all — failing the first item on your own checklist,
  visibly only if you look closely. Every key is now checked against what the
  render actually looked up, at every level, so a typo inside a series or a
  panel is caught too, and the message suggests the real spelling.
- **A series drawn without a name while its neighbours have one.** The
  legend names only the series that carry a `label`, so the rest are drawn
  and left unidentified — three series with two labelled shows blue, amber
  and green bars and names two colours. Nothing about the picture looks
  wrong, which is what makes it worth refusing. Naming none of them is fine:
  that is a chart with one meaning, and the y-label carries it.
- **A stated limit that crops the data.** `xlim`/`ylim` outside the values,
  `vmin`/`vmax` outside the matrix, or an explicit `levels` list narrower than
  `z`. Each one hides part of the finding while the axis or colourbar states a
  range the data does not have: `vmax: 0.3` on a matrix running 0.10..0.95
  painted 0.30 and 0.95 the identical yellow under a bar labelled
  0.100..0.300, and `levels: [2.6..3.2]` over a field of 2.3..4.6 left 70% of
  the plot area as bare page — the basin holding the optimum included, drawn
  exactly like no-data. Cropping is a legitimate wish; it just has to be a
  stated one, so widen the limit or drop it and let the axis fit.
- **Non-positive values on a log axis.** matplotlib MASKS them rather than
  complaining, so the figure comes back with fewer points than the data. Five
  points drawn trending up carried a fit annotation reading `y = -1.75x +
  53.2`, because the slope was still computed over the two at `x = 0` that the
  reader cannot see. Applies wherever `logx`/`logy` does — `line`, `scaling`,
  `scatter`, `pareto`.
- **A negative band in a stacked chart.** Bands and segments are drawn end to
  end, so a negative one folds back over the one beneath it and every height
  stops matching its value: 10 / -8 / 5 drew as three bands of 10 / 8 / 5,
  with a top edge of 10 where the total is 7. Use `line` with one line per
  part for signed quantities. Same for stacked `bar` and `stacked_pct`.
- **Tied scores in a `bump` chart.** It has one row per rank, so a tie can
  only be broken by the order the series happen to appear in — two models
  level at 80.0 drew as a permanent one-rank gap, and moving them past each
  other in the spec, numbers unchanged, showed a crossing that is not in the
  data. Crossings are what this chart type is read for. Use `line`, or
  `slope` for two periods, which draw the scores themselves.
- **Two series a reader cannot tell apart.** The palette holds eight colours
  and wraps; the dash pattern is a second channel and multiplies that to 32
  for line charts drawn with `"series_marks": false`, and the default marker
  and dash carry lines and points to 64, but a solid shape has no dash. A twelve-series `bar`
  shipped four PAIRS of identical swatches and a fifty-series `line` wrapped
  both channels at series 32. Measured on the drawn legend, so it holds for
  bars, lines and markers alike — and `bubble`'s size key, whose entries
  share a colour on purpose, is judged on size as well and passes.

Errors name the offending key and index (`series[1].values has 2 entries but
5 were expected`), so a bad spec is one edit from correct. Nothing partial is
ever written — a half-file would pass the downstream existence check.

## Legibility

- **Non-Latin scripts.** The default font covers Latin, Greek and Cyrillic —
  all three verified, not assumed. Hebrew and Arabic are refused even though
  the glyphs are there: matplotlib does no bidi reordering and no Arabic
  joining, so it draws the characters left to right in isolated forms and the
  label comes out reversed and unjoined, with every glyph present and nothing
  else noticing. Transliterate, or write the label in the paper's own script.
  For any other script set
  `font_family` (e.g. `"Noto Sans CJK JP"`), which goes ahead of the default
  families. Without it the figure is refused rather than shipped full of
  boxes.

  **`font_family` only helps where that font is installed, and the pipeline
  image has none.** It ships 23 families, not one of which covers CJK, Indic
  or Thai — so inside the image the escape hatch resolves to nothing and the
  figure is refused either way. The refusal now names the FONT rather than
  the script: a name that does not resolve is caught before anything is
  drawn, with the closest installed families listed, because matplotlib
  otherwise falls back in silence and the glyph gate then blames the text.
  Label it in Latin script, or add the font to
  `Dockerfile.pipeline` (Noto Sans CJK is ~20 MB). On a developer machine
  with the font present it works: verified rendering a Japanese title and
  Japanese category labels with no missing glyph.
- **Dense categories.** Labels wrap when long, tilt at 30° when that isn't
  enough, and stand up at 90° when even that collides — where neighbours
  cannot touch however long they get. Which of the three applies is decided
  by MEASURING the drawn labels against the axes after layout, so a panel
  cell gets the treatment its own width needs rather than the one the whole
  figure's width would suggest. Names past ~40 characters do not fit under a
  vertical bar at all and are refused with a pointer to `barh`, which puts
  the label on the y-axis where the full width is available.
- **Column-width figures.** `width_in: 3.5` works for the ordinary types —
  bar, barh, line, scatter, box, hist, ecdf, heatmap — provided the spec is
  written for that size: about four categories, two or three series, and a
  title under ~45 characters. These of the catalogue's own examples are
  refused at 3.5 inches, because each is written for the full text width —
  the list is pinned by a test that measures it, so it cannot go stale:

  > `bland_altman`, `cd_diagram`, `clustermap`, `corr`, `dumbbell`,
  > `funnel`, `panel`, `radar`, `sankey`, `seqheat`, `slope`, `speedup`,
  > `survival`, `timeline`, `treemap`, `upset`, `volcano`

  A leaner spec fits for every one of them — measured, including the
  label-dense ones (`corr`, `upset`, `sankey`, `treemap`, `parallel`,
  `radar`, `cd_diagram`), which only refuse above a lower ceiling than the
  ordinary types. Three one-letter categories draw at 3.5 inches; `upset`
  is the tightest, taking two sets before its own "Intersection size" axis
  label runs off the edge. What the list above says is that the SHIPPED
  EXAMPLES do not fit, because each is written for the full text width.
  Every refusal names what is in the way, and `upset` and `cd_diagram`
  quantify it ("the method names need 4.2 inches of margin") rather than
  shipping something unreadable.
- **Many series.** Past eight the palette wraps, so the marker and dash
  every line and point series carries are what is left — otherwise series 1
  and 9 were the same colour. Past six,
  the legend moves below the axes. Inside, it
  covered the data at twelve series and hid a tick label; outside, layout
  reserves real space for it.
- **Long titles** are measured after layout and wrapped. On a chart whose
  axes is a narrow strip (a `barh` with long names) the title is promoted to
  a figure heading, since an axes title would centre on the strip and run
  off the page.
- **`$` is safe.** A matched pair used to be read as mathtext, so
  "Cost $5 to $9" rendered as "Cost 5to9". All user text is now escaped, so
  dollars print verbatim. The trade: mathtext is unavailable — write
  superscripts in Unicode (`R²`, `10⁻³`), which the fits already do.

## What the house style already handles

Do not re-solve these; they are set globally in `chart_style.py`.

- **Colourblind-safe palette** (seaborn's `colorblind` set). Never override
  it with a red/green pair. The separations are measured, not assumed: the
  closest pair is ΔE*ab 14.0 under protanopia and 10.3 under deuteranopia,
  against a just-noticeable difference of ~1. **Greyscale print separates
  the first three colours and no more** — past that the lightnesses
  cluster, and violet against grey is ΔL* 0.3, the same shade in print.
  That is why line and point series also get their own marker and dash by
  default (see *Spec shape*): they carry the series through B&W print, a
  colour-blind reader and two lines drawn on top of each other.
- **The caption's typeface at the caption's size.** CMU Serif (Computer
  Modern, the paper template's font) with mathtext in `cm`, or DejaVu Sans
  where the paper's figure style is sans-serif, 11 pt on a
  figure 6.5 in wide, which the paper prints at 100% beside its 10.95 pt
  captions. Keep `width_in` at the default for a full-width figure: a wider
  one is shrunk on the page and its text with it. **No label may print under
  7 pt** once the page has scaled the figure to `\linewidth` (6.5 in): the
  legibility gate refuses it and names the label. A wider canvas is never the
  way to make room: shorten labels, use fewer panels or categories, or turn
  the chart on its side.
- **No chartjunk** — no 3D bars or surfaces, gradients, shadows, coloured plot
  background; faint horizontal grid behind the data only.
- **3-D scatter for geometry only.** When a claim is about representation or
  embedding geometry (which cluster lies closer to which, whether a point sits
  at the edge of a set) and the third principal component explains a share
  comparable to the second, one 2-D projection hides an axis the claim
  depends on. Then, and only then, hand-write a 3-D scatter
  (`mpl_toolkits.mplot3d`, house style applied) of PC1-PC3: in a static figure
  as two or three fixed-angle views side by side, each axis labelled with its
  explained variance; on an interactive page as one rotatable plot. Look at
  the rendered views before writing a "closer to" sentence, and state the
  distances in full dimension beside them.
- **Constrained layout**, so an axis label can never be clipped off the
  canvas. This was the single most common defect across every library
  surveyed, including in otherwise flawless output. Layout alone does not
  cover TITLES — it reflows axes but cannot wrap a line — so titles wider
  than their axes are measured after layout and wrapped.
- **TrueType (Type 42) fonts, never Type 3.** matplotlib emits Type 3 by
  default and **IEEE and ACM submission systems reject PDFs containing
  it**, so every default matplotlib figure is non-compliant.
- **Legend headroom** — the y-range is widened before an inside legend is
  placed, because `loc="best"` lands on the data when nothing is free. Where
  headroom cannot help — a horizontal chart, whose free space is on the
  x-axis, or a plot area that is full by construction — the placed legend is
  MEASURED against the drawn bars and moved below the axes if it covers any.
- **Very dense point clouds are drawn as a bitmap inside the vector file.**
  A scatter writes every marker as its own path — 360,000 points is a 5.7 MB
  PDF, and six of those do not fit a venue's upload limit. Past ~25,000
  points in one series the cloud alone is rasterized; the axes, ticks,
  labels and legend stay vector, so the text is still selectable and sharp
  at any zoom. Below that threshold the bitmap would be the *larger* of the
  two, so nothing changes.
- **Cell annotations are plain black or white, at tick size.** A heatmap's
  numbers take pure black or pure white, whichever contrasts better with the
  colour the map actually painted, and that alone clears 4.5:1 on any colour
  (4.58:1 at worst), so no outline is needed. They are set at the tick-label
  size and shrink only as far as small cells require, never below 7.5 pt.
- **Sub-decade log axes keep their tick labels.** A log axis spanning less
  than one decade — a loss curve from 2.90 to 2.05, say — contains no power
  of ten. matplotlib ticks only at powers of ten, so it places 10⁰ and 10¹,
  *both outside the view*, and the visible axis carries no label at all.
  Silently. Handled.

## Verify what you generated

Read the PNG back and look at it. The generator prevents the structural
defects above, but it cannot know that your data was wrong. Check:

- every number in the figure matches the number you meant to plot;
- axis labels state units;
- the caption describes what is actually drawn;
- the chart type still says what you meant once you can see it.

Two things that used to be on this list are now refused instead, so a figure
you can read back cannot have them: overlapping category labels, and a
series drawn without a name while its neighbours have one.

If a figure is crowded, widen `aspect` (`"21:9"`) or split it into a
`panel` — do not shrink the font.

## Limits

- **Hand-drawn architecture diagrams** (a pipeline, a block diagram, a
  flowchart with prose in the boxes) are out of scope: they have no
  underlying numbers and a layout engine has nothing to compute from. Those
  go to `aii-concept-fig-gen`. A graph whose edges ARE data — citations,
  message counts, co-occurrence — is a `network` here, because the picture
  has to match the edge list.
- **No LaTeX-native output.** PGFPlots produces the best camera-ready
  result of anything surveyed, because the figure text is typeset by the
  paper's own engine in the paper's own font. What is missing is a second
  backend behind 60 renderers, not the toolchain: `texlive-pictures` is
  pulled in as a dependency of `texlive-latex-extra`, and a pgfplots document
  compiles at exit 0 wherever that toolchain is present. (This entry used to
  say the package was absent and would cost +81 MB. Measured in the built
  image, both halves were wrong.) **Where it is present changed on
  2026-09-07**: TeX Live left the `aii_pipeline` runtime image for
  `amgrobelnik/aii_tex`, which `aii_pipeline.bundles.ensure_tex()` fetches at
  `gen_full_paper`. Figure generation runs in the invention loop, HOURS
  before that, so a pgfplots backend here could not assume `pdflatex` is on
  PATH — it would have to await the bundle first. One more reason the missing
  piece is a backend, not a package.
- **The legibility gate reads TEXT.** It refuses a label printed over another
  label or cut off by the canvas. A label printed over the DATA is only
  handled where a renderer registers it with `place_point_label`, which five
  types do: `pareto`, `network`, `tree`, `volcano` and `bubble`. If you
  hand-write a figure, call `fit_point_labels` too.
  `bubble` registers only the names it draws OUTSIDE their disc — a name
  small enough to sit inside its own bubble is already where it belongs and
  no nudge improves it. That registration became worth doing once the
  clearance test started measuring each marker against ITS OWN radius: with
  a single radius for the axes (the largest drawn) a bubble field running
  4 px to 88 px left no candidate position measuring clean, so every name
  stayed on its first guess.
  One limit remains, and it is the candidate SET rather than the model: the
  nudger tries corners a few pixels out, which cannot clear a very large
  neighbouring disc. On a crowded bubble chart a small bubble's name can
  still touch a big one — give those names in a legend, or space the points.
- Still uncovered: geographic/choropleth (needs a basemap and boundary data,
  neither of which is in the image). Add a renderer to its family's
  `chart_renderers*.py` rather than hand-writing matplotlib at the call site
  — that is what keeps every figure in a paper looking like a set.
````

### [6] SYSTEM-USER prompt · 2026-11-20 05:06:19 UTC

```
The headline does not hold up against the claim ledger and the run record:
- `Pride and Prejudice global word frequencies` does not appear in the run record. Name each set exactly as the report and iteration records name it, so its every use can be found.
- The abstract does not give the headline's number (`Pride and Prejudice global word frequencies` (art_dZwh_EvU-PqH): +21462 against Standard Zipf power law (1 parameter), a pre-registered test that passed, for 'Global Zipf-Mandelbrot model accuracy on Pride and Prejudice' (scope: Single novel (Pride and Prejudice), 123,350 tokens, 6,425 types; MLE with bootstrap CIs (n=100); confirmed across three independent evaluation artifacts)). State it with its interval and grade.

Correct `headline_candidates` first, then the title, abstract, summary and Results so they lead with the ledger's headline, its number, interval, grade and scope. Grade it exactly as the run record grades its test, and state in the abstract the scope the record gives it. Then resubmit every field.
```

### [7] SYSTEM-USER prompt · 2026-11-20 05:12:40 UTC

```
The headline does not hold up against the claim ledger and the run record:
- The abstract does not give the headline's number (`complete novel frequency table` (art_dZwh_EvU-PqH): +21462 against Standard Zipf power law (1 parameter), a pre-registered test that passed, for 'Global Zipf-Mandelbrot model accuracy on Pride and Prejudice' (scope: Single novel (Pride and Prejudice), 123,350 tokens, 6,425 types; MLE with bootstrap CIs (n=100); confirmed across three independent evaluation artifacts)). State it with its interval and grade.

Correct `headline_candidates` first, then the title, abstract, summary and Results so they lead with the ledger's headline, its number, interval, grade and scope. Grade it exactly as the run record grades its test, and state in the abstract the scope the record gives it. Then resubmit every field.
```
