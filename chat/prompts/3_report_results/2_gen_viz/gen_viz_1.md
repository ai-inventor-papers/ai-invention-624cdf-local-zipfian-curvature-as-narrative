# gen_viz_1 — report_results

> Phase: `gen_paper_repo` · `gen_viz`
> Run: `run_fODCbnzS2ZPQ` — Zipf's Law Falls Short: The Zipf-Mandelbrot Model Accurately Captures Word Frequencies in Pride and Prejudice
>
> Full, verbatim record of every prompt the AI Inventor pipeline gave this agent — system-user, human-user and skill-input — in the order they landed. Nothing truncated.

## Task: `gen_viz_1` (sdk_openhands_agent)

### [1] SYSTEM-USER prompt · 2026-11-20 05:22:50 UTC

````
<task>
Generate a publication-quality CONCEPT figure for a top-tier venue research paper that follows the provided specification.

Use the aii-concept-fig-gen skill to generate the figure in the aspect ratio from the spec. ALWAYS pass `--model sunburst --style concept` to EVERY concept_fig_gen.py call (this run uses the **sunburst** image tier).

ONLY THIS ROUTE. A drawn figure — a flowchart, an overview, a pipeline or method diagram, the flagship — is made by concept_fig_gen.py on the `sunburst` tier and nothing else. Never pass another `--model`, never call another image model or image API, and never draw it yourself: no matplotlib, graphviz, TikZ, SVG, HTML or PIL drawing. PIL is for the margin crop below and nothing more. `--style concept` appends the paper's concept style — lettered panels with bold titles, the field's own objects, meaningful saturated colour, large labels, the paper's typeface, its palette, a canvas filled edge to edge — so the tool carries it on every call instead of you having to remember it in every prompt.

CONCEPT RULES. The figure is printed 6.5 inches wide on a single-column page, and it looks like the Figure 1 of a top recent paper in this paper's field: concrete, rich and readable, never a sparse cartoon.
- The canvas is 16:9, and a concept figure is never wider than 2:1.
- 2 to 4 lettered panels, (a), (b), (c), read LEFT TO RIGHT in the order the specification gives, each with a short bold title that says what the panel shows.
- About 80 to 120 words in the whole figure, most of them the paper's own text: a real prompt or input, the decisive tokens, a one-line code snippet, the real method and model names. Labels are short role names or sentence fragments of at most about eight words; no paragraphs.
- Real numbers: at least one (in Figure 1, the headline number) and the few others the specification gives — a k, a temperature, a count, a share — each exactly as written there. Short plain notation (pass@k, T = 0.8) and one-line code or token snippets are welcome; display equations and derivations stay in the paper's text.
- Draw the field's own objects: token strips, code lines, layer stacks, probability bars, sampling trees, chat bubbles, a small result thumbnail. Never a generic icon (a gear, a brain, a light bulb, a thermometer, a question mark) and never a SLOGAN PANEL, a title over an icon, a lone word or an empty box.
- LARGE labels, the panel titles largest, every word readable at 6.5 inches wide. Never ask for small text, fine print, thin lines or generous white space: the figure fills its canvas.
- NO OVERALL FIGURE TITLE in the image: the caption is the figure's title. The `Title` in the specification names the figure for the paper and is never drawn; only the panels carry titles.
- COLOUR CARRIES MEANING: 2 to 4 saturated colours, each with one meaning that stays the same in every panel and that the caption can name; grey for what is shared or de-emphasised; a white background, thin grey rounded panel frames, and pastel only as a light tint behind text. ONE COLOUR MAP FOR THE WHOLE PAPER, the same one its data figures use, so a colour means the same thing in every figure. Where the figure contrasts the paper's own method with a baseline, colour each element by its ROLE:
  - `#0173B2` blue: the paper's own method, model or condition
  - `#949494` grey: the main baseline, control or reference
  - `#DE8F05` amber: the first further comparator
  - `#029E73` green: the second
  - `#CC78BC` violet: the third
  Where it has no such contrast, colour by STAGE or CONCEPT instead, one of the same colours per stage, outcome or branch, keeping any colour a data figure of the paper already gives the same thing.

DOMAIN EXEMPLARS — before your first generation, look at how this paper's field draws its Figure 1s (about 6 to 10 tool calls and a few minutes, no more):
1. Take the arXiv ids after "Style references:" in the specification. With fewer than two, run ONE web search for "<2 to 4 keywords of this paper's field> arXiv 2025 ICLR OR NeurIPS OR ICML" and add the ids of the most-cited recent papers that name a method or phenomenon.
2. Fetch their Figure 1 pages in one call, which returns on its own at its deadline: `$PY "$SKILL_DIR/scripts/fig1_exemplars.py" <id> <id> <id> --out exemplars`
3. Read each page it rendered. Keep the two or three whose Figure 1 is an overview (a pipeline, a worked example, a concept beside a result); for a pure results chart, fetch the next candidate rather than lowering the bar. If none can be fetched, go on with these rules alone.
4. Write `fig1_style_note.md`: 3 to 5 lines per kept exemplar (its layout, roughly how many words, the real objects and numbers it shows, what its colours mean), then which one this figure adapts and how, then the claim this figure must show and its draft caption.
5. Write your image prompt as that adaptation, drawn with this paper's own objects and numbers. Never copy an exemplar's content. You may add domain detail the specification, caption and summary support (a real token, a code line, a model name), never a result: no number they do not state.

Be detailed about LAYOUT and exact about TEXT in your image generation prompt: say where each panel, object and arrow sits, its shape and its colour, and give the exact text, tokens and numbers each one shows, copied from the specification. Then END the prompt with a separate sentence listing the words that must appear, verbatim — "The boxes read Tokenizer, Transformer, Classifier." Naming them inside the layout sentence instead is what turns Encoder into `Enc:der`; every measured run that stated them as their own closing sentence spelled all of them correctly, and word length made no difference either way.

Create a subfolder `fig1_all/` in your workspace for ALL attempts.

ONE PHASE — explore layouts (HARD LIMIT: 7 attempts). `sunburst` ignores `--image-size` and always renders at its own maximum size, so an edit-mode "upscale" of a chosen draft returns the same picture for the price of a new one. There is NO upscale phase on this tier: spend those generations on layouts instead.
- Generate at `--model sunburst --image-size 1K`. Save attempts as `fig1_all/fig1_v0_it1.jpg`, `fig1_all/fig1_v0_it2.jpg`, … up to `_it7.jpg`.
- Keep the specification's reading order, its panels left to right, in every attempt. Make your first two or three attempts genuinely DIFFERENT LAYOUTS within it — the panels' widths, where the worked example sits, how the panels are joined — not rewordings of one layout. Then refine the clearest layout with the attempts left.
- After EACH attempt, read the image back and verify it against the checklist below. Copy every attempt that passes it to the deliverable, crop it and run the print check (DELIVERABLE below). Stop only once one passes the checklist AND the print check, its style verdict included: a clean-looking render the style check fails is not done. Otherwise keep going until the 7 attempts are spent, and deliver the best. Every paid generation draws on this run's OpenRouter budget for Report results: $7 USD for the ENTIRE phase that writes the paper and repo, not just this figure. Every other agent and step in that phase spends from that SAME shared pot, so use fewer or cheaper attempts when you are unsure. $7 USD of it was held back for concept figures, and the pot is enforced by AI Inventor: a generation past it is refused (HTTP 403, 'AI Inventor per-run OpenRouter budget reached'), and retrying will not help.
- The chosen attempt is the final figure.

DELIVERABLE:
- Copy the chosen final image to your workspace root as: fig1_v0.jpg — the chosen attempt.
- Crop its blank margins in code before you deliver it: with PIL, take the bounding box of
  pixels darker than about 245, add a 20-pixel margin, and save it over fig1_v0.jpg as
  JPEG quality 95. Read the cropped file back. Never leave cropping to LaTeX options: the paper,
  the repo and the website must all show this one file.
- RUN THE PRINT CHECK on the cropped file, every time you crop, with your style note and the
  exemplar pages you kept:
  `$PY "$SKILL_DIR/scripts/concept_print_check.py" fig1_v0.jpg --aspect 16:9 --style-note fig1_style_note.md --reference exemplars/<id>/fig1_page.png`
  (`$PY` and `$SKILL_DIR` as the skill's Setup section sets them; one `--reference` per kept
  page). It re-checks the aspect after the crop — a crop more than 15% wider than 16:9
  means the drawing did not fill the canvas height — downscales the figure to 650 px wide, the
  size the page prints it at, and has a vision model list every label it cannot read there. Then
  it judges MEANING AND STYLE against the exemplars: does each panel show its point with the
  field's own objects and real numbers, and is the figure about as rich as they are? Exit 1 is a
  FAIL: read its reasons and regenerate (it counts as an attempt). An unreadable label or a strip
  wants shorter labels and larger text, filling the full height of the canvas; a style failure
  wants what its problems name. Exit 2 means it could not reach its vision model (or would bill on
  a $0 run): the aspect was still checked, so deliver. Read back the
  `fig1_v0_print650.png` it writes: that is what a reader sees.
- Leave `fig1_v0.print_check.json` beside the figure. The step reads it, and it vouches
  only for the exact file it checked. A figure delivered without a passing check, or checked
  without `--style-note`, gets ONE retry, and a figure still wider than 2.6:1 after its crop is
  refused outright.
- The file `fig1_v0.jpg` is the deliverable — everything in `fig1_all/` is reference only.

Verification checklist (apply after EVERY generation). Check for:
- MISSPELLED labels. Read every word in the image letter by letter against the word you asked for. This is the most common defect by a wide margin — `erooder` for Encoder, `routter` for Router, `conveged?` for converged? — and it is the one that survives a glance, because the shape of the word is right
- Invented text you never asked for. A prompt ending "no text of any kind" came back lettered with `Kat q` and fake axis ticks, so absence has to be checked too, not assumed
- A box, arrow or panel that is duplicated, missing, or pointing nowhere, even when every word in the image is spelled correctly
- Text too small to read at 6.5 inches wide, or a label over about eight words
- An overall title or headline drawn above the panels: the figure carries none, only its panel titles
- A SPARSE CARTOON: a slogan panel (a title over a generic icon, a lone word or an empty box), no real number, no worked example, or a figure far sparser than the exemplars you kept
- A panel whose drawing does not show the claim its title makes, or panels out of the left-to-right order the specification gives
- A display equation or derivation, or a code snippet longer than a line or two
- Empty bands or wide margins the drawing does not fill, or a strip much wider than 16:9
- Overlapping, touching, cut-off or blurry labels, and poor contrast
- Font: serif labels in the style of LaTeX's Computer Modern, the paper's typeface, which the style preset asks for, to match the paper's other figures. Do not spend an attempt on the typeface alone when everything else is right

If ANY issue is found — even minor — do another attempt (within the 7-attempt limit). Do NOT accept a figure with problems as the final figure.

Change the prompt only when the prompt is what was wrong — a word you never specified, an element you forgot to name. For a defect the prompt already rules out, re-run it UNCHANGED: the same prompt sent twice gave a correct three-box chain once and four boxes with one label repeated the other time. Rewriting a prompt that was already right spends one of your 5 attempts on a variable that was not the cause.
</task>

<figure_specification>
Figure ID: fig1
Title: Overview of the Zipf vs Zipf-Mandelbrot model comparison on Pride and Prejudice
Caption: The paper's core comparison: standard Zipf power law (one parameter) versus Zipf-Mandelbrot (two parameters) on Pride and Prejudice. (a) The novel's word frequency distribution on log-log axes: 123,350 tokens, 6,425 types, 2,447 hapax legomena. (b) Both models fitted via MLE: Zipf (s=0.992, R²=0.739) and Zipf-Mandelbrot (s=1.267, b=8.165, R²=0.966). (c) Residual analysis reveals Zipf systematically underestimates function words (head residual ≈ 15), while Zipf-Mandelbrot captures the head (residual ≈ 0.07). Blue marks Zipf-Mandelbrot, grey marks standard Zipf.
Image Generation Description: Three lettered panels (a), (b), (c) left to right on a white canvas with thin grey panel frames. Panel (a) titled '(a) Pride and Prejudice word frequencies': a log-log scatter plot showing rank (x-axis, 1 to 6,425) vs frequency (y-axis, 1 to 4,338) with black dots for the 6,425 word types. Inset text box: '123,350 tokens, 6,425 types, 2,447 hapax legomena'. Panel (b) titled '(b) Model fits via MLE': same log-log axes with two fitted curves — grey dashed line for Zipf (s=0.992, R²=0.739), solid blue line for Zipf-Mandelbrot (s=1.267, b=8.165, R²=0.966). Legend: 'Zipf (1 param)' grey, 'Zipf-Mandelbrot (2 params)' blue. Panel (c) titled '(c) Residuals by frequency rank bin': grouped horizontal bars for three bins — head (ranks 1–642), body (643–2570), tail (2571–6425). For each bin, two bars: grey 'Zipf' and blue 'Zipf-Mandelbrot' showing mean residual. Values: head Zipf=+15.2, ZM=+0.07; body Zipf=-1.67, ZM=+0.44; tail Zipf=-1.70, ZM=-0.37. Annotations: 'function words' arrow pointing to head bin. Colour map: blue for Zipf-Mandelbrot, grey for Zipf throughout. Style references: arXiv 2601.21678, 2501.06678.
Aspect Ratio: 16:9
Summary: Conceptual overview of the method and headline finding
</figure_specification>

<critical_requirements>
1. Draw what the specification describes, adding only domain detail the specification, caption and summary support (a real token, a code line, a model name). Invent no result: no number they do not state, and no stage the method does not have
2. The concept rules above are hard rules: 2 to 4 lettered panels read left to right, no overall figure title, about 80 to 120 words with at least one real number, the field's own objects and no slogan panels, no display equations, a canvas no wider than 2:1
3. FONT: serif labels in the style of LaTeX's Computer Modern, the paper's typeface — `--style concept` asks for it on every call
4. Publication camera-ready style, in the manner of the exemplars you kept: white background, thin grey rounded panel frames, saturated colour that means something, no 3D effects/shadows/gradients. Follow aii-concept-fig-gen skill for image generation, prompting best practices, and figure type templates
5. TEXT SIZE: LARGE labels that never overlap or touch. Give a crowded figure room by shortening labels or dropping a secondary one, never by dropping the worked example or the real numbers, never by going wider and never by asking for smaller text
6. RESOLUTION: Explore layouts at 1K in the one phase above; no upscale or edit passes on this tier. Keep the aspect ratio from the spec. The paper's Figure 1 overview stays 16:9, never wider than 2:1
7. MANDATORY VERIFICATION: After EVERY generation attempt, read the image and check spelling first, then layout and meaning, then the print check (legibility and style) on the cropped deliverable. If anything is wrong, regenerate. Do NOT stop at the first clean-looking attempt
</critical_requirements>


<comparison_completeness>
If this figure's title, caption or summary names specific checkpoints, models or
variants being COMPARED — "ours vs baseline", "the base and the abliterated model",
"across the three checkpoints" — every one of them named there MUST appear in the
rendered figure as its own bar, curve, point or panel. Before you render, list every
comparator the specification names and check each one off as you draw it. A
comparison figure that quietly drops one of its own named comparators is wrong even
when every bar it does draw is numerically correct — the missing one is invisible to
anyone who was not told to look for it, which is what makes it worse than an
obviously incomplete figure.
</comparison_completeness>


<caption_from_the_rendered_figure>
The caption in <figure_specification> is a DRAFT, written before this figure existed by a step
that never saw it. After your final render, read the final image back and write the figure's
caption into the `caption` field of your output. It replaces the draft caption everywhere this
figure appears: the paper, the report and the paper's website.
- AT MOST 60 WORDS. One sentence on what the figure shows and what it argues, then only
  what a reader cannot see for themselves: what a colour or marker means where no legend says it,
  what the error bars are, the sample size. Do not re-list the axes, panels or labels the image
  already names; the detail belongs in the paper's text. A 300-word caption that walks the reader
  through every panel is the failure this limit exists to stop.
- Name a colour, marker, panel or series only if it is in the image. Where the draft caption and
  the image disagree (colours said to encode models when they encode languages, an axis the panel
  does not plot, a grey series that was never drawn), the image wins.
- Keep what is still true of the draft: the data, the sample size, what the error bars are, and
  the takeaway. Keep it LaTeX-ready in the same form as the draft caption.
- State no number the figure and its data do not carry.
- A figure with no data behind it that could pass for data (a drawn network, a map, a snapshot)
  is called a schematic illustration in its caption.
</caption_from_the_rendered_figure>


---

Output the result as JSON to: `/Volumes/aii-s20018/work/run_fODCbnzS2ZPQ/4_gen_paper_repo/_3_gen_viz/gen_viz_1/.sdk_openhands_agent_struct_out.json`

JSON Schema:
```json
{
  "$defs": {
    "VizExpectedFiles": {
      "description": "Expected output files from viz generation.",
      "properties": {
        "image_path": {
          "description": "Path to the generated figure image file. Example: 'fig1_v0.jpg'",
          "title": "Image Path",
          "type": "string"
        }
      },
      "required": [
        "image_path"
      ],
      "title": "VizExpectedFiles",
      "type": "object"
    }
  },
  "description": "Structured output from viz figure generation agent.",
  "properties": {
    "title": {
      "description": "Figure title in plain, everyday language \u2014 short and jargon-free so a non-expert grasps it at a glance. Aim for about 4-8 words (~40 characters).",
      "maxLength": 90,
      "minLength": 12,
      "title": "Title",
      "type": "string"
    },
    "summary": {
      "description": "Brief summary of the generated figure: what it shows, style, any issues fixed",
      "maxLength": 5000,
      "minLength": 500,
      "title": "Summary",
      "type": "string"
    },
    "caption": {
      "description": "The figure's caption, written from the FINAL rendered image after you read it back, in the same LaTeX-ready form as the draft caption. AT MOST ABOUT 60 WORDS: one sentence on what the figure shows and argues, then only what a reader cannot see for themselves (an encoding no legend states, what the error bars are, the sample size). Do not re-list the axes, panels or labels the image already names. It replaces the draft caption in the paper, the report and the paper's website. Name only axes, colours, markers, panels and series that are in the image.",
      "maxLength": 480,
      "minLength": 20,
      "title": "Caption",
      "type": "string"
    },
    "out_expected_files": {
      "$ref": "#/$defs/VizExpectedFiles",
      "description": "Output file you created. Must include the generated figure image path."
    }
  },
  "required": [
    "title",
    "summary",
    "caption",
    "out_expected_files"
  ],
  "title": "VizFigureOutput",
  "type": "object"
}
```

IMPORTANT: this task is NOT complete until `/Volumes/aii-s20018/work/run_fODCbnzS2ZPQ/4_gen_paper_repo/_3_gen_viz/gen_viz_1/.sdk_openhands_agent_struct_out.json` exists and contains JSON matching the schema above.

How accurately does Zipf's law describe word frequencies in a public-domain novel such as Pride and Prejudice?
````

### [2] SKILL-INPUT — aii-concept-fig-gen · 2026-11-20 05:23:06 UTC

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

### [3] SYSTEM-USER prompt · 2026-11-20 05:33:34 UTC

```
<validation-feedback>
Attempt 1 failed validation.

Schema validation found 1 problem — fix ALL of them at once:
  - at `summary`: 'Conceptual overview of the Zipf vs Zipf-Mandelbrot model comparison on Pride and Prejudice. The figure illustrates the standard Zipf power law vs Zipf-Mandelbrot on a log-log word frequency distribution. NOTE: Figure generation failed due to a tool configuration error (invalid OpenRouter URL scheme). This figure is a placeholder.' is too short (at least 500 characters, got 331)
Every required field must be present and every field type must match the schema.

Produce `/Volumes/aii-s20018/work/run_fODCbnzS2ZPQ/4_gen_paper_repo/_3_gen_viz/gen_viz_1/.sdk_openhands_agent_struct_out.json` again so it contains corrected JSON that matches the schema. Do not invent new fields.
</validation-feedback>
```

### [4] SYSTEM-USER prompt · 2026-11-20 05:34:16 UTC

```
The following required files are missing:

- `fig1_v0.jpg`

Create these files now. The task is not complete until all required files exist.

IMPORTANT: When providing your structured output (title, summary, etc.), describe the ARTIFACT you built — NOT the file verification status. Your title and summary must describe what you created, not that you verified files.
```
