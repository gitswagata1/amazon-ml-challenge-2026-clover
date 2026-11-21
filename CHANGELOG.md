# Changelog

## 2026-11-21 · v1.0

- 22 unit tests for the competition pipeline's core: transliteration and normalisation (French, US and Indian
  address quirks), consonant skeletons across scripts, legal-word removal, house-number keys, blocking-key
  families and feature helpers. They run on every push alongside the examples.
- README badges render as images; credits line trimmed.
- Competition scale from Amazon's results email: 32,000+ teams, 89,000+ registrations, about 103 teams disqualified.

## 2026-11-20 · One self-contained repository

- The competition code, the experiments and the 72-hour logs now live in this repository (`code/`, `experiments/`,
  `logs/`) instead of being linked as submodules. A plain `git clone` gets everything.
- Every link and command updated for the new layout; the final package keeps its original structure, so
  `reproduce.py` runs from the repo root.
- Credits simplified to Team Amizon, Martial-Gautam and gitswagata1. Personal names, Kaggle handles and machine
  details removed from all files.
- README rebuilt: banner, palette badges, table of contents, one story from problem to results, and collapsible
  reproduction notes. Decorative emojis removed across the repo.

## 2026-11-20 · Run it anywhere, prove it runs

- `notebooks/build_from_scratch.ipynb`: the 6-step build guide as a notebook, with an "Open in Colab" badge. In Colab
  it clones the repo itself. Saved with outputs so GitHub shows the results. Tested from an empty folder.
- `.github/workflows/examples.yml`: CI runs the metric, the mini pipeline (and checks its 0.9361 score), the 6-step
  solution and the notebook on every push. A status badge is in the README.
- `docs/assets/social-preview.png`: a 1280×640 social preview image for link sharing.
- README: a "My role" statement that separates this repository's work from the team's competition pipeline.

## 2026-11-20 · Problem statement

- Added the official problem statement PDF in `problem-statement/`, unchanged and credited to the organisers. Linked
  from README §1, the repo map and `docs/problem.md`.

## 2026-11-20 · New solution diagram

- README §6 and TECHNICAL §0: the auto-laid-out Mermaid architecture chart is replaced by a custom SVG
  (`docs/assets/solution-architecture.svg`) in the project palette. It has numbered stage cards, separate US/India and
  France re-read lanes, the confident-pair bypass, and a results ribbon. Text was checked to fit in Inter, Helvetica,
  Arial and DejaVu Sans, so it renders cleanly on any operating system.

## 2026-11-20 · Deck style pass

- Dark deep-teal cover with silk and orange accents; a four-colour accent bar on every slide.
- Softly raised cards, larger headline numbers, dark table headers, and the problem table's "different business" row
  marked with a cedar edge.
- Shaded area under the leaderboard score chart.
- Staged motion when a slide appears (titles, then content, then call-outs; bars grow from the left). The PDF and
  PowerPoint show the finished slides.

## 2026-11-20 · Deck palette

- Pitch deck recoloured to the Coolors palette: steel teal #588B8B, white, unbleached silk #FFD5C2, cadmium orange
  #F28F3B and cedar chest #C8553D, with a deep teal ink (#1E3535 text, #234646 dark panels) derived from steel teal.
  Orange and silk carry dark text only; white text sits on deep teal or bold on cedar, so all text keeps readable contrast.

## 2026-11-20 · Quality pass and pitch deck

**Technical accuracy**
- Blocking weight: the docs said `w = log(N / df)` with N implied to be the number of records. In the code
  (`blocking.block`), N is the total number of (record, key) occurrences in a country. Corrected in TECHNICAL §2,
  its diagram and the glossary.
- `docs/data.md` stated as fact that the extra empty predictions were "mostly French". This isn't in the source
  files, so it's now marked as our interpretation and `[not in the files]`.
- Cross-checked all 185 precise numbers in the docs against the team's logs and documentation. Every number that
  isn't quoted verbatim was traced: toy-example outputs, our own profile of `matching_results.tsv`, constants in the
  code, or differences between logged scores.

**Teaching quality**
- Glossary: added 12 terms that the docs used without defining (ambiguity rejection, hash, learning rate,
  look-alike, pool, rescue rule, softmax, stage 1 / stage 2, submodule, temperature, leaf, data generator). There
  are now 92 terms.
- Learning-track answer for the consonant skeleton rewritten step by step.

**Organisation**
- Every page now ends with ← previous · Home · next → links that follow the learning path:
  start here → problem → data → build guide → TECHNICAL → learning track → journey → lessons → code walkthrough →
  debrief script → open questions.

**Engineering checks**
- Fixed the prep self-test path in `docs/code-walkthrough.md`, which must be run from the repo root as
  `code/code/...`, and noted that it needs `polars`.
- Verified in a fresh virtual environment with only `numpy` and `scikit-learn`: all three examples run and their
  output matches the docs.
- Verified `git clone --recurse-submodules` from GitHub pulls both submodules at their pinned commits.
- All Mermaid diagrams render; all internal links and anchors resolve.

**Added**
- `presentation/`: a 21-slide pitch deck, as PDF and as PowerPoint with speaker notes.
