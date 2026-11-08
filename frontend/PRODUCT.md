# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Stack

delegated: static HTML/CSS/vanilla JS, no build step. All match data flows through one data module (`assets/js/data.js`, shaped like a future API response) so a backend the user adds later can replace it without touching page code. Upload and processing are front-end simulations until that backend exists.

## Users

Coaches and players equally. Coaches review a player's or an opponent's match to prepare feedback and training. Club and competitive players review their own matches. Both use the same report.

## Product Purpose

Goodminton analyses an uploaded badminton match recording and organises it into rallies. Users can then review rally patterns, shot choices and the turning points of the match. Success means a coach or player quickly answers four questions: what happened, where the match turned, what patterns appeared, and which moments to rewatch.

## Positioning

The court is the interface. Analysis is shown on the court itself (rally paths, shot maps, score runs) and stays linked to video timestamps. It is not a generic AI stats dashboard. Measured facts and coaching interpretations are always kept separate.

## Operating Context

- Workflow: upload a match recording → processing (tracking play, organising rallies) → analysis in progress → interactive report → match library.
- The report keeps the video player, rally timeline and court map connected. Selecting a rally seeks the video and highlights its path. Filters cover game and player.
- Review happens on desktop and on phones, often courtside. On mobile, video, score and rally timeline come first.

## Capabilities and Constraints

- First version, four experiences: the homepage with an explorable sample analysis, match upload, the analysis-in-progress state, and the interactive report linked to video timestamps. A match library list follows.
- Supported video formats, size limits, processing time and recording limitations are **undecided**. The UI shows them as clearly marked placeholders and must not claim detection the system cannot yet do reliably.
- The backend is not built yet. Upload and processing states are simulated in the front end.

## Brand Commitments

- Name: **Goodminton**.
- Visual ambition (updated 2026-11-07): the user wants the site as immersive, interactive and distinctive as the HackUTD 2026 and HackUTA 2026 websites. That means an illustrated, themed world that owns the page, with custom lettering and motion; "no ordinary, only special, unique, attractive designs". This replaces the original paper/ink/one-accent constraint, which now lives only in the first build at the project root (kept unless the user asks to remove it).
- **Strung** (`strung/`) is the chosen world (decided 2026-11-07). The site is a strung racket head hanging in a night sky. Palette (changed 2026-11-07 at the user's request) matches the HackUTD 2026 site exactly: black / #0e011f ground, #f4f2ff strings and text, electric violet #7828ff for actions, magenta #ff2ee6 for neon lettering and plates, lavender #dbc4ff for accent words, and cyan #22e0ff reserved for the selection. The racket → court morph is time-driven (fixed duration, stage pinned while it plays), never scroll-scrubbed. It shares the product engine in `assets/js/`. A second world, Night Hall, was built for comparison and removed at the user's request.
- Still binding: no fake AI-dashboard clichés. Measured facts and suggestions stay visibly separate, and illustrative data stays labelled.
- Homepage headline: "Find the moments that changed the match."

## Evidence on Hand

- The user will supply real match footage and analysis data later. Until then the sample report uses a slot structure and must be labelled **illustrative**: fictional players, no customer claims.
- There are no testimonials, customers, benchmarks or pricing. Do not fabricate them.

## Product Principles

1. Facts and suggestions never share a voice. A count is a measurement; coaching guidance is phrased as a suggestion.
2. Every chart points back to the video. If a moment can't be rewatched, it doesn't belong in the report.
3. Be honest about process. Status, limits and confidence are stated plainly; nothing is implied.
4. Show the real product, not marketing about it.

## Accessibility & Inclusion

Deliberate keyboard focus, loading, error and empty states. Respect reduced-motion settings; motion only explains an action, such as tracing a selected rally.
