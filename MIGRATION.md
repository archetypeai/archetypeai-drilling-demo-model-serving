# Migration: Direct Query API → Inference API

This project is the [Volve drilling demo](https://github.com/archetypeai/archetypeai-drilling-demo) with its Omega calls moved from the Direct Query API (`POST /api/lens/query`) to the Inference API: Newton Omega 1.4 → **Newton Omega 1.5**. The UI, the wells, the reference split, the scaler and the KNN are unchanged.

All measurements below were taken on production on 2026-11-13.

## Summary

- **Endpoint and model:** the server opens one session on `omega-embedding` (Omega 1.5) with `window_size` 128 and sends **one query per window**, all nine channels together, instead of nine per-channel queries.
- **Faster:** a window takes about 0.3 s instead of 1–3 s, and building the library takes about 15 seconds.
- **New library:** Omega 1.5 embeddings aren't comparable with Omega 1.4 ones, so `data/knn-library.json` was rebuilt from the same 36 reference windows.
- **More accurate:** on the same held-out windows, accuracy across the 8 wells with both classes rises from 89% to 95%. See [Results](#results).
- **Open-source fonts:** Geist and Geist Mono replace the design system's PP Neue Montreal, whose files can't be redistributed.

## Direct Query API vs Inference API

|                     | Direct Query API                                  | Inference API                                                                                                                               |
| ------------------- | ------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------- |
| Endpoint            | `POST /api/lens/query`                            | `POST /api/inference/sessions` once, then `POST /api/inference/sessions/{id}/query`                                                         |
| Model               | `OmegaEncoder::omega_embeddings_1_4`              | `omega-embedding` (Omega 1.5)                                                                                                               |
| Window length       | 16–1024 per request; shorter is padded and masked | Set once by the session: `values.window_size` 128                                                                                           |
| Requests per window | One per channel: 9                                | 1                                                                                                                                           |
| Answer              | Flat `[768]` per channel request                  | `[1, 9 × 768]`: one row, channels in order                                                                                                  |
| `normalize_input`   | `false`, with the shared scaler                   | Not sent (it has no effect); the shared scaler stays                                                                                        |
| Session lifecycle   | None                                              | Opened on first use, reopened on `409 session_not_running`, cancelled when the server or a script stops (16 open sessions per organization) |

## What changed in the code

| File                           | Change                                                                                                                                                                                                                                                    |
| ------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `src/lib/inference-client.js`  | New, the same client as the [SWaT demo](https://github.com/archetypeai/archetypeai-swat-demo-inference-api)'s. `OmegaSession` opens the session, sends one-window queries through a bounded pool, reopens on `session_not_running`, retries, and cancels. |
| `src/lib/server/newton.js`     | Omega through one server-wide session (kept on `globalThis` so hot reload doesn't open more). The per-channel fan-out, `normalize_input`, the `/v0.5/query` URL and the unused `getApiKey` are gone.                                                      |
| `vite.config.js`               | The project's `.env` wins over `ATAI_*` shell variables, and the Omega session is cancelled during Vite's shutdown.                                                                                                                                       |
| `scripts/build-knn-library.js` | Embeds through `OmegaSession`, all windows in parallel through the pool; cancels the session at the end or on failure.                                                                                                                                    |
| `scripts/eval-wells.js`        | New: scores the library on every playable well, the way the README's table is measured.                                                                                                                                                                   |
| `scripts/windows.js`           | New: the scaler and window helpers, shared by the two scripts.                                                                                                                                                                                            |
| `src/routes/+page.svelte`      | The Advanced panel shows the library's real size (18 references per class, not 2,000) and search (brute force).                                                                                                                                           |
| `src/routes/layout.css`        | Geist and Geist Mono from `@fontsource-variable` instead of the design system's `fonts.css`; the font files in `static/fonts/` are gone.                                                                                                                  |
| Removed                        | The Lens and Direct Query skills in `.claude/skills/` (listed in `CLAUDE.md`), which pointed coding agents at the old APIs, and a Staging-only test script.                                                                                               |
| `LICENSE`                      | Apache License 2.0, for the code.                                                                                                                                                                                                                         |

## Results

### Per-well accuracy

`node scripts/eval-wells.js`, the same windows for both models (up to 15 unanimous-ACTC windows per class, window 128, k=3), each against its own library built from the same reference windows:

Accuracy / F1 per well:

| Well                         |     Omega 1.4 |     Omega 1.5 | Eval windows |
| ---------------------------- | ------------: | ------------: | -----------: |
| F-1                          |     83% / 80% |   100% / 100% |           30 |
| F-14                         |   100% / 100% |   100% / 100% |           30 |
| F-9                          |     93% / 93% |   100% / 100% |           30 |
| F-12                         |     93% / 93% |     97% / 97% |           30 |
| F-5 (NA-NA)                  |     90% / 91% |     93% / 94% |           30 |
| F-5 (StatoilHydro)           |     87% / 87% |     93% / 94% |           30 |
| F-7                          |     83% / 80% |     93% / 93% |           30 |
| F-15S                        |     83% / 81% |     87% / 86% |           30 |
| F-9 A †                      |     83% / 85% |     83% / 85% |           24 |
| F-15B †                      |     65% / 79% |     71% / 83% |           17 |
| F-15 †                       |     65% / 77% |     50% / 67% |           20 |
| F-15A †                      |     38% / 55% |     41% / 56% |           29 |
| **Mean, 8 both-class wells** | **89% / 88%** | **95% / 95%** |          240 |

The Omega 1.4 numbers match the Direct Query version's README exactly, so the two runs score the same windows the same way. Omega 1.5 is as good or better on all 8 both-class wells, and the biggest gains are on the wells 1.4 found hardest (F-1, F-7). Of the single-class wells, it's worse only on F-15, which has 5 not-drilling windows in total.

† Single-class wells (>99% drilling): only 2–14 not-drilling windows exist in the whole well, so a balanced sample over-weights them and the numbers are noisy.

### Speed

|                                   | Omega 1.4, Direct Query                                                    | Omega 1.5, Inference API |
| --------------------------------- | -------------------------------------------------------------------------- | ------------------------ |
| One window                        | 1–3 s with 9 calls in parallel (from its README); 8.5 s one call at a time | 0.3 s (median of 15)     |
| Opening the session               | n/a                                                                        | 0.7 s, once              |
| Library (36 windows)              | not measured                                                               | 12 s                     |
| Per-well evaluation (330 windows) | 47 minutes, one call at a time                                             | 103 s                    |

## Reproducing

```bash
node scripts/build-knn-library.js   # Omega 1.5 library, about 15 s
node scripts/eval-wells.js          # the per-well table, about 2 minutes
```

The Omega 1.4 numbers come from the [Direct Query version](https://github.com/archetypeai/archetypeai-drilling-demo) and its committed library, scored with the same windows. On the night of the measurement, Direct Query calls sent several at a time often hung until they timed out, so the 1.4 evaluation ran one call at a time; one call took about 1 s. The Inference API needs one query per window, so it wasn't affected.
