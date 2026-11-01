# Evaluation set rubric (own / openly licensed data)

**Status: final. Committed in `1f5a548` before any model run; unchanged since.**

## Task
One image in, one jersey number out. Prompt and scoring are unchanged from `benchmark/run_benchmark.py`.

## Inclusion rules
1. Source: Wikimedia Commons, with a license that allows reuse (CC0, public domain, CC BY, CC BY-SA). No NC/ND.
2. Provenance: the uploader must be the author (watermark and EXIF "Author/Artist" must match, or be explained). Otherwise skip.
3. Files that were upscaled, AI-enhanced, colorized or "restored" are excluded. Plain crops and colour/contrast correction are allowed.
4. Shorter side ≥ 400 px. Downloaded at 960 px (or full resolution if smaller) and saved as JPEG.
5. No youth, high-school or age-group games (minors). College, professional, national-team and adult amateur only.
6. Game action or in-game moments (incl. timeouts). No press events, ceremonies, posed groups or gift jerseys.
7. Max ~4–5 images per game.
8. At least one player jersey number must be readable with certainty from the pixels (not from knowing who the player is). If the labeler would not bet on it, skip.

## Labeling rules
1. **Targets are on-court player jersey numbers only.** Fans' replica jerseys, referees, ads, seats, scoreboards, score cards and shorts numbers are distractors, never labels.
2. **Target = the largest fully visible, clearly readable player jersey number.** Numbers cut off by the frame or hidden by bodies are not candidates.
3. **Alternates (`a|b`)**: another fully visible, clear player number is listed as an alternate only if its digit height is ≥ 75% of the target's. An answer matching any listed number is correct.
4. At most three numbers may be listed (target + two alternates). Images with 4+ near-equal candidates and no dominant number are excluded.
5. Upside-down or rotated numbers are labeled as printed on the jersey.

## Difficulty
- **easy**: target number large, sharp, facing the camera, unobstructed.
- **medium**: target small, angled or partly turned; or one competing number.
- **hard**: blur, low light, film grain/halftone, partial occlusion, crowding, or strong non-jersey number distractors.

Mix: 9 easy / 15 medium / 16 hard (hard-weighted on purpose; the on-device rate is a conservative lower bound).

## Split
`scripts/build_manifests.py`: stratified by difficulty, seed `20261101`, 15/40 of each difficulty to dev, the rest held-out. No hand-picking.

## Threshold
Chosen on dev only by `benchmark/select_threshold.py` (rule in its docstring), written to `config/threshold.json`, then frozen for the single held-out run.
