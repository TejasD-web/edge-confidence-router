# Locomotive number test set rubric (asset-ID transfer test, with a format hint)

**Status: final. Committed together with `labels.csv` and `benchmark/number_test.py` BEFORE any model run on
these images. The prompt, format rule and threshold were fixed before any image was collected.**

## Why this set
A field robot in a rail yard or plant has to identify the unit in front of it from the number painted on it.
Unlike jersey numbers, a real site knows what its IDs look like. This test asks: if the robot is told the ID
format, does the same training-free router read asset numbers on-device, and does a format check catch bad
reads offline? Locomotive running numbers are the proxy: large painted digits in a fixed format.

## Scope: one numbering scheme
Five-digit British Railways running numbers, digits only: TOPS numbers (shown as two digits, a space, three
digits, e.g. a class 66 is `66 xxx`) or the earlier BR series for steam and diesel (e.g. `80151`, `13059`).
One format, so it can be stated exactly.

## Fixed before any image (do not change after collecting)
- **Prompt** (edge and Muse):
  > This is a British locomotive. Its number has exactly five digits, usually painted as two digits, a space, then three digits. Read the locomotive's number from the image. Reply with only the five digits, no spaces and nothing else.
- **Confidence**: joint digit probability, the jersey router's signal (`benchmark/logprob_probe.confidence`).
- **Threshold**: read from `config/threshold.json` (0.80, chosen on the jersey dev set and frozen). Not re-selected.
- **Format rule**: the normalized answer must be exactly five digits (`^[0-9]{5}$`).
- **Routing rule (headline)**: keep on-device only if the format rule passes AND confidence >= threshold;
  otherwise queue / send to Muse Spark.
- **Also reported (pre-registered comparison)**: what confidence alone would have kept (no format rule), to show
  what the format check adds.

## Images
- Source: Wikimedia Commons only, license CC0, public domain, CC BY or CC BY-SA (no NC/ND). Record file page URL,
  author and license per image in `labels.csv`.
- Provenance: the uploader is the author (or the author is clearly credited). Skip anything else.
- No upscaled, AI-enhanced, colorized or "restored" files. Resize, JPEG and plain crops only (each crop noted in `labels.csv`), shorter side >= 400 px.
- 25 images: 7 easy / 8 medium / 10 hard (hard-weighted like the jersey set). Stop at 25.
  - **easy**: large cab-side number, sharp, roughly side-on, nothing in the way
  - **medium**: smaller or angled number, a front/end number, or a second (non-target) number in frame
  - **hard**: weathered or grimy paint, shadow or low light, motion blur, partly hidden, or a competing five-digit
    number (another locomotive) in frame
- Exclude: any recognizable face; multiple-unit trains (their unit numbers are six digits); pre-TOPS numbers with
  letter prefixes (e.g. `D1015`); images where the only readable number is a carriage or wagon number.
- At most 2 images of the same locomotive.

## Changes made during collection (before any model run)
- Image count and split set to 25 (7 / 8 / 10) before the last images were collected.
- Scope wording widened to include the earlier BR number series (one image, `lo_007`, number 13059).
- Plain crops allowed (one image, `lo_016`, cropped to exclude the front number so only the side number is read).
The prompt, format rule, confidence signal and threshold were not changed.

## Labeling
1. Target = the locomotive's running number, five digits, as painted (spaces removed).
2. If two different locomotives' numbers are equally prominent and both fully readable, list both as `a|b`.
3. Build plates, depot codes, class numbers on their own, nameplates and wagon numbers are never targets.
4. Truth is read from the pixels, not from knowing a famous engine. If you wouldn't bet on it, skip the image.

## Scoring
Normalize (uppercase, strip ` -_.,`), exact match, any listed alternate counts. `O` != `0`.

## Report
Per image: edge answer, joint confidence, format pass/fail, tier, final, correct. Headline: edge alone,
on-device count and confident errors under the routing rule; the same under confidence alone; after-sync
accuracy. n = 25, stated as small. One run.
