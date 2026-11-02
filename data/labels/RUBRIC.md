# Label test set rubric (task transfer check)

**Status: final. Committed before any model run on these images.**

## Purpose
A small qualitative check of whether the router transfers to a new task (reading identifiers on labels and tags) with a prompt change and no training. n = 9. No accuracy percentage is claimed from it; results are reported per image.

## Images
- Source: Pexels, under the Pexels License (free use and modification, attribution not required; photographers credited anyway). No iStock/sponsored results.
- `px_003` is cropped to one shelf so one area of the frame holds the candidates. No other edits besides resizing and JPEG conversion.
- Not used for any tuning. The jersey benchmark and its threshold are unaffected.

## Task and prompt
One image in, one identifier out. Prompt (fixed before any run):
> Read the main identifier on the label or tag in this image: a tracking number, barcode number, part number, location code, or postal code. Reply with only that identifier.

## Labeling rules
1. The target is a printed or handwritten identifier on a label, tag or bin. Names, street addresses, prices, dates and phone numbers are never targets.
2. Priority when several exist: tracking number, then barcode number, then part or location code, then postal code.
3. Alternates (`a|b`): other identifiers in the frame that are fully legible and comparably prominent, or genuinely ambiguous characters. At most four listed. Any listed answer counts as correct.
4. Truth must be readable from the pixels. One exception, disclosed: `px_007`'s tracking number was confirmed with `px_008`, which shows the same physical label.

## Scoring
Normalize both strings: uppercase, then remove spaces, `-`, `_`, `.`, `,`. Exact match after that. No letter/digit substitutions (`O` and `0` stay distinct).

## Routing
- Confidence = the **minimum** per-token probability over the answer tokens (joint probability shrinks with length, so it isn't comparable to the 1–2 digit jersey task).
- Threshold fixed in advance at **0.90**. Not tuned on these images.
- Jersey mode is unchanged: joint probability, threshold 0.85 from `config/threshold.json`.

## Difficulty
- easy: large, sharp, facing the camera (px_006, px_008, px_009)
- medium: small, angled, long or with competing identifiers (px_002, px_003, px_004, px_005)
- hard: handwritten or rotated and small (px_001, px_007)

## Credits
| File | Photographer | Source |
| --- | --- | --- |
| px_001 | Michael Orshan | [Pexels](https://www.pexels.com/photo/close-up-of-steel-beam-with-inspection-tag-36003959/) |
| px_002 | Kampus Production | [Pexels](https://www.pexels.com/photo/close-up-shot-of-a-delivery-package-7843978/) |
| px_003 | cottonbro studio | [Pexels](https://www.pexels.com/photo/labeled-shelves-in-a-warehouse-7018662/) |
| px_004 | RDNE Stock project | [Pexels](https://www.pexels.com/photo/close-up-photo-of-a-parcel-7362924/) |
| px_005 | RDNE Stock project | [Pexels](https://www.pexels.com/photo/a-deliveryman-carrying-packages-7363082/) |
| px_006 | Liza Summer | [Pexels](https://www.pexels.com/photo/woman-preparing-box-with-parcel-for-sending-6347513/) |
| px_007 | Polina Tanikilevitch | [Pexels](https://www.pexels.com/photo/a-person-writing-on-a-delivery-box-postal-details-4440889/) |
| px_008 | Polina Tanikilevitch | [Pexels](https://www.pexels.com/photo/boxes-with-paste-postal-stickers-4440874/) |
| px_009 | Liza Summer | [Pexels](https://www.pexels.com/photo/scissors-and-tape-placed-on-carton-box-with-parcel-6347514/) |
