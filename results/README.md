# Results index

Every run is committed here, including runs from before the switch to Meta's Perception LM. `plm_` files are the submitted project; `qwen_` and `cloud_` files are earlier work. Perception LM files also record `edge_model: facebook/Perception-LM-1B@2b1a854` inside the JSON.

The files were renamed from timestamped names (for example `route_20261103_052030.json`) on Nov 3 for readability; contents are unchanged, and `git log --follow <file>` shows each one's history and original name. `config/threshold.json` is left exactly as it was frozen, so it still refers to the dev threshold run by its original name, `threshold_probe_20261103_051705.json` (now `plm_jersey_dev_threshold.json`).

## Submitted project: Perception LM 1B on the edge (Nov 3)

| File | What it is |
| --- | --- |
| `plm_jersey_dev_threshold.json`, `.png` | Dev-set threshold selection (chose 0.80) |
| `plm_jersey_heldout_online.json` | Held-out jerseys, connection up: 24/25, 11 on-device, 14 Muse calls |
| `plm_jersey_heldout_offline.json` | Held-out jerseys, no connection: 22/25, 14 queued |
| `plm_jersey_heldout_sync.jsonl` | Sync of the offline queue once the connection was back: 23/25 |
| `plm_locomotives.json` | Locomotive numbers, pre-registered, one run |
| `plm_labels_preregistered.json` | Labels, pre-registered prompt |
| `plm_labels_exploratory_4tiles.json`, `plm_labels_exploratory_8tiles.json` | Labels, exploratory second prompt (reported separately) |
| `confidence_example.json` | Per-token probabilities behind the README's lo_002 example |

## Earlier work, before the switch to Perception LM (Nov 1–2)

Kept for the README's "Earlier comparison" section and for the record. Where an edge model was used, it was Qwen2.5-VL-3B-Instruct. Not part of the submitted project.

| File | What it is |
| --- | --- |
| `qwen_jersey_dev_threshold.json`, `.png` | Dev threshold selection |
| `qwen_jersey_heldout_online.json`, `qwen_jersey_heldout_offline.json`, `qwen_jersey_heldout_sync.jsonl` | Held-out jerseys: connection up, no connection, sync |
| `qwen_labels.json`, `qwen_labels_exploratory_*tok.json`, `qwen_labels_sam_crop_run1/2.json` | Label experiments |
| `qwen_describe_exploratory.json`, `qwen_describe_caption_scores.csv` | Scene-description experiment |
| `cloud_sam_jerseys.json`, `cloud_sam_labels.json` | Cloud SAM 3.1 crop-then-read experiment (no edge model) |
