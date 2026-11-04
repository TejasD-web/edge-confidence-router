#!/bin/bash
# One-off: give results/ files readable names (git mv keeps history; see `git log --follow`).
# Run from the repo root on a clean tree. Contents of every file are unchanged.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
declare -A M=(
  # Perception LM 1B (submitted project, Nov 3)
  [threshold_probe_20261103_051705.json]=plm_jersey_dev_threshold.json
  [threshold_sweep_20261103_051705.png]=plm_jersey_dev_threshold.png
  [route_20261103_052030.json]=plm_jersey_heldout_online.json
  [route_20261103_052242.json]=plm_jersey_heldout_offline.json
  [sync_heldout_20261103.jsonl]=plm_jersey_heldout_sync.jsonl
  [numbers_20261103_105432.json]=plm_locomotives.json
  [labels_20261103_052752.json]=plm_labels_preregistered.json
  [labels_explore_4tiles_20261103_053613.json]=plm_labels_exploratory_4tiles.json
  [labels_explore_8tiles_20261103_053740.json]=plm_labels_exploratory_8tiles.json
  # Earlier work, Qwen2.5-VL-3B edge or cloud-only (Nov 1-2)
  [threshold_probe_20261101_022045.json]=qwen_jersey_dev_threshold.json
  [threshold_sweep_20261101_022045.png]=qwen_jersey_dev_threshold.png
  [route_20261101_022348.json]=qwen_jersey_heldout_online.json
  [route_20261101_022746.json]=qwen_jersey_heldout_offline.json
  [sync_20261101_heldout_own.jsonl]=qwen_jersey_heldout_sync.jsonl
  [labels_20261102_004720.json]=qwen_labels.json
  [labels_explore_256tok_20261102_030244.json]=qwen_labels_exploratory_256tok.json
  [labels_explore_1024tok_20261102_005241.json]=qwen_labels_exploratory_1024tok.json
  [labels_sam_explore_20261102_032703.json]=qwen_labels_sam_crop_run1.json
  [labels_sam_explore_20261102_033219.json]=qwen_labels_sam_crop_run2.json
  [describe_explore_20261102_011628.json]=qwen_describe_exploratory.json
  [describe_caption_scores.csv]=qwen_describe_caption_scores.csv
  [cloud_sam_jerseys_20261102_043025.json]=cloud_sam_jerseys.json
  [cloud_sam_labels_20261102_043145.json]=cloud_sam_labels.json
)
for old in "${!M[@]}"; do
  if [ -e "results/$old" ]; then git mv -f "results/$old" "results/${M[$old]}"; fi
done
# Point links in the markdown files at the new names
for md in $(git ls-files '*.md'); do
  for old in "${!M[@]}"; do sed -i "s#$old#${M[$old]}#g" "$md"; done
done
git status --short
echo "Old names left in tracked files (should be none; config/threshold.json is left as frozen):"
git grep -n -E '(threshold_probe|threshold_sweep|route|labels|numbers|sync)_?(heldout_)?2026110' -- . ':!scripts/rename_results.sh' ':!config/threshold.json' || echo "  none"
