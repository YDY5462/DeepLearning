# Training Index

This index tracks the main saved training runs used in the thesis comparisons.

| run | final_epoch | final_rmse | best_epoch | best_rmse | input_count | video_descriptor | notes |
|---|---:|---:|---:|---:|---:|---|---|
| [run_20260407_150222](run_20260407_150222/RUN_README.md) | 170 | 47.782113 | 90 | 38.074737 | 4 | no_video_or_not_recorded | Contains epoch 25/35 files from older historical runs; treat with caution for strict comparability. |
| [noisy_video_backup_20260419_144958](noisy_video_backup_20260419_144958/RUN_README.md) | 190 | 80.572369 | 170 | 73.369644 | 5 | video_proxy_noisy_or_uncertain | Contains epoch 25/35 files from older historical runs; treat with caution for strict comparability. |
| [clean_video_run_20260419_163315](clean_video_run_20260419_163315/RUN_README.md) | 190 | 90.030728 | 170 | 79.766489 | 5 | video_proxy_clean |  |
| [baseline_no_video_run_20260419_210757](baseline_no_video_run_20260419_210757/RUN_README.md) | 250 | 73.206591 | 80 | 60.820846 | 4 | video_disabled | Current new-stack baseline (video branch off). |

## Backup Scope
- Code snapshot: current working tree files committed in this backup branch.
- Training results: metrics, predictions, plots, and logs for the four main runs above.
- Model weights (`*.h5`) remain ignored to avoid very large git objects.
