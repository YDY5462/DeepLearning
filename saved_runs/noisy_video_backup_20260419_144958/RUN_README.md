# Run README: noisy_video_backup_20260419_144958

## Run Identity
- run_dir: `saved_runs/noisy_video_backup_20260419_144958`
- result_dir: `saved_runs/noisy_video_backup_20260419_144958/testresult`
- video_descriptor: `video_proxy_noisy_or_uncertain`

## Metrics
- epoch_points: `25,35,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190`
- final_epoch: `190`
- final: `RMSE=80.572369, MAE=46.716963, R2=0.903761, WMAPE=0.171426`
- best_epoch(by RMSE): `170`
- best: `RMSE=73.369644, MAE=40.564799, R2=0.912753, WMAPE=0.150400`

## Model Signature (from .h5)
- model_file: `saved_runs/noisy_video_backup_20260419_144958/testresult/190-model-with-graph.h5`
- input_count: `5`
- layernorm_count: `5`
- dropout_layer_count: `7`
- dense_l2_count: `7`
- residual_attention_enhance: `True`
- has_video_input_name: `True`
- custom_layers: `LearnableScalar,PairwiseSubtract,ReduceMeanAxis2,ReduceSumAxis2`

## Notes
- Contains epoch 25/35 files from older historical runs; treat with caution for strict comparability.
