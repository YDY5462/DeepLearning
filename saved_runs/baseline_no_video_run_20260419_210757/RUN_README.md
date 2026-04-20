# Run README: baseline_no_video_run_20260419_210757

## Run Identity
- run_dir: `saved_runs/baseline_no_video_run_20260419_210757`
- result_dir: `saved_runs/baseline_no_video_run_20260419_210757/testresult`
- video_descriptor: `video_disabled`

## Metrics
- epoch_points: `50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200,210,220,230,240,250`
- final_epoch: `250`
- final: `RMSE=73.206591, MAE=40.519800, R2=0.919172, WMAPE=0.150230`
- best_epoch(by RMSE): `80`
- best: `RMSE=60.820846, MAE=33.795479, R2=0.933189, WMAPE=0.125514`

## Model Signature (from .h5)
- model_file: `saved_runs/baseline_no_video_run_20260419_210757/testresult/250-model-with-graph.h5`
- input_count: `4`
- layernorm_count: `4`
- dropout_layer_count: `6`
- dense_l2_count: `6`
- residual_attention_enhance: `True`
- has_video_input_name: `False`
- custom_layers: `LearnableScalar,PairwiseSubtract,ReduceMeanAxis2,ReduceSumAxis2`

## Notes
- no extra notes
