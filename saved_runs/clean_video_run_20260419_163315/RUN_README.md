# Run README: clean_video_run_20260419_163315

## Run Identity
- run_dir: `saved_runs/clean_video_run_20260419_163315`
- result_dir: `saved_runs/clean_video_run_20260419_163315/testresult`
- video_descriptor: `video_proxy_clean`

## Metrics
- epoch_points: `50,60,70,80,90,100,110,120,130,140,150,160,170,180,190`
- final_epoch: `190`
- final: `RMSE=90.030728, MAE=50.022107, R2=0.886305, WMAPE=0.184021`
- best_epoch(by RMSE): `170`
- best: `RMSE=79.766489, MAE=43.145336, R2=0.907116, WMAPE=0.160269`

## Model Signature (from .h5)
- model_file: `saved_runs/clean_video_run_20260419_163315/testresult/190-model-with-graph.h5`
- input_count: `5`
- layernorm_count: `5`
- dropout_layer_count: `7`
- dense_l2_count: `7`
- residual_attention_enhance: `True`
- has_video_input_name: `True`
- custom_layers: `LearnableScalar,PairwiseSubtract,ReduceMeanAxis2,ReduceSumAxis2`

## Notes
- no extra notes
