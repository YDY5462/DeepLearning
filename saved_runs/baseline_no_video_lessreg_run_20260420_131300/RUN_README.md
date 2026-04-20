# Run README: baseline_no_video_lessreg_run_20260420_131300

## Run Identity
- created_at: `2026-04-20T13:18:43`
- run_dir: `saved_runs/baseline_no_video_lessreg_run_20260420_131300`
- source_result_dir: `testresult/`
- mode: `baseline_no_video`

## Command
```powershell
python ResLSTM.py --tg 15 --time_lag 6 --tg_in_one_day 72 --forecast_day_number 5 --tg_in_one_week 360 --batch_size 64 --start_epoch 50 --total_rounds 21 --disable_video
```

## Regularization/Strategy (less-underfit version)
- `DROPOUT_RATE`: `0.2 -> 0.1`
- `L2_WEIGHT`: `1e-5 -> 3e-6`
- `Adam(clipnorm=1.0) -> Adam()`
- `EarlyStopping`: `patience 6/3 -> 10/5`, `min_delta 1e-4 -> 3e-5`
- `ReduceLROnPlateau`: `factor 0.5 -> 0.7`, `patience 3/2 -> 5/3`
- `validation_split`: `0.1 -> 0.05`

## Metrics
- epoch_points: `50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200,210,220,230,240,250`
- final_epoch: `250`
- final: `RMSE=54.598626, MAE=30.581578, R2=0.946343, WMAPE=0.113248`
- total_train_time_sec: `6992.985`
- best_epoch(by RMSE): `190`
- best: `RMSE=51.736690, MAE=30.243917, R2=0.946500, WMAPE=0.111834`

## Logs
- `baseline_novideo_lessreg_20260420_112055_out.log`
- `baseline_novideo_lessreg_20260420_112055_err.log`
