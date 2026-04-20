# 四次训练代码差异记录（用于毕设实验追溯）

生成时间：2026-04-20  
项目目录：`E:\TJ-CS\graduation project\ResNet-LSTM-GCN-master\ResNet-LSTM-GCN-master`

## 1. 对比的四次训练

1. `run_20260407_150222`（旧版4分支）
2. `noisy_video_backup_20260419_144958`（5分支，视频=噪声/缺失版）
3. `clean_video_run_20260419_163315`（5分支，视频=clean版）
4. `baseline_no_video_run_20260419_210757`（新结构4分支，禁用视频）

## 2. 关键结论（先看这个）

1. `noisy_video` 和 `clean_video` 的模型代码结构一致，差异主要是视频特征文件不同。  
2. `baseline_no_video` 与 `noisy/clean` 同属“新结构代码栈”，只是关闭了视频输入分支。  
3. `run_20260407_150222` 属于更早代码栈（旧结构），不能与新结构直接当作“仅改一个变量”的严格对照。  
4. `run_20260407_150222` 与 `noisy_video_backup` 中的 `25/35` epoch 文件来自 2026-01-10 的旧文件残留（不是这两次连续训练产生）。

## 3. 四次训练的代码与数据差异表

| 实验ID | 结果目录 | 模型输入分支（由`.h5`反查） | 结构特征（由`.h5`反查） | 优化器配置（由`.h5`反查） | 视频数据 | epoch记录点 |
|---|---|---|---|---|---|---|
| E1 旧版4分支 | `saved_runs/run_20260407_150222/testresult` | 4输入（in/out/graph/weather） | LayerNorm=0, Dropout层=0, Dense-L2=0；仅有`multi_source_attention_weights` | Adam（无`clipnorm`字段） | 无视频分支 | `25,35,50...170`（其中25/35为旧残留） |
| E2 noisy视频 | `saved_runs/noisy_video_backup_20260419_144958/testresult` | 5输入（含video） | LayerNorm=5, Dropout层=7, Dense-L2=7；含`base_modality_fusion/attention_modality_fusion/attention_residual_scale` | Adam + `clipnorm=1.0` | `video_15min_noisy.csv` | `25,35,50...190`（其中25/35为旧残留） |
| E3 clean视频 | `saved_runs/clean_video_run_20260419_163315/testresult` | 5输入（含video） | 与E2一致（层数/组件一致） | Adam + `clipnorm=1.0` | `video_15min_used.csv`（clean） | `50...190` |
| E4 新版4分支baseline | `saved_runs/baseline_no_video_run_20260419_210757/testresult` | 4输入（不含video） | LayerNorm=4, Dropout层=6, Dense-L2=6；保留残差增强注意力模块 | Adam + `clipnorm=1.0` | 强制禁用（`--disable_video`） | `50...250` |

## 4. 每次训练的结果（方便论文写“实验设置与结果”）

| 实验ID | 最终epoch指标 | 最佳epoch指标（按RMSE） |
|---|---|---|
| E1 | epoch170: RMSE=47.7821, MAE=28.6775, R2=0.9438, WMAPE=0.1070 | epoch90: RMSE=38.0747, MAE=22.6769, R2=0.9568, WMAPE=0.0849 |
| E2 | epoch190: RMSE=80.5724, MAE=46.7170, R2=0.9038, WMAPE=0.1714 | epoch170: RMSE=73.3696, MAE=40.5648, R2=0.9128, WMAPE=0.1504 |
| E3 | epoch190: RMSE=90.0307, MAE=50.0221, R2=0.8863, WMAPE=0.1840 | epoch170: RMSE=79.7665, MAE=43.1453, R2=0.9071, WMAPE=0.1603 |
| E4 | epoch250: RMSE=73.2066, MAE=40.5198, R2=0.9192, WMAPE=0.1502 | epoch80: RMSE=60.8208, MAE=33.7955, R2=0.9332, WMAPE=0.1255 |

## 5. 可复现实验命令（已知与推断）

### E4（已知，来自README记录）

```powershell
python ResLSTM.py --tg 15 --time_lag 6 --tg_in_one_day 72 --forecast_day_number 5 --tg_in_one_week 360 --batch_size 64 --start_epoch 50 --total_rounds 21 --disable_video
```

### E2 / E3（推断：同一代码栈，视频自动启用）

`ResLSTM.py` 当前默认参数为：`start_epoch=50, total_rounds=15, batch_size=64`，会输出 `50~190`。  
E2/E3的epoch分布与该行为一致（E2中额外`25/35`为历史残留文件）。

推断命令：

```powershell
python ResLSTM.py
```

### E1（旧版，脚本内写死参数）

历史代码中无CLI参数，文件尾部写死：
- `total_rounds = 15`
- `Run_epoch = 50`
- 每轮 `Run_epoch += 10`

即默认输出 `50~190`，但 E1 目录只到 `170`，且保留了更早的 `25/35` 文件。

## 6. 证据文件与哈希（保证“不是口述”）

### 6.1 代码文件哈希

- 当前新结构代码：
  - `ResLSTM.py`  
    `SHA256=E932984507293AC7E145E09DE0587E6CFD6322D55FB839DFFD642D93AEC48499`
  - `load_data.py`  
    `SHA256=1EBCA574F9AB498C4A8D30778D3652D802CF82D68F31AC6257E1719EDDBE665E`

- 旧结构参考快照（历史文件）：
  - `.history/ResLSTM_20260331181321.py`  
    `SHA256=2422BF1B18AF573677ABAF9991563E601A71D38BC797D5E2D42DAF03E0EF6E57`
  - `.history/load_data_20260407110558.py`  
    `SHA256=7D31C8DE463BE5A0BA9C2EE23F1FDF8A24B6256012983A7C8802777B63E1BF08`

### 6.2 视频特征文件哈希

- noisy版：
  - `data/videodata/video_15min.csv`  
    `SHA256=EB67AB5FA82E0A5BC81F2D03C213641CAEFB8B8D41E66FD4B3A7B54782A54B71`
  - `saved_runs/noisy_video_backup_20260419_144958/video_15min_noisy.csv`  
    `SHA256=EB67AB5FA82E0A5BC81F2D03C213641CAEFB8B8D41E66FD4B3A7B54782A54B71`

- clean版：
  - `data/videodata/video_15min_clean.csv`  
    `SHA256=5EECCA3FDD7DB63D85A428114053900926F56C008FC84DCAD6A6E2172E09EC6E`
  - `saved_runs/clean_video_run_20260419_163315/video_15min_used.csv`  
    `SHA256=5EECCA3FDD7DB63D85A428114053900926F56C008FC84DCAD6A6E2172E09EC6E`

## 7. 论文写作建议（可直接引用）

可把实验编号固定为：`E1(旧版4分支)`、`E2(noisy视频)`、`E3(clean视频)`、`E4(新版4分支baseline)`。  
并在“实验设置”中明确写：

1. E1与E2/E3/E4非同一代码栈，主要用于“演进对比”，不作为严格消融。  
2. 严格消融应在同一代码栈下比较：E4（无视频） vs E2（有视频-noisy） vs E3（有视频-clean）。  
3. 为避免历史文件混淆，后续每次训练前建议清空`testresult`或写入独立目录。

