# ResNet+LSTM+GCN (ResLSTM)

Keras implementation of ResNet

Keras implementation of Attention LSTM

Keras implementation of GCN


## [Deep-learning Architecture for Short-term Passenger Flow Forecasting in Urban Rail Transit](http://doi.org/10.1109/TITS.2020.3000761)

<img src="https://github.com/JinleiZhangBJTU/ResNet-LSTM-GCN/blob/master/pictures/Model%20structure.png" width = "722" height = "813" alt="model structure" 
align=center>

## Description

We propose a deep-learning architecture combined residual network (ResNet), graph convolutional network (GCN) and long short-term memory (LSTM) (called “**ResLSTM**”) to forecast short-term passenger flow in urban rail transit on a network scale. First, improved methodologies of ResNet, GCN, and attention LSTM models are presented. Then, model architecture is proposed, wherein ResNet is used to capture deep abstract spatial correlations between subway stations, GCN is applied to extract network-topology information, and attention LSTM is used to extract temporal correlations. Model architecture includes four branches for inflow, outflow, graph-network topology, as well as weather conditions and air quality. To the best of our knowledge, this is the first time that air-quality indicators have been taken into account, and their influences on prediction precision have been quantified. Finally, ResLSTM is applied to Beijing subway. Three time granularities (10, 15, and 30 min) are chosen to conduct short-term passenger flow forecasting. Comparison of prediction performance of ResLSTM with those of many state-of-the-art models shows the advancement and robustness of ResLSTM. Moreover, comparison of prediction precisions obtained from time granularities of 10, 15, and 30 min indicates that prediction precision increases with increasing time granularity. 

## Data

The dimension of inflow data is n*time steps, where n represents number of stations and time steps denote time steps in 25 weekdays.

The structure of outflow data is the same with inflow data.

The dimension of meteorology data is n*time steps, where n represents  the 11 meteorology indicators such as temperature, wind speed.

## Requirement

Keras == 2.2.4  
tensorflow-gpu == 1.10.0  
numpy == 1.14.5  
scipy == 1.3.3  
scikit-learn == 0.20.2  
protobuf == 3.6.0  

## Implementation

Just download this repository and using PyCharm to open it. Then run ResLSTM.py.

## Result

![Model comparison](https://github.com/JinleiZhangBJTU/ResNet-LSTM-GCN/blob/master/pictures/Model%20comparison.jpg)

## Reference

J. Zhang, F. Chen, Z. Cui, Y. Guo and Y. Zhu, "[Deep Learning Architecture for Short-Term Passenger Flow Forecasting in Urban Rail Transit](http://doi.org/10.1109/TITS.2020.3000761)," in IEEE Transactions on Intelligent Transportation Systems, doi: 10.1109/TITS.2020.3000761.

## Video Data Path (New)

Video processing is unified into a single file:
`video_feature_pipeline.py`

You can add video-source features as the 5th input branch in two ways:

1. Mapping CSV (recommended for controlled experiments):
   - Prepare `data/videodata/station_video_map.csv` from template `data/videodata/station_video_map.template.csv`
   - Run:
     `python video_feature_pipeline.py --map_csv data/videodata/station_video_map.csv --total_time_slots <time_steps> --output_csv data/videodata/video_15min.csv --time_granularity_min 15`

2. Auto-discover by filename in a directory:
   - Place files like `station_000.mp4`, `station_001.mp4`, ...
   - Run:
     `python video_feature_pipeline.py --video_dir data/video --total_time_slots <time_steps> --output_csv data/videodata/video_15min.csv --time_granularity_min 15`

Optional extraction settings:
- `--backend motion|multifeature|hybrid`
- `--slot_agg mean|max|median`
- `--fill_missing zero|ffill|interp`

If no real videos are available, you can generate proxy video features from AFC:
`python video_feature_pipeline.py --backend proxy --total_time_slots <time_steps> --output_csv data/videodata/video_15min.csv --proxy_tg 15 --proxy_noise_std 0.03 --proxy_delay_steps 1 --proxy_missing_rate 0.1 --proxy_weights 0.3,0.5,0.2 --fill_missing interp`

Then train as usual:
`python ResLSTM.py`

Force-disable video branch (ablation with same codebase):
`python ResLSTM.py --disable_video`

`load_data.py` will auto-load video matrix from:
- `data/videodata/video_15min.csv`
- fallback: `data/video_features/video_15min.csv`

If no file exists, the video branch is disabled automatically and training falls back to the 4-branch model.
