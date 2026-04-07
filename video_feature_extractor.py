"""
地铁客流视频特征提取模块
========================================
功能：从地铁站监控视频中提取客流密度特征，用于多模态客流预测模型训练

主要组件：
1. VideoFrameExtractor - 视频帧提取器
2. CrowdDensityEstimator - 人群密度估计器（基于预训练CNN）
3. VideoFeatureAggregator - 视频特征聚合器（按时间粒度聚合）
4. VideoDataGenerator - 视频数据生成器（生成训练数据）

作者：毕设项目
日期：2025
"""

import numpy as np
import os
import cv2
from datetime import datetime, timedelta
import pickle
import warnings
warnings.filterwarnings('ignore')

# ===================== 配置参数 =====================
class VideoConfig:
    """视频处理配置"""
    # 视频源设置
    VIDEO_DIR = "data/video/"           # 视频文件目录
    FRAME_WIDTH = 224                    # 帧宽度（适配预训练模型）
    FRAME_HEIGHT = 224                   # 帧高度
    FPS_SAMPLE_RATE = 1                  # 每秒采样帧数（降采样）
    
    # 时间粒度设置（与客流数据对齐）
    TIME_GRANULARITY = 15                # 时间粒度（分钟）
    TIME_LAG = 6                         # 历史时间步数
    
    # 站点设置
    NUM_STATIONS = 276                   # 站点数量
    
    # 特征维度
    FEATURE_DIM = 512                    # CNN特征维度
    OUTPUT_FEATURE_DIM = 1               # 最终输出特征维度（密度值）
    
    # 输出路径
    OUTPUT_DIR = "data/video_features/"
    

# ===================== 视频帧提取器 =====================
class VideoFrameExtractor:
    """
    视频帧提取器
    功能：从视频文件中按指定采样率提取帧
    """
    
    def __init__(self, config=VideoConfig):
        self.config = config
        self.frame_width = config.FRAME_WIDTH
        self.frame_height = config.FRAME_HEIGHT
        self.sample_rate = config.FPS_SAMPLE_RATE
        
    def extract_frames(self, video_path, start_time=None, end_time=None):
        """
        从视频中提取帧
        
        Args:
            video_path: 视频文件路径
            start_time: 开始时间（秒）
            end_time: 结束时间（秒）
            
        Returns:
            frames: numpy数组，形状为 (num_frames, height, width, 3)
            timestamps: 每帧对应的时间戳列表
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"视频文件不存在: {video_path}")
            
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"无法打开视频: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        
        print(f"视频信息: FPS={fps:.2f}, 总帧数={total_frames}, 时长={duration:.2f}秒")
        
        # 计算采样间隔
        sample_interval = int(fps / self.sample_rate)
        
        frames = []
        timestamps = []
        frame_idx = 0
        
        # 设置起止位置
        if start_time:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_time * fps))
            frame_idx = int(start_time * fps)
        
        end_frame = int(end_time * fps) if end_time else total_frames
        
        while frame_idx < end_frame:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_idx % sample_interval == 0:
                # 调整大小
                frame_resized = cv2.resize(frame, (self.frame_width, self.frame_height))
                # BGR to RGB
                frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
                # 归一化
                frame_normalized = frame_rgb.astype(np.float32) / 255.0
                
                frames.append(frame_normalized)
                timestamps.append(frame_idx / fps)
            
            frame_idx += 1
        
        cap.release()
        
        return np.array(frames), timestamps
    
    def extract_frames_by_time_slots(self, video_path, time_granularity_minutes=15):
        """
        按时间槽提取帧并聚合
        
        Args:
            video_path: 视频文件路径
            time_granularity_minutes: 时间粒度（分钟）
            
        Returns:
            time_slot_frames: dict，key为时间槽索引，value为该时间槽内的帧列表
        """
        frames, timestamps = self.extract_frames(video_path)
        
        time_slot_frames = {}
        slot_duration = time_granularity_minutes * 60  # 转换为秒
        
        for frame, ts in zip(frames, timestamps):
            slot_idx = int(ts // slot_duration)
            if slot_idx not in time_slot_frames:
                time_slot_frames[slot_idx] = []
            time_slot_frames[slot_idx].append(frame)
        
        # 转换为numpy数组
        for slot_idx in time_slot_frames:
            time_slot_frames[slot_idx] = np.array(time_slot_frames[slot_idx])
            
        return time_slot_frames


# ===================== 人群密度估计器 =====================
class CrowdDensityEstimator:
    """
    人群密度估计器
    功能：基于预训练CNN模型估计视频帧中的人群密度
    
    支持两种模式：
    1. 使用预训练的密度估计模型（如CSRNet）
    2. 使用简化的特征提取 + 回归方法
    """
    
    def __init__(self, model_type='simple', config=VideoConfig):
        """
        初始化密度估计器
        
        Args:
            model_type: 模型类型
                - 'simple': 简化模型（基于图像统计特征）
                - 'cnn': 使用预训练CNN提取特征
                - 'csrnet': 使用CSRNet密度估计模型（需要额外安装）
        """
        self.config = config
        self.model_type = model_type
        self.model = None
        
        if model_type == 'cnn':
            self._load_cnn_model()
        elif model_type == 'csrnet':
            self._load_csrnet_model()
            
    def _load_cnn_model(self):
        """加载预训练CNN模型（VGG16用于特征提取）"""
        try:
            from tensorflow.keras.applications import VGG16
            from tensorflow.keras.models import Model
            
            # 使用VGG16作为特征提取器
            base_model = VGG16(weights='imagenet', include_top=False, 
                              input_shape=(self.config.FRAME_HEIGHT, self.config.FRAME_WIDTH, 3))
            self.model = Model(inputs=base_model.input, 
                             outputs=base_model.get_layer('block5_pool').output)
            print("CNN特征提取模型加载成功 (VGG16)")
        except Exception as e:
            print(f"CNN模型加载失败: {e}")
            print("将使用简化的特征提取方法")
            self.model_type = 'simple'
            
    def _load_csrnet_model(self):
        """加载CSRNet密度估计模型"""
        # CSRNet需要额外的模型权重，这里提供接口
        print("CSRNet模型需要预训练权重，请确保已下载")
        print("可从 https://github.com/leeyeehoo/CSRNet-pytorch 获取")
        self.model_type = 'simple'
    
    def estimate_density_simple(self, frame):
        """
        简化的密度估计方法
        基于图像特征（边缘密度、颜色分布等）估计人群密度
        
        Args:
            frame: 输入帧，形状为 (H, W, 3)，值范围 [0, 1]
            
        Returns:
            density: 估计的人群密度值
            features: 提取的特征向量
        """
        # 转换为uint8格式
        frame_uint8 = (frame * 255).astype(np.uint8)
        
        # 1. 边缘检测 - 人群区域通常有更多边缘
        gray = cv2.cvtColor(frame_uint8, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.mean(edges) / 255.0
        
        # 2. 光流估计的替代 - 使用帧差分（需要连续帧）
        # 这里用纹理复杂度代替
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        texture_complexity = np.var(laplacian) / 10000.0
        
        # 3. 颜色分布 - 人群区域颜色通常更复杂
        hsv = cv2.cvtColor(frame_uint8, cv2.COLOR_RGB2HSV)
        color_variance = np.mean([np.var(hsv[:,:,i]) for i in range(3)]) / 10000.0
        
        # 4. 局部二值模式(LBP)特征 - 纹理特征
        lbp_value = self._compute_lbp_density(gray)
        
        # 5. 前景检测（假设背景较暗或较亮）
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        foreground_ratio = np.mean(thresh) / 255.0
        
        # 组合特征
        features = np.array([
            edge_density,
            texture_complexity,
            color_variance,
            lbp_value,
            foreground_ratio
        ])
        
        # 简单的密度估计公式（可根据实际数据调整权重）
        weights = np.array([0.3, 0.25, 0.15, 0.2, 0.1])
        density = np.dot(features, weights)
        
        # 归一化到 [0, 1]
        density = np.clip(density, 0, 1)
        
        return density, features
    
    def _compute_lbp_density(self, gray_image):
        """计算LBP纹理密度"""
        h, w = gray_image.shape
        lbp_sum = 0
        
        # 简化的LBP计算
        for i in range(1, h-1):
            for j in range(1, w-1):
                center = gray_image[i, j]
                binary = 0
                binary |= (gray_image[i-1, j-1] >= center) << 7
                binary |= (gray_image[i-1, j] >= center) << 6
                binary |= (gray_image[i-1, j+1] >= center) << 5
                binary |= (gray_image[i, j+1] >= center) << 4
                binary |= (gray_image[i+1, j+1] >= center) << 3
                binary |= (gray_image[i+1, j] >= center) << 2
                binary |= (gray_image[i+1, j-1] >= center) << 1
                binary |= (gray_image[i, j-1] >= center) << 0
                lbp_sum += bin(binary).count('1')
        
        return lbp_sum / (h * w * 8)
    
    def estimate_density_cnn(self, frames):
        """
        使用CNN提取特征并估计密度
        
        Args:
            frames: 输入帧批次，形状为 (N, H, W, 3)
            
        Returns:
            densities: 密度估计值数组
            features: CNN特征数组
        """
        if self.model is None:
            raise ValueError("CNN模型未加载")
        
        # 预处理
        from tensorflow.keras.applications.vgg16 import preprocess_input
        frames_preprocessed = preprocess_input(frames * 255)
        
        # 提取特征
        features = self.model.predict(frames_preprocessed, verbose=0)
        
        # 全局平均池化
        features_pooled = np.mean(features, axis=(1, 2))
        
        # 使用特征范数作为密度代理
        densities = np.linalg.norm(features_pooled, axis=1)
        densities = densities / np.max(densities)  # 归一化
        
        return densities, features_pooled
    
    def estimate_density(self, frames):
        """
        统一的密度估计接口
        
        Args:
            frames: 输入帧，可以是单帧 (H, W, 3) 或批次 (N, H, W, 3)
            
        Returns:
            densities: 密度估计值
            features: 特征向量
        """
        # 确保是批次格式
        if frames.ndim == 3:
            frames = np.expand_dims(frames, axis=0)
        
        if self.model_type == 'cnn' and self.model is not None:
            return self.estimate_density_cnn(frames)
        else:
            # 使用简化方法
            densities = []
            features_list = []
            for frame in frames:
                d, f = self.estimate_density_simple(frame)
                densities.append(d)
                features_list.append(f)
            return np.array(densities), np.array(features_list)


# ===================== 视频特征聚合器 =====================
class VideoFeatureAggregator:
    """
    视频特征聚合器
    功能：将帧级别特征聚合为时间槽级别特征
    """
    
    def __init__(self, time_granularity=15, config=VideoConfig):
        """
        Args:
            time_granularity: 时间粒度（分钟）
        """
        self.time_granularity = time_granularity
        self.config = config
        
    def aggregate_by_time_slot(self, densities, timestamps, aggregation='mean'):
        """
        按时间槽聚合密度值
        
        Args:
            densities: 密度值数组
            timestamps: 时间戳数组（秒）
            aggregation: 聚合方式 ('mean', 'max', 'median')
            
        Returns:
            slot_densities: dict，key为时间槽索引，value为聚合后的密度
        """
        slot_duration = self.time_granularity * 60
        slot_densities = {}
        
        for density, ts in zip(densities, timestamps):
            slot_idx = int(ts // slot_duration)
            if slot_idx not in slot_densities:
                slot_densities[slot_idx] = []
            slot_densities[slot_idx].append(density)
        
        # 聚合
        for slot_idx in slot_densities:
            values = slot_densities[slot_idx]
            if aggregation == 'mean':
                slot_densities[slot_idx] = np.mean(values)
            elif aggregation == 'max':
                slot_densities[slot_idx] = np.max(values)
            elif aggregation == 'median':
                slot_densities[slot_idx] = np.median(values)
                
        return slot_densities
    
    def create_time_series(self, slot_densities, total_slots):
        """
        创建完整的时间序列（填充缺失值）
        
        Args:
            slot_densities: 时间槽密度字典
            total_slots: 总时间槽数
            
        Returns:
            time_series: 完整的时间序列数组
        """
        time_series = np.zeros(total_slots)
        
        for slot_idx, density in slot_densities.items():
            if 0 <= slot_idx < total_slots:
                time_series[slot_idx] = density
        
        # 线性插值填充缺失值
        nonzero_indices = np.where(time_series > 0)[0]
        if len(nonzero_indices) > 1:
            time_series = np.interp(
                np.arange(total_slots),
                nonzero_indices,
                time_series[nonzero_indices]
            )
            
        return time_series


# ===================== 视频数据生成器 =====================
class VideoDataGenerator:
    """
    视频数据生成器
    功能：生成与客流数据对齐的视频特征数据，用于模型训练
    """
    
    def __init__(self, config=VideoConfig):
        self.config = config
        self.frame_extractor = VideoFrameExtractor(config)
        self.density_estimator = CrowdDensityEstimator('simple', config)
        self.aggregator = VideoFeatureAggregator(config.TIME_GRANULARITY, config)
        
    def process_station_video(self, video_path, station_id):
        """
        处理单个站点的视频
        
        Args:
            video_path: 视频文件路径
            station_id: 站点ID
            
        Returns:
            features: 该站点的视频特征时间序列
        """
        print(f"处理站点 {station_id} 的视频: {video_path}")
        
        # 提取帧
        frames, timestamps = self.frame_extractor.extract_frames(video_path)
        
        if len(frames) == 0:
            print(f"警告: 站点 {station_id} 没有提取到有效帧")
            return None
        
        # 估计密度
        densities, _ = self.density_estimator.estimate_density(frames)
        
        # 聚合
        slot_densities = self.aggregator.aggregate_by_time_slot(densities, timestamps)
        
        # 计算总时间槽数
        total_duration = timestamps[-1] if timestamps else 0
        total_slots = int(total_duration / (self.config.TIME_GRANULARITY * 60)) + 1
        
        # 创建时间序列
        time_series = self.aggregator.create_time_series(slot_densities, total_slots)
        
        return time_series
    
    def generate_training_data(self, video_dir=None, num_days=25, slots_per_day=72):
        """
        生成训练数据
        
        Args:
            video_dir: 视频文件目录
            num_days: 天数
            slots_per_day: 每天的时间槽数（15分钟粒度 = 96，10分钟 = 144）
            
        Returns:
            X_train_video: 训练集视频特征
            X_test_video: 测试集视频特征
        """
        if video_dir is None:
            video_dir = self.config.VIDEO_DIR
            
        num_stations = self.config.NUM_STATIONS
        time_lag = self.config.TIME_LAG
        total_slots = num_days * slots_per_day
        
        # 检查是否有真实视频数据
        if os.path.exists(video_dir) and len(os.listdir(video_dir)) > 0:
            return self._process_real_videos(video_dir, num_stations, total_slots, time_lag)
        else:
            print("未找到视频数据目录，生成模拟数据用于测试...")
            return self._generate_simulated_data(num_stations, total_slots, time_lag, slots_per_day)
    
    def _process_real_videos(self, video_dir, num_stations, total_slots, time_lag):
        """处理真实视频数据"""
        all_features = np.zeros((num_stations, total_slots))
        
        for station_id in range(num_stations):
            video_pattern = f"station_{station_id:03d}*.mp4"
            video_files = [f for f in os.listdir(video_dir) 
                          if f.startswith(f"station_{station_id:03d}")]
            
            if video_files:
                video_path = os.path.join(video_dir, video_files[0])
                features = self.process_station_video(video_path, station_id)
                if features is not None:
                    # 对齐长度
                    min_len = min(len(features), total_slots)
                    all_features[station_id, :min_len] = features[:min_len]
            
        return self._create_samples(all_features, time_lag)
    
    def _generate_simulated_data(self, num_stations, total_slots, time_lag, slots_per_day):
        """
        生成模拟的视频特征数据
        用于在没有真实视频时测试模型架构
        
        模拟逻辑：
        - 基于时间的周期性模式（早晚高峰）
        - 站点间的空间相关性
        - 随机噪声
        """
        print(f"生成模拟视频特征: {num_stations}站点 x {total_slots}时间槽")
        
        all_features = np.zeros((num_stations, total_slots))
        
        for station_id in range(num_stations):
            for t in range(total_slots):
                # 时间因素（一天内的位置）
                time_of_day = (t % slots_per_day) / slots_per_day
                
                # 早高峰 (7:00-9:00) 和晚高峰 (17:00-19:00)
                morning_peak = np.exp(-((time_of_day - 0.33) ** 2) / 0.01)  # ~8:00
                evening_peak = np.exp(-((time_of_day - 0.75) ** 2) / 0.01)  # ~18:00
                
                # 基础密度
                base_density = 0.2 + 0.4 * (morning_peak + evening_peak)
                
                # 站点特性（某些站点更繁忙）
                station_factor = 0.5 + 0.5 * np.sin(station_id * 0.1)
                
                # 随机噪声
                noise = np.random.normal(0, 0.05)
                
                density = np.clip(base_density * station_factor + noise, 0, 1)
                all_features[station_id, t] = density
        
        return self._create_samples(all_features, time_lag)
    
    def _create_samples(self, all_features, time_lag):
        """
        从特征矩阵创建训练样本
        
        Args:
            all_features: 形状为 (num_stations, total_slots) 的特征矩阵
            time_lag: 历史时间步数
            
        Returns:
            X_train: 训练集
            X_test: 测试集
        """
        num_stations, total_slots = all_features.shape
        
        # 创建滑动窗口样本
        samples = []
        for t in range(time_lag - 1, total_slots):
            # 提取历史 time_lag-1 个时间步
            sample = all_features[:, t - (time_lag - 1):t]
            samples.append(sample)
        
        samples = np.array(samples)  # (num_samples, num_stations, time_lag-1)
        
        # 添加通道维度
        samples = np.expand_dims(samples, axis=-1)  # (num_samples, num_stations, time_lag-1, 1)
        
        # 划分训练集和测试集（最后5天为测试集）
        # 假设每天72个时间槽（15分钟粒度）
        test_samples = 5 * 72  # 5天
        
        X_train = samples[:-test_samples]
        X_test = samples[-test_samples:]
        
        print(f"视频特征数据形状: X_train={X_train.shape}, X_test={X_test.shape}")
        
        return X_train, X_test
    
    def save_features(self, X_train, X_test, output_dir=None):
        """保存提取的特征"""
        if output_dir is None:
            output_dir = self.config.OUTPUT_DIR
            
        os.makedirs(output_dir, exist_ok=True)
        
        np.save(os.path.join(output_dir, 'X_train_video.npy'), X_train)
        np.save(os.path.join(output_dir, 'X_test_video.npy'), X_test)
        
        print(f"视频特征已保存至: {output_dir}")
        
    def load_features(self, output_dir=None):
        """加载已保存的特征"""
        if output_dir is None:
            output_dir = self.config.OUTPUT_DIR
            
        X_train = np.load(os.path.join(output_dir, 'X_train_video.npy'))
        X_test = np.load(os.path.join(output_dir, 'X_test_video.npy'))
        
        return X_train, X_test


# ===================== 与现有数据加载模块集成 =====================
def get_video_features(TG=15, time_lag=6, forecast_day_number=5):
    """
    获取视频特征数据（供 load_data.py 调用）
    
    Args:
        TG: 时间粒度（分钟）
        time_lag: 历史时间步数
        forecast_day_number: 预测天数
        
    Returns:
        X_train_5: 训练集视频特征
        X_test_5: 测试集视频特征
    """
    # 更新配置
    VideoConfig.TIME_GRANULARITY = TG
    VideoConfig.TIME_LAG = time_lag
    
    # 计算每天的时间槽数
    slots_per_day = int(24 * 60 / TG)
    
    # 检查是否已有保存的特征
    feature_path = os.path.join(VideoConfig.OUTPUT_DIR, 'X_train_video.npy')
    
    if os.path.exists(feature_path):
        print("加载已保存的视频特征...")
        generator = VideoDataGenerator(VideoConfig)
        X_train_5, X_test_5 = generator.load_features()
    else:
        print("生成视频特征数据...")
        generator = VideoDataGenerator(VideoConfig)
        X_train_5, X_test_5 = generator.generate_training_data(
            num_days=25,
            slots_per_day=slots_per_day
        )
        # 保存以供后续使用
        generator.save_features(X_train_5, X_test_5)
    
    return X_train_5, X_test_5


# ===================== 主函数（测试用） =====================
if __name__ == "__main__":
    print("=" * 60)
    print("地铁客流视频特征提取模块测试")
    print("=" * 60)
    
    # 测试生成模拟数据
    generator = VideoDataGenerator(VideoConfig)
    
    # 生成训练数据
    X_train_video, X_test_video = generator.generate_training_data(
        num_days=25,
        slots_per_day=72  # 15分钟粒度
    )
    
    print(f"\n生成的数据形状:")
    print(f"  X_train_video: {X_train_video.shape}")
    print(f"  X_test_video: {X_test_video.shape}")
    
    # 保存特征
    generator.save_features(X_train_video, X_test_video)
    
    # 测试密度估计器
    print("\n测试密度估计器:")
    estimator = CrowdDensityEstimator('simple')
    
    # 创建测试帧
    test_frame = np.random.rand(224, 224, 3).astype(np.float32)
    density, features = estimator.estimate_density(test_frame)
    
    print(f"  测试帧密度估计: {density[0]:.4f}")
    print(f"  特征维度: {features.shape}")
    
    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)
