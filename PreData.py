import os
import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset
import torch.nn.functional as F

class GTZANDataset(Dataset):
    def __init__(self,
                 annotations_file,  # CSV索引文件路径
                 base_dir,          # 数据集根目录（GTZAN_processed）
                 transformation,    # 保留参数但实际不使用，冗余项，重构代码之后没用到了
                 target_sample_rate,
                 num_samples,
                 device="cpu"):
        """
        Args:
            annotations_file (str): train_index.csv 或 test_index.csv 的路径
            base_dir (str): GTZAN_processed 目录的路径
            transformation: 保留参数但不使用（为保持接口一致）
            target_sample_rate (int): 目标采样率
            num_samples (int): 每个样本的采样点数（22050 * 3 = 66150）
            device (str): 计算设备（cpu/cuda）
        """
        self.annotations = pd.read_csv(annotations_file)
        self.base_dir = base_dir
        self.device = device
        self.target_sample_rate = target_sample_rate
        self.num_samples = num_samples
        
        # STFT参数
        self.n_fft = 1024       # FFT窗口大小
        self.hop_length = 512    # 50%重叠
        self.win_length = 1024   # 窗口长度
        self.n_freq_bins = 513   # 频率维度（n_fft//2 +1）
        self.n_time_frames = 128 # 时间维度
        
       
        
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, index):
        # 获取音频绝对路径
        relative_path = self.annotations.iloc[index]['file_path']
        audio_sample_path = os.path.join(self.base_dir, relative_path)
        
        # 获取标签
        label = self.annotations.iloc[index]['label']
        
        # 加载音频（自动转换为目标采样率）
        signal, sr = torchaudio.load(audio_sample_path)
        signal = signal.to(self.device)
        
        # 预处理流程
        signal = self._resample_if_necessary(signal, sr)
        signal = self._mix_down_if_necessary(signal)
        signal = self._cut_if_necessary(signal)
        signal = self._right_pad_if_necessary(signal)
        
        # 应用STFT变换
        stft = torch.stft(
            input=signal,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=torch.hann_window(self.win_length).to(self.device),
            center=False,  # 禁用填充以确保精确128帧
            return_complex=True
        )  # 输出形状 [1, 513, 128]
        
        # 计算幅度谱并转置为[1, 128, 513]
        spectrogram = torch.abs(stft).permute(0, 2, 1)
        
        # 转换为dB单位
        spectrogram = torchaudio.functional.amplitude_to_DB(
            spectrogram, 
            multiplier=20.0, 
            amin=1e-10, 
            db_multiplier=0.0,
            top_db=80.0
        )
        
        return spectrogram, label, audio_sample_path
    
    # --- 音频预处理方法 ---
    def _cut_if_necessary(self, signal):
        """裁剪到目标长度"""
        if signal.shape[1] > self.num_samples:
            signal = signal[:, :self.num_samples]
        return signal
    
    def _right_pad_if_necessary(self, signal):
        """右侧填充到目标长度"""
        if signal.shape[1] < self.num_samples:
            num_missing = self.num_samples - signal.shape[1]
            signal = F.pad(signal, (0, num_missing))
        return signal
    
    def _resample_if_necessary(self, signal, sr):
        """重采样到目标采样率"""
        if sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sr, 
                new_freq=self.target_sample_rate
            ).to(self.device)
            signal = resampler(signal)
        return signal
    
    def _mix_down_if_necessary(self, signal):
        """混音为单声道"""
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
        return signal


if __name__ == "__main__":
    # 参数配置
    ANNOTATIONS_FILE = "GTZAN_processed/train_index.csv"  # 可修改为你的
    BASE_DIR = "GTZAN_processed"
    SAMPLE_RATE = 22050
    NUM_SAMPLES = (128 - 1) * 512 + 1024  # 计算得到66150（3秒）
    
    # 设备检测
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    # 初始化数据集
    gtzan = GTZANDataset(
        annotations_file=ANNOTATIONS_FILE,
        base_dir=BASE_DIR,
        transformation=None,
        target_sample_rate=SAMPLE_RATE,
        num_samples=NUM_SAMPLES,
        device=device
    )
    
    # 测试样本
    print(f"数据集大小: {len(gtzan)}")
    spectrogram, label, path = gtzan[0]
    print(spectrogram)
    print(f"频谱图形状: {spectrogram.shape}")  # 应为 torch.Size([1, 128, 513])
    print(f"标签: {label}")
    print(f"文件路径: {path}")

    # 验证参数
    print("\n验证STFT参数:")
    print(f"音频长度: {NUM_SAMPLES} samples ({NUM_SAMPLES/SAMPLE_RATE:.2f}s)")
    print(f"理论帧数: {(NUM_SAMPLES - 1024) // 512 + 1} (应为128)")