import sys
import random
import os
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QLabel, QPushButton, QTableWidget,
                            QTableWidgetItem, QHeaderView, QSlider, QFileDialog,
                            QMessageBox, QProgressBar)
from PyQt5.QtCore import Qt, QUrl, QTimer, QThread, pyqtSignal
from PyQt5.QtGui import QIcon, QColor, QFont
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
import torch
import torchaudio
from torch.nn import functional as F
from net import PRCNN_dpo
from paperPreData import GTZANDataset

# 配置参数
ANNOTATIONS_FILE = "gtzan_dataset.csv"
AUDIO_DIR = ""
SAMPLE_RATE = 22050
NUM_SAMPLES = 22050 * 3
MODEL_PATH = "best_model_final.pth"

# 音乐类别映射
CLASS_MAPPING = [
    'blues', 'classical', 'country', 'disco', 'hiphop',
    'jazz', 'metal', 'pop', 'reggae', 'rock'
]

# 颜色映射 (用于不同音乐风格)
STYLE_COLORS = {
    'blues': '#1f77b4',
    'classical': '#ff7f0e',
    'country': '#2ca02c',
    'disco': '#d62728',
    'hiphop': '#9467bd',
    'jazz': '#8c564b',
    'metal': '#e377c2',
    'pop': '#7f7f7f',
    'reggae': '#bcbd22',
    'rock': '#17becf'
}

class MusicRecommender:
    def __init__(self):
        # 初始化模型
        self.model = PRCNN_dpo()
        self.model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
        self.model.eval()
        
        # 加载数据集
        self.dataset = GTZANDataset(
            annotations_file=ANNOTATIONS_FILE,
            base_dir=AUDIO_DIR,
            transformation=None,
            target_sample_rate=SAMPLE_RATE,
            num_samples=NUM_SAMPLES,
            device="cpu"
        )
        
        # STFT参数
        self.n_fft = 1024
        self.hop_length = 512
        self.win_length = 1024
        self.n_time_frames = 128  # 目标时间帧数
        self.n_freq_bins = 513    # 频率维度大小

    def _load_audio(self, file_path):
        """加载音频文件并预处理"""
        waveform, sr = torchaudio.load(file_path)
        if waveform.shape[0] > 1:  # 转换为单声道
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        if sr != SAMPLE_RATE:      # 重采样
            waveform = torchaudio.transforms.Resample(sr, SAMPLE_RATE)(waveform)
        return waveform

    def _compute_spectrogram(self, waveform):
        """计算频谱图"""
        stft = torch.stft(
            waveform, n_fft=self.n_fft, hop_length=self.hop_length,
            win_length=self.win_length, window=torch.hann_window(self.win_length),
            center=False, return_complex=True
        )
        spectrogram = torch.abs(stft).permute(0, 2, 1)  # [1, time, freq]
        spectrogram = torchaudio.functional.amplitude_to_DB(
            spectrogram, multiplier=20.0, amin=1e-10, db_multiplier=0.0, top_db=80.0
        )
        return spectrogram

    def _segment_spectrogram(self, spectrogram, overlap=32):
        """将频谱图分割为多个[1, 128, 513]的片段"""
        segments = []
        step = self.n_time_frames - overlap  # 滑动步长
        
        for start in range(0, spectrogram.shape[1] - overlap, step):
            end = start + self.n_time_frames
            if end > spectrogram.shape[1]:  # 处理最后一个不足128帧的片段
                segment = F.pad(
                    spectrogram[:, start:], 
                    (0, 0, 0, end - spectrogram.shape[1])
                )
            else:
                segment = spectrogram[:, start:end]
            segments.append(segment)
        
        return torch.stack(segments) if segments else None

    def predict_custom_file(self, file_path, batch_size=32):
        """预测自定义文件的音乐风格"""
        try:
            # 1. 加载并预处理音频
            waveform = self._load_audio(file_path)
            
            # 2. 计算完整频谱图
            spectrogram = self._compute_spectrogram(waveform)
            
            # 3. 分割频谱图为多个[1, 128, 513]的片段
            segments = self._segment_spectrogram(spectrogram)
            if segments is None:
                raise ValueError("音频太短，无法分割成有效片段")
            
            # 4. 批量预测所有片段
            all_probs = []
            with torch.no_grad():
                for i in range(0, len(segments), batch_size):
                    batch = segments[i:i+batch_size]
                    probs = torch.softmax(self.model(batch), dim=1)
                    all_probs.append(probs)
            
            # 5. 合并所有片段的预测结果
            all_probs = torch.cat(all_probs, dim=0)
            avg_probs = all_probs.mean(dim=0)  # 平均所有片段的概率
            predicted_index = torch.argmax(avg_probs).item()
            
            return CLASS_MAPPING[predicted_index], predicted_index, avg_probs.tolist()
            
        except Exception as e:
            raise RuntimeError(f"音频处理失败: {str(e)}")

    def predict_genre(self, audio_index):
        """预测指定音频的风格类别"""
        with torch.no_grad():
            spectrogram, true_label, file_name = self.dataset[audio_index]
            prediction = self.model(spectrogram.unsqueeze(0))
            predicted_index = torch.argmax(prediction).item()
            return CLASS_MAPPING[predicted_index], predicted_index
    
    def recommend_similar(self, target_index, num_recommendations=5):
        """
        推荐与目标风格相似的歌曲
        :param target_index: 目标风格索引(0-9)
        :param num_recommendations: 推荐数量
        :return: 推荐歌曲列表，包含字典{name, style, similarity, index}
        """
        # 随机选取候选歌曲
        candidate_indices = random.sample(range(len(self.dataset)), min(20, len(self.dataset)))
        recommendations = []
        
        for idx in candidate_indices:
            with torch.no_grad():
                spectrogram, true_label, file_name = self.dataset[idx]
                prediction = self.model(spectrogram.unsqueeze(0))
                similarity = prediction[0][target_index].item()
                
                recommendations.append({
                    'name': CLASS_MAPPING[true_label],
                    'style': true_label,
                    'similarity': similarity,
                    'index': idx,
                    'file_name': file_name
                })
        
        # 按相似度排序并返回前N个
        return sorted(recommendations, key=lambda x: x['similarity'], reverse=True)[:num_recommendations]

class MusicRecommenderGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.recommender = MusicRecommender()
        self.init_ui()
        self.setWindowTitle("音乐推荐应用模块——自然晴-xl")
        icon_path="logo.png"
        self.setWindowIcon(QIcon(icon_path))
        self.setMinimumSize(1200, 800)
        
        # 初始化媒体播放器
        self.player = QMediaPlayer()
        self.player.setVolume(50)
        self.playback_timer = QTimer()
        self.playback_timer.setInterval(1000)  # 1秒更新一次
        self.playback_timer.timeout.connect(self.update_playback_position)
        
        # 当前播放状态
        self.current_playing_index = None
        self.is_playing = False
        self.custom_file_path = None
        self.current_playing_file = ""
        self.recommended_songs = []  # 存储当前推荐的所有歌曲
    
    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # 顶部控制区域
        control_layout = QHBoxLayout()
        
        # 添加自定义文件选择按钮
        self.custom_file_btn = QPushButton("选择自定义音乐")
        self.custom_file_btn.setStyleSheet("""
            QPushButton {
                padding: 8px 15px; 
                background-color: #9C27B0; 
                color: white;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #7B1FA2;
            }
        """)
        self.custom_file_btn.clicked.connect(self.select_custom_file)
        control_layout.addWidget(self.custom_file_btn)
        
        self.random_btn = QPushButton("随机选择歌曲")
        self.random_btn.setStyleSheet("""
            QPushButton {
                padding: 8px 15px; 
                background-color: #4CAF50; 
                color: white;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        self.random_btn.clicked.connect(self.random_select)
        control_layout.addWidget(self.random_btn)
        
        self.recommend_btn = QPushButton("推荐相似歌曲")
        self.recommend_btn.setStyleSheet("""
            QPushButton {
                padding: 8px 15px; 
                background-color: #2196F3; 
                color: white;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #0b7dda;
            }
            QPushButton:disabled {
                background-color: #cccccc;
            }
        """)
        self.recommend_btn.clicked.connect(self.recommend_similar)
        self.recommend_btn.setEnabled(False)
        control_layout.addWidget(self.recommend_btn)
        
        layout.addLayout(control_layout)
        
        # 播放控制区域
        playback_layout = QVBoxLayout()
        
        # 当前播放文件显示
        self.now_playing_label = QLabel("当前播放: 无")
        self.now_playing_label.setStyleSheet("font-size: 14px; color: #333; font-weight: bold;")
        playback_layout.addWidget(self.now_playing_label)
        
        # 播放控制按钮区域
        playback_controls = QHBoxLayout()
        
        self.play_btn = QPushButton("播放")
        self.play_btn.setStyleSheet("""
            QPushButton {
                padding: 8px 15px; 
                background-color: #4CAF50; 
                color: white;
                border-radius: 5px;
                min-width: 80px;
            }
        """)
        self.play_btn.clicked.connect(self.toggle_playback)
        playback_controls.addWidget(self.play_btn)
        
        self.pause_btn = QPushButton("暂停")
        self.pause_btn.setStyleSheet("""
            QPushButton {
                padding: 8px 15px; 
                background-color: #FF9800; 
                color: white;
                border-radius: 5px;
                min-width: 80px;
            }
        """)
        self.pause_btn.clicked.connect(self.pause_playback)
        playback_controls.addWidget(self.pause_btn)
        
        self.stop_btn = QPushButton("停止")
        self.stop_btn.setStyleSheet("""
            QPushButton {
                padding: 8px 15px; 
                background-color: #F44336; 
                color: white;
                border-radius: 5px;
                min-width: 80px;
            }
        """)
        self.stop_btn.clicked.connect(self.stop_playback)
        playback_controls.addWidget(self.stop_btn)
        
        # 音量控制
        volume_layout = QHBoxLayout()
        volume_layout.addWidget(QLabel("音量:"))
        self.volume_slider = QSlider(Qt.Horizontal)
        self.volume_slider.setRange(0, 100)
        self.volume_slider.setValue(50)
        self.volume_slider.setFixedWidth(100)
        self.volume_slider.valueChanged.connect(self.set_volume)
        volume_layout.addWidget(self.volume_slider)
        playback_controls.addLayout(volume_layout)
        
        playback_layout.addLayout(playback_controls)
        
        # 播放进度条
        progress_layout = QHBoxLayout()
        progress_layout.addWidget(QLabel("进度:"))
        
        self.position_slider = QSlider(Qt.Horizontal)
        self.position_slider.setRange(0, 100)
        self.position_slider.sliderMoved.connect(self.set_position)
        progress_layout.addWidget(self.position_slider, stretch=4)
        
        # 时间标签
        self.time_label = QLabel("00:00 / 00:00")
        self.time_label.setStyleSheet("min-width: 100px;")
        progress_layout.addWidget(self.time_label)
        
        playback_layout.addLayout(progress_layout)
        
        layout.addLayout(playback_layout)
        
        # 主内容区域
        content_layout = QHBoxLayout()
        
        # 左侧 - 当前歌曲信息
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        self.current_song_label = QLabel("当前歌曲: 未选择")
        self.current_song_label.setStyleSheet("font-size: 20px; font-weight: bold;")
        left_layout.addWidget(self.current_song_label)
        
        self.file_name_label = QLabel("文件名: -")
        self.file_name_label.setStyleSheet("font-size: 18px; color: #666;")
        left_layout.addWidget(self.file_name_label)
        
        self.pediction_label = QLabel("预测风格: -")
        self.pediction_label.setStyleSheet("font-size: 18px;")
        left_layout.addWidget(self.pediction_label)
        
        # 风格概率表
        self.prob_table = QTableWidget()
        self.prob_table.setColumnCount(2)
        self.prob_table.setHorizontalHeaderLabels(["音乐风格", "概率"])
        self.prob_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.prob_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.prob_table.setEditTriggers(QTableWidget.NoEditTriggers)
        left_layout.addWidget(self.prob_table)
        
        content_layout.addWidget(left_panel, stretch=1)
        
        # 右侧 - 推荐结果
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        self.recommendation_label = QLabel("推荐结果")
        self.recommendation_label.setStyleSheet("font-size: 20px; font-weight: bold;")
        right_layout.addWidget(self.recommendation_label)
        
        # 推荐结果表格
        self.recommend_table = QTableWidget()
        self.recommend_table.setColumnCount(4)
        self.recommend_table.setHorizontalHeaderLabels(["文件名", "风格", "相似度", "索引"])
        self.recommend_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.recommend_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.recommend_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.recommend_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self.recommend_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.recommend_table.cellDoubleClicked.connect(self.play_selected_song)
        right_layout.addWidget(self.recommend_table)
        
        content_layout.addWidget(right_panel, stretch=1)
        layout.addLayout(content_layout, stretch=1)
        
        # 初始化状态
        self.current_index = None
        self.current_prediction = None
    
    def select_custom_file(self):
        """选择自定义音乐文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择音乐文件", "", 
            "音频文件 (*.mp3 *.wav *.ogg *.flac);;所有文件 (*)"
        )
        
        if file_path:
            self.custom_file_path = file_path
            self.current_index = None  # 表示当前是自定义文件
            self.current_prediction = None
            
            # 更新文件名显示
            file_name = file_path.split('/')[-1]
            self.current_song_label.setText("当前歌曲: 自定义音乐")
            self.file_name_label.setText(f"文件名: {file_name}")
            self.current_playing_file = file_name
            self.update_now_playing_label()
            
            # 预测风格
            self.predict_custom_file(file_path)
            
            # 启用推荐按钮
            self.recommend_btn.setEnabled(True)
            
            # 清空之前的推荐结果
            self.recommend_table.setRowCount(0)
            self.recommended_songs = []
            
            # 停止当前播放
            self.stop_playback()
    
    def update_now_playing_label(self):
        """更新当前播放标签"""
        if self.current_playing_file:
            self.now_playing_label.setText(f"当前播放: {self.current_playing_file}")
        else:
            self.now_playing_label.setText("当前播放: 无")
    
    def predict_custom_file(self, file_path):
        """预测自定义文件的音乐风格"""
        try:
            # 使用新的分段预测方法
            predicted_genre, predicted_index, probs = self.recommender.predict_custom_file(file_path)
            
            # 更新UI
            self.pediction_label.setText(f"预测风格: {predicted_genre}")
            self.current_prediction = predicted_index
            
            # 更新概率表
            self.update_probability_table(probs, predicted_index)
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"无法处理音频文件:\n{str(e)}")
    
    def random_select(self):
        """随机选择一首歌曲并预测其风格"""
        self.custom_file_path = None  # 清除自定义文件
        self.current_index = random.randint(0, len(self.recommender.dataset)-1)
        predicted_genre, genre_index = self.recommender.predict_genre(self.current_index)
        self.current_prediction = genre_index
        
        # 获取文件名
        _, _, file_name = self.recommender.dataset[self.current_index]
        self.current_playing_file = file_name
        self.update_now_playing_label()
        
        # 更新UI显示
        self.current_song_label.setText(f"当前歌曲: #{self.current_index}")
        self.file_name_label.setText(f"文件名: {file_name}")
        self.pediction_label.setText(f"预测风格: {predicted_genre}")
        
        # 更新概率表
        spectrogram, _, _ = self.recommender.dataset[self.current_index]
        with torch.no_grad():
            prediction = self.recommender.model(spectrogram.unsqueeze(0))
            probs = torch.softmax(prediction, dim=1).squeeze().tolist()
        self.update_probability_table(probs, genre_index)
        
        # 启用推荐按钮
        self.recommend_btn.setEnabled(True)
        
        # 清空之前的推荐结果
        self.recommend_table.setRowCount(0)
        self.recommended_songs = []
        
        # 停止当前播放
        self.stop_playback()
    
    def update_probability_table(self, probs, predicted_index):
        """更新风格概率表格"""
        self.prob_table.setRowCount(len(CLASS_MAPPING))
        
        for i, (genre, prob) in enumerate(zip(CLASS_MAPPING, probs)):
            # 添加表格项
            self.prob_table.setItem(i, 0, QTableWidgetItem(genre))
            self.prob_table.setItem(i, 1, QTableWidgetItem(f"{prob:.4f}"))
            
            # 高亮显示预测风格
            if i == predicted_index:
                for col in range(2):
                    self.prob_table.item(i, col).setBackground(QColor(STYLE_COLORS[genre]))
                    self.prob_table.item(i, col).setForeground(QColor("white"))
    
    def recommend_similar(self):
        """推荐相似歌曲"""
        if self.current_prediction is None:
            return
            
        # 无论是自定义音乐还是随机歌曲，都使用相同的推荐逻辑
        self.recommended_songs = self.recommender.recommend_similar(self.current_prediction)
        
        # 清空表格
        self.recommend_table.setRowCount(0)
        
        # 添加推荐歌曲到表格
        self.recommend_table.setRowCount(len(self.recommended_songs))
        for row, song in enumerate(self.recommended_songs):
            # 获取文件名
            file_name = song['file_name']
            
            # 添加表格项
            self.recommend_table.setItem(row, 0, QTableWidgetItem(file_name))
            self.recommend_table.setItem(row, 1, QTableWidgetItem(song['name']))
            self.recommend_table.setItem(row, 2, QTableWidgetItem(f"{song['similarity']:.4f}"))
            self.recommend_table.setItem(row, 3, QTableWidgetItem(str(song['index'])))
            
            # 设置行颜色
            for col in range(4):
                self.recommend_table.item(row, col).setBackground(QColor(STYLE_COLORS[song['name']]))
                self.recommend_table.item(row, col).setForeground(QColor("white"))
    
    def play_selected_song(self, row, col):
        """播放选中的歌曲"""
        if row < len(self.recommended_songs):
            song = self.recommended_songs[row]
            self.play_song(song['index'])
    
    def play_song(self, index):
        """播放指定索引的歌曲"""
        if index == -1 and self.custom_file_path is not None:
            # 播放自定义文件
            media_content = QMediaContent(QUrl.fromLocalFile(self.custom_file_path))
            file_name = self.custom_file_path.split('/')[-1]
        else:
            # 播放数据集中的文件
            _, _, file_path = self.recommender.dataset[index]
            media_content = QMediaContent(QUrl.fromLocalFile(file_path))
            file_name = file_path.split('/')[-1]
        
        # 更新当前播放文件名
        self.current_playing_file = file_name
        self.update_now_playing_label()
        
        # 设置媒体内容
        self.player.setMedia(media_content)
        
        # 开始播放
        self.player.play()
        self.playback_timer.start()
        
        # 更新状态
        self.current_playing_index = index
        self.is_playing = True
        
        # 更新UI
        self.play_btn.setText("播放中")
    
    def toggle_playback(self):
        """切换播放状态"""
        if self.current_index is None and self.custom_file_path is None:
            return
            
        if self.is_playing:
            if self.player.state() == QMediaPlayer.PausedState:
                self.player.play()
                self.play_btn.setText("播放中")
                self.is_playing = True
            else:
                self.player.pause()
                self.play_btn.setText("继续")
                self.is_playing = False
        else:
            if self.custom_file_path is not None:
                self.play_song(-1)  # 播放自定义文件
            else:
                self.play_song(self.current_index)
    
    def pause_playback(self):
        """暂停播放"""
        if self.player.state() == QMediaPlayer.PlayingState:
            self.player.pause()
            self.play_btn.setText("继续")
            self.is_playing = False
    
    def stop_playback(self):
        """停止播放"""
        self.player.stop()
        self.playback_timer.stop()
        self.position_slider.setValue(0)
        self.time_label.setText("00:00 / 00:00")
        self.play_btn.setText("播放")
        self.is_playing = False
        self.current_playing_file = ""
        self.update_now_playing_label()
    
    def set_volume(self, value):
        """设置音量"""
        self.player.setVolume(value)
    
    def set_position(self, position):
        """设置播放位置"""
        if self.player.duration() > 0:
            self.player.setPosition(position * self.player.duration() // 100)
    
    def update_playback_position(self):
        """更新播放进度"""
        if self.player.duration() > 0:
            position = self.player.position()
            duration = self.player.duration()
            
            # 更新滑块位置
            self.position_slider.setValue(position * 100 // duration)
            
            # 更新时间标签
            current_time = self.format_time(position)
            total_time = self.format_time(duration)
            self.time_label.setText(f"{current_time} / {total_time}")
    
    def format_time(self, milliseconds):
        """格式化时间显示 (mm:ss)"""
        seconds = milliseconds // 1000
        minutes = seconds // 60
        seconds = seconds % 60
        return f"{minutes:02d}:{seconds:02d}"

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    # 设置全局字体
    font = app.font()
    font.setPointSize(10)
    app.setFont(font)
    
    window = MusicRecommenderGUI()
    window.show()
    sys.exit(app.exec_())