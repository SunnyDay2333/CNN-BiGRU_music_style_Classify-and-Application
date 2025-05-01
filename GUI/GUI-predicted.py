import sys
import torch
import torchaudio
from torch.nn import functional as F
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                            QHBoxLayout, QLabel, QPushButton, QFileDialog,
                            QTableWidget, QTableWidgetItem, QHeaderView,
                            QProgressBar, QMessageBox, QTabWidget, QScrollArea)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QPixmap, QIcon, QColor
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from net import PRCNN_dpo  # 导入模型类

# 模型配置
MODEL_PATH = "best_model_final.pth"
SAMPLE_RATE = 22050
CLASS_MAPPING = [
    'blues', 'classical', 'country', 'disco', 'hiphop',
    'jazz', 'metal', 'pop', 'reggae', 'rock'
]

class MusicRecommender:
    def __init__(self, model_path, sample_rate):
        self.model = PRCNN_dpo()
        self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
        self.model.eval()
        
        # STFT参数
        self.n_fft = 1024
        self.hop_length = 512
        self.win_length = 1024
        self.n_time_frames = 128

    def _load_audio(self, file_path):
        waveform, sr = torchaudio.load(file_path)
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        if sr != SAMPLE_RATE:
            waveform = torchaudio.transforms.Resample(sr, SAMPLE_RATE)(waveform)
        return waveform

    def _compute_spectrogram(self, waveform):
        stft = torch.stft(
            waveform, n_fft=self.n_fft, hop_length=self.hop_length,
            win_length=self.win_length, window=torch.hann_window(self.win_length),
            center=False, return_complex=True
        )
        spectrogram = torch.abs(stft).permute(0, 2, 1)
        spectrogram = torchaudio.functional.amplitude_to_DB(
            spectrogram, multiplier=20.0, amin=1e-10, db_multiplier=0.0, top_db=80.0
        )
        return spectrogram

    def segment_audio(self, file_path, overlap=32):
        waveform = self._load_audio(file_path)
        spectrogram = self._compute_spectrogram(waveform)
        
        segments = []
        step = self.n_time_frames - overlap
        for start in range(0, spectrogram.shape[1] - overlap, step):
            end = start + self.n_time_frames
            if end > spectrogram.shape[1]:
                segment = F.pad(
                    spectrogram[:, start:], 
                    (0, 0, 0, end - spectrogram.shape[1])
                )
            else:
                segment = spectrogram[:, start:end]
            segments.append(segment)
        return torch.stack(segments), waveform.shape[-1]

    def predict(self, segments, batch_size=32):
        all_probs = []
        with torch.no_grad():
            for i in range(0, len(segments), batch_size):
                batch = segments[i:i+batch_size]
                probs = torch.softmax(self.model(batch), dim=1)
                all_probs.append(probs)
        return torch.cat(all_probs, dim=0)

class AnalysisThread(QThread):
    update_progress = pyqtSignal(int)
    finished = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)

    def __init__(self, recommender, file_path):
        super().__init__()
        self.recommender = recommender
        self.file_path = file_path

    def run(self):
        try:
            segments, length = self.recommender.segment_audio(self.file_path)
            self.update_progress.emit(50)
            
            probs = self.recommender.predict(segments)
            self.update_progress.emit(80)
            
            result = {
                'genre': CLASS_MAPPING[torch.argmax(probs.mean(dim=0)).item()],
                'probs': probs.mean(dim=0).tolist(),
                'segments': segments,
                'duration': length / SAMPLE_RATE,
                'segment_probs': probs.tolist()  # 添加每个片段的概率
            }
            self.finished.emit(result)
        except Exception as e:
            self.error_occurred.emit(str(e))

class MusicGenreApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.recommender = MusicRecommender(MODEL_PATH, SAMPLE_RATE)
        self.init_ui()
        self.setWindowTitle("音乐预测应用模块——自然晴-xl")
        icon_path = "logo.png"
        self.setWindowIcon(QIcon(icon_path))
        self.setStyleSheet("background-color: #f0f0f0;")
        self.setMinimumSize(1200, 800)

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # 文件选择区域
        file_layout = QHBoxLayout()
        self.file_label = QLabel("未选择文件")
        self.file_label.setStyleSheet("font-size: 20px; color: #666; border: 1px solid #ddd; padding: 5px;")
        file_layout.addWidget(self.file_label, stretch=4)
        
        browse_btn = QPushButton("选择音频")
        browse_btn.setStyleSheet("padding: 5px 10px; background-color: #4CAF50; color: white; border-radius: 5px;")
        browse_btn.clicked.connect(self.browse_file)
        file_layout.addWidget(browse_btn)
        layout.addLayout(file_layout)

        # 控制区域
        control_layout = QHBoxLayout()
        self.analyze_btn = QPushButton("开始分析")
        self.analyze_btn.setStyleSheet("padding: 8px 15px; font-weight: bold; background-color: #2196F3; color: white; border-radius: 5px;")
        self.analyze_btn.clicked.connect(self.start_analysis)
        self.analyze_btn.setEnabled(False)
        control_layout.addWidget(self.analyze_btn)
        
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        control_layout.addWidget(self.progress)
        layout.addLayout(control_layout)

        # 创建标签页
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)

        # 第一页 - 主要结果
        self.main_tab = QWidget()
        self.tabs.addTab(self.main_tab, "主要结果")
        self.init_main_tab()

        # 第二页 - 所有频谱图
        self.spectrograms_tab = QWidget()
        self.tabs.addTab(self.spectrograms_tab, "所有频谱图")
        self.init_spectrograms_tab()

    def init_main_tab(self):
        layout = QVBoxLayout(self.main_tab)
        
        # 结果显示区域
        result_layout = QHBoxLayout()
        
        # 左侧面板 - 频谱图和主要结果
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        self.spec_canvas = QLabel()
        self.spec_canvas.setAlignment(Qt.AlignCenter)
        self.spec_canvas.setStyleSheet("border: 1px solid #ddd; background: white;")
        left_layout.addWidget(self.spec_canvas)
        
        self.result_display = QLabel()
        self.result_display.setAlignment(Qt.AlignCenter)
        self.result_display.setStyleSheet("font-size: 28px; margin: 10px 0;")
        left_layout.addWidget(self.result_display)
        result_layout.addWidget(left_panel, stretch=2)
        
        # 右侧面板 - 概率表格
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        self.prob_table = QTableWidget()
        self.prob_table.setColumnCount(2)
        self.prob_table.setHorizontalHeaderLabels(["音乐风格", "概率"])
        self.prob_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.prob_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.prob_table.setEditTriggers(QTableWidget.NoEditTriggers)
        right_layout.addWidget(self.prob_table)
        result_layout.addWidget(right_panel, stretch=1)
        
        layout.addLayout(result_layout, stretch=1)

    def init_spectrograms_tab(self):
        # 创建一个滚动区域来容纳所有频谱图
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        
        container = QWidget()
        self.spectrograms_layout = QVBoxLayout(container)
        self.spectrograms_layout.setAlignment(Qt.AlignTop)
        
        scroll.setWidget(container)
        
        layout = QVBoxLayout(self.spectrograms_tab)
        layout.addWidget(scroll)

    def browse_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择音频文件", "", 
            "音频文件 (*.wav *.mp3 *.flac);;所有文件 (*)"
        )
        if file_path:
            self.file_path = file_path
            self.file_label.setText(file_path.split('/')[-1])
            self.analyze_btn.setEnabled(True)
            self.clear_results()

    def clear_results(self):
        # 清除主标签页内容
        self.spec_canvas.clear()
        self.result_display.clear()
        self.prob_table.setRowCount(0)
        self.spec_canvas.setText("频谱图将在此显示")
        
        # 清除频谱图标签页内容
        for i in reversed(range(self.spectrograms_layout.count())): 
            self.spectrograms_layout.itemAt(i).widget().setParent(None)

    def start_analysis(self):
        self.analyze_btn.setEnabled(False)
        self.progress.setVisible(True)
        self.progress.setValue(0)
        
        self.thread = AnalysisThread(self.recommender, self.file_path)
        self.thread.update_progress.connect(self.progress.setValue)
        self.thread.finished.connect(self.show_results)
        self.thread.error_occurred.connect(self.show_error)
        self.thread.start()

    def show_results(self, result):
        self.progress.setVisible(False)
        self.analyze_btn.setEnabled(True)
        
        # 显示主要结果
        self.result_display.setText(
            f"<b>预测风格：</b>{result['genre']}<br>"
            f"<small>音频时长：{result['duration']:.2f}秒</small>"
        )
        
        # 显示第一个频谱图
        self.plot_spectrogram(result['segments'][0][0].numpy().T, self.spec_canvas)
        
        # 填充概率表格
        self.prob_table.setRowCount(len(CLASS_MAPPING))
        for i, (genre, prob) in enumerate(zip(CLASS_MAPPING, result['probs'])):
            self.prob_table.setItem(i, 0, QTableWidgetItem(genre))
            self.prob_table.setItem(i, 1, QTableWidgetItem(f"{prob:.4f}"))
            
            if genre == result['genre']:
                for col in range(2):
                    self.prob_table.item(i, col).setBackground(Qt.yellow)
        
        # 显示所有频谱图
        self.show_all_spectrograms(result['segments'], result['segment_probs'])

    def show_all_spectrograms(self, segments, segment_probs):
        # 清除旧内容
        for i in reversed(range(self.spectrograms_layout.count())): 
            self.spectrograms_layout.itemAt(i).widget().setParent(None)
        
        # 为每个片段创建频谱图
        for i, (segment, probs) in enumerate(zip(segments, segment_probs)):
            # 创建容器
            container = QWidget()
            container.setStyleSheet("border: 1px solid #ddd; margin: 5px; padding: 5px;")
            layout = QVBoxLayout(container)
            
            # 添加标题
            pred_genre = CLASS_MAPPING[torch.argmax(torch.tensor(probs)).item()]
            title = QLabel(f"片段 {i+1} - 预测风格: {pred_genre}")
            title.setStyleSheet("font-weight: bold;")
            layout.addWidget(title)
            
            # 创建频谱图
            canvas = QLabel()
            canvas.setAlignment(Qt.AlignCenter)
            canvas.setStyleSheet("background: white;")
            layout.addWidget(canvas)
            
            # 绘制频谱图
            self.plot_spectrogram(segment[0].numpy().T, canvas, figsize=(8, 3))
            
            # 添加概率信息
            prob_text = ", ".join([f"{CLASS_MAPPING[j]}: {p:.2f}" for j, p in enumerate(probs)])
            prob_label = QLabel(prob_text)
            prob_label.setStyleSheet("font-size: 10px; color: #666;")
            layout.addWidget(prob_label)
            
            self.spectrograms_layout.addWidget(container)

    def plot_spectrogram(self, spectrogram, target_widget, figsize=(6, 3)):
        plt.rcParams.update({'font.family': 'sans-serif'})
        fig, ax = plt.subplots(figsize=figsize)
        
        ax.imshow(spectrogram, origin='lower', aspect='auto', cmap='viridis')
        ax.set_title("1st Cut's Audio Spectrogram")
        ax.set_xlabel("Time Frame")
        ax.set_ylabel("Frequency Bin")
        
        plt.tight_layout()
        
        canvas = FigureCanvas(fig)
        canvas.draw()
        pixmap = QPixmap(canvas.grab().toImage())
        
        # 根据目标控件大小调整
        target_widget.setPixmap(pixmap.scaled(
            target_widget.width(), 
            target_widget.height(),
            Qt.KeepAspectRatio
        ))
        plt.close(fig)

    def show_error(self, message):
        self.progress.setVisible(False)
        self.analyze_btn.setEnabled(True)
        QMessageBox.critical(self, "错误", f"分析过程中发生错误:\n{message}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    # 设置全局字体
    font = app.font()
    font.setPointSize(10)
    app.setFont(font)
    
    window = MusicGenreApp()
    window.show()
    sys.exit(app.exec_())