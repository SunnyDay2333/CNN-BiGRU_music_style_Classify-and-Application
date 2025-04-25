# CNN-BiGRU_music_style_Classify-and-Application
# 基于并行CNN-BiGRU的音乐风格分类与简单应用

本仓库的数据集经过了预处理，预处理方式为：把GTZAN的音频文件按50%的overlap切分成3秒的小文件，一首30秒的音频会被切分成19个切片，这样大大扩充了数据集的大小，以便于神经网络模型能够更好的学习。
注意：由于存在overlap如果同一个音频的不同切片分别出现在了训练集和测试集中，那么可能会存在数据泄露的风险，事实也的确如此，作者在一开始划分时没有注意到，测试集的准确率一度碾压训练集（）所以本项目划分的数据集在划分时规避了这种情况，因此理论上不存在数据泄露。

## 代码说明-2025.4.25 ver

本次提交四个文件，`PreData.py`是构建数据集的方法，`bestmodel.py`是作者最终选定的模型（CNN-BiRNN），`train.py`是训练代码，主要是作者进行实验和消融时使用的，不保存模型，只记录训练过程，`train_save.py`是保存最佳模型的训练代码，保存test_loss值最小的作为最佳模型。这四个代码已经能复现结果，作者也尝试了其他模型（CNN-LSTM,CNN-SRU,纯CNN）这些代码就不上传了，其主要用作对比实验和消融实验。

## 注意注意

训练前，请保证你拥有数据集，它们保持如下的目录结构（当然，你也可以自己定义目录结构，只是记得在代码里修改路径）
GTZAN_processed/

├── split_80_20/

│   ├── test/                # 存放测试集音频文件（WAV格式）

│   └── train/               # 存放训练集音频文件（WAV格式）

├── test_index.csv           # 测试集索引（格式：file_path,label）

├── train_index.csv          # 训练集索引（格式：file_path,label）

PreData.py              # 数据集构建脚本（生成索引文件/音频预处理）

bestmodel.py            # 最终模型定义（CNN-BiRNN架构）

train.py                # 实验训练代码（不保存模型，仅记录过程）

train_save.py           # 正式训练代码（保存最佳模型参数）

现成数据集下载地址：

百度网盘链接：https://pan.baidu.com/s/1TixJaVoTiTya-uGbWKaZKA 

提取码：2920 
