'''这段代码主要是预测算法的设计，在终端显示，用来调试的，
其主要功能已经部署到GUI界面了，因此这个代码实际没啥用了'''


import random
import torch
from net import PRCNN_dpo
from paperPreData import GTZANDataset

# 配置参数
ANNOTATIONS_FILE = "gtzan_dataset.csv"  # 标签文件路径
AUDIO_DIR = ""  # 音频文件根目录
SAMPLE_RATE = 22050  # 采样率
NUM_SAMPLES = 22050 * 3  # 3秒音频
MODEL_PATH = "best_model_final.pth"  # 模型路径

# 音乐类别映射
CLASS_MAPPING = [
    'blues', 'classical', 'country', 'disco', 'hiphop',
    'jazz', 'metal', 'pop', 'reggae', 'rock'
]

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
    
    def predict_genre(self, audio_index):
        """预测指定音频的风格类别"""
        with torch.no_grad():
            spectrogram, _, _ = self.dataset[audio_index]
            prediction = self.model(spectrogram.unsqueeze(0))
            predicted_index = torch.argmax(prediction).item()
            return CLASS_MAPPING[predicted_index], predicted_index
    
    def recommend_similar(self, target_index, num_recommendations=5):
        """
        推荐与目标风格相似的歌曲
        :param target_index: 目标风格索引(0-9)
        :param num_recommendations: 推荐数量
        :return: 推荐歌曲列表，包含字典{name, style, similarity}
        """
        # 随机选取候选歌曲
        candidate_indices = random.sample(range(len(self.dataset)), 20)
        recommendations = []
        
        for idx in candidate_indices:
            with torch.no_grad():
                spectrogram, true_label, _ = self.dataset[idx]
                prediction = self.model(spectrogram.unsqueeze(0))
                similarity = prediction[0][target_index].item()
                
                recommendations.append({
                    'name': CLASS_MAPPING[true_label],
                    'style': true_label,
                    'similarity': similarity,
                    'index': idx
                })
        
        # 按相似度排序并返回前N个
        return sorted(recommendations, key=lambda x: x['similarity'], reverse=True)[:num_recommendations]

def main():
    recommender = MusicRecommender()
    
    # 示例：随机选择一首歌预测其风格
    test_index = random.randint(0, len(recommender.dataset)-1)
    predicted_genre, genre_index = recommender.predict_genre(test_index)
    print(f"\n测试歌曲 #{test_index} 的预测风格: {predicted_genre} (类别 {genre_index})")
    
    # 推荐相似歌曲
    print(f"\n推荐与 {predicted_genre} 相似的歌曲:")
    similar_songs = recommender.recommend_similar(genre_index)
    
    for i, song in enumerate(similar_songs, 1):
        print(f"{i}. {song['name']} (相似度: {song['similarity']:.4f}, 索引: {song['index']})")

if __name__ == "__main__":
    main()