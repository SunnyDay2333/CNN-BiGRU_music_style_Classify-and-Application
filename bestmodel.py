import torch
import torch.nn as nn
import torch.nn.functional as F
import torchsummary as summary

class PRCNN_dpo(nn.Module):
    def __init__(self, num_classes=10, dropout_prob=0.5):
        super(PRCNN_dpo, self).__init__()
        
        # CNN Block
        self.cnn_block = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),  # 输入通道1，输出16
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=4, stride=4),
            
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=4, stride=4),
        )
        
        # BI-RNN Block
        self.birnn_block = nn.Sequential(
            nn.Linear(513, 256),  # 嵌入层
            nn.MaxPool1d(kernel_size=2, stride=2),  # 变为128x128
        )
        self.birnn = nn.GRU(input_size=128, hidden_size=128, 
                           num_layers=1, bidirectional=True, batch_first=True)
        
        # Dropout layers
        self.dropout = nn.Dropout(p=dropout_prob)  # 添加Dropout
        
        # Classifier Block
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),  # 合并CNN和RNN的输出
            nn.Dropout(p=dropout_prob),  # 在全连接层后添加Dropout
            nn.Linear(256, 10),    # 输出10类
            nn.Softmax(dim=1)
        )
    
    def forward(self, x):
        # 输入x的形状应该是(batch_size, 1, 128, 513) - STFT频谱图
        
        # CNN路径
        cnn_out = self.cnn_block(x)
        cnn_out = cnn_out.view(cnn_out.size(0), -1)  # 展平
        
        # Dropout后再传递到下一层
        cnn_out = self.dropout(cnn_out)
        
        # BI-RNN路径
        # 首先处理频谱图
        birnn_in = x.squeeze(1)  # 移除通道维度 (batch, 128, 513)
        birnn_in = self.birnn_block(birnn_in)  # 处理后变为(batch, 128, 128)
        
        # 转置维度以适应GRU输入 (batch, seq_len, features)
        birnn_in = birnn_in.transpose(1, 2)  # (batch, 128, 128)
        
        # 双向RNN
        birnn_out, _ = self.birnn(birnn_in)
        
        # 取最后一个时间步的输出
        birnn_out = birnn_out[:, -1, :]  # (batch, 256)
        
        # Dropout后再传递到下一层
        birnn_out = self.dropout(birnn_out)
        
        # 合并两个路径的特征
        combined = torch.cat((cnn_out, birnn_out), dim=1)
        
        # 分类器
        output = self.classifier(combined)
        
        return output


