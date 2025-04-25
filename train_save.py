from bestmodel import PRCNN_dpo
import torch
from PreData import GTZANDataset
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchaudio
from torch.utils.tensorboard import SummaryWriter  
from typing import Dict
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import pandas as pd
import torchvision

# 参数配置
ANNOTATIONS_FILE = "GTZAN_processed\\train_index.csv"  # 测试用训练集索引
ANNOTATIONS_FILE_test = "GTZAN_processed\\test_index.csv"  # 测试集索引
AUDIO_DIR = "GTZAN_processed"  # 数据集根目录
AUDIO_DIR_test = "GTZAN_processed"                         
SAMPLE_RATE = 22050
NUM_SAMPLES = 22050 * 3  # 3秒音频

# 设备检测
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")
 

gtzan = GTZANDataset(ANNOTATIONS_FILE,AUDIO_DIR,None,target_sample_rate=SAMPLE_RATE,num_samples=NUM_SAMPLES)
gtzan_test = GTZANDataset(ANNOTATIONS_FILE_test,AUDIO_DIR_test,None,target_sample_rate=SAMPLE_RATE,num_samples=NUM_SAMPLES)


train_dataloader = DataLoader(gtzan,batch_size=64,shuffle=True)
test_dataloader = DataLoader(gtzan_test,batch_size=64,shuffle=False)



model = PRCNN_dpo().to(device) 

optimizer = torch.optim.Adam(model.parameters(), lr=0.0001,weight_decay=0.01)



def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    optimizer: optim.Optimizer,
    device: str,
    num_epochs: int = 100,
    log_dir: str = '2025-4-25-bestmodel_testsave',
    save_path: str = 'best_model.pth'
) -> Dict[str, list]:
    """
    封装完整的模型训练流程
    
    参数:
        model: 要训练的模型
        train_loader: 训练数据加载器
        test_loader: 测试数据加载器
        optimizer: 优化器
        device: 训练设备
        num_epochs: 训练轮数
        log_dir: TensorBoard日志目录
        save_path: 模型保存路径
    
    返回:
        包含训练指标的字典
    """
    # 初始化TensorBoard
    writer = SummaryWriter(log_dir=log_dir)
    
    # 记录训练过程
    history = {
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': []
    }
    
    # 固定随机种子
    torch.manual_seed(128)
    
    # 初始化最好的测试损失为一个较大的值
    best_test_loss = float('inf')
    
    for epoch in range(num_epochs):
        print('started')
        # 训练阶段
        model.train()
        epoch_train_loss = 0.0
        correct_train = 0
        total_samples = 0
        
        for batch_idx, batch in enumerate(train_loader):
            inputs, targets = batch[0].to(device), batch[1].to(device)
            
            # 前向传播
            preds = model(inputs)
            loss = F.cross_entropy(preds, targets)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 记录训练信息
            epoch_train_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(preds.data, 1)
            correct_train += (predicted == targets).sum().item()
            total_samples += targets.size(0)
        
        # 计算训练指标
        avg_train_loss = epoch_train_loss / len(train_loader.dataset)
        train_accuracy = 100 * correct_train / total_samples
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(train_accuracy)
        
        # 评估阶段
        model.eval()
        epoch_test_loss = 0.0
        correct_test = 0
        total_samples = 0
        
        # 清空上一轮的预测结果
        all_targets = []
        all_preds = []
        
        with torch.no_grad():
            for test_batch in test_loader:
                inputs, targets = test_batch[0].to(device), test_batch[1].to(device)
                preds = model(inputs)
                test_loss = F.cross_entropy(preds, targets)
                
                epoch_test_loss += test_loss.item() * inputs.size(0)
                _, predicted = torch.max(preds.data, 1)
                correct_test += (predicted == targets).sum().item()
                total_samples += inputs.size(0)
                
                # 收集预测结果和真实标签
                all_targets.extend(targets.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())
        
        # 计算测试指标
        avg_test_loss = epoch_test_loss / len(test_loader.dataset)
        test_accuracy = 100 * correct_test / total_samples
        history['test_loss'].append(avg_test_loss)
        history['test_acc'].append(test_accuracy)
        
        # 记录到TensorBoard
        writer.add_scalar('Loss/train', avg_train_loss, epoch)
        writer.add_scalar('Accuracy/train', train_accuracy, epoch)
        writer.add_scalar('Loss/test', avg_test_loss, epoch)
        writer.add_scalar('Accuracy/test', test_accuracy, epoch)
        
        # 打印epoch总结
        print(f"\nEpoch: {epoch+1}/{num_epochs} | "
              f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.2f}% | "
              f"Test Loss: {avg_test_loss:.4f}, Test Acc: {test_accuracy:.2f}%\n")
        
        # 如果当前测试损失更小，则保存模型
        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            if save_path:
                torch.save(model.state_dict(), save_path)
                print(f"Saved new best model with Test Loss: {best_test_loss:.4f}")
        
    writer.close()
    return history

if __name__ == "__main__":
    # 训练模型
    train_model(model, train_dataloader, test_dataloader, optimizer=optimizer, device='cuda', num_epochs=300, save_path='best_model.pth')
