import os
import pandas as pd

# 数据集路径
dataset_path = "GTZAN"

# 类别文件夹名称和对应的数字标签
class_mapping = {
    'blues': 0,
    'classical': 1,
    'country': 2,
    'disco': 3,
    'hiphop': 4,
    'jazz': 5,
    'metal': 6,
    'pop': 7,
    'reggae': 8,
    'rock': 9
}

# 收集所有WAV文件路径和标签
file_paths = []
labels = []

# 遍历每个类别文件夹
for class_name, label in class_mapping.items():
    class_dir = os.path.join(dataset_path, class_name)
    
    # 检查文件夹是否存在
    if not os.path.isdir(class_dir):
        print(f"警告: 类别文件夹 '{class_dir}' 不存在")
        continue
    
    # 遍历文件夹中的WAV文件
    for filename in os.listdir(class_dir):
        if filename.endswith('.wav'):
            file_path = os.path.join(class_dir, filename)
            file_paths.append(file_path)
            labels.append(label)

# 创建DataFrame并保存为CSV
data = {'file_path': file_paths, 'label': labels}
df = pd.DataFrame(data)

# 保存为CSV文件
output_csv = "gtzan_dataset.csv"
df.to_csv(output_csv, index=False)

print(f"成功生成标注文件: {output_csv}")
print(f"共处理 {len(df)} 个音频文件")
print("类别映射关系:")
for name, label in class_mapping.items():
    print(f"{name}: {label}")