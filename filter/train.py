import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import torchvision.transforms as transforms
from torchvision.datasets import Caltech101
import numpy as np
import pickle
import os
from PIL import Image
from tqdm import tqdm
import time

# 检查GPU是否可用，但我们会确保模型适合CPU运行
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on: {device}")

# 加载之前生成的软标签数据集
print("Loading soft label dataset...")
with open('caltech101_person_binary_soft_labels.pkl', 'rb') as f:
    binary_dataset = pickle.load(f)

# 定义一个轻量级的CNN模型，适合CPU运行
class LightweightCNN(nn.Module):
    def __init__(self):
        super(LightweightCNN, self).__init__()
        # 使用较少的特征图以保持模型轻量
        self.features = nn.Sequential(
            # 输入: 3x224x224
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),  # 16x112x112
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 16x56x56
            
            nn.Conv2d(16, 32, kernel_size=3, padding=1),  # 32x56x56
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 32x28x28
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # 64x28x28
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 64x14x14
            
            nn.Conv2d(64, 32, kernel_size=3, padding=1),  # 32x14x14
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 32x7x7
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, 1),
            nn.Sigmoid()  # 输出0-1之间的概率
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x.squeeze()

# 创建一个用于训练的数据集类，确保处理图像为3通道
class SoftLabelDataset(Dataset):
    def __init__(self, binary_data, root_dir='./data', transform=None):
        self.soft_labels = binary_data['soft_labels']
        self.true_labels = binary_data['true_labels']
        self.image_indices = binary_data['image_indices']
        self.class_names = binary_data['class_names']
        self.root_dir = root_dir
        self.transform = transform
        self.dataset = Caltech101(root=root_dir, download=False)
        
    def __len__(self):
        return len(self.soft_labels)
    
    def __getitem__(self, idx):
        # 获取原始图像
        orig_idx = self.image_indices[idx]
        img, _ = self.dataset[orig_idx]
        
        # 确保图像是RGB模式（3通道）
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # 应用变换
        if self.transform:
            img = self.transform(img)
        
        # 返回图像、软标签
        return img, self.soft_labels[idx]

# 数据增强和预处理
transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform_val = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 创建数据集和划分训练/验证集
full_dataset = SoftLabelDataset(binary_dataset, root_dir='./data', transform=transform_train)

# 分割数据集: 80% 训练, 20% 验证
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

# 创建一个新的验证数据集，确保使用验证用的变换
class ValidationDataset(Dataset):
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform
    
    def __len__(self):
        return len(self.subset)
    
    def __getitem__(self, idx):
        # 获取原始数据
        orig_idx = self.subset.indices[idx]
        img, _ = self.subset.dataset.dataset[orig_idx]
        soft_label = self.subset.dataset.soft_labels[orig_idx]
        
        # 确保图像是RGB模式
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # 应用验证变换
        if self.transform:
            img = self.transform(img)
        
        return img, soft_label

# 使用包装的验证数据集
val_dataset_wrapped = ValidationDataset(val_dataset, transform_val)

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset_wrapped, batch_size=32, shuffle=False, num_workers=2)

print(f"Training on {len(train_dataset)} samples, validating on {len(val_dataset)} samples")

# 创建模型
model = LightweightCNN()
model = model.to(device)

# 打印模型架构和参数数量
total_params = sum(p.numel() for p in model.parameters())
print(f"Model architecture:\n{model}")
print(f"Total parameters: {total_params:,}")

# 定义损失函数和优化器
# 使用MSE损失来处理软标签
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)

# 训练函数
def train(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    
    for images, soft_labels in tqdm(train_loader, desc="Training"):
        # 检查输入是否有效
        if images.shape[1] != 3:
            print(f"Warning: Encountered image with {images.shape[1]} channels instead of 3")
            continue
            
        images = images.to(device)
        soft_labels = soft_labels.float().to(device)
        
        # 清除梯度
        optimizer.zero_grad()
        
        # 前向传播
        outputs = model(images)
        
        # 计算损失
        loss = criterion(outputs, soft_labels)
        
        # 反向传播和优化
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    return running_loss / len(train_loader)

# 验证函数
def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    
    # 用于计算准确度
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, soft_labels in tqdm(val_loader, desc="Validating"):
            # 检查输入是否有效
            if images.shape[1] != 3:
                print(f"Warning: Encountered image with {images.shape[1]} channels instead of 3")
                continue
                
            images = images.to(device)
            soft_labels = soft_labels.float().to(device)
            
            # 前向传播
            outputs = model(images)
            
            # 计算损失
            loss = criterion(outputs, soft_labels)
            running_loss += loss.item()
            
            # 计算准确度（使用0.5作为阈值）
            predicted = (outputs > 0.5).float()
            true_labels = (soft_labels > 0.5).float()
            total += true_labels.size(0)
            correct += (predicted == true_labels).sum().item()
    
    accuracy = 100 * correct / total if total > 0 else 0
    avg_loss = running_loss / len(val_loader) if len(val_loader) > 0 else float('inf')
    
    return avg_loss, accuracy

# 训练模型
num_epochs = 20
best_val_loss = float('inf')

# 创建保存模型的目录
if not os.path.exists('models'):
    os.makedirs('models')

for epoch in range(num_epochs):
    print(f"\nEpoch {epoch+1}/{num_epochs}")
    
    # 训练循环
    train_loss = train(model, train_loader, criterion, optimizer, device)
    
    # 验证循环
    val_loss, val_accuracy = validate(model, val_loader, criterion, device)
    
    # 更新学习率
    scheduler.step(val_loss)
    
    # 打印结果
    print(f"Train Loss: {train_loss:.4f}")
    print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")
    
    # 保存最佳模型
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        # 保存模型到CPU，确保可在CPU上运行
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.cpu().state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
            'val_accuracy': val_accuracy
        }, 'models/person_classifier_best.pth')
        
        # 如果我们在GPU上训练，需要将模型移回设备
        model = model.to(device)
        print(f"Model saved with validation loss: {val_loss:.4f}")

# 保存最终模型
torch.save({
    'epoch': num_epochs,
    'model_state_dict': model.cpu().state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'val_loss': val_loss, 
    'val_accuracy': val_accuracy
}, 'models/person_classifier_final.pth')

print("Training completed!")

# 创建一个简单的推理函数，以便在CPU上使用
def load_and_run_model_on_cpu(image_path):
    # 加载模型
    model = LightweightCNN()
    checkpoint = torch.load('models/person_classifier_best.pth', map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # 图像预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 加载并处理图像
    image = Image.open(image_path).convert('RGB')  # 确保图像是RGB
    image = transform(image).unsqueeze(0)  # 添加batch维度
    
    # 检查图像通道数
    if image.shape[1] != 3:
        raise ValueError(f"Expected 3 channels, got {image.shape[1]}")
    
    # 推理
    with torch.no_grad():
        start_time = time.time()
        output = model(image)
        inference_time = time.time() - start_time
    
    person_probability = output.item()
    is_person = person_probability > 0.5
    
    result = {
        'is_person': bool(is_person),
        'probability': person_probability,
        'inference_time': inference_time
    }
    
    return result

# 使用示例
print("\nInference Demo:")
print("# 加载和使用训练好的模型 (CPU推理)")
print("result = load_and_run_model_on_cpu('path_to_image.jpg')")
print("print(f\"Is Person: {result['is_person']}\")")
print("print(f\"Probability: {result['probability']:.4f}\")")
print("print(f\"Inference time: {result['inference_time']*1000:.2f} ms\")")