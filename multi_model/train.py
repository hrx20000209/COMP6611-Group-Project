import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

from dataset import build_dataset
from model import MultiModalActionModel
import torchvision.transforms as T


def train_model(model, train_loader, val_loader, num_epochs=30, lr=1e-3, device='cuda', save_dir='./checkpoints'):
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        for radar, video, labels in pbar:
            radar = radar.to(device)
            video = video.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(radar, video)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * labels.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            # 实时更新tqdm显示
            current_acc = correct / total * 100
            current_loss = train_loss / total
            pbar.set_postfix({'loss': f'{current_loss:.3f}', 'acc': f'{current_acc:.3f}'})

        scheduler.step()

        train_acc = correct / total * 100
        train_loss = train_loss / total
        print(f"Train Loss: {train_loss:.3f}, Train Acc: {train_acc:.3f}")

        # Evaluate
        val_loss, val_acc = evaluate_model(model, val_loader, criterion, device)
        print(f"Val Loss: {val_loss:.3f}, Val Acc: {val_acc:.3f}")

        # 保存best checkpoint
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = os.path.join(save_dir, 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss
            }, save_path)
            print(f"Saved Best Model at Epoch {epoch + 1} with Val Acc: {val_acc:.3f}")

def evaluate_model(model, data_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        pbar = tqdm(data_loader, desc="[Validation]")
        for radar, video, labels in pbar:
            radar = radar.to(device)
            video = video.to(device)
            labels = labels.to(device)

            outputs = model(radar, video)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * labels.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            # 实时更新val进度条
            current_acc = correct / total * 100
            current_loss = val_loss / total
            pbar.set_postfix({'loss': f'{current_loss:.3f}', 'acc': f'{current_acc:.3f}'})

    val_acc = correct / total * 100
    val_loss = val_loss / total
    return val_loss, val_acc

if __name__ == "__main__":
    root_dir = '/home/hrx/Data/6611_Data'
    batch_size = 8
    num_epochs = 50

    train_transform = T.Compose([
        T.ToPILImage(),  # 注意如果原始数据是 np.array，要先转 PIL
        T.Resize((256, 256)),
        T.RandomResizedCrop(224, scale=(0.8, 1.0)),  # 随机裁剪后resize
        T.RandomHorizontalFlip(p=0.5),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])

    # 验证集 transform
    val_transform = T.Compose([
        T.ToPILImage(),
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])

    train_set, val_set, label_map = build_dataset(root_dir, train_ratio=0.8, radar_max_points=32, target_len=30, train_transform=train_transform, val_transform=val_transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=4)

    model = MultiModalActionModel(num_classes=len(label_map))
    train_model(model, train_loader, val_loader, num_epochs=num_epochs, device='cuda')
