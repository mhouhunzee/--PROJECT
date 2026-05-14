"""
乳腺癌浸润性导管癌(IDC)分类器 - 带弹性正则化
使用L1/L2组合正则化（远离最低点用L1，接近用L2）
"""

import os
import re
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from collections import defaultdict

# 固定随机种子
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 配置参数
DATA_DIR = r"C:\Users\44199\.cache\kagglehub\datasets\paultimothymooney\breast-histopathology-images\versions\1\IDC_regular_ps50_idx5"
MODEL_PATH = r"C:\Users\44199\OneDrive\文档\学习\人工智能的编程基础\project\idc_model_elastic.pth"
BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 0.001
IMAGE_SIZE = 50
L1_LAMBDA = 1e-5  # L1正则化系数
L2_LAMBDA = 1e-4  # L2正则化系数
LOSS_THRESHOLD = 0.3  # 损失阈值，低于此值认为接近最低点


def parse_filename(filename):
    """从文件名解析患者ID、坐标和类别"""
    match = re.match(r'(\d+_idx\d+)_x(\d+)_y(\d+)_class(\d)\.png', filename)
    if match:
        return match.group(1), int(match.group(2)), int(match.group(3)), int(match.group(4))
    return None, None, None, None


def collect_data_samples(data_dir):
    """收集数据样本信息"""
    samples = []
    for patient_folder in os.listdir(data_dir):
        patient_path = os.path.join(data_dir, patient_folder)
        if not os.path.isdir(patient_path):
            continue
        for class_label in ['0', '1']:
            class_path = os.path.join(patient_path, class_label)
            if not os.path.exists(class_path):
                continue
            for filename in os.listdir(class_path):
                if not filename.endswith('.png'):
                    continue
                pid, x, y, label = parse_filename(filename)
                if pid is not None:
                    filepath = os.path.join(class_path, filename)
                    samples.append((pid, x, y, label, filepath))
    print(f"找到 {len(samples)} 个样本")
    return samples


class IDCDataset(Dataset):
    """IDC数据集"""
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        _, x, y, label, filepath = self.samples[idx]
        image = Image.open(filepath).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label, x, y


def split_by_patient(samples, train_ratio=0.7, val_ratio=0.1, test_ratio=0.2):
    """按患者划分数据集"""
    patient_samples = defaultdict(list)
    for sample in samples:
        patient_samples[sample[0]].append(sample)

    patient_ids = list(patient_samples.keys())
    train_patients, temp_patients = train_test_split(
        patient_ids, test_size=(1 - train_ratio), random_state=SEED
    )
    val_patients, test_patients = train_test_split(
        temp_patients, test_size=(test_ratio / (val_ratio + test_ratio)), random_state=SEED
    )

    train_samples = [s for pid in train_patients for s in patient_samples[pid]]
    val_samples = [s for pid in val_patients for s in patient_samples[pid]]
    test_samples = [s for pid in test_patients for s in patient_samples[pid]]

    print(f"训练集: {len(train_samples)} | 验证集: {len(val_samples)} | 测试集: {len(test_samples)}")
    return train_samples, val_samples, test_samples


class SimpleCNN(nn.Module):
    """CNN模型"""
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 6 * 6, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 2)
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def elastic_regularization(model, current_loss):
    """弹性正则化：远离最低点用L1，接近用L2"""
    l1_reg = 0
    l2_reg = 0
    for param in model.parameters():
        if param.requires_grad:
            l1_reg += torch.sum(torch.abs(param))
            l2_reg += torch.sum(param ** 2)

    if current_loss > LOSS_THRESHOLD:
        alpha = min(1.0, (current_loss - LOSS_THRESHOLD) / LOSS_THRESHOLD)
        reg_loss = alpha * L1_LAMBDA * l1_reg + (1 - alpha) * L2_LAMBDA * l2_reg
    else:
        beta = min(1.0, (LOSS_THRESHOLD - current_loss) / LOSS_THRESHOLD)
        reg_loss = (1 - beta) * L1_LAMBDA * l1_reg + beta * L2_LAMBDA * l2_reg

    return reg_loss


def calculate_metrics(all_labels, all_preds):
    """计算精确度和召回率"""
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    return precision, recall


def custom_score(precision, recall):
    """自定义评分：3倍召回率 + 1倍精确度"""
    return 3 * recall + precision


def evaluate_metrics(model, data_loader, device, criterion):
    """评估模型并返回各项指标"""
    model.eval()
    total_loss = 0.0
    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for images, labels, _, _ in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())

    avg_loss = total_loss / len(data_loader)
    precision, recall = calculate_metrics(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_probs)
    score = custom_score(precision, recall)

    return avg_loss, precision, recall, auc, score, all_labels, all_probs


def train_model(model, train_loader, val_loader, device):
    """训练模型"""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2)

    # 记录每轮指标
    history = {
        'train_loss': [], 'val_loss': [],
        'train_precision': [], 'val_precision': [],
        'train_recall': [], 'val_recall': [],
        'train_score': [], 'val_score': [],
        'train_auc': [], 'val_auc': []
    }

    best_score = -float('inf')
    best_epoch = 0

    print("\n开始训练...")
    print("=" * 100)
    print(f"{'Epoch':<6} {'Train Loss':<12} {'Val Loss':<12} {'Train P':<10} {'Val P':<10} {'Train R':<10} {'Val R':<10} {'Train S':<10} {'Val S':<10}")
    print("=" * 100)

    for epoch in range(EPOCHS):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_labels = []
        train_preds = []

        for images, labels, _, _ in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)

            # 添加弹性正则化
            reg_loss = elastic_regularization(model, loss.item())
            total_loss = loss + reg_loss

            total_loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_labels.extend(labels.cpu().numpy())
            train_preds.extend(predicted.cpu().numpy())

        train_avg_loss = train_loss / len(train_loader)
        train_precision, train_recall = calculate_metrics(train_labels, train_preds)
        train_score = custom_score(train_precision, train_recall)

        # 验证阶段
        val_avg_loss, val_precision, val_recall, val_auc, val_score, _, _ = evaluate_metrics(
            model, val_loader, device, criterion
        )

        # 记录历史
        history['train_loss'].append(train_avg_loss)
        history['val_loss'].append(val_avg_loss)
        history['train_precision'].append(train_precision)
        history['val_precision'].append(val_precision)
        history['train_recall'].append(train_recall)
        history['val_recall'].append(val_recall)
        history['train_score'].append(train_score)
        history['val_score'].append(val_score)

        print(f"{epoch+1:<6} {train_avg_loss:<12.4f} {val_avg_loss:<12.4f} {train_precision:<10.4f} {val_precision:<10.4f} {train_recall:<10.4f} {val_recall:<10.4f} {train_score:<10.4f} {val_score:<10.4f}")

        # 保存最佳模型（基于自定义评分）
        if val_score > best_score:
            best_score = val_score
            best_epoch = epoch + 1
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"  -> [保存最佳模型] Epoch {best_epoch}, Score={best_score:.4f} (3*R+P)")

        scheduler.step(val_avg_loss)

    print("=" * 100)
    print(f"训练完成! 最佳模型在第 {best_epoch} 轮，评分={best_score:.4f}")

    return model, history, best_epoch


def plot_training_history(history, save_path):
    """绘制训练历史图表"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    epochs = range(1, len(history['train_loss']) + 1)

    # Loss
    axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Train Loss')
    axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Val Loss')
    axes[0, 0].set_title('Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # Precision
    axes[0, 1].plot(epochs, history['train_precision'], 'b-', label='Train Precision')
    axes[0, 1].plot(epochs, history['val_precision'], 'r-', label='Val Precision')
    axes[0, 1].set_title('Precision')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Precision')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # Recall
    axes[0, 2].plot(epochs, history['train_recall'], 'b-', label='Train Recall')
    axes[0, 2].plot(epochs, history['val_recall'], 'r-', label='Val Recall')
    axes[0, 2].set_title('Recall')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('Recall')
    axes[0, 2].legend()
    axes[0, 2].grid(True)

    # Custom Score (3*R + P)
    axes[1, 0].plot(epochs, history['train_score'], 'b-', label='Train Score (3R+P)')
    axes[1, 0].plot(epochs, history['val_score'], 'r-', label='Val Score (3R+P)')
    axes[1, 0].set_title('Custom Score (3*Recall + Precision)')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Score')
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    # Combined Metrics
    ax2 = axes[1, 1]
    ax3 = ax2.twinx()
    ax2.plot(epochs, history['val_precision'], 'g-', label='Val Precision')
    ax2.plot(epochs, history['val_recall'], 'b-', label='Val Recall')
    ax3.plot(epochs, history['val_score'], 'r-', linewidth=2, label='Val Score (3R+P)')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Precision / Recall', color='k')
    ax3.set_ylabel('Score', color='r')
    ax2.set_title('Validation Metrics')
    ax2.legend(loc='upper left')
    ax3.legend(loc='upper right')
    ax2.grid(True)

    # Summary Table
    axes[1, 2].axis('off')
    summary_text = "Training Summary\n\n"
    summary_text += f"Best Epoch: {np.argmax(history['val_score']) + 1}\n"
    summary_text += f"Best Val Score: {max(history['val_score']):.4f}\n\n"
    summary_text += f"Final Train Loss: {history['train_loss'][-1]:.4f}\n"
    summary_text += f"Final Val Loss: {history['val_loss'][-1]:.4f}\n\n"
    summary_text += f"Final Val Precision: {history['val_precision'][-1]:.4f}\n"
    summary_text += f"Final Val Recall: {history['val_recall'][-1]:.4f}"
    axes[1, 2].text(0.1, 0.5, summary_text, fontsize=12, verticalalignment='center',
                    family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"训练历史图表已保存到: {save_path}")
    plt.close()


def plot_roc(labels, probs, dataset_name, save_path):
    """绘制ROC曲线"""
    fpr, tpr, _ = roc_curve(labels, probs)
    auc = roc_auc_score(labels, probs)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC (AUC = {auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC - {dataset_name}')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()


def main():
    print("=" * 60)
    print("乳腺癌IDC分类器 (弹性L1/L2正则化)")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")

    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    print("\n收集数据...")
    samples = collect_data_samples(DATA_DIR)

    print("\n划分数据集...")
    train_samples, val_samples, test_samples = split_by_patient(samples)

    train_loader = DataLoader(IDCDataset(train_samples, transform), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(IDCDataset(val_samples, transform), batch_size=BATCH_SIZE)
    test_loader = DataLoader(IDCDataset(test_samples, transform), batch_size=BATCH_SIZE)

    print("\n创建模型...")
    model = SimpleCNN().to(device)
    print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 训练
    model, history, best_epoch = train_model(model, train_loader, val_loader, device)

    # 绘制训练历史
    output_dir = r"C:\Users\44199\OneDrive\文档\学习\人工智能的编程基础\project"
    plot_training_history(history, os.path.join(output_dir, "training_history.png"))

    # 加载最佳模型并评估
    print("\n加载最佳模型进行评估...")
    model.load_state_dict(torch.load(MODEL_PATH))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    # 验证集评估
    val_loss, val_p, val_r, val_auc, val_score, val_labels, val_probs = evaluate_metrics(
        model, val_loader, device, criterion
    )
    print(f"\n验证集结果:")
    print(f"  Loss: {val_loss:.4f} | Precision: {val_p:.4f} | Recall: {val_r:.4f} | AUC: {val_auc:.4f} | Score: {val_score:.4f}")

    # 测试集评估
    test_loss, test_p, test_r, test_auc, test_score, test_labels, test_probs = evaluate_metrics(
        model, test_loader, device, criterion
    )
    print(f"\n测试集结果:")
    print(f"  Loss: {test_loss:.4f} | Precision: {test_p:.4f} | Recall: {test_r:.4f} | AUC: {test_auc:.4f} | Score: {test_score:.4f}")

    # 绘制ROC曲线
    plot_roc(val_labels, val_probs, "Val", os.path.join(output_dir, "roc_val_elastic.png"))
    plot_roc(test_labels, test_probs, "Test", os.path.join(output_dir, "roc_test_elastic.png"))

    # 保存结果
    with open(os.path.join(output_dir, "results_elastic.txt"), 'w') as f:
        f.write("弹性L1/L2正则化实验结果\n")
        f.write(f"L1系数: {L1_LAMBDA}, L2系数: {L2_LAMBDA}, 阈值: {LOSS_THRESHOLD}\n")
        f.write(f"最佳轮次: {best_epoch}\n\n")
        f.write(f"验证集 - Loss: {val_loss:.4f}, Precision: {val_p:.4f}, Recall: {val_r:.4f}, AUC: {val_auc:.4f}, Score: {val_score:.4f}\n")
        f.write(f"测试集 - Loss: {test_loss:.4f}, Precision: {test_p:.4f}, Recall: {test_r:.4f}, AUC: {test_auc:.4f}, Score: {test_score:.4f}\n")

    print("\n" + "=" * 60)
    print("完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
