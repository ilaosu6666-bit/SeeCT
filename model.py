from tqdm import tqdm
import os
import json
import random
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights

import cv2


# =========================
# 1. 基础配置
# =========================
root_dir = r"D:\CTapp\cases"  # 你的CT总目录
lesion_dir = os.path.join(root_dir, "Lesion")   # 有病灶
normal_dir = os.path.join(root_dir, "Normal", "Normal")   # 正常

loss_fig_root = r".\loss_fig"
model_parameter_root = r".\model_parameter"

for folder in [loss_fig_root, model_parameter_root]:
    if not os.path.exists(folder):
        os.makedirs(folder)
        print(f"创建文件夹: {folder}")

net_name = "resnet18_ct_lesion_normal"

# 训练参数
image_size = 224
batch_size = 16
num_epochs = 50
learning_rate = 1e-4
train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15
random_seed = 42

# 类别映射
class_to_idx = {
    "Normal": 0,
    "Lesion": 1
}
idx_to_class = {v: k for k, v in class_to_idx.items()}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("当前设备：", device)


# =========================
# 2. 固定随机种子
# =========================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(random_seed)


# =========================
# 3. 收集所有图片路径
# =========================
def collect_image_paths(folder, label_name):
    valid_ext = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
    image_list = []

    if not os.path.exists(folder):
        raise FileNotFoundError(f"文件夹不存在: {folder}")

    for root, dirs, files in os.walk(folder):
        for filename in files:
            if filename.lower().endswith(valid_ext):
                image_path = os.path.join(root, filename)
                image_list.append((image_path, class_to_idx[label_name]))

    return image_list


# =========================
# 4. 自定义数据集
# =========================
class CTDataset(Dataset):
    def __init__(self, samples, transform=None):
        """
        samples: [(image_path, label), ...]
        """
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, label = self.samples[idx]

        try:
            image = Image.open(image_path).convert("RGB")  # 灰度CT也转3通道，适配ResNet
        except Exception as e:
            raise RuntimeError(f"图片读取失败: {image_path}, 错误: {e}")

        if self.transform is not None:
            image = self.transform(image)

        label = torch.tensor(label, dtype=torch.long)
        return image, label


# =========================
# 5. 数据增强 / 预处理
# =========================
train_transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

val_test_transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


# =========================
# 6. 构建数据集
# =========================
lesion_samples = collect_image_paths(lesion_dir, "Lesion")
normal_samples = collect_image_paths(normal_dir, "Normal")
all_samples = lesion_samples + normal_samples

if len(all_samples) == 0:
    raise ValueError("没有找到任何图片，请检查数据路径。")

print(f"有病灶图片数量: {len(lesion_samples)}")
print(f"正常图片数量: {len(normal_samples)}")
print(f"总图片数量: {len(all_samples)}")

random.shuffle(all_samples)

full_dataset = CTDataset(all_samples, transform=None)

total_size = len(full_dataset)
train_size = int(total_size * train_ratio)
val_size = int(total_size * val_ratio)
test_size = total_size - train_size - val_size

train_subset, val_subset, test_subset = random_split(
    full_dataset,
    [train_size, val_size, test_size],
    generator=torch.Generator().manual_seed(random_seed)
)

# 重新包装，使不同子集使用不同transform
train_dataset = CTDataset([full_dataset.samples[i] for i in train_subset.indices], transform=train_transform)
val_dataset = CTDataset([full_dataset.samples[i] for i in val_subset.indices], transform=val_test_transform)
test_dataset = CTDataset([full_dataset.samples[i] for i in test_subset.indices], transform=val_test_transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

print(f"训练集数量: {len(train_dataset)}")
print(f"验证集数量: {len(val_dataset)}")
print(f"测试集数量: {len(test_dataset)}")


# =========================
# 7. 定义模型
# =========================
def build_model():
    model = resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, 2)
    return model




net = build_model().to(device)


# =========================
# 7.5. Grad-CAM 辅助类 & 训练历史保存
# =========================
class GradCAMHelper:
    def __init__(self, model):
        self.model = model
        self.feature_maps = []
        self.gradients = []
        self._register_hooks()

    def _register_hooks(self):
        target_layer = self.model.layer4[-1]

        def forward_hook(module, inputs, output):
            self.feature_maps.clear()
            self.feature_maps.append(output)

        def backward_hook(module, grad_input, grad_output):
            self.gradients.clear()
            self.gradients.append(grad_output[0])

        target_layer.register_forward_hook(forward_hook)
        target_layer.register_full_backward_hook(backward_hook)

    def generate(self, image_tensor, class_idx):
        self.model.zero_grad(set_to_none=True)
        output = self.model(image_tensor)
        score = output[:, class_idx]
        score.backward(retain_graph=True)
        fmap = self.feature_maps[-1][0]
        grad = self.gradients[-1][0]
        weights = torch.mean(grad, dim=(1, 2))
        cam = torch.zeros(fmap.shape[1:], dtype=torch.float32, device=fmap.device)
        for i, w in enumerate(weights):
            cam += w * fmap[i]
        cam = torch.relu(cam)
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
        return cam.detach().cpu().numpy()


def overlay_heatmap(image, cam, alpha=0.45):
    image_np = np.array(image.convert("RGB"))
    cam_resized = cv2.resize(cam, (image_np.shape[1], image_np.shape[0]))
    heatmap = cv2.applyColorMap(np.uint8(cam_resized * 255), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = np.clip(image_np * (1 - alpha) + heatmap * alpha, 0, 255).astype(np.uint8)
    return overlay


def get_param_norms(model):
    norms = {}
    for name, param in model.named_parameters():
        if "weight" in name and ("layer" in name or "fc" in name):
            key = name.replace(".weight", "")
            norms[key] = float(torch.norm(param, p=2).item())
    return norms


def save_training_history(history, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)


def save_epoch_gradcam(model, fixed_images, device, epoch, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    gradcam_helper = GradCAMHelper(model)
    transform = val_test_transform
    for i, (img, label, name) in enumerate(fixed_images):
        img_tensor = transform(img).unsqueeze(0).to(device)
        cam = gradcam_helper.generate(img_tensor, label)
        np.save(os.path.join(save_dir, f"epoch_{epoch:03d}_{name}_cam.npy"), cam)
        if epoch == 1 or epoch % 5 == 0:
            overlay = overlay_heatmap(img, cam)
            Image.fromarray(overlay).save(
                os.path.join(save_dir, f"epoch_{epoch:03d}_{name}_overlay.png")
            )


# =========================
# 8. 训练与验证函数
# =========================
def train_one_epoch(model, loader, optimizer, criterion, device, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    progress_bar = tqdm(loader, desc=f"Epoch {epoch+1} [Train]", unit="batch")

    for images, labels in progress_bar:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = torch.argmax(outputs, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        progress_bar.set_postfix(
            loss=f"{loss.item():.4f}",
            acc=f"{100.0 * correct / total:.2f}%"
        )

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


@torch.no_grad()
def evaluate(model, loader, criterion, device, mode="Val"):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc=f"[{mode}]", unit="batch"):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        preds = torch.argmax(outputs, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


# =========================
# 9. 开始训练
# =========================
def train_model():
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)

    best_val_acc = 0.0
    train_loss_list = []
    val_loss_list = []
    train_acc_list = []
    val_acc_list = []
    param_norms_history = {"epochs": []}

    gradcam_save_dir = os.path.join(loss_fig_root, "gradcam_epochs")
    os.makedirs(gradcam_save_dir, exist_ok=True)

    # 选取固定样本用于Grad-CAM快照
    def _load_8bit_rgb(path):
        img = Image.open(path)
        if img.mode in ("I;16", "I;16B", "I"):
            arr = np.array(img, dtype=np.float32)
            arr_max = arr.max()
            if arr_max > 0:
                arr = (arr / arr_max * 255).astype(np.uint8)
            else:
                arr = arr.astype(np.uint8)
            img = Image.fromarray(arr)
        return img.convert("RGB")

    fixed_samples = []
    sample_lesion = [s for s in train_dataset.samples if s[1] == class_to_idx["Lesion"]]
    sample_normal = [s for s in train_dataset.samples if s[1] == class_to_idx["Normal"]]
    for s in sample_lesion[:3]:
        fixed_samples.append((_load_8bit_rgb(s[0]), s[1], "lesion"))
    for s in sample_normal[:3]:
        fixed_samples.append((_load_8bit_rgb(s[0]), s[1], "normal"))

    for epoch in range(num_epochs):
        train_loss, train_acc = train_one_epoch(net, train_loader, optimizer, criterion, device, epoch)
        val_loss, val_acc = evaluate(net, val_loader, criterion, device, mode="Val")

        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)
        train_acc_list.append(train_acc)
        val_acc_list.append(val_acc)

        # 记录参数范数
        norms = get_param_norms(net)
        param_norms_history["epochs"].append(epoch + 1)
        for k, v in norms.items():
            if k not in param_norms_history:
                param_norms_history[k] = []
            param_norms_history[k].append(v)

        print(f"\nEpoch [{epoch+1}/{num_epochs}]")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f}%")
        print(f"Val   Loss: {val_loss:.4f}, Val   Acc: {val_acc*100:.2f}%")

        # 保存训练历史
        history = {
            "epochs": list(range(1, epoch + 2)),
            "train_loss": train_loss_list,
            "val_loss": val_loss_list,
            "train_acc": train_acc_list,
            "val_acc": val_acc_list,
            "param_norms": param_norms_history,
        }
        save_training_history(history, os.path.join(loss_fig_root, "training_history.json"))

        # 保存每轮Grad-CAM快照
        net.eval()
        save_epoch_gradcam(net, fixed_samples, device, epoch + 1, gradcam_save_dir)
        net.train()

        # 每5轮保存checkpoint
        if (epoch + 1) % 5 == 0:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
            }, os.path.join(model_parameter_root, f"{net_name}_checkpoint_epoch_{epoch+1}.pt"))

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(net.state_dict(), os.path.join(model_parameter_root, f"{net_name}_best.pt"))
            print(f"已保存最佳模型，Val Acc = {best_val_acc*100:.2f}%")

    # 画loss曲线
    plt.figure(figsize=(8, 5))
    plt.plot(train_loss_list, label="Train Loss")
    plt.plot(val_loss_list, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.savefig(os.path.join(loss_fig_root, f"{net_name}_loss.png"))
    plt.close()

    # 画accuracy曲线
    plt.figure(figsize=(8, 5))
    plt.plot(train_acc_list, label="Train Acc")
    plt.plot(val_acc_list, label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.legend()
    plt.savefig(os.path.join(loss_fig_root, f"{net_name}_acc.png"))
    plt.close()

    print(f"\n训练完成。最佳验证准确率: {best_val_acc*100:.2f}%")


# =========================
# 10. 测试集评估
# =========================
@torch.no_grad()
def test_model():
    model_path = os.path.join(model_parameter_root, f"{net_name}_best.pt")
    if not os.path.exists(model_path):
        print("❌ 没有找到训练好的模型，请先训练。")
        return

    ckpt = torch.load(model_path, map_location=device)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        net.load_state_dict(ckpt["model_state_dict"])
    else:
        net.load_state_dict(ckpt)
    criterion = nn.CrossEntropyLoss()

    test_loss, test_acc = evaluate(net, test_loader, criterion, device, mode="Test")
    print(f"\n📌 Test Loss: {test_loss:.4f}")
    print(f"📌 Test Acc : {test_acc*100:.2f}%")


# =========================
# 11. 单张图片预测
# =========================
@torch.no_grad()
def predict_single_image(image_path):
    model_path = os.path.join(model_parameter_root, f"{net_name}_best.pt")
    if not os.path.exists(model_path):
        print("❌ 没有找到训练好的模型，请先训练。")
        return

    if not os.path.exists(image_path):
        print("❌ 图片路径不存在：", image_path)
        return

    ckpt = torch.load(model_path, map_location=device)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        net.load_state_dict(ckpt["model_state_dict"])
    else:
        net.load_state_dict(ckpt)
    net.eval()

    transform = val_test_transform

    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    outputs = net(image_tensor)
    probs = F.softmax(outputs, dim=1)
    pred_idx = torch.argmax(probs, dim=1).item()

    pred_class = idx_to_class[pred_idx]
    lesion_prob = probs[0][1].item()
    normal_prob = probs[0][0].item()

    print(f"预测结果: {pred_class}")
    print(f"Normal 概率: {normal_prob:.4f}")
    print(f"Lesion 概率: {lesion_prob:.4f}")

    plt.imshow(image, cmap='gray')
    plt.title(f"Prediction: {pred_class}")
    plt.axis("off")
    # plt.show()


# =========================
# 12. 主程序
# =========================
if __name__ == '__main__':
    print("程序启动成功")
    print("当前设备:", device)
    print("root_dir:", root_dir)
    print("Lesion目录存在吗:", os.path.exists(lesion_dir))
    print("Normal目录存在吗:", os.path.exists(normal_dir))

    training = True
    testing = False
    predicting = False

    if training:
        train_model()

    if testing:
        test_model()

    if predicting:
        image_path = r"D:\python\python\CT_data\Lesion\001.png"  # 改成你要预测的图片
        predict_single_image(image_path)