import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import random
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Установка seed для воспроизводимости
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)

# ==========================================
# 1. ГЕНЕРАЦИЯ СИНТЕТИЧЕСКИХ ДАННЫХ (64x64) - БЕЗ CV2
# ==========================================

class MilitaryVehicleGenerator:
    """
    Генератор синтетических военных объектов для спутниковых снимков.
    Использует PIL вместо OpenCV.
    """
    
    def __init__(self, img_size=64):
        self.img_size = img_size
        self.classes = {
            0: 'T-72 (Tank)',
            1: 'BTR-82A (APC)', 
            2: 'Ural-4320 (Truck)'
        }
        
    def add_gaussian_noise(self, image: np.ndarray, sigma=0.1) -> np.ndarray:
        """Добавление гауссовского шума для имитации атмосферных помех"""
        noise = np.random.normal(0, sigma, image.shape)
        noisy = np.clip(image + noise, 0, 1)
        return noisy
    
    def generate_background(self) -> np.ndarray:
        """Генерация фона (земля, трава, песок)"""
        bg_type = random.choice(['grass', 'sand', 'urban', 'forest'])
        
        if bg_type == 'grass':
            base = np.array([0.2, 0.4, 0.15])
            noise = np.random.normal(0, 0.05, (self.img_size, self.img_size, 3))
            bg = base + noise
        elif bg_type == 'sand':
            base = np.array([0.7, 0.6, 0.4])
            noise = np.random.normal(0, 0.03, (self.img_size, self.img_size, 3))
            bg = base + noise
        elif bg_type == 'urban':
            base = np.array([0.4, 0.4, 0.4])
            noise = np.random.normal(0, 0.08, (self.img_size, self.img_size, 3))
            bg = base + noise
        else:  # forest
            base = np.array([0.1, 0.25, 0.1])
            noise = np.random.normal(0, 0.06, (self.img_size, self.img_size, 3))
            bg = base + noise
            
        return np.clip(bg, 0, 1)
    
    def draw_rotated_rectangle(self, draw, center, size, angle, color, outline=None):
        """Рисование повёрнутого прямоугольника"""
        cx, cy = center
        w, h = size
        angle_rad = np.radians(angle)
        
        # Углы прямоугольника
        corners = [
            (-w/2, -h/2), (w/2, -h/2), (w/2, h/2), (-w/2, h/2)
        ]
        
        # Поворот и смещение
        rotated_corners = []
        for x, y in corners:
            rx = x * np.cos(angle_rad) - y * np.sin(angle_rad) + cx
            ry = x * np.sin(angle_rad) + y * np.cos(angle_rad) + cy
            rotated_corners.append((rx, ry))
        
        draw.polygon(rotated_corners, fill=color, outline=outline)
        return rotated_corners
    
    def draw_tank(self, img_array: np.ndarray, center: Tuple[int, int], 
                  angle: float, scale: float = 1.0) -> np.ndarray:
        """Рисование T-72: корпус + башня + пушка"""
        img = Image.fromarray((img_array * 255).astype(np.uint8))
        draw = ImageDraw.Draw(img)
        cx, cy = int(center[0]), int(center[1])
        
        # Цвета (RGB 0-255)
        body_color = (38, 51, 38)  # Тёмно-зелёный
        turret_color = (51, 64, 51)
        gun_color = (25, 38, 25)
        
        w, h = int(30 * scale), int(15 * scale)
        
        # Корпус
        self.draw_rotated_rectangle(draw, (cx, cy), (w, h), angle, body_color)
        
        # Башня (круг)
        r = int(9 * scale)
        draw.ellipse([cx-r, cy-r, cx+r, cy+r], fill=turret_color)
        
        # Пушка
        rad = np.radians(angle)
        gun_length = int(20 * scale)
        x2 = int(cx + gun_length * np.cos(rad))
        y2 = int(cy + gun_length * np.sin(rad))
        draw.line([(cx, cy), (x2, y2)], fill=gun_color, width=max(2, int(3*scale)))
        
        # Деталь башни
        r2 = int(4 * scale)
        draw.ellipse([cx-r2-2, cy-r2-2, cx+r2-2, cy+r2-2], fill=(25, 30, 25))
        
        return np.array(img).astype(np.float32) / 255.0
    
    def draw_btr(self, img_array: np.ndarray, center: Tuple[int, int],
                 angle: float, scale: float = 1.0) -> np.ndarray:
        """Рисование BTR-82A: колёсная база, корпус, башня"""
        img = Image.fromarray((img_array * 255).astype(np.uint8))
        draw = ImageDraw.Draw(img)
        cx, cy = int(center[0]), int(center[1])
        
        body_color = (46, 56, 46)
        wheel_color = (13, 13, 13)
        turret_color = (56, 66, 56)
        
        w, h = int(28 * scale), int(12 * scale)
        
        # Корпус
        self.draw_rotated_rectangle(draw, (cx, cy), (w, h), angle, body_color)
        
        # Колёса
        rad = np.radians(angle)
        perp = rad + np.pi/2
        
        for i in range(4):
            offset = (i - 1.5) * 6 * scale
            wheel_cx = cx + offset * np.cos(rad)
            wheel_cy = cy + offset * np.sin(rad)
            
            for sign in [-1, 1]:
                wx = int(wheel_cx + sign * 5 * scale * np.cos(perp))
                wy = int(wheel_cy + sign * 5 * scale * np.sin(perp))
                r = int(2 * scale)
                draw.ellipse([wx-r, wy-r, wx+r, wy+r], fill=wheel_color)
        
        # Башня
        r = int(6 * scale)
        draw.ellipse([cx-r, cy-r, cx+r, cy+r], fill=turret_color)
        
        return np.array(img).astype(np.float32) / 255.0
    
    def draw_truck(self, img_array: np.ndarray, center: Tuple[int, int],
                   angle: float, scale: float = 1.0) -> np.ndarray:
        """Рисование Урал-4320: кабина + кузов"""
        img = Image.fromarray((img_array * 255).astype(np.uint8))
        draw = ImageDraw.Draw(img)
        cx, cy = int(center[0]), int(center[1])
        
        cab_color = (64, 51, 38)  # Светлее
        body_color = (38, 46, 30)  # Темнее
        wheel_color = (13, 13, 13)
        
        rad = np.radians(angle)
        
        # Кабина
        cab_w, cab_h = int(10 * scale), int(8 * scale)
        cab_cx = cx - int(8 * scale * np.cos(rad))
        cab_cy = cy - int(8 * scale * np.sin(rad))
        self.draw_rotated_rectangle(draw, (cab_cx, cab_cy), (cab_w, cab_h), angle, cab_color)
        
        # Кузов
        body_w, body_h = int(16 * scale), int(9 * scale)
        body_cx = cx + int(6 * scale * np.cos(rad))
        body_cy = cy + int(6 * scale * np.sin(rad))
        self.draw_rotated_rectangle(draw, (body_cx, body_cy), (body_w, body_h), angle, body_color)
        
        # Колёса
        perp = rad + np.pi/2
        positions = [
            (cab_cx, cab_cy, 3),
            (cx, cy, 3),
            (int(body_cx + 4*scale*np.cos(rad)), int(body_cy + 4*scale*np.sin(rad)), 3)
        ]
        
        for wx, wy, dist in positions:
            for sign in [-1, 1]:
                wheel_x = int(wx + sign * dist * scale * np.cos(perp))
                wheel_y = int(wy + sign * dist * scale * np.sin(perp))
                r = int(2.5 * scale)
                draw.ellipse([wheel_x-r, wheel_y-r, wheel_x+r, wheel_y+r], fill=wheel_color)
        
        return np.array(img).astype(np.float32) / 255.0
    
    def generate_sample(self, class_id: int, add_noise=True) -> Tuple[np.ndarray, int]:
        """Генерация одного образца"""
        bg = self.generate_background()
        
        margin = 15
        cx = random.randint(margin, self.img_size - margin)
        cy = random.randint(margin, self.img_size - margin)
        angle = random.uniform(0, 360)
        scale = random.uniform(0.8, 1.2)
        
        brightness = random.uniform(0.8, 1.2)
        
        if class_id == 0:
            img = self.draw_tank(bg, (cx, cy), angle, scale)
        elif class_id == 1:
            img = self.draw_btr(bg, (cx, cy), angle, scale)
        else:
            img = self.draw_truck(bg, (cx, cy), angle, scale)
            
        img = np.clip(img * brightness, 0, 1)
        
        if add_noise:
            img = self.add_gaussian_noise(img, sigma=0.1)
            
        return img.astype(np.float32), class_id


class MilitaryDataset(Dataset):
    """PyTorch Dataset для военной техники"""
    
    def __init__(self, num_samples=100, img_size=64, augment=True):
        self.generator = MilitaryVehicleGenerator(img_size)
        self.num_samples = num_samples
        self.img_size = img_size
        self.augment = augment
        
        self.labels = [i % 3 for i in range(num_samples)]
        random.shuffle(self.labels)
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        ])
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        label = self.labels[idx]
        img, _ = self.generator.generate_sample(label, add_noise=True)
        
        if self.augment:
            img_tensor = self.transform(img)
        else:
            img_tensor = transforms.ToTensor()(img)
            
        return img_tensor, label


# ==========================================
# 2. АРХИТЕКТУРА СЕТИ С ATTENTION
# ==========================================

class SpatialAttention(nn.Module):
    """Пространственный механизм внимания"""
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels // 8, 1),
            nn.ReLU(),
            nn.Conv2d(channels // 8, 1, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        attn = self.conv(x)
        return x * attn, attn


class LightweightCNN(nn.Module):
    """Лёгкая CNN для бортовых компьютеров"""
    def __init__(self, num_classes=3):
        super().__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.pool1 = nn.MaxPool2d(2)
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.pool2 = nn.MaxPool2d(2)
        self.attention2 = SpatialAttention(64)
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.pool3 = nn.MaxPool2d(2)
        self.attention3 = SpatialAttention(128)
        
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
        
        self.feature_maps = {}
        
    def forward(self, x, return_attention=False):
        x = self.conv1(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x, attn2 = self.attention2(x)
        x = self.pool2(x)
        
        x = self.conv3(x)
        x, attn3 = self.attention3(x)
        x = self.pool3(x)
        
        self.feature_maps['conv3'] = x.detach()
        self.feature_maps['attention3'] = attn3.detach()
        
        x = self.global_pool(x).flatten(1)
        x = self.classifier(x)
        
        # ИСПРАВЛЕНИЕ: всегда возвращаем одинаковую структуру для TorchScript
        if return_attention:
            return x, {'attn2': attn2, 'attn3': attn3}
        else:
            return x, {}  # Пустой словарь вместо одиночного тензора
    
    def get_attention_maps(self):
        return self.feature_maps


# ==========================================
# 3. ОБУЧЕНИЕ И РОБАСТНОСТЬ
# ==========================================

class RobustTrainer:
    def __init__(self, model, device='cpu'):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        
    def random_masking(self, images, max_ratio=0.2):
        """Случайное маскирование частей изображения"""
        b, c, h, w = images.shape
        masked = images.clone()
        
        for i in range(b):
            mh = int(h * random.uniform(0.1, max_ratio))
            mw = int(w * random.uniform(0.1, max_ratio))
            y = random.randint(0, h - mh)
            x = random.randint(0, w - mw)
            
            masked[i, :, y:y+mh, x:x+mw] = torch.rand(c, mh, mw).to(self.device)
            
        return masked
    
    def train_epoch(self, dataloader, optimizer, use_robust=True):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for images, labels in dataloader:
            images, labels = images.to(self.device), labels.to(self.device)
            
            if use_robust and random.random() > 0.5:
                images = self.random_masking(images)
            
            optimizer.zero_grad()
            # ИСПРАВЛЕНИЕ: теперь forward всегда возвращает кортеж
            outputs, _ = self.model(images, return_attention=False)
            loss = self.criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
        return total_loss / len(dataloader), 100. * correct / total
    
    def evaluate(self, dataloader):
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in dataloader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs, _ = self.model(images, return_attention=False)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
        return 100. * correct / total


# ==========================================
# 4. ВИЗУАЛИЗАЦИЯ И ОБЪЯСНИМОСТЬ
# ==========================================

class Explainer:
    def __init__(self, model, class_names=None):
        self.model = model
        self.class_names = class_names or ['T-72', 'BTR-82A', 'Ural-4320']
        
    def visualize_attention(self, image_tensor, save_path=None):
        """Визуализация карт внимания"""
        self.model.eval()
        
        with torch.no_grad():
            output, attentions = self.model(image_tensor.unsqueeze(0), return_attention=True)
            pred = output.argmax(dim=1).item()
            
        attn_map = attentions['attn3'][0, 0].cpu().numpy()
        
        # Интерполяция до размера изображения (без cv2)
        from scipy.ndimage import zoom
        zoom_factor = 64 / attn_map.shape[0]
        attn_resized = zoom(attn_map, zoom_factor, order=1)
        
        attn_resized = (attn_resized - attn_resized.min()) / (attn_resized.max() - attn_resized.min() + 1e-8)
        
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        
        img_np = image_tensor.permute(1, 2, 0).cpu().numpy()
        axes[0].imshow(np.clip(img_np, 0, 1))
        axes[0].set_title('Input Image')
        axes[0].axis('off')
        
        im = axes[1].imshow(attn_resized, cmap='hot')
        axes[1].set_title('Attention Map')
        axes[1].axis('off')
        plt.colorbar(im, ax=axes[1])
        
        axes[2].imshow(np.clip(img_np, 0, 1))
        axes[2].imshow(attn_resized, cmap='hot', alpha=0.6)
        axes[2].set_title(f'Overlay (Pred: {self.class_names[pred]})')
        axes[2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            
        return fig, attn_resized, pred
    
    def generate_report(self, image_tensor, true_label=None):
        """Генерация текстового отчёта"""
        self.model.eval()
        
        with torch.no_grad():
            output, _ = self.model(image_tensor.unsqueeze(0), return_attention=True)
            probs = F.softmax(output, dim=1)[0].cpu().numpy()
            
        pred = np.argmax(probs)
        confidence = probs[pred]
        
        report = {
            'predicted_class': self.class_names[pred],
            'confidence': float(confidence),
            'all_probabilities': {name: float(p) for name, p in zip(self.class_names, probs)},
            'attention_focus': 'Центр объекта' if confidence > 0.8 else 'Требуется проверка',
            'recommendation': 'Подтверждено' if confidence > 0.9 else 'Низкая уверенность - ручная проверка'
        }
        
        if true_label is not None:
            report['true_class'] = self.class_names[true_label]
            report['correct'] = (pred == true_label)
            
        return report


# ==========================================
# 5. ПОЛНЫЙ ПАЙПЛАЙН
# ==========================================

def main():
    print("=" * 60)
    print("ВОЕННАЯ СИСТЕМА ДЕТЕКЦИИ ТЕХНИКИ (PIL версия)")
    print("Версия: 1.0 (Оффлайн-режим)")
    print("=" * 60)
    
    # 1. Генерация данных
    print("\n[1] Генерация синтетических данных...")
    train_dataset = MilitaryDataset(num_samples=500, augment=True)
    test_dataset = MilitaryDataset(num_samples=100, augment=False)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    # 2. Инициализация модели
    print("[2] Инициализация лёгкой CNN...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LightweightCNN(num_classes=3)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"    Параметров модели: {total_params:,} (~{total_params/1e6:.2f}M)")
    print(f"    Размер модели: ~{total_params * 4 / 1024 / 1024:.2f} MB (fp32)")
    
    # 3. Обучение
    print("\n[3] Обучение модели...")
    trainer = RobustTrainer(model, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
    
    best_acc = 0
    for epoch in range(20):
        train_loss, train_acc = trainer.train_epoch(train_loader, optimizer, use_robust=True)
        test_acc = trainer.evaluate(test_loader)
        scheduler.step()
        
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), 'best_military_model.pth')
            
        if (epoch + 1) % 5 == 0:
            print(f"    Epoch {epoch+1}: Loss={train_loss:.3f}, Train Acc={train_acc:.1f}%, Test Acc={test_acc:.1f}%")
    
    print(f"\n    Лучшая точность: {best_acc:.1f}%")
    
    # 4. Тестирование робастности
    print("\n[4] Тестирование робастности...")
    model.load_state_dict(torch.load('best_military_model.pth'))
    
    noisy_dataset = MilitaryDataset(num_samples=100, augment=False)
    noisy_acc = trainer.evaluate(DataLoader(noisy_dataset, batch_size=16))
    print(f"    Точность на зашумлённых данных: {noisy_acc:.1f}%")
    
    # 5. Демонстрация объяснимости
    print("\n[5] Генерация отчётов объяснимости...")
    explainer = Explainer(model)
    
    sample_indices = [0, 5, 10]
    for idx in sample_indices:
        img, label = test_dataset[idx]
        fig, attn_map, pred = explainer.visualize_attention(img, save_path=f'attention_sample_{idx}.png')
        report = explainer.generate_report(img, label)
        
        print(f"\n    Образец {idx}:")
        print(f"    Истинный класс: {report.get('true_class', 'N/A')}")
        print(f"    Предсказание: {report['predicted_class']} ({report['confidence']*100:.1f}%)")
        print(f"    Рекомендация: {report['recommendation']}")
        
    # 6. Оценка задержки
    print("\n[6] Оценка задержки обработки...")
    model.eval()
    dummy_input = torch.randn(1, 3, 64, 64).to(device)
    
    for _ in range(10):
        _ = model(dummy_input)
        
    import time
    times = []
    with torch.no_grad():
        for _ in range(100):
            start = time.time()
            _ = model(dummy_input)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            times.append((time.time() - start) * 1000)
            
    avg_time = np.mean(times)
    print(f"    Средняя задержка: {avg_time:.2f} ms")
    print(f"    FPS: {1000/avg_time:.1f}")
    
    # 7. Экспорт модели для бортовых систем
    print("\n[7] Экспорт модели для бортовых систем...")
    
    # TorchScript - ИСПРАВЛЕННАЯ ВЕРСИЯ
    try:
        model_scripted = torch.jit.script(model.cpu())
        model_scripted.save('military_model_scripted.pt')
        print("    Сохранено: military_model_scripted.pt (TorchScript)")
    except Exception as e:
        print(f"    TorchScript пропущен: {str(e)[:50]}...")
        print("    Используйте ONNX формат")
    
    # ONNX
    dummy_input = torch.randn(1, 3, 64, 64)
    torch.onnx.export(model, dummy_input, 'military_model.onnx',
                      input_names=['input'],
                      output_names=['output'],
                      dynamic_axes={'input': {0: 'batch_size'},
                                   'output': {0: 'batch_size'}})
    print("    Сохранено: military_model.onnx (ONNX)")
    
    print("\n" + "=" * 60)
    print("ГОТОВО К РАЗВЁРТЫВАНИЮ")
    print("=" * 60)
    print("Файлы:")
    print("  - best_military_model.pth (PyTorch)")
    print("  - military_model_scripted.pt (TorchScript, если создался)")
    print("  - military_model.onnx (ONNX - универсальный)")
    print("  - attention_sample_*.png (Визуализация)")
    
    return model, explainer


if __name__ == "__main__":
    model, explainer = main()