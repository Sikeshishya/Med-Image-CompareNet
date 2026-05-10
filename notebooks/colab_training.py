# ============================================================================
# Med-Image CompareNet — Google Colab Training Script
# ============================================================================
# HOW TO USE:
#   1. Open Google Colab (colab.research.google.com)
#   2. File → Upload Notebook → upload this as .py or paste into cells
#   3. Runtime → Change runtime type → GPU (T4)
#   4. Run All
#
# Each section below = one Colab cell. Copy-paste between the
# ═══ markers into separate cells.
# ============================================================================


# ═══════════════════════════════════════════════════════════════════════════
# CELL 1: Mount Google Drive & Setup
# ═══════════════════════════════════════════════════════════════════════════

from google.colab import drive
drive.mount('/content/drive')

import os
PROJECT_DIR = '/content/MedImageCompareNet'
DRIVE_DIR = '/content/drive/MyDrive/MedImageCompareNet'
os.makedirs(DRIVE_DIR, exist_ok=True)
os.makedirs(f'{DRIVE_DIR}/checkpoints', exist_ok=True)
os.makedirs(f'{DRIVE_DIR}/results', exist_ok=True)
os.makedirs(f'{DRIVE_DIR}/data', exist_ok=True)

# Check GPU
import torch
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU!'}")
print(f"PyTorch: {torch.__version__}")


# ═══════════════════════════════════════════════════════════════════════════
# CELL 2: Install Dependencies
# ═══════════════════════════════════════════════════════════════════════════

!pip install -q transformers timm albumentations scikit-image torchmetrics fpdf2 omegaconf tqdm plotly seaborn


# ═══════════════════════════════════════════════════════════════════════════
# CELL 3A: Setup Kaggle API
# ═══════════════════════════════════════════════════════════════════════════
import os
os.environ['KAGGLE_API_TOKEN'] = 'KGAT_76cb18f216985925ea8af77920c3bae5'
!pip install -q kaggle
!mkdir -p /content/data
print("✅ Kaggle ready")


# ═══════════════════════════════════════════════════════════════════════════
# CELL 3B: Download & Unzip X-Ray Dataset (~2GB — takes ~2 min)
# ═══════════════════════════════════════════════════════════════════════════
!kaggle datasets download -d paultimothymooney/chest-xray-pneumonia -p /content/data/
!unzip -q -n /content/data/chest-xray-pneumonia.zip -d /content/data/
!rm -f /content/data/chest-xray-pneumonia.zip
!ls /content/data/
!ls /content/data/chest_xray/
print("✅ X-Ray dataset ready")


# ═══════════════════════════════════════════════════════════════════════════
# CELL 3C: Download & Unzip Pathology Dataset (~1.6GB — takes ~5 min)
# ═══════════════════════════════════════════════════════════════════════════
!kaggle datasets download -d paultimothymooney/breast-histopathology-images -p /content/data/
!unzip -q -n /content/data/breast-histopathology-images.zip -d /content/data/
!rm -f /content/data/breast-histopathology-images.zip
!ls /content/data/
!df -h /content
print("✅ Pathology dataset ready")


# ═══════════════════════════════════════════════════════════════════════════
# CELL 4: Shared Utilities
# ═══════════════════════════════════════════════════════════════════════════

import random, logging, copy, time, json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models
from PIL import Image
from pathlib import Path
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             classification_report)
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

SEED = 42
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
set_seed(SEED)

def save_ckpt(model, opt, epoch, metrics, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({'model': model.state_dict(), 'opt': opt.state_dict(),
                'epoch': epoch, 'metrics': metrics}, path)
    print(f"  💾 Saved → {path}")

print(f"✅ Utilities ready | Device: {DEVICE}")


# ═══════════════════════════════════════════════════════════════════════════
# CELL 5: Dataset Loaders
# ═══════════════════════════════════════════════════════════════════════════

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

train_tf = transforms.Compose([
    transforms.Resize((256, 256)), transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(), transforms.RandomRotation(15),
    transforms.ColorJitter(0.2, 0.2, 0.1, 0.05),
    transforms.ToTensor(), transforms.Normalize(MEAN, STD),
])
eval_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(), transforms.Normalize(MEAN, STD),
])

class XRayDataset(Dataset):
    def __init__(self, root, split='train', transform=None):
        self.transform = transform
        self.paths, self.labels = [], []
        for cls_idx, cls_name in enumerate(['NORMAL', 'PNEUMONIA']):
            d = Path(root) / split / cls_name
            if d.exists():
                for p in d.glob('*'):
                    if p.suffix.lower() in ('.jpg', '.jpeg', '.png'):
                        self.paths.append(str(p)); self.labels.append(cls_idx)
        print(f"  XRay [{split}]: {len(self)} images")

    def __len__(self): return len(self.paths)
    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert('RGB')
        if self.transform: img = self.transform(img)
        return img, self.labels[i]

class PathoDataset(Dataset):
    def __init__(self, root, split='train', transform=None, max_per_class=15000):
        self.transform = transform
        p0, p1 = [], []
        for pd in Path(root).iterdir():
            if not pd.is_dir(): continue
            b, m = pd / '0', pd / '1'
            if b.exists(): p0.extend([str(x) for x in b.glob('*.png')])
            if m.exists(): p1.extend([str(x) for x in m.glob('*.png')])
        rng = np.random.RandomState(SEED)
        if len(p0) > max_per_class: p0 = list(rng.choice(p0, max_per_class, False))
        if len(p1) > max_per_class: p1 = list(rng.choice(p1, max_per_class, False))
        all_p, all_l = p0 + p1, [0]*len(p0) + [1]*len(p1)
        tr_p, tmp_p, tr_l, tmp_l = train_test_split(all_p, all_l, test_size=0.3, stratify=all_l, random_state=SEED)
        va_p, te_p, va_l, te_l = train_test_split(tmp_p, tmp_l, test_size=0.5, stratify=tmp_l, random_state=SEED)
        m = {'train': (tr_p, tr_l), 'val': (va_p, va_l), 'test': (te_p, te_l)}
        self.paths, self.labels = m[split]
        print(f"  Patho [{split}]: {len(self)} patches")

    def __len__(self): return len(self.paths)
    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert('RGB')
        if self.transform: img = self.transform(img)
        return img, self.labels[i]

def make_loaders(ds_class, root, batch_size=32, **kwargs):
    tr = ds_class(root, 'train', train_tf, **kwargs)
    va = ds_class(root, 'val', eval_tf, **kwargs)
    te = ds_class(root, 'test', eval_tf, **kwargs)
    counts = np.bincount(tr.labels)
    w = 1.0 / counts; sw = [w[l] for l in tr.labels]
    sampler = WeightedRandomSampler(sw, len(sw), replacement=True)
    return {
        'train': DataLoader(tr, batch_size, sampler=sampler, num_workers=0, pin_memory=True, drop_last=True),
        'val': DataLoader(va, batch_size, shuffle=False, num_workers=0, pin_memory=True),
        'test': DataLoader(te, batch_size, shuffle=False, num_workers=0, pin_memory=True),
    }

print("✅ Dataset classes ready")


# ═══════════════════════════════════════════════════════════════════════════
# CELL 6: Train & Evaluate Function (shared by CNN and ViT)
# ═══════════════════════════════════════════════════════════════════════════

def train_model(model, loaders, optimizer, scheduler, epochs, save_path, patience=7, is_hf=False):
    """Generic training loop. is_hf=True for HuggingFace models (.logits)."""
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler()
    best_acc, wait, best_w = 0, 0, None
    history = {'train_acc': [], 'val_acc': []}

    for epoch in range(1, epochs + 1):
        model.train()
        correct, total = 0, 0
        for imgs, lbls in tqdm(loaders['train'], desc=f'Epoch {epoch}/{epochs}', leave=False):
            imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
            optimizer.zero_grad()
            with torch.amp.autocast(device_type='cuda'):
                out = model(imgs).logits if is_hf else model(imgs)
                loss = criterion(out, lbls)
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            correct += out.argmax(1).eq(lbls).sum().item(); total += lbls.size(0)

        if scheduler: scheduler.step()
        train_acc = correct / total

        # Validate
        model.eval(); vc, vt = 0, 0
        with torch.no_grad():
            for imgs, lbls in loaders['val']:
                imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
                out = model(imgs).logits if is_hf else model(imgs)
                vc += out.argmax(1).eq(lbls).sum().item(); vt += lbls.size(0)
        val_acc = vc / vt
        history['train_acc'].append(train_acc); history['val_acc'].append(val_acc)
        print(f'  Epoch {epoch}: Train {train_acc:.4f} | Val {val_acc:.4f}')

        if val_acc > best_acc:
            best_acc = val_acc; wait = 0
            best_w = copy.deepcopy(model.state_dict())
            save_ckpt(model, optimizer, epoch, {'val_acc': val_acc}, save_path)
        else:
            wait += 1
            if wait >= patience:
                print(f'  ⏹ Early stop at epoch {epoch}'); break

    if best_w: model.load_state_dict(best_w)
    return history

def evaluate_model(model, loader, is_hf=False):
    """Full evaluation with all metrics."""
    model.eval()
    preds, labels, probs = [], [], []
    t = 0
    with torch.no_grad():
        for imgs, lbls in loader:
            imgs = imgs.to(DEVICE)
            start = time.time()
            out = model(imgs).logits if is_hf else model(imgs)
            t += time.time() - start
            p = torch.softmax(out, 1)
            preds.extend(out.argmax(1).cpu().numpy())
            labels.extend(lbls.numpy())
            probs.extend(p[:, 1].cpu().numpy())
    y, yp, ypr = np.array(labels), np.array(preds), np.array(probs)
    n = len(y)
    return {
        'accuracy': accuracy_score(y, yp),
        'precision': precision_score(y, yp, average='binary', zero_division=0),
        'recall': recall_score(y, yp, average='binary', zero_division=0),
        'f1': f1_score(y, yp, average='binary', zero_division=0),
        'auc_roc': roc_auc_score(y, ypr) if len(set(y)) > 1 else 0,
        'inference_time_ms': (t / n) * 1000,
        'confusion_matrix': confusion_matrix(y, yp).tolist(),
    }

print("✅ Training functions ready")


# ═══════════════════════════════════════════════════════════════════════════
# CELL 7: MODULE 1 — U-Net + GAN Enhancement
# ═══════════════════════════════════════════════════════════════════════════
# Safety imports + variables (each cell is self-contained)
import os, random, copy, time, json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models
from PIL import Image
from tqdm.auto import tqdm
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED = 42
DRIVE_DIR = '/content/drive/MyDrive/MedImageCompareNet'
os.makedirs(f'{DRIVE_DIR}/checkpoints', exist_ok=True)
os.makedirs('/content/data', exist_ok=True)

import subprocess

# ── Download + Unzip if data is missing ──
if len(os.listdir('/content/data')) == 0:
    print("  ⚠️ /content/data is EMPTY — downloading datasets now...")
    os.environ['KAGGLE_API_TOKEN'] = 'KGAT_76cb18f216985925ea8af77920c3bae5'
    subprocess.run(['pip', 'install', '-q', 'kaggle'], check=True)
    # Download
    subprocess.run(['kaggle', 'datasets', 'download', '-d',
                    'paultimothymooney/chest-xray-pneumonia', '-p', '/content/data/'], check=True)
    subprocess.run(['kaggle', 'datasets', 'download', '-d',
                    'paultimothymooney/breast-histopathology-images', '-p', '/content/data/'], check=True)
    # Unzip
    for zf in os.listdir('/content/data'):
        if zf.endswith('.zip'):
            print(f"  📦 Unzipping {zf}...")
            subprocess.run(['unzip', '-q', '-n', f'/content/data/{zf}', '-d', '/content/data/'])
    print(f"  ✅ Download complete!")
else:
    # Also try unzipping from Drive if zips are there
    drive_data = f'{DRIVE_DIR}/data'
    if os.path.isdir(drive_data):
        for zf in os.listdir(drive_data):
            if zf.endswith('.zip'):
                print(f"  📦 Unzipping {zf} from Drive...")
                subprocess.run(['unzip', '-q', '-n', os.path.join(drive_data, zf), '-d', '/content/data/'])

print(f"  /content/data contents: {os.listdir('/content/data/')}")

# ── Auto-detect X-Ray path ──
XRAY_DIR = '/content/data'
for root, dirs, _ in os.walk('/content/data'):
    if 'train' in dirs:
        train_path = os.path.join(root, 'train')
        if os.path.isdir(train_path):
            tc = os.listdir(train_path)
            if 'NORMAL' in tc or 'PNEUMONIA' in tc:
                XRAY_DIR = root; break
print(f"  ✅ X-Ray root: {XRAY_DIR}")

# ── Auto-detect Pathology path ──
PATHO_DIR = '/content/data'
for root, dirs, _ in os.walk('/content/data'):
    for d in dirs:
        c = os.path.join(root, d)
        try:
            if os.path.isdir(c) and '0' in os.listdir(c) and '1' in os.listdir(c):
                PATHO_DIR = root; break
        except: pass
    if PATHO_DIR != '/content/data': break
print(f"  ✅ Pathology root: {PATHO_DIR}")

def save_ckpt(model, opt, epoch, metrics, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({'model': model.state_dict(), 'opt': opt.state_dict(),
                'epoch': epoch, 'metrics': metrics}, path)
    print(f'  💾 Saved → {path}')

print("🖼️ MODULE 1: X-Ray Enhancement (U-Net + GAN)")

class NoisyDS(Dataset):
    def __init__(self, root, split='train', noise=25, size=256):
        self.noise, self.size = noise, size
        self.paths = []
        r = os.path.join(root, split)
        if os.path.exists(r):
            for dp, _, fns in os.walk(r):
                self.paths.extend([os.path.join(dp, f) for f in fns if f.lower().endswith(('.jpg','.jpeg','.png'))])
        print(f"  NoisyDS [{split}]: {len(self)} images")
    def __len__(self): return len(self.paths)
    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert('L').resize((self.size, self.size))
        clean = transforms.ToTensor()(img)
        noisy = torch.clamp(clean + torch.randn_like(clean) * (self.noise/255), 0, 1)
        return noisy, clean

class DConv(nn.Module):
    def __init__(self, ci, co):
        super().__init__()
        self.c = nn.Sequential(nn.Conv2d(ci,co,3,1,1,bias=False), nn.BatchNorm2d(co), nn.ReLU(True),
                               nn.Conv2d(co,co,3,1,1,bias=False), nn.BatchNorm2d(co), nn.ReLU(True))
    def forward(self, x): return self.c(x)

class UNet(nn.Module):
    def __init__(self, ci=1, co=1, fs=None):
        super().__init__()
        fs = fs or [64,128,256,512]
        self.enc, self.dec, self.ups = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        self.pool = nn.MaxPool2d(2)
        p = ci
        for f in fs: self.enc.append(DConv(p, f)); p = f
        self.bn = DConv(fs[-1], fs[-1]*2)
        for f in reversed(fs):
            self.ups.append(nn.ConvTranspose2d(f*2, f, 2, 2))
            self.dec.append(DConv(f*2, f))
        self.out = nn.Sequential(nn.Conv2d(fs[0], co, 1), nn.Sigmoid())
    def forward(self, x):
        sk = []
        for e in self.enc: x = e(x); sk.append(x); x = self.pool(x)
        x = self.bn(x)
        for u, d, s in zip(self.ups, self.dec, reversed(sk)):
            x = u(x)
            if x.shape != s.shape: x = nn.functional.interpolate(x, s.shape[2:])
            x = d(torch.cat([s, x], 1))
        return self.out(x)

class PatchDisc(nn.Module):
    def __init__(self, ci=1):
        super().__init__()
        self.m = nn.Sequential(
            nn.Conv2d(ci,64,4,2,1), nn.LeakyReLU(0.2,True),
            nn.Conv2d(64,128,4,2,1,bias=False), nn.InstanceNorm2d(128), nn.LeakyReLU(0.2,True),
            nn.Conv2d(128,256,4,2,1,bias=False), nn.InstanceNorm2d(256), nn.LeakyReLU(0.2,True),
            nn.Conv2d(256,512,4,1,1,bias=False), nn.InstanceNorm2d(512), nn.LeakyReLU(0.2,True),
            nn.Conv2d(512,1,4,1,1))
    def forward(self, x): return self.m(x)

# Train Enhancement
enh_train = NoisyDS(XRAY_DIR, 'train', 25, 256)
enh_val = NoisyDS(XRAY_DIR, 'val', 25, 256)
enh_tl = DataLoader(enh_train, 8, shuffle=True, num_workers=0, pin_memory=True)
enh_vl = DataLoader(enh_val, 8, shuffle=False, num_workers=0, pin_memory=True)

gen = UNet(1, 1, [64,128,256,512]).to(DEVICE)
disc = PatchDisc(1).to(DEVICE)
opt_g = optim.Adam(gen.parameters(), 2e-4, betas=(0.5, 0.999))
opt_d = optim.Adam(disc.parameters(), 2e-4, betas=(0.5, 0.999))
l1_loss = nn.L1Loss()
adv_loss = nn.BCEWithLogitsLoss()

ENH_EPOCHS = 30
best_psnr = 0

for epoch in range(1, ENH_EPOCHS + 1):
    gen.train(); disc.train()
    for noisy, clean in tqdm(enh_tl, desc=f'Enh Epoch {epoch}', leave=False):
        noisy, clean = noisy.to(DEVICE), clean.to(DEVICE)
        # Disc
        opt_d.zero_grad()
        with torch.no_grad(): fake = gen(noisy)
        ld = (adv_loss(disc(clean), torch.ones_like(disc(clean))) +
              adv_loss(disc(fake), torch.zeros_like(disc(fake)))) / 2
        ld.backward(); opt_d.step()
        # Gen
        opt_g.zero_grad()
        fake = gen(noisy)
        lg = 100 * l1_loss(fake, clean) + adv_loss(disc(fake), torch.ones_like(disc(fake)))
        lg.backward(); opt_g.step()

    # Validate
    gen.eval(); psnrs, ssims = [], []
    with torch.no_grad():
        for n, c in enh_vl:
            n, c = n.to(DEVICE), c.to(DEVICE)
            o = gen(n)
            for i in range(c.size(0)):
                cn, dn = c[i,0].cpu().numpy(), o[i,0].cpu().numpy()
                psnrs.append(peak_signal_noise_ratio(cn, dn, data_range=1.0))
                ssims.append(structural_similarity(cn, dn, data_range=1.0))
    mp, ms = np.mean(psnrs), np.mean(ssims)
    print(f'  Epoch {epoch}: PSNR={mp:.2f} dB | SSIM={ms:.4f}')
    if mp > best_psnr:
        best_psnr = mp
        save_ckpt(gen, opt_g, epoch, {'psnr': mp, 'ssim': ms},
                  f'{DRIVE_DIR}/checkpoints/best_generator.pth')

print(f"✅ Enhancement done! Best PSNR: {best_psnr:.2f} dB")


# ═══════════════════════════════════════════════════════════════════════════
# CELL 8: MODULE 2 — CNN Classification (ResNet-50 & DenseNet-121)
# ═══════════════════════════════════════════════════════════════════════════
import os, random, copy, time, json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models
from PIL import Image
from pathlib import Path
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED = 42
DRIVE_DIR = '/content/drive/MyDrive/MedImageCompareNet'
os.makedirs(f'{DRIVE_DIR}/checkpoints', exist_ok=True)

def find_dataset_root(base, marker_subdir='train'):
    if os.path.isdir(os.path.join(base, marker_subdir)): return base
    for root, dirs, _ in os.walk(base):
        if marker_subdir in dirs: return root
    return base

XRAY_DIR = find_dataset_root('/content/data', 'train')
PATHO_BASE = '/content/data'
PATHO_DIR = '/content/data/breast_histopathology'
for d in os.listdir(PATHO_BASE):
    full = os.path.join(PATHO_BASE, d)
    if os.path.isdir(full):
        subdirs = os.listdir(full)
        if any(os.path.isdir(os.path.join(full, s, '0')) for s in subdirs[:5] if os.path.isdir(os.path.join(full, s))):
            PATHO_DIR = full; break

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
print(f"  X-Ray: {XRAY_DIR} | Pathology: {PATHO_DIR}")

def save_ckpt(model, opt, epoch, metrics, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({'model': model.state_dict(), 'opt': opt.state_dict(),
                'epoch': epoch, 'metrics': metrics}, path)
    print(f'  💾 Saved → {path}')

train_tf = transforms.Compose([
    transforms.Resize((256, 256)), transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(), transforms.RandomRotation(15),
    transforms.ColorJitter(0.2, 0.2, 0.1, 0.05),
    transforms.ToTensor(), transforms.Normalize(MEAN, STD),
])
eval_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(), transforms.Normalize(MEAN, STD),
])

class XRayDataset(Dataset):
    def __init__(self, root, split='train', transform=None):
        self.transform = transform
        self.paths, self.labels = [], []
        for cls_idx, cls_name in enumerate(['NORMAL', 'PNEUMONIA']):
            d = Path(root) / split / cls_name
            if d.exists():
                for p in d.glob('*'):
                    if p.suffix.lower() in ('.jpg', '.jpeg', '.png'):
                        self.paths.append(str(p)); self.labels.append(cls_idx)
        print(f'  XRay [{split}]: {len(self)} images')
    def __len__(self): return len(self.paths)
    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert('RGB')
        if self.transform: img = self.transform(img)
        return img, self.labels[i]

class PathoDataset(Dataset):
    def __init__(self, root, split='train', transform=None, max_per_class=15000):
        self.transform = transform
        p0, p1 = [], []
        for pd in Path(root).iterdir():
            if not pd.is_dir(): continue
            b, m = pd / '0', pd / '1'
            if b.exists(): p0.extend([str(x) for x in b.glob('*.png')])
            if m.exists(): p1.extend([str(x) for x in m.glob('*.png')])
        rng = np.random.RandomState(SEED)
        if len(p0) > max_per_class: p0 = list(rng.choice(p0, max_per_class, False))
        if len(p1) > max_per_class: p1 = list(rng.choice(p1, max_per_class, False))
        all_p, all_l = p0 + p1, [0]*len(p0) + [1]*len(p1)
        tr_p, tmp_p, tr_l, tmp_l = train_test_split(all_p, all_l, test_size=0.3, stratify=all_l, random_state=SEED)
        va_p, te_p, va_l, te_l = train_test_split(tmp_p, tmp_l, test_size=0.5, stratify=tmp_l, random_state=SEED)
        m = {'train': (tr_p, tr_l), 'val': (va_p, va_l), 'test': (te_p, te_l)}
        self.paths, self.labels = m[split]
        print(f'  Patho [{split}]: {len(self)} patches')
    def __len__(self): return len(self.paths)
    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert('RGB')
        if self.transform: img = self.transform(img)
        return img, self.labels[i]

def make_loaders(ds_class, root, batch_size=32, **kwargs):
    tr = ds_class(root, 'train', train_tf, **kwargs)
    va = ds_class(root, 'val', eval_tf, **kwargs)
    te = ds_class(root, 'test', eval_tf, **kwargs)
    counts = np.bincount(tr.labels)
    w = 1.0 / counts; sw = [w[l] for l in tr.labels]
    sampler = WeightedRandomSampler(sw, len(sw), replacement=True)
    return {
        'train': DataLoader(tr, batch_size, sampler=sampler, num_workers=0, pin_memory=True, drop_last=True),
        'val': DataLoader(va, batch_size, shuffle=False, num_workers=0, pin_memory=True),
        'test': DataLoader(te, batch_size, shuffle=False, num_workers=0, pin_memory=True),
    }

def train_model(model, loaders, optimizer, scheduler, epochs, save_path, patience=7, is_hf=False):
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler()
    best_acc, wait, best_w = 0, 0, None
    for epoch in range(1, epochs + 1):
        model.train(); correct, total = 0, 0
        for imgs, lbls in tqdm(loaders['train'], desc=f'Epoch {epoch}/{epochs}', leave=False):
            imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
            optimizer.zero_grad()
            with torch.amp.autocast(device_type='cuda'):
                out = model(imgs).logits if is_hf else model(imgs)
                loss = criterion(out, lbls)
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            correct += out.argmax(1).eq(lbls).sum().item(); total += lbls.size(0)
        if scheduler: scheduler.step()
        train_acc = correct / total
        model.eval(); vc, vt = 0, 0
        with torch.no_grad():
            for imgs, lbls in loaders['val']:
                imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
                out = model(imgs).logits if is_hf else model(imgs)
                vc += out.argmax(1).eq(lbls).sum().item(); vt += lbls.size(0)
        val_acc = vc / vt
        print(f'  Epoch {epoch}: Train {train_acc:.4f} | Val {val_acc:.4f}')
        if val_acc > best_acc:
            best_acc = val_acc; wait = 0
            best_w = copy.deepcopy(model.state_dict())
            save_ckpt(model, optimizer, epoch, {'val_acc': val_acc}, save_path)
        else:
            wait += 1
            if wait >= patience: print(f'  ⏹ Early stop at epoch {epoch}'); break
    if best_w: model.load_state_dict(best_w)

def evaluate_model(model, loader, is_hf=False):
    model.eval(); preds, labels, probs = [], [], []; t = 0
    with torch.no_grad():
        for imgs, lbls in loader:
            imgs = imgs.to(DEVICE)
            start = time.time()
            out = model(imgs).logits if is_hf else model(imgs)
            t += time.time() - start
            p = torch.softmax(out, 1)
            preds.extend(out.argmax(1).cpu().numpy())
            labels.extend(lbls.numpy())
            probs.extend(p[:, 1].cpu().numpy())
    y, yp, ypr = np.array(labels), np.array(preds), np.array(probs)
    n = len(y)
    return {
        'accuracy': accuracy_score(y, yp),
        'precision': precision_score(y, yp, average='binary', zero_division=0),
        'recall': recall_score(y, yp, average='binary', zero_division=0),
        'f1': f1_score(y, yp, average='binary', zero_division=0),
        'auc_roc': roc_auc_score(y, ypr) if len(set(y)) > 1 else 0,
        'inference_time_ms': (t / n) * 1000,
        'confusion_matrix': confusion_matrix(y, yp).tolist(),
    }

print('\n🧬 MODULE 2: CNN Classification')
ALL_RESULTS = {}

for ds_name, ds_cls, ds_root, bs in [
    ('xray', XRayDataset, XRAY_DIR, 32),
    ('pathology', PathoDataset, PATHO_DIR, 32),
]:
    print(f"\n  📁 Loading {ds_name} dataset...")
    loaders = make_loaders(ds_cls, ds_root, bs)

    for model_name in ['resnet50', 'densenet121']:
        print(f"\n  🔬 Training {model_name} on {ds_name}")
        if model_name == 'resnet50':
            m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            m.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(m.fc.in_features, 2))
        else:
            m = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
            m.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(m.classifier.in_features, 2))
        m = m.to(DEVICE)
        opt = optim.Adam(m.parameters(), lr=1e-4, weight_decay=1e-4)
        sched = optim.lr_scheduler.StepLR(opt, 10, 0.1)

        train_model(m, loaders, opt, sched, epochs=30,
                    save_path=f'{DRIVE_DIR}/checkpoints/{model_name}_{ds_name}_best.pth')

        results = evaluate_model(m, loaders['test'])
        ALL_RESULTS[f'{model_name}_{ds_name}'] = results
        print(f"  ✅ {model_name}/{ds_name} — Acc: {results['accuracy']:.4f} | F1: {results['f1']:.4f} | AUC: {results['auc_roc']:.4f}")

print("\n✅ CNN training complete!")


# ── ViT Training (same cell, shares loaders & functions) ──
print("\n🤖 MODULE 3: ViT Classification")
from transformers import ViTForImageClassification

for ds_name, ds_cls, ds_root, bs in [
    ('xray', XRayDataset, XRAY_DIR, 16),
    ('pathology', PathoDataset, PATHO_DIR, 16),
]:
    print(f"\n  📁 Loading {ds_name} dataset...")
    loaders = make_loaders(ds_cls, ds_root, bs)

    print(f"  🔬 Training ViT on {ds_name}")
    vit = ViTForImageClassification.from_pretrained(
        'google/vit-base-patch16-224', num_labels=2, ignore_mismatched_sizes=True
    ).to(DEVICE)
    opt = optim.AdamW(vit.parameters(), lr=5e-5, weight_decay=0.01)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=30)

    train_model(vit, loaders, opt, sched, epochs=30,
                save_path=f'{DRIVE_DIR}/checkpoints/vit_{ds_name}_best.pth', is_hf=True)

    results = evaluate_model(vit, loaders['test'], is_hf=True)
    ALL_RESULTS[f'vit_{ds_name}'] = results
    print(f"  ✅ ViT/{ds_name} — Acc: {results['accuracy']:.4f} | F1: {results['f1']:.4f} | AUC: {results['auc_roc']:.4f}")

print("\n✅ All model training complete!")

# Save results immediately (in case next cell has issues)
import json
results_path = f'{DRIVE_DIR}/results/all_results.json'
import os; os.makedirs(os.path.dirname(results_path), exist_ok=True)
with open(results_path, 'w') as f:
    json.dump({k: {kk: (float(vv) if hasattr(vv, 'item') else vv)
                   for kk, vv in v.items()}
               for k, v in ALL_RESULTS.items()}, f, indent=2, default=str)
print(f"💾 Results auto-saved → {results_path}")


# ═══════════════════════════════════════════════════════════════════════════
# CELL 9: Final Summary & Save
# ═══════════════════════════════════════════════════════════════════════════
import json, os
import numpy as np
DRIVE_DIR = '/content/drive/MyDrive/MedImageCompareNet'

# Load results if ALL_RESULTS isn't in memory
results_path = f'{DRIVE_DIR}/results/all_results.json'
try:
    ALL_RESULTS
except NameError:
    with open(results_path) as f:
        ALL_RESULTS = json.load(f)

print("\n📊 FINAL RESULTS SUMMARY")
print("=" * 70)
print(f"{'Model':<20} {'Dataset':<12} {'Accuracy':>10} {'F1':>10} {'AUC-ROC':>10} {'ms/img':>10}")
print("-" * 70)
for key, r in ALL_RESULTS.items():
    parts = key.rsplit('_', 1)
    print(f"{parts[0]:<20} {parts[1]:<12} {r['accuracy']:>10.4f} {r['f1']:>10.4f} {r['auc_roc']:>10.4f} {r['inference_time_ms']:>10.1f}")
print("=" * 70)

print(f"\n💾 Results saved → {results_path}")
print(f"💾 Checkpoints saved → {DRIVE_DIR}/checkpoints/")
print("\n🎉 ALL DONE! Download checkpoints folder to your laptop and run:")
print("   streamlit run dashboard/app.py")
