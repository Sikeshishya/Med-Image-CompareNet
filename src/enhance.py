"""
============================================================================
MODULE 1 — U-Net + GAN X-Ray Image Enhancement
============================================================================
Denoises low-dose X-ray images using a U-Net generator with a PatchGAN
discriminator.  Combined loss = L1 + Perceptual (VGG) + Adversarial.
Metrics: PSNR, SSIM.
============================================================================
"""

import os, logging
from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from src.utils import load_config, set_seed, get_device, save_checkpoint

logger = logging.getLogger("MedImageCompareNet")

# ── Noisy X-Ray Dataset ──────────────────────────────────────────────────

class NoisyXRayDataset(Dataset):
    """Creates (noisy, clean) pairs by adding Gaussian noise to clean X-rays."""
    def __init__(self, root_dir, noise_level=25, image_size=256, split="train"):
        self.noise_level = noise_level
        self.image_size = image_size
        self.image_paths = []
        root = os.path.join(root_dir, split)
        if os.path.exists(root):
            for dp, _, fns in os.walk(root):
                for f in fns:
                    if f.lower().endswith((".jpg", ".jpeg", ".png")):
                        self.image_paths.append(os.path.join(dp, f))
        logger.info(f"NoisyXRayDataset [{split}]: {len(self)} images")

    def __len__(self): return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("L")
        img = img.resize((self.image_size, self.image_size), Image.BILINEAR)
        clean = transforms.ToTensor()(img)
        noise = torch.randn_like(clean) * (self.noise_level / 255.0)
        noisy = torch.clamp(clean + noise, 0.0, 1.0)
        return noisy, clean

# ── Double Convolution Block ─────────────────────────────────────────────

class DoubleConv(nn.Module):
    """Two consecutive Conv-BN-ReLU blocks — U-Net building block."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.conv(x)

# ── U-Net Generator ──────────────────────────────────────────────────────

class UNet(nn.Module):
    """
    U-Net encoder-decoder with skip connections.
    Skip connections preserve fine anatomical detail lost during downsampling.
    """
    def __init__(self, in_ch=1, out_ch=1, features=None):
        super().__init__()
        features = features or [64, 128, 256, 512]
        self.enc = nn.ModuleList()
        self.dec = nn.ModuleList()
        self.pool = nn.MaxPool2d(2, 2)
        self.ups = nn.ModuleList()

        prev = in_ch
        for f in features:
            self.enc.append(DoubleConv(prev, f)); prev = f
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)
        for f in reversed(features):
            self.ups.append(nn.ConvTranspose2d(f * 2, f, 2, 2))
            self.dec.append(DoubleConv(f * 2, f))

        self.final = nn.Sequential(nn.Conv2d(features[0], out_ch, 1), nn.Sigmoid())

    def forward(self, x):
        skips = []
        for enc in self.enc:
            x = enc(x); skips.append(x); x = self.pool(x)
        x = self.bottleneck(x)
        for up, dec, skip in zip(self.ups, self.dec, reversed(skips)):
            x = up(x)
            if x.shape != skip.shape:
                x = nn.functional.interpolate(x, size=skip.shape[2:])
            x = torch.cat([skip, x], dim=1)
            x = dec(x)
        return self.final(x)

# ── PatchGAN Discriminator ───────────────────────────────────────────────

class PatchDiscriminator(nn.Module):
    """PatchGAN — classifies 70x70 patches as real/fake for sharp outputs."""
    def __init__(self, in_ch=1):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_ch, 64, 4, 2, 1), nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False), nn.InstanceNorm2d(128), nn.LeakyReLU(0.2, True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False), nn.InstanceNorm2d(256), nn.LeakyReLU(0.2, True),
            nn.Conv2d(256, 512, 4, 1, 1, bias=False), nn.InstanceNorm2d(512), nn.LeakyReLU(0.2, True),
            nn.Conv2d(512, 1, 4, 1, 1),
        )
    def forward(self, x): return self.model(x)

# ── Perceptual Loss ──────────────────────────────────────────────────────

class PerceptualLoss(nn.Module):
    """VGG-16 feature-matching loss for perceptually sharp results."""
    def __init__(self):
        super().__init__()
        from torchvision.models import vgg16, VGG16_Weights
        vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        self.features = nn.Sequential(*list(vgg.features)[:10]).eval()
        for p in self.features.parameters(): p.requires_grad = False
        self.criterion = nn.L1Loss()

    def forward(self, gen, tgt):
        if gen.shape[1] == 1:
            gen, tgt = gen.repeat(1, 3, 1, 1), tgt.repeat(1, 3, 1, 1)
        return self.criterion(self.features(gen), self.features(tgt))

# ── Training Engine ──────────────────────────────────────────────────────

class EnhancementTrainer:
    """U-Net + GAN training: L_total = λ_pix·L1 + λ_perc·VGG + λ_adv·GAN"""
    def __init__(self, config):
        self.config = config
        self.device = get_device(config)
        cfg = config["enhancement"]
        u = cfg["unet"]
        self.gen = UNet(u["in_channels"], u["out_channels"], u["features"]).to(self.device)
        self.disc = PatchDiscriminator(u["out_channels"]).to(self.device)
        self.l1 = nn.L1Loss()
        self.adv_loss = nn.BCEWithLogitsLoss()
        self.perc_loss = PerceptualLoss().to(self.device)
        g = cfg["gan"]
        self.lp, self.lperc, self.ladv = g["lambda_pixel"], g["lambda_perceptual"], g["lambda_adversarial"]
        t = cfg["training"]
        self.opt_g = optim.Adam(self.gen.parameters(), lr=t["learning_rate"], betas=(t["beta1"], t["beta2"]))
        self.opt_d = optim.Adam(self.disc.parameters(), lr=t["learning_rate"], betas=(t["beta1"], t["beta2"]))
        self.epochs, self.save_interval = t["epochs"], t["save_interval"]

    def train(self, train_loader, val_loader):
        history = {"g_loss": [], "d_loss": [], "psnr": [], "ssim": []}
        best_psnr = 0.0
        for epoch in range(1, self.epochs + 1):
            self.gen.train(); self.disc.train()
            eg, ed = 0.0, 0.0
            for noisy, clean in tqdm(train_loader, desc=f"Epoch {epoch}/{self.epochs}", leave=False):
                noisy, clean = noisy.to(self.device), clean.to(self.device)
                # Discriminator
                self.opt_d.zero_grad()
                with torch.no_grad(): fake = self.gen(noisy)
                pr, pf = self.disc(clean), self.disc(fake)
                ld = (self.adv_loss(pr, torch.ones_like(pr)) + self.adv_loss(pf, torch.zeros_like(pf))) / 2
                ld.backward(); self.opt_d.step()
                # Generator
                self.opt_g.zero_grad()
                fake = self.gen(noisy)
                lg = self.lp * self.l1(fake, clean) + self.lperc * self.perc_loss(fake, clean) + self.ladv * self.adv_loss(self.disc(fake), torch.ones_like(pr))
                lg.backward(); self.opt_g.step()
                eg += lg.item(); ed += ld.item()

            vm = self.evaluate(val_loader)
            history["g_loss"].append(eg / len(train_loader))
            history["d_loss"].append(ed / len(train_loader))
            history["psnr"].append(vm["psnr"]); history["ssim"].append(vm["ssim"])
            logger.info(f"Epoch {epoch} — G:{eg/len(train_loader):.4f} D:{ed/len(train_loader):.4f} PSNR:{vm['psnr']:.2f} SSIM:{vm['ssim']:.4f}")
            if vm["psnr"] > best_psnr:
                best_psnr = vm["psnr"]
                save_checkpoint(self.gen, self.opt_g, epoch, vm, "checkpoints/enhancement/best_generator.pth")
            if epoch % self.save_interval == 0:
                save_checkpoint(self.gen, self.opt_g, epoch, vm, f"checkpoints/enhancement/gen_e{epoch}.pth")
        return history

    @torch.no_grad()
    def evaluate(self, loader):
        self.gen.eval()
        psnrs, ssims = [], []
        for noisy, clean in loader:
            noisy, clean = noisy.to(self.device), clean.to(self.device)
            out = self.gen(noisy)
            for i in range(clean.size(0)):
                c, d = clean[i, 0].cpu().numpy(), out[i, 0].cpu().numpy()
                psnrs.append(peak_signal_noise_ratio(c, d, data_range=1.0))
                ssims.append(structural_similarity(c, d, data_range=1.0))
        return {"psnr": np.mean(psnrs), "ssim": np.mean(ssims)}

    @torch.no_grad()
    def enhance_single(self, image):
        """Enhance a single X-ray image. Returns (enhanced_uint8, metrics_dict)."""
        self.gen.eval()
        if len(image.shape) == 3: image = np.mean(image, axis=2)
        h, w = image.shape
        t = torch.tensor(image, dtype=torch.float32)
        if t.max() > 1.0: t = t / 255.0
        t = t.unsqueeze(0).unsqueeze(0).to(self.device)
        t_r = nn.functional.interpolate(t, size=(256, 256), mode="bilinear")
        out = self.gen(t_r)
        out = nn.functional.interpolate(out, size=(h, w), mode="bilinear")
        out_np = out[0, 0].cpu().numpy()
        inp_np = nn.functional.interpolate(t, size=(h, w), mode="bilinear")[0, 0].cpu().numpy()
        metrics = {
            "psnr": float(peak_signal_noise_ratio(inp_np, out_np, data_range=1.0)),
            "ssim": float(structural_similarity(inp_np, out_np, data_range=1.0)),
        }
        return (out_np * 255).astype(np.uint8), metrics

def main():
    config = load_config("config.yaml"); set_seed(config["project"]["seed"])
    cfg, dc = config["enhancement"], config["data"]["xray"]
    train_ds = NoisyXRayDataset(dc["root_dir"], cfg["noise_level"], 256, "train")
    val_ds = NoisyXRayDataset(dc["root_dir"], cfg["noise_level"], 256, "val")
    tl = DataLoader(train_ds, cfg["training"]["batch_size"], shuffle=True, num_workers=4, pin_memory=True)
    vl = DataLoader(val_ds, cfg["training"]["batch_size"], shuffle=False, num_workers=4, pin_memory=True)
    trainer = EnhancementTrainer(config)
    h = trainer.train(tl, vl)
    logger.info(f"Done! Best PSNR: {max(h['psnr']):.2f} dB, SSIM: {max(h['ssim']):.4f}")

if __name__ == "__main__":
    main()
