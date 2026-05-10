"""
============================================================================
MODULE 3 — Vision Transformer (ViT) Classification
============================================================================
Fine-tunes google/vit-base-patch16-224 from HuggingFace on the same
datasets as Module 2 for a fair CNN-vs-ViT comparison.

ViT splits each image into 16x16 patches, embeds them linearly, adds
positional embeddings, and processes them through Transformer encoder
blocks with multi-head self-attention.

Key advantage: Global receptive field from the first layer (vs CNNs
which build it gradually through pooling).

Includes patch-based inference with majority voting for pathology slides.
============================================================================
"""

import os, time, copy, logging
from typing import Dict, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import ViTForImageClassification, ViTConfig
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix,
)
from src.utils import load_config, set_seed, get_device, save_checkpoint
from src.datasets import create_dataloaders

logger = logging.getLogger("MedImageCompareNet")


def build_vit_model(config: Dict, num_classes: int) -> nn.Module:
    """
    Load pretrained ViT-B/16 and replace the classification head.
    
    ViT-B/16 architecture:
      - 12 Transformer encoder layers
      - 768 hidden dimension
      - 12 attention heads
      - 16×16 patch size → 14×14 = 196 patches for 224×224 input
      - [CLS] token prepended for classification
    """
    vit_cfg = config["vit"]
    model_name = vit_cfg["model_name"]
    
    model = ViTForImageClassification.from_pretrained(
        model_name,
        num_labels=num_classes,
        ignore_mismatched_sizes=True,
        attn_implementation="eager",
    )
    logger.info(f"Built ViT from {model_name} (classes={num_classes})")
    return model


class ViTTrainer:
    """Full training + evaluation pipeline for Vision Transformer."""

    def __init__(self, config: Dict, dataset_name: str):
        self.config = config
        self.device = get_device(config)
        self.dataset_name = dataset_name
        
        num_classes = config["data"]["xray" if dataset_name == "xray" else "pathology"]["num_classes"]
        self.model = build_vit_model(config, num_classes).to(self.device)
        
        t = config["vit"]["training"]
        self.optimizer = optim.AdamW(
            self.model.parameters(), lr=t["learning_rate"], weight_decay=t["weight_decay"]
        )
        
        # Cosine schedule with warmup
        self.epochs = t["epochs"]
        self.warmup_steps = t["warmup_steps"]
        self.patience = t["early_stopping_patience"]
        self.grad_accum = t["gradient_accumulation_steps"]
        self.criterion = nn.CrossEntropyLoss()
        self.use_amp = config["device"].get("mixed_precision", False)
        self.scaler = torch.amp.GradScaler() if self.use_amp else None

    def _get_scheduler(self, total_steps):
        from torch.optim.lr_scheduler import LambdaLR
        warmup = self.warmup_steps
        def lr_lambda(step):
            if step < warmup:
                return float(step) / float(max(1, warmup))
            progress = float(step - warmup) / float(max(1, total_steps - warmup))
            return max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))
        return LambdaLR(self.optimizer, lr_lambda)

    def train(self, loaders: Dict[str, DataLoader]) -> Dict:
        """Training loop with cosine warmup scheduling and early stopping."""
        total_steps = len(loaders["train"]) * self.epochs // self.grad_accum
        scheduler = self._get_scheduler(total_steps)
        
        history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
        best_val_acc, patience_counter, best_weights = 0.0, 0, None

        for epoch in range(1, self.epochs + 1):
            self.model.train()
            running_loss, correct, total, step = 0.0, 0, 0, 0
            self.optimizer.zero_grad()

            for images, labels in tqdm(loaders["train"], desc=f"[ViT] Epoch {epoch}", leave=False):
                images, labels = images.to(self.device), labels.to(self.device)
                
                if self.use_amp:
                    with torch.amp.autocast(device_type="cuda"):
                        outputs = self.model(images).logits
                        loss = self.criterion(outputs, labels) / self.grad_accum
                    self.scaler.scale(loss).backward()
                    step += 1
                    if step % self.grad_accum == 0:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        self.optimizer.zero_grad()
                        scheduler.step()
                else:
                    outputs = self.model(images).logits
                    loss = self.criterion(outputs, labels) / self.grad_accum
                    loss.backward()
                    step += 1
                    if step % self.grad_accum == 0:
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                        scheduler.step()
                
                running_loss += loss.item() * self.grad_accum * images.size(0)
                _, preds = outputs.max(1)
                correct += preds.eq(labels).sum().item()
                total += labels.size(0)

            train_loss = running_loss / total
            train_acc = correct / total
            val_metrics = self._eval_epoch(loaders["val"])
            
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_metrics["loss"])
            history["train_acc"].append(train_acc)
            history["val_acc"].append(val_metrics["accuracy"])

            logger.info(
                f"[ViT/{self.dataset_name}] Epoch {epoch} — "
                f"Train: {train_acc:.4f} | Val: {val_metrics['accuracy']:.4f}"
            )

            if val_metrics["accuracy"] > best_val_acc:
                best_val_acc = val_metrics["accuracy"]
                best_weights = copy.deepcopy(self.model.state_dict())
                patience_counter = 0
                save_checkpoint(
                    self.model, self.optimizer, epoch, val_metrics,
                    f"checkpoints/vit_{self.dataset_name}_best.pth"
                )
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    logger.info(f"Early stopping at epoch {epoch}"); break

        if best_weights: self.model.load_state_dict(best_weights)
        return history

    @torch.no_grad()
    def _eval_epoch(self, loader):
        self.model.eval()
        loss, correct, total = 0.0, 0, 0
        for images, labels in loader:
            images, labels = images.to(self.device), labels.to(self.device)
            outputs = self.model(images).logits
            loss += self.criterion(outputs, labels).item() * images.size(0)
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)
        return {"loss": loss / total, "accuracy": correct / total}

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> Dict:
        """Full evaluation with all metrics."""
        self.model.eval()
        all_preds, all_labels, all_probs = [], [], []
        total_time = 0.0

        for images, labels in loader:
            images = images.to(self.device)
            start = time.time()
            outputs = self.model(images).logits
            total_time += time.time() - start
            probs = torch.softmax(outputs, dim=1)
            _, preds = outputs.max(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())

        y_true, y_pred, y_prob = np.array(all_labels), np.array(all_preds), np.array(all_probs)
        n = len(y_true)
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, average="binary", zero_division=0),
            "recall": recall_score(y_true, y_pred, average="binary", zero_division=0),
            "f1": f1_score(y_true, y_pred, average="binary", zero_division=0),
            "auc_roc": roc_auc_score(y_true, y_prob) if len(set(y_true)) > 1 else 0.0,
            "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
            "classification_report": classification_report(y_true, y_pred, output_dict=True),
            "inference_time_ms": (total_time / n) * 1000,
            "total_samples": n,
        }

    @torch.no_grad()
    def predict_single(self, image_tensor: torch.Tensor) -> Dict:
        """Single-image inference."""
        self.model.eval()
        image_tensor = image_tensor.unsqueeze(0).to(self.device)
        start = time.time()
        output = self.model(image_tensor).logits
        inference_ms = (time.time() - start) * 1000
        probs = torch.softmax(output, dim=1)[0]
        pred_class = probs.argmax().item()
        return {
            "predicted_class": pred_class,
            "confidence": probs[pred_class].item(),
            "probabilities": probs.cpu().numpy().tolist(),
            "inference_time_ms": inference_ms,
        }

    @torch.no_grad()
    def predict_patches_with_voting(self, patch_tensors: list) -> Dict:
        """
        Patch-based inference with majority voting for pathology slides.
        Each patch gets a prediction; final class = majority vote.
        """
        self.model.eval()
        votes, confidences = [], []
        for pt in patch_tensors:
            result = self.predict_single(pt)
            votes.append(result["predicted_class"])
            confidences.append(result["confidence"])
        
        from collections import Counter
        vote_counts = Counter(votes)
        final_class = vote_counts.most_common(1)[0][0]
        avg_conf = np.mean([c for v, c in zip(votes, confidences) if v == final_class])
        return {
            "final_class": final_class,
            "confidence": float(avg_conf),
            "vote_distribution": dict(vote_counts),
            "num_patches": len(patch_tensors),
        }

    def get_attention_weights(self, image_tensor: torch.Tensor):
        """Extract attention weights from all layers for visualization."""
        self.model.eval()
        image_tensor = image_tensor.unsqueeze(0).to(self.device)
        outputs = self.model(image_tensor, output_attentions=True)
        # attentions: tuple of (batch, heads, seq_len, seq_len) per layer
        attentions = [a.cpu().numpy() for a in outputs.attentions]
        return attentions


def main():
    config = load_config("config.yaml")
    set_seed(config["project"]["seed"])
    for ds in ["xray", "pathology"]:
        loaders = create_dataloaders(config, ds)
        logger.info(f"\n{'='*60}\nTraining ViT on {ds}\n{'='*60}")
        trainer = ViTTrainer(config, ds)
        trainer.train(loaders)
        results = trainer.evaluate(loaders["test"])
        logger.info(f"Test — Acc: {results['accuracy']:.4f} | F1: {results['f1']:.4f} | AUC: {results['auc_roc']:.4f}")

if __name__ == "__main__":
    main()
