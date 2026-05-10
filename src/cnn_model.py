"""
============================================================================
MODULE 2 — CNN Classification Baseline (ResNet-50 & DenseNet-121)
============================================================================
Fine-tunes pretrained ResNet-50 and DenseNet-121 on:
  (a) Chest X-Ray (Normal vs Pneumonia)
  (b) Breast Histopathology (Benign vs Malignant)

Uses ImageNet weights → replaces final FC layer → fine-tunes entire network.
Metrics: Accuracy, Precision, Recall, F1, AUC-ROC.
============================================================================
"""

import os, time, logging, copy
from typing import Dict, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix,
)
from src.utils import load_config, set_seed, get_device, save_checkpoint
from src.datasets import create_dataloaders

logger = logging.getLogger("MedImageCompareNet")


def build_cnn_model(model_name: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    """
    Build a CNN classifier with pretrained ImageNet weights.
    
    ResNet-50: Deep residual network — skip connections solve vanishing gradients.
    DenseNet-121: Dense connections — each layer receives features from ALL previous layers.
    Both are excellent for medical imaging due to strong feature hierarchies.
    """
    if model_name == "resnet50":
        weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.resnet50(weights=weights)
        model.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(model.fc.in_features, num_classes),
        )
    elif model_name == "densenet121":
        weights = models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.densenet121(weights=weights)
        model.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(model.classifier.in_features, num_classes),
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    logger.info(f"Built {model_name} (pretrained={pretrained}, classes={num_classes})")
    return model


class CNNTrainer:
    """Full training + evaluation pipeline for CNN classifiers."""

    def __init__(self, config: Dict, model_name: str, dataset_name: str):
        self.config = config
        self.device = get_device(config)
        self.model_name = model_name
        self.dataset_name = dataset_name
        
        num_classes = config["data"]["xray" if dataset_name == "xray" else "pathology"]["num_classes"]
        
        # Find model config
        model_cfg = None
        for m in config["cnn"]["models"]:
            if m["name"] == model_name:
                model_cfg = m; break
        if model_cfg is None:
            raise ValueError(f"Model {model_name} not in config")
        
        self.model = build_cnn_model(model_name, num_classes, model_cfg["pretrained"]).to(self.device)
        
        t = config["cnn"]["training"]
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=t["learning_rate"], weight_decay=t["weight_decay"]
        )
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, step_size=t["lr_step_size"], gamma=t["lr_gamma"]
        )
        self.criterion = nn.CrossEntropyLoss()
        self.epochs = t["epochs"]
        self.patience = t["early_stopping_patience"]
        self.use_amp = config["device"].get("mixed_precision", False)
        self.scaler = torch.amp.GradScaler() if self.use_amp else None

    def train(self, loaders: Dict[str, DataLoader]) -> Dict:
        """Full training loop with early stopping. Returns history dict."""
        history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
        best_val_acc, patience_counter = 0.0, 0
        best_weights = None

        for epoch in range(1, self.epochs + 1):
            # ── Train ──
            self.model.train()
            running_loss, correct, total = 0.0, 0, 0
            for images, labels in tqdm(loaders["train"], desc=f"[{self.model_name}] Epoch {epoch}", leave=False):
                images, labels = images.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                
                if self.use_amp:
                    with torch.amp.autocast(device_type="cuda"):
                        outputs = self.model(images)
                        loss = self.criterion(outputs, labels)
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                    loss.backward()
                    self.optimizer.step()
                
                running_loss += loss.item() * images.size(0)
                _, preds = outputs.max(1)
                correct += preds.eq(labels).sum().item()
                total += labels.size(0)

            self.scheduler.step()
            train_loss = running_loss / total
            train_acc = correct / total

            # ── Validate ──
            val_metrics = self._eval_epoch(loaders["val"])
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_metrics["loss"])
            history["train_acc"].append(train_acc)
            history["val_acc"].append(val_metrics["accuracy"])

            logger.info(
                f"[{self.model_name}/{self.dataset_name}] Epoch {epoch} — "
                f"Train Acc: {train_acc:.4f} | Val Acc: {val_metrics['accuracy']:.4f}"
            )

            # Early stopping
            if val_metrics["accuracy"] > best_val_acc:
                best_val_acc = val_metrics["accuracy"]
                best_weights = copy.deepcopy(self.model.state_dict())
                patience_counter = 0
                save_checkpoint(
                    self.model, self.optimizer, epoch,
                    val_metrics, f"checkpoints/{self.model_name}_{self.dataset_name}_best.pth"
                )
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break

        # Restore best weights
        if best_weights:
            self.model.load_state_dict(best_weights)
        return history

    @torch.no_grad()
    def _eval_epoch(self, loader):
        self.model.eval()
        loss, correct, total = 0.0, 0, 0
        for images, labels in loader:
            images, labels = images.to(self.device), labels.to(self.device)
            outputs = self.model(images)
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
            outputs = self.model(images)
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
        """Single-image inference. Returns class, confidence, time."""
        self.model.eval()
        image_tensor = image_tensor.unsqueeze(0).to(self.device)
        start = time.time()
        output = self.model(image_tensor)
        inference_ms = (time.time() - start) * 1000
        probs = torch.softmax(output, dim=1)[0]
        pred_class = probs.argmax().item()
        return {
            "predicted_class": pred_class,
            "confidence": probs[pred_class].item(),
            "probabilities": probs.cpu().numpy().tolist(),
            "inference_time_ms": inference_ms,
        }


def main():
    """Train all CNN models on all datasets."""
    config = load_config("config.yaml")
    set_seed(config["project"]["seed"])

    for dataset_name in ["xray", "pathology"]:
        loaders = create_dataloaders(config, dataset_name)
        for model_cfg in config["cnn"]["models"]:
            name = model_cfg["name"]
            logger.info(f"\n{'='*60}\nTraining {name} on {dataset_name}\n{'='*60}")
            trainer = CNNTrainer(config, name, dataset_name)
            trainer.train(loaders)
            results = trainer.evaluate(loaders["test"])
            logger.info(f"Test Results — Acc: {results['accuracy']:.4f} | F1: {results['f1']:.4f} | AUC: {results['auc_roc']:.4f}")

if __name__ == "__main__":
    main()
