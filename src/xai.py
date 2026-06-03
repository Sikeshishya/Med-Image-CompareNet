"""
============================================================================
MODULE 4 — Explainable AI (XAI) Audit
============================================================================
Implements:
  1. Grad-CAM for CNNs (ResNet/DenseNet) — gradient-weighted class activation maps
  2. Attention Rollout for ViT — aggregated multi-layer attention visualization
  3. Pointing Game — quantitative interpretability metric (hit/miss on clinical ROI)
  4. Clinical interpretation generator — plain-English explanations

Grad-CAM: Uses gradients flowing into the final conv layer to produce a
coarse localization map highlighting important regions for the prediction.

Attention Rollout: Multiplies attention matrices across all layers to track
how information flows from input patches to the [CLS] token.
============================================================================
"""

import logging
from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend

logger = logging.getLogger("MedImageCompareNet")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  GRAD-CAM FOR CNNs                                                    ║
# ╚══════════════════════════════════════════════════════════════════════════╝

class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (Grad-CAM).
    
    How it works:
    1. Forward pass → get feature maps from target layer
    2. Backward pass → get gradients w.r.t. target layer
    3. Global-average-pool the gradients → channel importance weights
    4. Weighted combination of feature maps → heatmap
    5. ReLU → only keep positive influences
    """
    
    def __init__(self, model: nn.Module, target_layer_name: str):
        self.model = model
        self.model.eval()
        self.gradients = None
        self.activations = None
        
        # Register hooks on the target layer
        target_layer = self._find_layer(model, target_layer_name)
        target_layer.register_forward_hook(self._forward_hook)
        target_layer.register_full_backward_hook(self._backward_hook)
    
    def _find_layer(self, model, name):
        """Navigate nested module hierarchy to find target layer."""
        parts = name.split(".")
        module = model
        for part in parts:
            if part.isdigit():
                module = module[int(part)]
            else:
                module = getattr(module, part)
        return module
    
    def _forward_hook(self, module, input, output):
        self.activations = output.detach()
    
    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def generate(self, input_tensor: torch.Tensor, target_class: Optional[int] = None) -> np.ndarray:
        """
        Generate Grad-CAM heatmap.
        
        Parameters
        ----------
        input_tensor : torch.Tensor
            Shape (1, C, H, W) — single image batch.
        target_class : int, optional
            Class to explain. If None, uses the predicted class.
            
        Returns
        -------
        np.ndarray
            Heatmap of shape (H, W) with values in [0, 1].
        """
        self.model.zero_grad()
        output = self.model(input_tensor)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Backward pass for the target class
        target_score = output[0, target_class]
        target_score.backward()
        
        # Channel importance weights = global average pooling of gradients
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)  # (1, C, 1, 1)
        
        # Weighted combination of activation maps
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # (1, 1, h, w)
        cam = F.relu(cam)  # Only positive influences
        
        # Normalize to [0, 1]
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        return cam
    
    def overlay_on_image(
        self, image: np.ndarray, heatmap: np.ndarray,
        colormap: str = "jet", alpha: float = 0.5,
    ) -> np.ndarray:
        """Overlay heatmap on original image."""
        # Downscale to prevent Out-Of-Memory on large inputs
        max_dim = 512
        h, w = image.shape[:2]
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            new_h, new_w = int(h * scale), int(w * scale)
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
            h, w = new_h, new_w
            
        heatmap_resized = cv2.resize(heatmap, (w, h))
        
        # Apply colormap
        cmap = plt.get_cmap(colormap)
        heatmap_colored = cmap(heatmap_resized)[:, :, :3]  # Drop alpha
        heatmap_colored = (heatmap_colored * 255).astype(np.uint8)
        
        # Normalize image to uint8
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        overlay = cv2.addWeighted(image, 1 - alpha, heatmap_colored, alpha, 0)
        return overlay


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  ATTENTION ROLLOUT FOR ViT                                             ║
# ╚══════════════════════════════════════════════════════════════════════════╝

class AttentionRollout:
    """
    Attention Rollout visualization for Vision Transformers.
    
    Concept: Multiply attention matrices across all layers to see how
    information flows from input patches to the [CLS] token.
    
    At each layer, we account for residual connections by averaging
    the attention matrix with the identity matrix (because skip
    connections mean ~50% of information passes through unchanged).
    """
    
    def __init__(self, head_fusion: str = "mean", discard_ratio: float = 0.9):
        self.head_fusion = head_fusion
        self.discard_ratio = discard_ratio
    
    def compute(self, attention_matrices: List[np.ndarray]) -> np.ndarray:
        """
        Compute attention rollout from a list of attention matrices.
        
        Parameters
        ----------
        attention_matrices : list of np.ndarray
            Each element: (batch, heads, seq_len, seq_len)
            
        Returns
        -------
        np.ndarray
            Attention map of shape (num_patches,) showing how much
            each patch contributes to the [CLS] token.
        """
        result = None
        
        for attn in attention_matrices:
            # attn shape: (1, num_heads, seq_len, seq_len)
            attn = attn[0]  # Remove batch dim → (heads, seq, seq)
            
            # Fuse heads
            if self.head_fusion == "mean":
                attn_fused = attn.mean(axis=0)
            elif self.head_fusion == "max":
                attn_fused = attn.max(axis=0)
            elif self.head_fusion == "min":
                attn_fused = attn.min(axis=0)
            else:
                attn_fused = attn.mean(axis=0)
            
            # Discard low-attention entries (keep top 1-discard_ratio)
            flat = attn_fused.flatten()
            threshold = np.quantile(flat, self.discard_ratio)
            attn_fused[attn_fused < threshold] = 0
            
            # Account for residual connections: 0.5*attention + 0.5*identity
            I = np.eye(attn_fused.shape[0])
            attn_fused = 0.5 * attn_fused + 0.5 * I
            
            # Normalize rows to sum to 1
            attn_fused = attn_fused / attn_fused.sum(axis=-1, keepdims=True)
            
            # Rollout: multiply across layers
            if result is None:
                result = attn_fused
            else:
                result = result @ attn_fused
        
        # Extract attention from [CLS] token (index 0) to all patches
        cls_attention = result[0, 1:]  # Skip [CLS]-to-[CLS]
        
        # Reshape to 2D grid (ViT-B/16 with 224x224 → 14x14 patches)
        num_patches = len(cls_attention)
        grid_size = int(np.sqrt(num_patches))
        attention_map = cls_attention.reshape(grid_size, grid_size)
        
        # Normalize to [0, 1]
        attention_map = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min() + 1e-8)
        
        return attention_map
    
    def overlay_on_image(
        self, image: np.ndarray, attention_map: np.ndarray,
        colormap: str = "inferno", alpha: float = 0.5,
    ) -> np.ndarray:
        """Overlay attention map on original image."""
        # Downscale to prevent Out-Of-Memory on large inputs
        max_dim = 512
        h, w = image.shape[:2]
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            new_h, new_w = int(h * scale), int(w * scale)
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
            h, w = new_h, new_w
            
        attn_resized = cv2.resize(attention_map, (w, h), interpolation=cv2.INTER_CUBIC)
        
        cmap = plt.get_cmap(colormap)
        attn_colored = cmap(attn_resized)[:, :, :3]
        attn_colored = (attn_colored * 255).astype(np.uint8)
        
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        overlay = cv2.addWeighted(image, 1 - alpha, attn_colored, alpha, 0)
        return overlay


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  POINTING GAME METRIC                                                 ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def pointing_game_score(
    heatmap: np.ndarray,
    roi_mask: np.ndarray,
    tolerance: int = 15,
) -> Dict[str, float]:
    """
    Pointing Game: Does the maximum attention point fall within the clinical ROI?
    
    A "hit" means the model is looking at the right region. Higher hit rate =
    better interpretability alignment with clinical ground truth.
    
    Parameters
    ----------
    heatmap : np.ndarray
        XAI heatmap (H, W), values in [0, 1].
    roi_mask : np.ndarray
        Binary mask (H, W) of the clinically annotated region of interest.
    tolerance : int
        Pixel tolerance around the max point for a "hit".
        
    Returns
    -------
    dict
        {"hit": bool, "score": float, "max_point": (y, x)}
    """
    h, w = roi_mask.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w, h))
    
    # Find the point of maximum activation
    max_idx = np.unravel_index(heatmap_resized.argmax(), heatmap_resized.shape)
    max_y, max_x = max_idx
    
    # Check if max point is within tolerance of ROI
    roi_dilated = cv2.dilate(
        roi_mask.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (tolerance * 2 + 1, tolerance * 2 + 1)),
    )
    hit = bool(roi_dilated[max_y, max_x] > 0)
    
    # IoU between thresholded heatmap and ROI
    heatmap_binary = (heatmap_resized > 0.5).astype(np.uint8)
    roi_binary = (roi_mask > 0).astype(np.uint8)
    intersection = np.logical_and(heatmap_binary, roi_binary).sum()
    union = np.logical_or(heatmap_binary, roi_binary).sum()
    iou = intersection / (union + 1e-8)
    
    return {
        "hit": hit,
        "score": 1.0 if hit else 0.0,
        "iou": float(iou),
        "max_point": (int(max_y), int(max_x)),
    }


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  CLINICAL INTERPRETATION GENERATOR                                     ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def generate_clinical_interpretation(
    model_name: str,
    predicted_class: str,
    confidence: float,
    heatmap: np.ndarray,
    dataset_type: str = "xray",
) -> str:
    """
    Generate a plain-English clinical interpretation of the model's prediction.
    
    Analyzes the heatmap to identify which anatomical regions the model focused on,
    then produces a human-readable explanation suitable for panel Q&A.
    """
    h, w = heatmap.shape
    
    # Divide image into quadrants and find highest-activation region
    quadrants = {
        "upper-left": heatmap[:h//2, :w//2].mean(),
        "upper-right": heatmap[:h//2, w//2:].mean(),
        "lower-left": heatmap[h//2:, :w//2].mean(),
        "lower-right": heatmap[h//2:, w//2:].mean(),
    }
    primary_region = max(quadrants, key=quadrants.get)
    secondary_region = sorted(quadrants, key=quadrants.get, reverse=True)[1]
    
    # Map regions to anatomical descriptions
    if dataset_type == "xray":
        region_map = {
            "upper-left": "right upper lung field (apical zone)",
            "upper-right": "left upper lung field (apical zone)",
            "lower-left": "right lower lung field (basal zone)",
            "lower-right": "left lower lung field (basal zone)",
        }
        finding_map = {
            "PNEUMONIA": "consolidation pattern consistent with pneumonia presentation",
            "NORMAL": "no significant abnormalities",
        }
    else:
        region_map = {
            "upper-left": "upper-left tissue region",
            "upper-right": "upper-right tissue region",
            "lower-left": "lower-left tissue region",
            "lower-right": "lower-right tissue region",
        }
        finding_map = {
            "malignant": "cellular irregularities suggestive of invasive ductal carcinoma",
            "benign": "normal cellular morphology with no signs of malignancy",
        }
    
    primary_anatomy = region_map.get(primary_region, primary_region)
    finding = finding_map.get(predicted_class, "undetermined findings")
    
    conf_level = "high" if confidence > 0.85 else "moderate" if confidence > 0.65 else "low"
    
    interpretation = (
        f"🔍 **{model_name} Analysis** — {conf_level}-confidence prediction "
        f"({confidence:.1%})\n\n"
        f"The model classified this image as **{predicted_class}** and focused "
        f"primarily on the **{primary_anatomy}**, with secondary attention to the "
        f"**{region_map.get(secondary_region, secondary_region)}**.\n\n"
        f"This activation pattern suggests {finding}."
    )
    
    if conf_level == "low":
        interpretation += (
            "\n\n⚠️ *Low confidence — clinical correlation recommended. "
            "This prediction should be reviewed by a radiologist.*"
        )
    
    return interpretation


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  SIDE-BY-SIDE COMPARISON VISUALIZATION                                ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def create_comparison_figure(
    original_image: np.ndarray,
    cnn_heatmap: np.ndarray,
    vit_attention: np.ndarray,
    cnn_name: str = "ResNet-50",
    save_path: Optional[str] = None,
) -> np.ndarray:
    """
    Create a side-by-side comparison: Original | CNN Grad-CAM | ViT Attention.
    Returns the figure as a numpy array for display in Streamlit.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Original
    if len(original_image.shape) == 2:
        axes[0].imshow(original_image, cmap="gray")
    else:
        axes[0].imshow(original_image)
    axes[0].set_title("Original Image", fontsize=14, fontweight="bold")
    axes[0].axis("off")
    
    # CNN Grad-CAM
    gradcam_vis = GradCAM.__new__(GradCAM)  # Just use overlay method
    cnn_overlay = gradcam_vis.overlay_on_image(original_image.copy(), cnn_heatmap)
    axes[1].imshow(cnn_overlay)
    axes[1].set_title(f"{cnn_name} Grad-CAM", fontsize=14, fontweight="bold")
    axes[1].axis("off")
    
    # ViT Attention
    rollout = AttentionRollout()
    vit_overlay = rollout.overlay_on_image(original_image.copy(), vit_attention)
    axes[2].imshow(vit_overlay)
    axes[2].set_title("ViT Attention Rollout", fontsize=14, fontweight="bold")
    axes[2].axis("off")
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    
    # Convert figure to numpy array
    fig.canvas.draw()
    img_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img_array = img_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    
    return img_array
