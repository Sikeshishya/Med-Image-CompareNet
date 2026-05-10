"""
============================================================================
COMPARATIVE ANALYSIS ENGINE
============================================================================
Produces structured comparison reports: CNN vs ViT across both modalities.
Generates evidence-based recommendations and exportable PDF reports.
============================================================================
"""

import os, json, logging
from typing import Dict, List, Optional
from datetime import datetime
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger("MedImageCompareNet")


class ComparativeAnalyzer:
    """Compares CNN and ViT performance across X-ray and Pathology datasets."""

    def __init__(self):
        self.results = {}

    def add_result(self, model_name: str, dataset: str, metrics: Dict):
        """Store evaluation results for a model-dataset pair."""
        key = f"{model_name}_{dataset}"
        self.results[key] = {
            "model": model_name, "dataset": dataset,
            "metrics": metrics, "timestamp": datetime.now().isoformat(),
        }

    def generate_comparison_table(self) -> Dict:
        """Build a structured comparison across all model-dataset pairs."""
        table = {"headers": ["Model", "Dataset", "Accuracy", "F1", "AUC-ROC", "Inference (ms)"], "rows": []}
        for key, r in self.results.items():
            m = r["metrics"]
            table["rows"].append([
                r["model"], r["dataset"],
                f"{m.get('accuracy', 0):.4f}",
                f"{m.get('f1', 0):.4f}",
                f"{m.get('auc_roc', 0):.4f}",
                f"{m.get('inference_time_ms', 0):.1f}",
            ])
        return table

    def generate_recommendation(self) -> str:
        """Evidence-based recommendation: which model wins on which modality."""
        xray_results, path_results = {}, {}
        for key, r in self.results.items():
            bucket = xray_results if r["dataset"] == "xray" else path_results
            bucket[r["model"]] = r["metrics"]

        rec = "# 📊 Evidence-Based Recommendations\n\n"

        # X-ray recommendation
        if xray_results:
            best_xray = max(xray_results, key=lambda m: xray_results[m].get("f1", 0))
            f1 = xray_results[best_xray].get("f1", 0)
            rec += (
                f"## X-Ray Classification\n"
                f"**Recommended: {best_xray}** (F1={f1:.4f})\n\n"
                f"CNNs typically excel on X-ray images because their inductive biases "
                f"(locality, translation equivariance) align well with the spatial "
                f"patterns in radiographic images. The hierarchical feature extraction "
                f"captures both local texture (lung markings) and global structure.\n\n"
            )

        # Pathology recommendation
        if path_results:
            best_path = max(path_results, key=lambda m: path_results[m].get("f1", 0))
            f1 = path_results[best_path].get("f1", 0)
            rec += (
                f"## Pathology Classification\n"
                f"**Recommended: {best_path}** (F1={f1:.4f})\n\n"
                f"For histopathology patches, ViT's global self-attention can capture "
                f"long-range dependencies between cellular structures. However, CNNs "
                f"remain competitive on small patches where local features dominate.\n\n"
            )

        return rec

    def plot_comparison_charts(self, save_dir: str = "figures") -> Dict[str, str]:
        """Generate publication-quality comparison visualizations."""
        os.makedirs(save_dir, exist_ok=True)
        paths = {}

        # Grouped bar chart: Accuracy comparison
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        for idx, dataset in enumerate(["xray", "pathology"]):
            models_list, accs, f1s, aucs = [], [], [], []
            for key, r in self.results.items():
                if r["dataset"] == dataset:
                    models_list.append(r["model"])
                    accs.append(r["metrics"].get("accuracy", 0))
                    f1s.append(r["metrics"].get("f1", 0))
                    aucs.append(r["metrics"].get("auc_roc", 0))
            
            if not models_list:
                continue
            
            x = np.arange(len(models_list))
            w = 0.25
            axes[idx].bar(x - w, accs, w, label="Accuracy", color="#4ECDC4")
            axes[idx].bar(x, f1s, w, label="F1-Score", color="#FF6B6B")
            axes[idx].bar(x + w, aucs, w, label="AUC-ROC", color="#45B7D1")
            axes[idx].set_xticks(x)
            axes[idx].set_xticklabels(models_list, rotation=15)
            axes[idx].set_title(f"{'X-Ray' if dataset == 'xray' else 'Pathology'} — Model Comparison", fontsize=13, fontweight="bold")
            axes[idx].legend()
            axes[idx].set_ylim(0, 1.05)
            axes[idx].grid(axis="y", alpha=0.3)

        plt.tight_layout()
        path = os.path.join(save_dir, "metric_comparison.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        paths["metric_comparison"] = path

        # Inference time comparison
        fig, ax = plt.subplots(figsize=(8, 5))
        models_list, times = [], []
        for key, r in self.results.items():
            models_list.append(f"{r['model']}\n({r['dataset']})")
            times.append(r["metrics"].get("inference_time_ms", 0))
        
        colors = ["#4ECDC4" if "resnet" in m.lower() or "dense" in m.lower() else "#FF6B6B" for m in models_list]
        ax.barh(models_list, times, color=colors)
        ax.set_xlabel("Inference Time (ms)", fontsize=12)
        ax.set_title("Inference Speed Comparison", fontsize=13, fontweight="bold")
        ax.grid(axis="x", alpha=0.3)
        
        plt.tight_layout()
        path = os.path.join(save_dir, "inference_time.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        paths["inference_time"] = path

        return paths

    def export_pdf_report(self, save_path: str = "exports/comparison_report.pdf"):
        """Export the full comparison report as a PDF."""
        import re
        from fpdf import FPDF

        def strip_emoji(text):
            """Remove emoji and non-Latin1 characters for PDF compatibility."""
            # Remove emoji unicode ranges
            text = re.sub(r'[^\x00-\x7F\x80-\xFF]+', '', text)
            return text.strip()

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        
        # Title page
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 24)
        pdf.cell(0, 30, "Med-Image CompareNet", ln=True, align="C")
        pdf.set_font("Helvetica", "", 14)
        pdf.cell(0, 10, "Comparative Analysis Report", ln=True, align="C")
        pdf.cell(0, 10, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True, align="C")
        pdf.ln(20)

        # Comparison table
        pdf.set_font("Helvetica", "B", 14)
        pdf.cell(0, 10, "Performance Comparison", ln=True)
        pdf.ln(5)
        
        table = self.generate_comparison_table()
        pdf.set_font("Helvetica", "B", 9)
        col_widths = [35, 25, 25, 20, 25, 30]
        for i, h in enumerate(table["headers"]):
            pdf.cell(col_widths[i], 8, strip_emoji(h), border=1, align="C")
        pdf.ln()
        
        pdf.set_font("Helvetica", "", 9)
        for row in table["rows"]:
            for i, cell in enumerate(row):
                pdf.cell(col_widths[i], 8, strip_emoji(str(cell)), border=1, align="C")
            pdf.ln()
        
        pdf.ln(10)

        # Recommendation
        pdf.set_font("Helvetica", "B", 14)
        pdf.cell(0, 10, "Recommendations", ln=True)
        pdf.set_font("Helvetica", "", 10)
        rec = self.generate_recommendation()
        # Strip markdown formatting and emojis for PDF
        for line in rec.split("\n"):
            line = line.replace("#", "").replace("**", "").replace("*", "").strip()
            line = strip_emoji(line)
            if line:
                pdf.multi_cell(0, 6, line)
        
        # Add charts if they exist
        chart_paths = self.plot_comparison_charts()
        for name, path in chart_paths.items():
            if os.path.exists(path):
                pdf.add_page()
                pdf.set_font("Helvetica", "B", 14)
                pdf.cell(0, 10, name.replace("_", " ").title(), ln=True)
                pdf.image(path, x=10, w=190)
        
        pdf.output(save_path)
        logger.info(f"PDF report saved to {save_path}")
        return save_path

    def save_json(self, path: str = "results/comparison_results.json"):
        """Save all results as JSON for programmatic access."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # Convert any numpy types for JSON serialization
        def convert(obj):
            if isinstance(obj, (np.integer,)): return int(obj)
            if isinstance(obj, (np.floating,)): return float(obj)
            if isinstance(obj, np.ndarray): return obj.tolist()
            return obj
        
        with open(path, "w") as f:
            json.dump(self.results, f, indent=2, default=convert)
        logger.info(f"Results saved → {path}")
