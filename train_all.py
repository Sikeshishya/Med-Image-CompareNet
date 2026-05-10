"""
============================================================================
Med-Image CompareNet — Master Training Script
============================================================================
Runs all modules in sequence:
  1. Enhancement training (U-Net + GAN)
  2. CNN training (ResNet-50 + DenseNet-121 on both datasets)
  3. ViT training (ViT-B/16 on both datasets)
  4. Comparative analysis & report generation

Usage:
    python train_all.py
    python train_all.py --skip-enhancement
    python train_all.py --module cnn --dataset xray
============================================================================
"""

import argparse
import logging
from src.utils import load_config, set_seed, setup_logging
from src.datasets import create_dataloaders


def main():
    parser = argparse.ArgumentParser(description="Med-Image CompareNet Training")
    parser.add_argument("--module", choices=["all", "enhance", "cnn", "vit", "compare"], default="all")
    parser.add_argument("--dataset", choices=["all", "xray", "pathology"], default="all")
    parser.add_argument("--skip-enhancement", action="store_true")
    args = parser.parse_args()

    config = load_config("config.yaml")
    set_seed(config["project"]["seed"])
    logger = setup_logging(config["paths"]["logs"])

    logger.info("=" * 60)
    logger.info("Med-Image CompareNet — Training Pipeline")
    logger.info("=" * 60)

    datasets = ["xray", "pathology"] if args.dataset == "all" else [args.dataset]

    # Module 1: Enhancement
    if args.module in ["all", "enhance"] and not args.skip_enhancement:
        logger.info("\n🖼️  MODULE 1: X-Ray Enhancement (U-Net + GAN)")
        from src.enhance import EnhancementTrainer, NoisyXRayDataset
        from torch.utils.data import DataLoader
        cfg = config["enhancement"]
        dc = config["data"]["xray"]
        train_ds = NoisyXRayDataset(dc["root_dir"], cfg["noise_level"], 256, "train")
        val_ds = NoisyXRayDataset(dc["root_dir"], cfg["noise_level"], 256, "val")
        tl = DataLoader(train_ds, cfg["training"]["batch_size"], shuffle=True, num_workers=4)
        vl = DataLoader(val_ds, cfg["training"]["batch_size"], shuffle=False, num_workers=4)
        trainer = EnhancementTrainer(config)
        trainer.train(tl, vl)

    # Module 2: CNN Classification
    if args.module in ["all", "cnn"]:
        logger.info("\n🧬  MODULE 2: CNN Classification")
        from src.cnn_model import CNNTrainer
        for ds in datasets:
            loaders = create_dataloaders(config, ds)
            for model_cfg in config["cnn"]["models"]:
                name = model_cfg["name"]
                logger.info(f"Training {name} on {ds}...")
                trainer = CNNTrainer(config, name, ds)
                trainer.train(loaders)
                results = trainer.evaluate(loaders["test"])
                logger.info(f"  → Accuracy: {results['accuracy']:.4f} | F1: {results['f1']:.4f}")

    # Module 3: ViT Classification
    if args.module in ["all", "vit"]:
        logger.info("\n🤖  MODULE 3: ViT Classification")
        from src.vit_model import ViTTrainer
        for ds in datasets:
            loaders = create_dataloaders(config, ds)
            trainer = ViTTrainer(config, ds)
            trainer.train(loaders)
            results = trainer.evaluate(loaders["test"])
            logger.info(f"  → Accuracy: {results['accuracy']:.4f} | F1: {results['f1']:.4f}")

    # Comparative Analysis
    if args.module in ["all", "compare"]:
        logger.info("\n📊  Generating Comparative Analysis")
        from src.compare import ComparativeAnalyzer
        analyzer = ComparativeAnalyzer()
        # Load results from checkpoints or use latest training results
        analyzer.plot_comparison_charts()
        report_path = analyzer.export_pdf_report()
        logger.info(f"Report saved → {report_path}")

    logger.info("\n✅ Training pipeline complete!")


if __name__ == "__main__":
    main()
