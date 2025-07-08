#!/usr/bin/env python3
"""
YOLOv11 training script for GANs-YOLO-HC project
Handles model training with various configurations
"""

import os
import argparse
import sys
from pathlib import Path
from ultralytics import YOLO

def train_yolo(args):
    """Train YOLOv11 model with specified parameters"""
    print("Starting YOLOv11 training...")
    
    # Validate dataset path
    config_path = Path(args.data) / "config.yaml"
    if not config_path.exists():
        print(f"✗ Config file not found: {config_path}")
        print("Please make sure you have copied the dataset to data/output_320/")
        return False
    
    # Load or create model
    if args.pretrained and Path(args.pretrained).exists():
        print(f"Loading pretrained model: {args.pretrained}")
        model = YOLO(args.pretrained)
    else:
        print(f"Creating new YOLOv11{args.model_size} model...")
        model = YOLO(f'yolo11{args.model_size}.pt')
    
    # Set training parameters
    train_params = {
        'data': str(config_path),
        'epochs': args.epochs,
        'imgsz': args.img_size,
        'batch': args.batch_size,
        'device': args.device,
        'project': 'runs/train',
        'name': args.name,
        'save': True,
        'plots': True,
        'val': True
    }
    
    # Add optional parameters
    if args.lr:
        train_params['lr0'] = args.lr
    if args.patience:
        train_params['patience'] = args.patience
    
    print(f"Training parameters: {train_params}")
    
    try:
        # Train the model
        results = model.train(**train_params)
        
        print("✓ Training completed successfully!")
        print(f"✓ Model saved to: runs/train/{args.name}")
        print(f"✓ Best weights: runs/train/{args.name}/weights/best.pt")
        
        # Run validation if requested
        if args.validate:
            print("\nRunning validation...")
            val_results = model.val()
            print(f"✓ Validation completed")
            
        return True
        
    except Exception as e:
        print(f"✗ Training failed: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Train YOLOv11 model for hair cell detection")
    
    # Dataset parameters
    parser.add_argument("--data", default="data/output_320", help="Dataset directory")
    
    # Model parameters
    parser.add_argument("--model-size", choices=['n', 's', 'm', 'l', 'x'], default='l', 
                       help="YOLOv11 model size (n=nano, s=small, m=medium, l=large, x=extra-large)")
    parser.add_argument("--pretrained", help="Path to pretrained model weights")
    
    # Training parameters
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--img-size", type=int, default=640, help="Image size")
    parser.add_argument("--device", default="", help="Device to use (cuda:0, cpu, etc.)")
    parser.add_argument("--name", default="yolov11_hair_cells", help="Experiment name")
    
    # Optimization parameters
    parser.add_argument("--lr", type=float, help="Learning rate")
    parser.add_argument("--patience", type=int, help="Early stopping patience")
    
    # Actions
    parser.add_argument("--validate", action="store_true", help="Run validation after training")
    parser.add_argument("--resume", help="Resume training from checkpoint")
    
    args = parser.parse_args()
    
    # Check if resume is specified
    if args.resume:
        print(f"Resuming training from: {args.resume}")
        try:
            model = YOLO(args.resume)
            results = model.train(resume=True)
            print("✓ Resume training completed!")
            return 0
        except Exception as e:
            print(f"✗ Resume training failed: {e}")
            return 1
    
    # Regular training
    success = train_yolo(args)
    
    if success:
        print("\n🎉 Training pipeline completed successfully!")
        print("\nNext steps:")
        print("1. Check training results in runs/train/")
        print("2. Run inference: python scripts/inference_yolo.py")
        print("3. Visualize results: python scripts/visualize_results.py")
        return 0
    else:
        print("\n❌ Training failed!")
        return 1

if __name__ == "__main__":
    exit(main())
