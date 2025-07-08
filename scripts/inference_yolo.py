#!/usr/bin/env python3
"""
YOLOv11 inference script for GANs-YOLO-HC project
Handles model inference and prediction on images
"""

import os
import argparse
import sys
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO

def run_inference(model_path, source_path, args):
    """Run inference on source images"""
    print(f"Loading model from: {model_path}")
    
    # Load model
    try:
        model = YOLO(model_path)
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        return False
    
    # Check source
    source = Path(source_path)
    if not source.exists():
        print(f"✗ Source not found: {source_path}")
        return False
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process based on source type
    if source.is_file():
        # Single image
        print(f"Processing single image: {source}")
        results = process_image(model, source, output_dir, args)
        
    elif source.is_dir():
        # Directory of images
        print(f"Processing directory: {source}")
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
            image_files.extend(source.glob(ext))
        
        if not image_files:
            print("✗ No image files found in directory")
            return False
        
        print(f"Found {len(image_files)} images")
        
        if args.batch:
            # Batch processing
            results = process_batch(model, image_files, output_dir, args)
        else:
            # Individual processing
            results = []
            for img_file in image_files:
                result = process_image(model, img_file, output_dir, args)
                if result:
                    results.append(result)
    
    else:
        print(f"✗ Invalid source: {source_path}")
        return False
    
    print(f"✓ Inference completed. Results saved to: {output_dir}")
    return True

def process_image(model, image_path, output_dir, args):
    """Process a single image"""
    try:
        # Run prediction
        results = model(str(image_path), conf=args.confidence)
        
        # Save results
        result = results[0]
        
        # Generate output filename
        output_name = f"{Path(image_path).stem}_predicted{Path(image_path).suffix}"
        output_path = output_dir / output_name
        
        # Save annotated image
        result.save(str(output_path))
        
        # Print detailed results if requested
        if args.detailed:
            print(f"\nResults for {image_path.name}:")
            print(f"  Detections: {len(result.boxes) if result.boxes else 0}")
            
            if result.boxes:
                for i, box in enumerate(result.boxes):
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    coords = box.xyxy[0].tolist()
                    
                    # Get class name
                    class_name = model.names[cls] if cls in model.names else f"class_{cls}"
                    
                    print(f"    Detection {i+1}: {class_name} ({conf:.3f}) at {coords}")
            
            print(f"  Saved to: {output_path}")
        
        return {
            'image': str(image_path),
            'output': str(output_path),
            'detections': len(result.boxes) if result.boxes else 0
        }
        
    except Exception as e:
        print(f"✗ Failed to process {image_path}: {e}")
        return None

def process_batch(model, image_files, output_dir, args):
    """Process multiple images in batch"""
    try:
        # Convert to string paths
        image_paths = [str(img) for img in image_files]
        
        # Run batch prediction
        results = model(image_paths, conf=args.confidence)
        
        # Save results
        batch_results = []
        for i, result in enumerate(results):
            image_path = Path(image_files[i])
            output_name = f"{image_path.stem}_predicted{image_path.suffix}"
            output_path = output_dir / output_name
            
            # Save annotated image
            result.save(str(output_path))
            
            batch_results.append({
                'image': str(image_path),
                'output': str(output_path),
                'detections': len(result.boxes) if result.boxes else 0
            })
            
            if args.detailed:
                print(f"✓ Processed {image_path.name}: {len(result.boxes) if result.boxes else 0} detections")
        
        return batch_results
        
    except Exception as e:
        print(f"✗ Batch processing failed: {e}")
        return []

def main():
    parser = argparse.ArgumentParser(description="Run YOLOv11 inference on images")
    
    # Required arguments
    parser.add_argument("--model", required=True, help="Path to trained model file (.pt)")
    parser.add_argument("--source", required=True, help="Path to image file or directory")
    
    # Optional arguments
    parser.add_argument("--output", default="results", help="Output directory for results")
    parser.add_argument("--confidence", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--batch", action="store_true", help="Process all images in batch")
    parser.add_argument("--detailed", action="store_true", help="Show detailed detection information")
    
    args = parser.parse_args()
    
    # Validate model file
    if not Path(args.model).exists():
        print(f"✗ Model file not found: {args.model}")
        print("Available models:")
        
        # Look for trained models
        runs_dir = Path("runs/train")
        if runs_dir.exists():
            for exp_dir in runs_dir.iterdir():
                if exp_dir.is_dir():
                    weights_dir = exp_dir / "weights"
                    if weights_dir.exists():
                        for weight_file in weights_dir.glob("*.pt"):
                            print(f"  - {weight_file}")
        
        return 1
    
    # Run inference
    success = run_inference(args.model, args.source, args)
    
    if success:
        print("\n✅ Inference completed successfully!")
        return 0
    else:
        print("\n❌ Inference failed!")
        return 1

if __name__ == "__main__":
    exit(main())
