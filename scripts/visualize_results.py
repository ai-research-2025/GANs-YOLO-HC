#!/usr/bin/env python3
"""
Results visualization script for GANs-YOLO-HC project
Handles comparison between predictions and ground truth
"""

import os
import argparse
import sys
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from ultralytics import YOLO

# Define colors for each class (matching the notebook)
COLORS = [
    (255, 255, 255),   # Class 0: IHC - White
    (255, 0, 0),       # Class 1: OHC-1 - Red
    (0, 255, 0),       # Class 2: OHC-2 - Green
    (0, 0, 255),       # Class 3: OHC-3 - Blue
    (255, 255, 0)      # Class 4: OHC-4 - Yellow
]

# Class names (matching the user's configuration)
CLASS_NAMES = ['IHC', 'OHC-1', 'OHC-2', 'OHC-3', 'OHC-4']

def visualize_predictions(model_path, image_path, label_path=None, output_path=None):
    """
    Visualize model predictions vs ground truth
    Based on the comparison code from the notebook
    """
    # Load model
    try:
        model = YOLO(model_path)
        print(f"✓ Model loaded from: {model_path}")
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        return False
    
    # Load and process image
    try:
        # Read image
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"✗ Failed to load image: {image_path}")
            return False
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_height, image_width = image.shape[:2]
        
        print(f"✓ Image loaded: {image_path} ({image_width}x{image_height})")
        
    except Exception as e:
        print(f"✗ Failed to process image: {e}")
        return False
    
    # Create color mapping
    color_codes = {name: color for name, color in zip(CLASS_NAMES, COLORS)}
    
    # Create copies for visualization
    image_pred = image.copy()
    image_gt = image.copy() if label_path else None
    
    # Run prediction
    try:
        results = model(image)
        result = results[0]
        
        # Draw prediction bounding boxes
        if result.boxes:
            print(f"✓ Found {len(result.boxes)} predictions")
            
            for box in result.boxes:
                # Get box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().numpy()
                cls = int(box.cls[0].cpu().numpy())
                
                # Get class name and color
                class_name = CLASS_NAMES[cls] if cls < len(CLASS_NAMES) else f"class_{cls}"
                color = COLORS[cls % len(COLORS)]
                
                # Draw bounding box
                cv2.rectangle(image_pred, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                
                # Add label (optional)
                label = f"{class_name} {conf:.2f}"
                cv2.putText(image_pred, label, (int(x1), int(y1)-10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        else:
            print("✓ No predictions found")
        
    except Exception as e:
        print(f"✗ Prediction failed: {e}")
        return False
    
    # Process ground truth if available
    annotations = []
    if label_path and Path(label_path).exists():
        try:
            with open(label_path, 'r') as f:
                annotation_data = f.read()
            
            # Parse annotations
            for line in annotation_data.strip().split('\n'):
                if line.strip():
                    parts = line.split()
                    cls_id, x_center, y_center, width, height = map(float, parts)
                    annotations.append((int(cls_id), x_center, y_center, width, height))
            
            print(f"✓ Loaded {len(annotations)} ground truth annotations")
            
            # Draw ground truth bounding boxes
            for cls_id, x_center, y_center, width, height in annotations:
                # Convert YOLO format to pixel coordinates
                x = int((x_center - width / 2) * image_width)
                y = int((y_center - height / 2) * image_height)
                w = int(width * image_width)
                h = int(height * image_height)
                
                # Get color
                color = COLORS[cls_id % len(COLORS)]
                
                # Draw bounding box
                cv2.rectangle(image_gt, (x, y), (x + w, y + h), color, 2)
                
                # Add label
                class_name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else f"class_{cls_id}"
                cv2.putText(image_gt, class_name, (x, y-10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
        except Exception as e:
            print(f"✗ Failed to process ground truth: {e}")
            image_gt = None
    
    # Create visualization
    try:
        # Create legend patches
        legend_patches = [
            mpatches.Patch(color=np.array(color)/255, label=class_name) 
            for class_name, color in zip(CLASS_NAMES, COLORS)
        ]
        
        # Create subplot layout
        if image_gt is not None:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
            
            # Ground truth
            ax1.imshow(image_gt)
            ax1.set_title('Ground Truth', fontsize=16)
            ax1.axis('off')
            
            # Predictions
            ax2.imshow(image_pred)
            ax2.set_title('Model Predictions', fontsize=16)
            ax2.axis('off')
            
        else:
            fig, ax = plt.subplots(1, 1, figsize=(12, 8))
            
            # Only predictions
            ax.imshow(image_pred)
            ax.set_title('Model Predictions', fontsize=16)
            ax.axis('off')
        
        # Add legend
        fig.legend(handles=legend_patches, loc='upper center', 
                  ncol=len(CLASS_NAMES), fontsize=12)
        
        plt.tight_layout()
        
        # Save or show
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"✓ Visualization saved to: {output_path}")
        else:
            plt.show()
        
        plt.close()
        return True
        
    except Exception as e:
        print(f"✗ Visualization failed: {e}")
        return False

def process_test_images(model_path, test_dir, output_dir, args):
    """Process all images in test directory"""
    print(f"Processing test images from: {test_dir}")
    
    # Get all image files
    test_path = Path(test_dir)
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
        image_files.extend(test_path.glob(ext))
    
    if not image_files:
        print("✗ No image files found in test directory")
        return False
    
    print(f"Found {len(image_files)} images")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Process each image
    processed_count = 0
    for img_path in image_files:
        # Look for corresponding label file
        label_path = None
        if args.labels_dir:
            label_path = Path(args.labels_dir) / f"{img_path.stem}.txt"
            if not label_path.exists():
                label_path = None
        
        # Generate output filename
        output_filename = f"{img_path.stem}_comparison.png"
        output_file = output_path / output_filename
        
        # Process image
        success = visualize_predictions(model_path, img_path, label_path, output_file)
        
        if success:
            processed_count += 1
            print(f"✓ Processed: {img_path.name}")
        else:
            print(f"✗ Failed to process: {img_path.name}")
    
    print(f"\n✓ Processed {processed_count}/{len(image_files)} images")
    return processed_count > 0

def main():
    parser = argparse.ArgumentParser(description="Visualize YOLOv11 results")
    
    # Required arguments
    parser.add_argument("--model", required=True, help="Path to trained model file (.pt)")
    
    # Input options
    parser.add_argument("--image", help="Path to single image file")
    parser.add_argument("--test-dir", help="Directory containing test images")
    parser.add_argument("--labels-dir", help="Directory containing ground truth labels")
    
    # Output options
    parser.add_argument("--output", default="visualizations", help="Output directory for visualizations")
    parser.add_argument("--save", help="Save visualization to specific file")
    
    args = parser.parse_args()
    
    # Validate model file
    if not Path(args.model).exists():
        print(f"✗ Model file not found: {args.model}")
        return 1
    
    # Process based on input type
    if args.image:
        # Single image
        image_path = Path(args.image)
        if not image_path.exists():
            print(f"✗ Image file not found: {args.image}")
            return 1
        
        # Look for corresponding label
        label_path = None
        if args.labels_dir:
            label_path = Path(args.labels_dir) / f"{image_path.stem}.txt"
            if not label_path.exists():
                label_path = None
        
        output_path = args.save if args.save else None
        success = visualize_predictions(args.model, image_path, label_path, output_path)
        
    elif args.test_dir:
        # Directory of images
        if not Path(args.test_dir).exists():
            print(f"✗ Test directory not found: {args.test_dir}")
            return 1
        
        success = process_test_images(args.model, args.test_dir, args.output, args)
        
    else:
        print("✗ Please specify either --image or --test-dir")
        return 1
    
    if success:
        print("\n✅ Visualization completed successfully!")
        return 0
    else:
        print("\n❌ Visualization failed!")
        return 1

if __name__ == "__main__":
    exit(main())
