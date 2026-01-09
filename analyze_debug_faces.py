#!/usr/bin/env python3
"""
Analyze debug face images to verify preprocessing quality.
Shows sample images and statistics about predictions.
"""

import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from collections import Counter
import re

def parse_filename(filename):
    """Extract emotion and confidence from filename."""
    # Format: HHMMSS_mmm_EMOTION_CONFIDENCE.png
    parts = filename.stem.split('_')
    if len(parts) >= 4:
        emotion = parts[2]
        confidence = float(parts[3])
        return emotion, confidence
    return None, None

def analyze_debug_faces(debug_dir="debug_faces", show_samples=True):
    """Analyze all debug face images."""
    
    debug_path = Path(debug_dir)
    if not debug_path.exists():
        print(f"❌ Directory {debug_dir} not found!")
        return
    
    image_files = list(debug_path.glob("*.png"))
    if not image_files:
        print(f"❌ No PNG files found in {debug_dir}!")
        return
    
    print("=" * 60)
    print(f"Debug Faces Analysis: {len(image_files)} images")
    print("=" * 60)
    
    # Parse all filenames
    emotions = []
    confidences = []
    
    for img_file in image_files:
        emotion, conf = parse_filename(img_file)
        if emotion and conf:
            emotions.append(emotion)
            confidences.append(conf)
    
    # Statistics
    print(f"\n📊 Prediction Statistics:")
    print(f"   Total images: {len(image_files)}")
    
    emotion_counts = Counter(emotions)
    print(f"\n   Emotion distribution:")
    for emotion, count in emotion_counts.most_common():
        pct = (count / len(emotions)) * 100
        print(f"      {emotion}: {count} ({pct:.1f}%)")
    
    if confidences:
        print(f"\n   Confidence statistics:")
        print(f"      Mean: {np.mean(confidences):.3f}")
        print(f"      Min:  {np.min(confidences):.3f}")
        print(f"      Max:  {np.max(confidences):.3f}")
        print(f"      Std:  {np.std(confidences):.3f}")
    
    # Check image quality
    print(f"\n🔍 Image Quality Check:")
    
    # Load a few sample images
    sample_files = sorted(image_files)[:5]
    
    for img_file in sample_files:
        img = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            emotion, conf = parse_filename(img_file)
            print(f"\n   {img_file.name}")
            print(f"      Shape: {img.shape}")
            print(f"      Dtype: {img.dtype}")
            print(f"      Range: [{img.min()}, {img.max()}]")
            print(f"      Mean:  {img.mean():.1f}")
            print(f"      Prediction: {emotion} ({conf:.3f})")
    
    # Visual check
    if show_samples:
        print(f"\n📸 Showing sample images...")
        show_sample_grid(image_files[:16])
    
    # Final verdict
    print("\n" + "=" * 60)
    print("✅ Verdict:")
    print("=" * 60)
    
    if len(emotion_counts) == 1 and "sad" in emotion_counts:
        print("⚠️  ALL predictions are 'sad'")
        print("   → This confirms FER2013 neutral/sad confusion issue")
        print("   → If faces in images look neutral/normal, the model is wrong")
        print("   → Solution: Retrain with FER+ (see fer_plus_guide.md)")
    elif np.mean(confidences) < 0.6:
        print("⚠️  Low average confidence ({:.1f}%)".format(np.mean(confidences) * 100))
        print("   → Model is uncertain about predictions")
        print("   → Suggests training data quality issues (FER2013)")
    else:
        print("✅ Predictions look reasonable")
        print("   → Good confidence levels")
        print("   → Multiple emotions detected")
    
    print("\n💡 Next Steps:")
    print("   1. View sample images above")
    print("   2. If faces look neutral but predicted as 'sad' → FER2013 issue")
    print("   3. Follow fer_plus_guide.md to retrain with better data")
    print("=" * 60)

def show_sample_grid(image_files, grid_size=(4, 4)):
    """Display a grid of sample images."""
    
    n_images = min(len(image_files), grid_size[0] * grid_size[1])
    
    fig, axes = plt.subplots(grid_size[0], grid_size[1], figsize=(12, 12))
    fig.suptitle('Debug Face Samples (48x48 preprocessed)', fontsize=16)
    
    for idx, ax in enumerate(axes.flat):
        if idx < n_images:
            img_file = image_files[idx]
            img = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
            
            emotion, conf = parse_filename(img_file)
            
            ax.imshow(img, cmap='gray', vmin=0, vmax=255)
            ax.set_title(f'{emotion}\n{conf:.2f}', fontsize=10)
            ax.axis('off')
        else:
            ax.axis('off')
    
    plt.tight_layout()
    
    # Save to file
    output_file = Path("debug_faces_analysis.png")
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"   Saved grid to: {output_file}")
    
    # Try to open (macOS)
    try:
        import subprocess
        subprocess.run(['open', str(output_file)], check=False)
        print(f"   Opening in default viewer...")
    except:
        print(f"   View it manually: open {output_file}")

def open_random_sample():
    """Open a random debug image for inspection."""
    debug_path = Path("debug_faces")
    images = list(debug_path.glob("*.png"))
    
    if images:
        import random
        sample = random.choice(images)
        print(f"\n🎲 Opening random sample: {sample.name}")
        
        try:
            import subprocess
            subprocess.run(['open', str(sample)], check=False)
        except:
            print(f"   Manually open: open {sample}")

if __name__ == "__main__":
    import sys
    
    show_samples = True
    if len(sys.argv) > 1 and sys.argv[1] == '--no-plot':
        show_samples = False
    
    analyze_debug_faces(show_samples=show_samples)
    
    # Ask if user wants to see a random sample
    print("\n" + "=" * 60)
    print("Commands to view images:")
    print("=" * 60)
    print("# View grid visualization")
    print("open debug_faces_analysis.png")
    print("\n# View individual images")
    print("open debug_faces/")
    print("\n# View specific image")
    print("open debug_faces/144616_029_sad_0.516.png")
    print("=" * 60)
