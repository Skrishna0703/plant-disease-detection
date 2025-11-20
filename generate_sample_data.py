"""
Generate sample synthetic plant leaf images for training.
Creates realistic-looking leaf images with various disease patterns.
"""

import os
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import random
from pathlib import Path

# Configuration
DATA_DIR = 'data/train'
IMG_SIZE = 224
SAMPLES_PER_CLASS = 50  # 50 images per class for quick training
SEED = 42

random.seed(SEED)
np.random.seed(SEED)

# Disease classes (matching your labels)
CLASSES = [
    'Apple___Apple_scab',
    'Apple___Black_rot',
    'Apple___Cedar_apple_rust',
    'Apple___healthy',
    'Blueberry___healthy',
    'Cherry_(including_sour)___healthy',
    'Cherry_(including_sour)___Powdery_mildew',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_(maize)___Common_rust_',
    'Corn_(maize)___healthy',
    'Corn_(maize)___Northern_Leaf_Blight',
    'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)',
    'Grape___healthy',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Orange___Haunglongbing_(Citrus_greening)',
    'Peach___Bacterial_spot',
    'Peach___healthy',
    'Pepper,_bell___Bacterial_spot',
    'Pepper,_bell___healthy',
    'Potato___Early_blight',
    'Potato___healthy',
    'Potato___Late_blight',
    'Raspberry___healthy',
    'Soybean___healthy',
    'Squash___Powdery_mildew',
    'Strawberry___healthy',
    'Strawberry___Leaf_scorch',
    'Tomato___Bacterial_spot',
    'Tomato___Early_blight',
    'Tomato___healthy',
    'Tomato___Late_blight',
    'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot',
    'Tomato___Tomato_mosaic_virus',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
]

def create_leaf_shape(size):
    """Create a basic leaf shape."""
    img = Image.new('RGB', (size, size), color=(255, 255, 255))
    draw = ImageDraw.Draw(img, 'RGBA')
    
    # Create leaf-like ellipse
    margin = size // 6
    draw.ellipse(
        [(margin, margin), (size - margin, size - margin)],
        fill=(34, 139, 34, 200),  # Forest green
        outline=(0, 100, 0, 255)
    )
    
    return img

def add_healthy_texture(img):
    """Add texture to healthy leaf."""
    arr = np.array(img, dtype=np.float32)
    noise = np.random.normal(0, 5, arr.shape)
    arr = np.clip(arr + noise, 0, 255)
    return Image.fromarray(arr.astype(np.uint8))

def add_disease_spots(img, disease_type):
    """Add disease-specific spots/patterns."""
    arr = np.array(img, dtype=np.float32)
    h, w = arr.shape[:2]
    
    if 'healthy' in disease_type.lower():
        # Just add subtle texture
        noise = np.random.normal(0, 3, arr.shape)
        arr = np.clip(arr + noise, 0, 255)
    
    elif 'scab' in disease_type.lower():
        # Brown scab spots
        for _ in range(random.randint(5, 15)):
            y, x = random.randint(h//4, 3*h//4), random.randint(w//4, 3*w//4)
            r = random.randint(10, 30)
            yy, xx = np.ogrid[-r:r+1, -r:r+1]
            mask = xx*xx + yy*yy <= r*r
            py, px = np.where(mask)
            py, px = np.clip(y + py, 0, h-1), np.clip(x + px, 0, w-1)
            arr[py, px] = [139, 69, 19]  # Brown
    
    elif 'black_rot' in disease_type.lower() or 'black' in disease_type.lower():
        # Black spots
        for _ in range(random.randint(3, 8)):
            y, x = random.randint(h//4, 3*h//4), random.randint(w//4, 3*w//4)
            r = random.randint(15, 40)
            yy, xx = np.ogrid[-r:r+1, -r:r+1]
            mask = xx*xx + yy*yy <= r*r
            py, px = np.where(mask)
            py, px = np.clip(y + py, 0, h-1), np.clip(x + px, 0, w-1)
            arr[py, px] = [0, 0, 0]  # Black
    
    elif 'blight' in disease_type.lower():
        # Yellow/brown blight patterns
        for _ in range(random.randint(8, 15)):
            y, x = random.randint(h//4, 3*h//4), random.randint(w//4, 3*w//4)
            r = random.randint(8, 25)
            yy, xx = np.ogrid[-r:r+1, -r:r+1]
            mask = xx*xx + yy*yy <= r*r
            py, px = np.where(mask)
            py, px = np.clip(y + py, 0, h-1), np.clip(x + px, 0, w-1)
            arr[py, px] = [184, 134, 11]  # Dark goldenrod
    
    elif 'spot' in disease_type.lower():
        # Brown spots
        for _ in range(random.randint(10, 20)):
            y, x = random.randint(h//4, 3*h//4), random.randint(w//4, 3*w//4)
            r = random.randint(5, 20)
            yy, xx = np.ogrid[-r:r+1, -r:r+1]
            mask = xx*xx + yy*yy <= r*r
            py, px = np.where(mask)
            py, px = np.clip(y + py, 0, h-1), np.clip(x + px, 0, w-1)
            arr[py, px] = [165, 42, 42]  # Brown
    
    elif 'mildew' in disease_type.lower() or 'powdery' in disease_type.lower():
        # White powder-like spots
        for _ in range(random.randint(15, 30)):
            y, x = random.randint(h//4, 3*h//4), random.randint(w//4, 3*w//4)
            r = random.randint(3, 12)
            yy, xx = np.ogrid[-r:r+1, -r:r+1]
            mask = xx*xx + yy*yy <= r*r
            py, px = np.where(mask)
            py, px = np.clip(y + py, 0, h-1), np.clip(x + px, 0, w-1)
            arr[py, px] = [200, 200, 200]  # Light gray/white
    
    elif 'rust' in disease_type.lower():
        # Orange/rust colored spots
        for _ in range(random.randint(10, 20)):
            y, x = random.randint(h//4, 3*h//4), random.randint(w//4, 3*w//4)
            r = random.randint(5, 18)
            yy, xx = np.ogrid[-r:r+1, -r:r+1]
            mask = xx*xx + yy*yy <= r*r
            py, px = np.where(mask)
            py, px = np.clip(y + py, 0, h-1), np.clip(x + px, 0, w-1)
            arr[py, px] = [255, 140, 0]  # Dark orange
    
    else:
        # Generic disease pattern
        noise = np.random.normal(0, 10, arr.shape)
        arr = np.clip(arr + noise, 0, 255)
    
    return Image.fromarray(arr.astype(np.uint8))

def generate_sample_image(disease_class):
    """Generate a synthetic sample image for a disease class."""
    # Create base leaf
    img = create_leaf_shape(IMG_SIZE)
    
    # Add disease pattern
    img = add_disease_spots(img, disease_class)
    
    # Add some blur to make it more realistic
    img = img.filter(ImageFilter.GaussianBlur(radius=1))
    
    return img

def main():
    """Generate sample dataset."""
    print(f"Generating {len(CLASSES)} disease classes with {SAMPLES_PER_CLASS} samples each...")
    print(f"Total images: {len(CLASSES) * SAMPLES_PER_CLASS}")
    
    for class_name in CLASSES:
        class_dir = os.path.join(DATA_DIR, class_name)
        os.makedirs(class_dir, exist_ok=True)
        
        # Generate samples
        for i in range(SAMPLES_PER_CLASS):
            img = generate_sample_image(class_name)
            img_path = os.path.join(class_dir, f"sample_{i:03d}.jpg")
            img.save(img_path, 'JPEG')
        
        print(f"✓ Generated {SAMPLES_PER_CLASS} images for {class_name}")
    
    print("\n✅ Sample dataset generation complete!")
    print(f"Dataset location: {os.path.abspath(DATA_DIR)}")

if __name__ == "__main__":
    main()
