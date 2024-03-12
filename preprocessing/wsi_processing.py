import os
import cv2
import numpy as np
import openslide
from concurrent.futures import ThreadPoolExecutor

def is_tissue(patch, threshold=220):
    """
    Simple tissue detection using HSV saturation and value.
    """
    hsv = cv2.cvtColor(patch, cv2.COLOR_RGB2HSV)
    saturation = hsv[:, :, 1]
    
    # Tissue usually has higher saturation than background
    return np.mean(saturation) > 20 and np.mean(patch) < threshold

def tile_wsi(slide_path, output_dir, patch_size=256, level=0):
    """
    Tiles a WSI into patches, saving only those with tissue.
    """
    slide = openslide.OpenSlide(slide_path)
    w, h = slide.level_dimensions[level]
    
    os.makedirs(output_dir, exist_ok=True)
    
    patches_saved = 0
    
    for y in range(0, h, patch_size):
        for x in range(0, w, patch_size):
            # Read region
            patch = slide.read_region((x, y), level, (patch_size, patch_size))
            patch = np.array(patch.convert('RGB'))
            
            if is_tissue(patch):
                save_path = os.path.join(output_dir, f"{x}_{y}.png")
                cv2.imwrite(save_path, cv2.cvtColor(patch, cv2.COLOR_RGB2BGR))
                patches_saved += 1
                
    print(f"Processed {slide_path}: {patches_saved} patches saved.")
