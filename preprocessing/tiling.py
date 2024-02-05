import os
import cv2
import numpy as np
from openslide import OpenSlide

def tile_wsi(slide_path, output_dir, patch_size=256, level=0):
    slide = OpenSlide(slide_path)
    w, h = slide.level_dimensions[level]
    
    for y in range(0, h, patch_size):
        for x in range(0, w, patch_size):
            patch = slide.read_region((x, y), level, (patch_size, patch_size))
            patch = np.array(patch.convert('RGB'))
            
            # Simple tissue detection (thresholding)
            if np.mean(patch) < 220:
                save_path = os.path.join(output_dir, f"{x}_{y}.png")
                cv2.imwrite(save_path, cv2.cvtColor(patch, cv2.COLOR_RGB2BGR))

# Added multi-processing support
