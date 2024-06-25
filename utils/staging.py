import numpy as np

def calculate_tumor_diameter(tumor_area_mm2):
    """
    Estimates tumor diameter from area assuming a roughly circular shape.
    """
    return 2 * np.sqrt(tumor_area_mm2 / np.pi)

def get_pn_stage(tumor_size_mm):
    """
    Determines pN stage based on tumor size (mm).
    
    Rules:
    - pN0: No tumor cells
    - pN0(i+): <= 0.2 mm
    - pN1mi: > 0.2 mm and <= 2.0 mm
    - pN1: > 2.0 mm (Macrometastasis)
    """
    if tumor_size_mm <= 0:
        return "pN0"
    elif tumor_size_mm <= 0.2:
        return "pN0(i+)"
    elif tumor_size_mm <= 2.0:
        return "pN1mi"
    else:
        return "pN1"

def aggregate_predictions(patch_preds, patch_size=256, pixel_size_microns=0.25):
    """
    Aggregates patch-level predictions to slide-level staging.
    """
    # Count positive patches
    positive_patches = np.sum(patch_preds > 0.5)
    
    # Calculate area in mm^2
    patch_area_mm2 = (patch_size * pixel_size_microns / 1000) ** 2
    total_tumor_area = positive_patches * patch_area_mm2
    
    # Estimate diameter
    diameter_mm = calculate_tumor_diameter(total_tumor_area)
    
    return get_pn_stage(diameter_mm), diameter_mm
