import numpy as np

def calculate_pn_stage(metastasis_size_mm):
    # TNM Staging Rules for Breast Cancer (pN)
    if metastasis_size_mm <= 0.2:
        return "pN0(i+)" # Isolated Tumor Cells
    elif 0.2 < metastasis_size_mm <= 2.0:
        return "pN1mi"   # Micrometastasis
    elif metastasis_size_mm > 2.0:
        return "pN1"     # Macrometastasis
    else:
        return "pN0"

def aggregate_slide_predictions(slide_preds, pixel_size_microns=0.25):
    # Logic to aggregate patch predictions into a metastasis size
    tumor_patches = np.sum(slide_preds)
    area_mm2 = tumor_patches * (256 * pixel_size_microns / 1000)**2
    # Simplified diameter estimation
    diameter_mm = 2 * np.sqrt(area_mm2 / np.pi)
    
    return calculate_pn_stage(diameter_mm)
