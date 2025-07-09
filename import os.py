import os
import numpy as np
import rasterio
from rasterio.enums import Resampling
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

# Set your root directory where data is stored
root = r'D:\ISRO Forest Fire\GEE_Train'

# Helper functions
def load_raster(path):
    with rasterio.open(path) as src:
        return src.read(1).astype(np.float32)

def resample_to_ref(src_path, ref_array):
    with rasterio.open(src_path) as src:
        data = src.read(
            out_shape=(1, ref_array.shape[0], ref_array.shape[1]),
            resampling=Resampling.nearest
        )[0].astype(np.float32)
    return data

def load_dynamic(feature_prefix, year):
    filename = f"{feature_prefix}_{year}.tif"
    return load_raster(os.path.join(root, filename))

# === Load static features ===
slope = load_raster(os.path.join(root, 'Slope.tif'))
aspect = load_raster(os.path.join(root, 'Aspect.tif'))
lulc = resample_to_ref(os.path.join(root, 'LULC.tif'), slope)

# Stack static features: shape (H, W, 3)
static_stack = np.stack([slope, aspect, lulc], axis=-1)
print("Static stack shape:", static_stack.shape)
