import os
import numpy as np
import rasterio
from rasterio.enums import Resampling

# Root directory where all TIFFs are stored
root = r'D:\ISRO Forest Fire\GEE_Train'

# Load reference layer to align all rasters
def load_raster(path):
    with rasterio.open(path) as src:
        return src.read(1).astype(np.float32)

# Static files
slope = load_raster(os.path.join(root, 'Slope.tif'))
aspect = load_raster(os.path.join(root, 'Aspect.tif'))
with rasterio.open(os.path.join(root, 'LULC.tif')) as src:
    lulc = src.read(
        out_shape=(1, slope.shape[0], slope.shape[1]),
        resampling=Resampling.nearest
    )[0].astype(np.float32)

# Stack static once
static_stack = np.stack([slope, aspect, lulc], axis=-1)
print("✅ Static stack ready:", static_stack.shape)

def prepare_year_stack(year, ref_array):
    dynamic_features = [
        ('NDVI', Resampling.bilinear),
        ('LSTDay', Resampling.bilinear),
        ('Temp2m', Resampling.bilinear),
        ('WindU', Resampling.bilinear),
        ('WindV', Resampling.bilinear),
        ('Train_Precip', Resampling.bilinear),
    ]
    dynamic_stack = []
    for prefix, method in dynamic_features:
        path = os.path.join(root, f"{prefix}_{year}.tif")
        if not os.path.exists(path):
            print(f"⚠️ Missing: {path}")
            return None
        with rasterio.open(path) as src:
            arr = src.read(
                out_shape=(1, ref_array.shape[0], ref_array.shape[1]),
                resampling=method
            )[0].astype(np.float32)
        dynamic_stack.append(arr)
    # Combine dynamic + static (H, W, C)
    full_stack = np.concatenate([np.stack(dynamic_stack, axis=-1), static_stack], axis=-1)
    return full_stack

def load_label(year, ref_array):
    path = os.path.join(root, f"FireMask_{year}.tif")
    with rasterio.open(path) as src:
        firemask = src.read(
            out_shape=(1, ref_array.shape[0], ref_array.shape[1]),
            resampling=Resampling.nearest
        )[0]
    label = (firemask >= 7).astype(np.uint8)
    return label

# Example Test
if __name__ == "__main__":
    year = 2017
    X = prepare_year_stack(year, slope)
    if X is not None:
        y = load_label(year, slope)
        print(f"✅ Year {year} -> Features: {X.shape}, Labels: {y.shape}")
    else:
        print(f"❌ Feature stack for year {year} could not be created (missing data).")
