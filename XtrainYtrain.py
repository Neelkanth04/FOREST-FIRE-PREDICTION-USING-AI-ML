import os
import numpy as np
import rasterio
from rasterio.enums import Resampling
from tqdm import tqdm

# 📁 Root directory for aligned rasters
root = r"D:/ISRO Forest Fire/GEE_Train"

# 🔁 Years to process
YEARS = [2017, 2018, 2019, 2020, 2021]

# 📌 Static rasters (same every year)
def load_raster(path):
    with rasterio.open(path) as src:
        return src.read(1).astype(np.float32)

slope = load_raster(os.path.join(root, 'Slope.tif'))
aspect = load_raster(os.path.join(root, 'Aspect.tif'))
with rasterio.open(os.path.join(root, 'LULC.tif')) as src:
    lulc = src.read(
        out_shape=(1, slope.shape[0], slope.shape[1]),
        resampling=Resampling.nearest
    )[0].astype(np.float32)

# Combine static into one stack (H, W, 3)
static_stack = np.stack([slope, aspect, lulc], axis=-1)
print("✅ Static stack ready:", static_stack.shape)

# 🧠 Function to load and align dynamic features for a year
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
    # Combine dynamic and static into full stack (H, W, C)
    full_stack = np.concatenate([np.stack(dynamic_stack, axis=-1), static_stack], axis=-1)
    return full_stack

# 🎯 Load fire labels for a year
def load_label(year, ref_array):
    path = os.path.join(root, f"FireMask_{year}.tif")
    if not os.path.exists(path):
        print(f"❌ FireMask missing: {path}")
        return None
    with rasterio.open(path) as src:
        firemask = src.read(
            out_shape=(1, ref_array.shape[0], ref_array.shape[1]),
            resampling=Resampling.nearest
        )[0]
    label = (firemask >= 7).astype(np.uint8)  # 🔥 Fire if value ≥ 7
    return label

# 🔁 Combine all years
X_all = []
y_all = []

for year in tqdm(YEARS, desc="📦 Building training dataset"):
    X_stack = prepare_year_stack(year, slope)
    if X_stack is None:
        continue
    y_mask = load_label(year, slope)
    if y_mask is None:
        continue

    H, W, C = X_stack.shape
    X_flat = X_stack.reshape(-1, C)
    y_flat = y_mask.reshape(-1)

    valid = np.all(np.isfinite(X_flat), axis=1)
    fire_valid = (y_flat == 0) | (y_flat == 1)
    mask = valid & fire_valid

    X_all.append(X_flat[mask])
    y_all.append(y_flat[mask])

# 🧱 Stack and save
X_train = np.vstack(X_all)
y_train = np.concatenate(y_all)

np.save(os.path.join(root, "X_train.npy"), X_train)
np.save(os.path.join(root, "y_train.npy"), y_train)

print("\n✅ Done! Training data saved:")
print("🔹 X_train shape:", X_train.shape)
print("🔹 y_train shape:", y_train.shape)
