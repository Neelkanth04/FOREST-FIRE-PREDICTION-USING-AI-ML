import os
import numpy as np
import rasterio
from rasterio.enums import Resampling
from sklearn.metrics import confusion_matrix, classification_report
import joblib

# === Define paths ===
base = "D:/ISRO Forest Fire/GEE_Train/"
model_path = os.path.join(base, "fire_risk_model.pkl")
output_prob_path = os.path.join(base, "fire_risk_2022_rf_prob.tif")
output_bin_path = os.path.join(base, "fire_risk_2022_rf_bin.tif")

feature_files = [
    "NDVI_2022.tif",
    "LSTDay_2022.tif",
    "Temp2m_2022.tif",
    "WindU_2022.tif",
    "WindV_2022.tif",
    "Train_Precip_2022.tif",
    "Slope.tif",
    "Aspect.tif",
    "LULC.tif"
]

print("📦 Loading and resampling features...")

# === Reference raster (Slope.tif) ===
ref_path = os.path.join(base, "Slope.tif")
with rasterio.open(ref_path) as ref:
    ref_profile = ref.profile
    ref_transform = ref.transform
    ref_crs = ref.crs
    ref_shape = (ref.height, ref.width)

arrays = []
for fname in feature_files:
    path = os.path.join(base, fname)
    with rasterio.open(path) as src:
        if (src.height, src.width) != ref_shape or src.crs != ref_crs:
            print(f"⚠️ Resampling {fname} to match Slope.tif")
            arr = src.read(1, out_shape=ref_shape, resampling=Resampling.bilinear)
        else:
            arr = src.read(1)
        arrays.append(arr)

# === Stack and flatten ===
stacked = np.stack(arrays, axis=-1)
print(f"✅ Stacked shape: {stacked.shape}")
H, W, C = stacked.shape
flat = stacked.reshape(-1, C).astype(np.float32)

# === Load model and predict ===
print("🧠 Loading model...")
model = joblib.load(model_path)
print("🔮 Predicting fire probabilities...")
probs = model.predict_proba(flat)[:, 1]

# === Save probability map ===
prob_img = probs.reshape(H, W).astype(np.float32)
ref_profile.update(dtype=rasterio.float32, count=1)
with rasterio.open(output_prob_path, "w", **ref_profile) as dst:
    dst.write(prob_img, 1)

# === Save binary map with default threshold 0.05 ===
binary = (probs >= 0.05).astype(np.uint8)
bin_img = binary.reshape(H, W)
ref_profile.update(dtype=rasterio.uint8, count=1)
with rasterio.open(output_bin_path, "w", **ref_profile) as dst:
    dst.write(bin_img, 1)

print("\n✅ Saved outputs:")
print(" • Probability map:", output_prob_path)
print(" • Binary map (0.05):", output_bin_path)

# === Load FireMask ground truth ===
print("\n📊 Evaluating multiple thresholds against FireMask_2022...")
with rasterio.open(os.path.join(base, "FireMask_2022.tif")) as src:
    y_true = src.read(1).flatten().astype(np.uint8)

# === Filter valid pixels (0 or 1 only) ===
mask = (y_true == 0) | (y_true == 1)
y_true = y_true[mask]
probs = probs[mask]

# === Evaluate thresholds ===
thresholds = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]
for thresh in thresholds:
    y_pred = (probs >= thresh).astype(np.uint8)
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, digits=4, zero_division=0)

    print(f"\n🔎 Threshold = {thresh:.2f}")
    print("Confusion Matrix:")
    print(cm)
    print(report)
